# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Unit tests for the nowcast.lightning.NowcastLightning plugin."""


import unittest
from datetime import datetime as dt

import numpy as np
from iris.coords import CellMethod
from iris.cube import Cube, CubeList
from iris.exceptions import ConstraintMismatchError, CoordinateNotFoundError
from iris.tests import IrisTest
from iris.util import squeeze

from improver.metadata.probabilistic import find_threshold_coordinate
from improver.nowcasting.lightning import NowcastLightning as Plugin
from improver.utilities.cube_checker import find_dimension_coordinate_mismatch

from ...set_up_test_cubes import set_up_probability_cube, set_up_variable_cube


def set_up_lightning_test_cubes(validity_time=dt(2015, 11, 23, 7),
                                fg_frt=dt(2015, 11, 23, 3),
                                grid_points=3):
    """Set up five cubes for testing nowcast lightning.

    The cube coordinates look like this:
        Dimension coordinates:
            projection_y_coordinate: grid_points;
            projection_x_coordinate: grid_points;
        Scalar coordinates:
            time: 2015-11-23 07:00:00
            forecast_reference_time: 2015-11-23 07:00:00
            forecast_period: 0 seconds

    Args:
        grid_points (int):
            Number of points along each spatial axis (square grid)
        validity_time (datetime.datetime):
            Time to use for test cubes
        fg_frt (datetime.datetime):
            Forecast reference time for first_guess_cube, which needs
            to have different forecast periods for different tests

    Returns:
        template_cube (iris.cube.Cube)
        first_guess_cube (iris.cube.Cube)
        lightning_rate_cube (iris.cube.Cube)
        prob_precip_cube (iris.cube.Cube):
            Has extra coordinate of length(3) "threshold" containing
            points [0.5, 7., 35.] mm h-1
        prob_vii_cube (iris.cube.Cube):
            Has extra coordinate of length(3) "threshold" containing
            points [0.5, 1., 2.] kg m-2
    """
    # template cube with metadata matching desired output
    data = np.ones((grid_points, grid_points), dtype=np.float32)
    template_cube = set_up_variable_cube(
        data.copy(), name='metadata_template', units=None,
        time=validity_time, frt=validity_time, spatial_grid='equalarea')

    # first guess lightning rate probability cube with flexible forecast
    # period (required for level 2 lighting risk index)
    prob_fg = np.array([data.copy()], dtype=np.float32)
    first_guess_cube = set_up_probability_cube(
        prob_fg, np.array([0], dtype=np.float32), threshold_units='s-1',
        variable_name='rate_of_lightning', time=validity_time, frt=fg_frt,
        spatial_grid='equalarea')
    first_guess_cube = squeeze(first_guess_cube)

    # lightning rate cube full of ones
    lightning_rate_cube = set_up_variable_cube(
        data.copy(), name='rate_of_lightning', units='min-1',
        time=validity_time, frt=validity_time, spatial_grid='equalarea')

    # probability of precip rate exceedance cube with higher rate probabilities
    # set to zero, and central point of low rate probabilities set to zero
    precip_data = np.ones((3, grid_points, grid_points), dtype=np.float32)
    precip_thresholds = np.array([0.5, 7.0, 35.0], dtype=np.float32)
    prob_precip_cube = set_up_probability_cube(
        precip_data, precip_thresholds,
        variable_name='lwe_precipitation_rate', threshold_units='mm h-1',
        time=validity_time, frt=validity_time, spatial_grid='equalarea')
    prob_precip_cube.data[0, 1, 1] = 0.
    prob_precip_cube.data[1:, ...] = 0.

    # probability of VII exceedance cube full of zeros
    vii_data = np.zeros((3, grid_points, grid_points), dtype=np.float32)
    vii_thresholds = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    prob_vii_cube = set_up_probability_cube(
        vii_data, vii_thresholds,
        variable_name='vertical_integral_of_ice', threshold_units='kg m-2',
        time=validity_time, frt=validity_time, spatial_grid='equalarea')

    return (template_cube, first_guess_cube, lightning_rate_cube,
            prob_precip_cube, prob_vii_cube)


class Test__init__(IrisTest):

    """Test the __init__ method accepts keyword arguments."""

    def test_with_radius(self):
        """
        Test that the radius keyword is accepted.
        """
        radius = 20000.
        plugin = Plugin(radius=radius)
        self.assertEqual(plugin.radius, radius)


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        plugin = Plugin()
        result = str(plugin)
        msg = ("""<NowcastLightning: radius={radius},
 lightning mapping (lightning rate in "min^-1"):
   upper: lightning rate {lthru} => min lightning prob {lprobu}
   lower: lightning rate {lthrl} => min lightning prob {lprobl}
>""".format(radius=10000.,
            lthru="<class 'function'>", lthrl=0.,
            lprobu=1., lprobl=0.25)
              )
        self.assertEqual(result, msg)


class Test__update_metadata(IrisTest):

    """Test the _update_metadata method."""

    def setUp(self):
        """Create a cube like this:
        probability_of_lwe_precipitation_rate_above_threshold / (1)
        Dimension coordinates:
            lwe_precipitation_rate: 1;
            projection_y_coordinate: 16;
            projection_x_coordinate: 16;
        Scalar coordinates:
            forecast_period: 14400 seconds
            forecast_reference_time: 2017-11-10 00:00:00
            time: 2017-11-10 04:00:00
        Cell methods:
            mean: realization

        The lwe_precipitation_rate coordinate will have the attribute:
            spp__relative_to_threshold: above.
        """
        data = np.ones((1, 16, 16), dtype=np.float32)
        thresholds = np.array([0.5], dtype=np.float32)
        self.cube = set_up_probability_cube(
            data, thresholds, variable_name='lwe_precipitation_rate',
            threshold_units='mm h-1')
        self.cube.add_cell_method(CellMethod('mean', coords='realization'))
        self.plugin = Plugin()

    def test_basic(self):
        """Test that the method returns the expected cube type
        and that the metadata are as expected.
        We expect a new name, the threshold coord to be removed and
        cell methods to be discarded."""
        result = self.plugin._update_metadata(self.cube)
        self.assertIsInstance(result, Cube)
        self.assertEqual(
            result.name(), "probability_of_rate_of_lightning_above_threshold")
        msg = "No threshold coord found"
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            find_threshold_coordinate(result)
        self.assertEqual(result.cell_methods, ())

    def test_input(self):
        """Test that the method does not modify the input cube data."""
        incube = self.cube.copy()
        self.plugin._update_metadata(incube)
        self.assertArrayAlmostEqual(incube.data, self.cube.data)
        self.assertEqual(incube.metadata, self.cube.metadata)

    def test_missing_threshold_coord(self):
        """Test that the method raises an error in Iris if the cube doesn't
        have a threshold coordinate to remove."""
        self.cube.remove_coord(find_threshold_coordinate(self.cube))
        msg = "No threshold coord found"
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            self.plugin._update_metadata(self.cube)


class Test__modify_first_guess(IrisTest):

    """Test the _modify_first_guess method."""

    def setUp(self):
        """Create test cubes and plugin instance.
        The cube coordinates look like this:
        Dimension coordinates:
            projection_y_coordinate: 3;
            projection_x_coordinate: 3;
        Scalar coordinates:
            time: 2015-11-23 07:00:00
            forecast_reference_time: 2015-11-23 07:00:00
            forecast_period: 0 seconds

        self.cube:
            Metadata describes the nowcast lightning fields to be calculated.
            forecast_period: 0 seconds (simulates nowcast data)
        self.fg_cube:
            Has 4 hour forecast period, to test impact at lr2
        self.ltng_cube:
            forecast_period: 0 seconds (simulates nowcast data)
        self.precip_cube:
            Has extra coordinate of length(3) "threshold" containing
            points [0.5, 7., 35.] mm h-1.
        self.vii_cube:
            Has extra coordinate of length(3) "threshold" containing
            points [0.5, 1., 2.] kg m-2.
        """
        (self.cube, self.fg_cube, self.ltng_cube, self.precip_cube,
            self.vii_cube) = set_up_lightning_test_cubes()
        self.plugin = Plugin()

    def test_basic(self):
        """Test that the method returns the expected cube type"""
        result = self.plugin._modify_first_guess(self.cube,
                                                 self.fg_cube,
                                                 self.ltng_cube,
                                                 self.precip_cube,
                                                 self.vii_cube)
        self.assertIsInstance(result, Cube)

    def test_input_with_vii(self):
        """Test that the method does not modify the input cubes."""
        cube_a = self.cube.copy()
        cube_b = self.fg_cube.copy()
        cube_c = self.ltng_cube.copy()
        cube_d = self.precip_cube.copy()
        cube_e = self.vii_cube.copy()
        self.plugin._modify_first_guess(cube_a, cube_b, cube_c, cube_d, cube_e)
        self.assertArrayAlmostEqual(cube_a.data, self.cube.data)
        self.assertArrayAlmostEqual(cube_b.data, self.fg_cube.data)
        self.assertArrayAlmostEqual(cube_c.data, self.ltng_cube.data)
        self.assertArrayAlmostEqual(cube_d.data, self.precip_cube.data)
        self.assertArrayAlmostEqual(cube_e.data, self.vii_cube.data)

    def test_missing_lightning(self):
        """Test that the method raises an error if the lightning cube doesn't
        match the meta-data cube time coordinate."""
        self.ltng_cube.coord('time').points = [1.0]
        msg = ("No matching lightning cube for")
        with self.assertRaisesRegex(ConstraintMismatchError, msg):
            self.plugin._modify_first_guess(self.cube,
                                            self.fg_cube,
                                            self.ltng_cube,
                                            self.precip_cube,
                                            None)

    def test_missing_first_guess(self):
        """Test that the method raises an error if the first-guess cube doesn't
        match the meta-data cube time coordinate."""
        self.fg_cube.coord('time').points = [1.0]
        msg = ("is not available within the input cube within the "
               "allowed difference")
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin._modify_first_guess(self.cube,
                                            self.fg_cube,
                                            self.ltng_cube,
                                            self.precip_cube,
                                            None)

    def test_cube_has_no_time_coord(self):
        """Test that the method raises an error if the meta-data cube has no
        time coordinate."""
        self.cube.remove_coord('time')
        msg = ("Expected to find exactly 1 time coordinate, but found none.")
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            self.plugin._modify_first_guess(self.cube,
                                            self.fg_cube,
                                            self.ltng_cube,
                                            self.precip_cube,
                                            None)

    def test_precip_zero(self):
        """Test that apply_precip is being called"""
        # Set lightning data to "no-data" so it has a Null impact
        self.ltng_cube.data = np.full_like(self.ltng_cube.data, -1.)
        # No halo - we're only testing this method.
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[1, 1] = 0.0067
        result = self.plugin._modify_first_guess(self.cube,
                                                 self.fg_cube,
                                                 self.ltng_cube,
                                                 self.precip_cube,
                                                 None)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_vii_large(self):
        """Test that ApplyIce is being called"""
        # Set lightning data to zero so it has a Null impact
        self.vii_cube.data[:, 1, 1] = 1.
        self.ltng_cube.data[1, 1] = -1.
        self.fg_cube.data[1, 1] = 0.
        # No halo - we're only testing this method.
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[1, 1] = 0.9
        result = self.plugin._modify_first_guess(self.cube,
                                                 self.fg_cube,
                                                 self.ltng_cube,
                                                 self.precip_cube,
                                                 self.vii_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_null(self):
        """Test that large precip probs and -1 lrates have no impact"""
        # Set prob(precip) data for lowest threshold to to 0.1, the highest
        # value that has no impact.
        self.precip_cube.data[0, 1, 1] = 0.1
        # Set lightning data to -1 so it has a Null impact
        self.ltng_cube.data = np.full_like(self.ltng_cube.data, -1.)
        # No halo - we're only testing this method.
        expected = self.fg_cube.copy()
        # expected.data should be an unchanged copy of fg_cube.
        result = self.plugin._modify_first_guess(self.cube,
                                                 self.fg_cube,
                                                 self.ltng_cube,
                                                 self.precip_cube,
                                                 None)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_lrate_large(self):
        """Test that large lightning rates increase zero lightning risk"""
        expected = self.fg_cube.copy()
        # expected.data contains all ones
        # Set prob(precip) data for lowest threshold to to 1., so it has a Null
        # impact when lightning is present.
        self.precip_cube.data[0, 1, 1] = 1.
        # Set first-guess data zero point that will be increased
        self.fg_cube.data[1, 1] = 0.
        # No halo - we're only testing this method.
        result = self.plugin._modify_first_guess(self.cube,
                                                 self.fg_cube,
                                                 self.ltng_cube,
                                                 self.precip_cube,
                                                 None)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_lrate_large_shortfc(self):
        """Test that nearly-large lightning rates increases zero lightning risk
        when forecast_period is non-zero"""
        expected = self.fg_cube.copy()
        # expected.data contains all ones
        # Set precip data to 1. so it has a Null impact
        # Set prob(precip) data for lowest threshold to to 1., so it has a Null
        # impact when lightning is present.
        self.precip_cube.data[0, 1, 1] = 1.
        # Test the impact of the lightning-rate function.
        # A highish lightning value at one-hour lead time isn't high enough to
        # get to the high lightning category.
        self.ltng_cube.data[1, 1] = 0.8
        self.cube.coord('forecast_period').points = [3600.]  # seconds
        # Set first-guess data zero point that will be increased
        self.fg_cube.data[1, 1] = 0.
        # This time, lightning probability increases only to 0.25, not 1.
        expected.data[1, 1] = 0.25
        # No halo - we're only testing this method.
        result = self.plugin._modify_first_guess(self.cube,
                                                 self.fg_cube,
                                                 self.ltng_cube,
                                                 self.precip_cube,
                                                 None)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_lrate_large_null(self):
        """Test that large lightning rates do not increase high lightning
        risk"""
        expected = self.fg_cube.copy()
        # expected.data contains all ones
        # Set precip data to 1. so it has a Null impact
        # Set prob(precip) data for lowest threshold to to 1., so it has a Null
        # impact when lightning is present.
        self.precip_cube.data[0, 1, 1] = 1.
        # Set first-guess data zero point that will be increased
        self.fg_cube.data[1, 1] = 1.
        # No halo - we're only testing this method.
        result = self.plugin._modify_first_guess(self.cube,
                                                 self.fg_cube,
                                                 self.ltng_cube,
                                                 self.precip_cube,
                                                 None)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_lrate_small(self):
        """Test that lightning nearby (encoded as lightning rate zero)
        increases lightning risk at point"""
        # Set prob(precip) data for lowest threshold to to 1., so it has a Null
        # impact when lightning is present.
        self.precip_cube.data[0, 1, 1] = 1.
        # Set lightning data to zero to represent the data halo
        self.ltng_cube.data[1, 1] = 0.
        # Set first-guess data zero point that will be increased
        self.fg_cube.data[1, 1] = 0.
        # No halo - we're only testing this method.
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[1, 1] = 0.25
        result = self.plugin._modify_first_guess(self.cube,
                                                 self.fg_cube,
                                                 self.ltng_cube,
                                                 self.precip_cube,
                                                 None)
        self.assertArrayAlmostEqual(result.data, expected.data)


class Test_apply_precip(IrisTest):

    """Test the apply_precip method."""

    def setUp(self):
        """Create test cubes and plugin instance.
        The cube coordinates look like this:
        Dimension coordinates:
            projection_y_coordinate: 3;
            projection_x_coordinate: 3;
        Scalar coordinates:
            time: 2015-11-23 07:00:00
            forecast_reference_time: 2015-11-23 07:00:00
            forecast_period: 0 seconds

        self.fg_cube:
            Has 4 hour forecast period
        self.precip_cube:
            Has extra coordinate of length(3) "threshold" containing
            points [0.5, 7., 35.] mm h-1.
        """
        (_, self.fg_cube, _, self.precip_cube, _) = (
            set_up_lightning_test_cubes())
        self.plugin = Plugin()
        self.precip_threshold_coord = find_threshold_coordinate(
            self.precip_cube)

    def test_basic(self):
        """Test that the method returns the expected cube type"""
        result = self.plugin.apply_precip(self.fg_cube, self.precip_cube)
        self.assertIsInstance(result, Cube)

    def test_input(self):
        """Test that the method does not modify the input cubes."""
        cube_a = self.fg_cube.copy()
        cube_b = self.precip_cube.copy()
        self.plugin.apply_precip(cube_a, cube_b)
        self.assertArrayAlmostEqual(cube_a.data, self.fg_cube.data)
        self.assertArrayAlmostEqual(cube_b.data, self.precip_cube.data)

    def test_nearby_threshold_low(self):
        """Test that the method accepts a threshold point within machine
        tolerance."""
        self.precip_threshold_coord.points = [0.5000000001, 7., 35.]
        self.plugin.apply_precip(self.fg_cube, self.precip_cube)

    def test_missing_threshold_low(self):
        """Test that the method raises an error if the precip_cube doesn't
        have a threshold coordinate for 0.5."""
        self.precip_threshold_coord.points = [1.0, 7., 35.]
        msg = ("No matching any precip cube for")
        with self.assertRaisesRegex(ConstraintMismatchError, msg):
            self.plugin.apply_precip(self.fg_cube, self.precip_cube)

    def test_missing_threshold_mid(self):
        """Test that the method raises an error if the precip_cube doesn't
        have a threshold coordinate for 7.0."""
        self.precip_threshold_coord.points = [0.5, 8., 35.]
        msg = ("No matching high precip cube for")
        with self.assertRaisesRegex(ConstraintMismatchError, msg):
            self.plugin.apply_precip(self.fg_cube, self.precip_cube)

    def test_missing_threshold_high(self):
        """Test that the method raises an error if the precip_cube doesn't
        have a threshold coordinate for 35.0."""
        self.precip_threshold_coord.points = [0.5, 7., 20.]
        msg = ("No matching intense precip cube for")
        with self.assertRaisesRegex(ConstraintMismatchError, msg):
            self.plugin.apply_precip(self.fg_cube, self.precip_cube)

    def test_precip_zero(self):
        """Test that zero precip probs reduce high lightning risk a lot"""
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[1, 1] = 0.0067
        result = self.plugin.apply_precip(self.fg_cube, self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_precip_small(self):
        """Test that small precip probs reduce high lightning risk a bit"""
        self.precip_cube.data[:, 1, 1] = 0.
        self.precip_cube.data[0, 1, 1] = 0.075
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[1, 1] = 0.625
        result = self.plugin.apply_precip(self.fg_cube, self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_precip_light(self):
        """Test that high probs of light precip do not reduce high lightning
        risk"""
        self.precip_cube.data[:, 1, 1] = 0.
        self.precip_cube.data[0, 1, 1] = 0.8
        expected = self.fg_cube.copy()
        # expected.data contains all ones
        result = self.plugin.apply_precip(self.fg_cube, self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_precip_heavy(self):
        """Test that prob of heavy precip increases zero lightning risk"""
        self.precip_cube.data[0, 1, 1] = 1.0
        self.precip_cube.data[1, 1, 1] = 0.5
        # Set first-guess to zero
        self.fg_cube.data[1, 1] = 0.0
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[1, 1] = 0.25
        result = self.plugin.apply_precip(self.fg_cube, self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_precip_heavy_null(self):
        """Test that low prob of heavy precip does not increase
        low lightning risk"""
        self.precip_cube.data[0, 1, 1] = 1.0
        self.precip_cube.data[1, 1, 1] = 0.3
        # Set first-guess to zero
        self.fg_cube.data[1, 1] = 0.1
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[1, 1] = 0.1
        result = self.plugin.apply_precip(self.fg_cube, self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_precip_intense(self):
        """Test that prob of intense precip increases zero lightning risk"""
        expected = self.fg_cube.copy()
        # expected.data contains all ones
        self.precip_cube.data[0, 1, 1] = 1.0
        self.precip_cube.data[1, 1, 1] = 1.0
        self.precip_cube.data[2, 1, 1] = 0.5
        # Set first-guess to zero
        self.fg_cube.data[1, 1] = 0.0
        result = self.plugin.apply_precip(self.fg_cube, self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_precip_intense_null(self):
        """Test that low prob of intense precip does not increase
        low lightning risk"""
        self.precip_cube.data[0, 1, 1] = 1.0
        self.precip_cube.data[1, 1, 1] = 1.0
        self.precip_cube.data[2, 1, 1] = 0.1
        # Set first-guess to zero
        self.fg_cube.data[1, 1] = 0.1
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[1, 1] = 0.25  # Heavy-precip result only
        result = self.plugin.apply_precip(self.fg_cube, self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)


class Test_apply_ice(IrisTest):

    """Test the apply_ice method."""

    def setUp(self):
        """Create test cubes and plugin instance.
        The cube coordinates look like this:
        Dimension coordinates:
            projection_y_coordinate: 3;
            projection_x_coordinate: 3;
        Scalar coordinates:
            time: 2015-11-23 07:00:00
            forecast_reference_time: 2015-11-23 07:00:00
            forecast_period: 0 seconds

        self.vii_cube:
            Has extra coordinate of length(3) "threshold" containing
            points [0.5, 1., 2.] kg m-2.
        """
        (_, self.fg_cube, _, _, self.ice_cube) = (
            set_up_lightning_test_cubes(validity_time=dt(2015, 11, 23, 7),
                                        fg_frt=dt(2015, 11, 23, 7)))
        self.plugin = Plugin()
        self.ice_threshold_coord = find_threshold_coordinate(self.ice_cube)

    def test_basic(self):
        """Test that the method returns the expected cube type"""
        result = self.plugin.apply_ice(self.fg_cube, self.ice_cube)
        self.assertIsInstance(result, Cube)

    def test_input(self):
        """Test that the method does not modify the input cubes."""
        cube_a = self.fg_cube.copy()
        cube_b = self.ice_cube.copy()
        self.plugin.apply_ice(cube_a, cube_b)
        self.assertArrayAlmostEqual(cube_a.data, self.fg_cube.data)
        self.assertArrayAlmostEqual(cube_b.data, self.ice_cube.data)

    def test_missing_threshold_low(self):
        """Test that the method raises an error if the ice_cube doesn't
        have a threshold coordinate for 0.5."""
        self.ice_threshold_coord.points = [0.4, 1., 2.]
        msg = (r"No matching prob\(Ice\) cube for threshold 0.5")
        with self.assertRaisesRegex(ConstraintMismatchError, msg):
            self.plugin.apply_ice(self.fg_cube, self.ice_cube)

    def test_missing_threshold_mid(self):
        """Test that the method raises an error if the ice_cube doesn't
        have a threshold coordinate for 1.0."""
        self.ice_threshold_coord.points = [0.5, 0.9, 2.]
        msg = (r"No matching prob\(Ice\) cube for threshold 1.")
        with self.assertRaisesRegex(ConstraintMismatchError, msg):
            self.plugin.apply_ice(self.fg_cube, self.ice_cube)

    def test_missing_threshold_high(self):
        """Test that the method raises an error if the ice_cube doesn't
        have a threshold coordinate for 2.0."""
        self.ice_threshold_coord.points = [0.5, 1., 4.]
        msg = (r"No matching prob\(Ice\) cube for threshold 2.")
        with self.assertRaisesRegex(ConstraintMismatchError, msg):
            self.plugin.apply_ice(self.fg_cube, self.ice_cube)

    def test_ice_null(self):
        """Test that small VII probs do not increase moderate lightning risk"""
        self.ice_cube.data[:, 1, 1] = 0.
        self.ice_cube.data[0, 1, 1] = 0.5
        self.fg_cube.data[1, 1] = 0.25
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[1, 1] = 0.25
        result = self.plugin.apply_ice(self.fg_cube,
                                       self.ice_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_ice_zero(self):
        """Test that zero VII probs do not increase zero lightning risk"""
        self.ice_cube.data[:, 1, 1] = 0.
        self.fg_cube.data[1, 1] = 0.
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[1, 1] = 0.
        result = self.plugin.apply_ice(self.fg_cube,
                                       self.ice_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_ice_small(self):
        """Test that small VII probs do increase zero lightning risk"""
        self.ice_cube.data[:, 1, 1] = 0.
        self.ice_cube.data[0, 1, 1] = 0.5
        self.fg_cube.data[1, 1] = 0.
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[1, 1] = 0.05
        result = self.plugin.apply_ice(self.fg_cube,
                                       self.ice_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_ice_large(self):
        """Test that large VII probs do increase zero lightning risk"""
        self.ice_cube.data[:, 1, 1] = 1.
        self.fg_cube.data[1, 1] = 0.
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[1, 1] = 0.9
        result = self.plugin.apply_ice(self.fg_cube,
                                       self.ice_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_ice_large_with_fc(self):
        """Test that large VII probs do increase zero lightning risk when
        forecast lead time is non-zero (three forecast_period points)"""
        self.ice_cube.data[:, 1, 1] = 1.
        self.fg_cube.data[1, 1] = 0.
        frt_point = self.fg_cube.coord('forecast_reference_time').points[0]
        fg_cube_input = CubeList([])
        for fc_time in np.array([1, 2.5, 3]) * 3600:  # seconds
            fg_cube_next = self.fg_cube.copy()
            fg_cube_next.coord('time').points = [frt_point + fc_time]
            fg_cube_next.coord('forecast_period').points = [fc_time]
            fg_cube_input.append(squeeze(fg_cube_next))
        fg_cube_input = fg_cube_input.merge_cube()
        expected = fg_cube_input.copy()
        # expected.data contains all ones except:
        expected.data[0, 1, 1] = 0.54
        expected.data[1, 1, 1] = 0.0
        expected.data[2, 1, 1] = 0.0
        result = self.plugin.apply_ice(fg_cube_input,
                                       self.ice_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)


class Test_process(IrisTest):

    """Test the nowcast lightning plugin."""

    def setUp(self):
        """Create test cubes and plugin instance.
        The cube coordinates look like this:
        Dimension coordinates:
            projection_y_coordinate: 16;
            projection_x_coordinate: 16;
        Scalar coordinates:
            time: 2015-11-23 07:00:00
            forecast_reference_time: 2015-11-23 07:00:00
            forecast_period: 0 seconds

        self.fg_cube:
            Has 4 hour forecast period, to test impact on "lightning risk
            2 level" output (see improver.nowcasting.lightning for details)
        self.ltng_cube:
            forecast_period: 0 seconds (simulates nowcast data)
        self.precip_cube:
            Has extra coordinate of length(3) "threshold" containing
            points [0.5, 7., 35.] mm h-1.  Has a 4 hour forecast period.
        self.vii_cube:
            Has extra coordinate of length(3) "threshold" containing
            points [0.5, 1., 2.] kg m-2.
        """
        (_, self.fg_cube, self.ltng_cube, self.precip_cube,
            self.vii_cube) = set_up_lightning_test_cubes(grid_points=16)
        # reset some data and give precip cube a 4 hour forecast period
        self.precip_cube.data[0, ...] = 1.
        self.precip_cube.coord("forecast_period").points = [4*3600.]

        # sort out spatial coordinates - need smaller grid length (set to 2 km)
        for cube in [self.fg_cube, self.ltng_cube,
                     self.precip_cube, self.vii_cube]:
            points_array = 2000.*np.arange(16).astype(np.float32)
            cube.coord(axis='x').points = points_array
            cube.coord(axis='y').points = points_array
        self.plugin = Plugin()

    def set_up_vii_input_output(self):
        """Used to modify setUp() to set up four standard VII tests."""

        # Repeat all tests relating to vii from Test__modify_first_guess
        expected = self.fg_cube.copy()
        # expected.data contains all ones except where modified below:

        # Set up precip_cube with increasing intensity along x-axis
        # y=5; no precip
        self.precip_cube.data[:, 5:9, 5] = 0.
        # y=6; light precip
        self.precip_cube.data[0, 5:9, 6] = 0.1
        self.precip_cube.data[1, 5:9, 6] = 0.
        # y=7; heavy precip
        self.precip_cube.data[:2, 5:9, 7] = 1.
        self.precip_cube.data[2, 5:9, 7] = 0.
        # y=8; intense precip
        self.precip_cube.data[:, 5:9, 8] = 1.

        # test_vii_null - with lightning-halo
        self.vii_cube.data[:, 5, 5:9] = 0.
        self.vii_cube.data[0, 5, 5:9] = 0.5
        self.ltng_cube.data[5, 5:9] = 0.
        self.fg_cube.data[5, 5:9] = 0.
        expected.data[5, 5:9] = [0.05, 0.25, 0.25, 1.]

        # test_vii_zero
        self.vii_cube.data[:, 6, 5:9] = 0.
        self.ltng_cube.data[6, 5:9] = -1.
        self.fg_cube.data[6, 5:9] = 0.
        expected.data[6, 5:9] = [0., 0., 0.25, 1.]

        # test_vii_small
        # Set lightning data to -1 so it has a Null impact
        self.vii_cube.data[:, 7, 5:9] = 0.
        self.vii_cube.data[0, 7, 5:9] = 0.5
        self.ltng_cube.data[7, 5:9] = -1.
        self.fg_cube.data[7, 5:9] = 0.
        expected.data[7, 5:9] = [0.05, 0.05, 0.25, 1.]

        # test_vii_large
        # Set lightning data to -1 so it has a Null impact
        self.vii_cube.data[:, 8, 5:9] = 1.
        self.ltng_cube.data[8, 5:9] = -1.
        self.fg_cube.data[8, 5:9] = 0.
        expected.data[8, 5:9] = [0.9, 0.9, 0.9, 1.]
        return expected

    def test_basic(self):
        """Test that the method returns the expected cube type with coords"""
        result = self.plugin(CubeList([
            self.fg_cube,
            self.ltng_cube,
            self.precip_cube]))
        self.assertIsInstance(result, Cube)
        # We expect the threshold coordinate to have been removed.
        threshold_coord = find_threshold_coordinate(self.precip_cube).name()
        self.assertCountEqual(find_dimension_coordinate_mismatch(
                result, self.precip_cube), [threshold_coord])
        self.assertEqual(
            result.name(), 'probability_of_rate_of_lightning_above_threshold')
        self.assertEqual(result.units, '1')

    def test_basic_with_vii(self):
        """Test that the method returns the expected cube type when vii is
        present"""
        result = self.plugin(CubeList([
            self.fg_cube,
            self.ltng_cube,
            self.precip_cube,
            self.vii_cube]))
        self.assertIsInstance(result, Cube)
        # We expect the threshold coordinate to have been removed.
        threshold_coord = find_threshold_coordinate(self.precip_cube).name()
        self.assertCountEqual(find_dimension_coordinate_mismatch(
                result, self.precip_cube), [threshold_coord])
        self.assertEqual(
            result.name(), 'probability_of_rate_of_lightning_above_threshold')
        self.assertEqual(result.units, '1')

    def test_no_first_guess_cube(self):
        """Test that the method raises an error if the first_guess cube is
        omitted from the cubelist"""
        msg = (r"Got 0 cubes for constraint Constraint\(name=\'probability_of_"
               r"rate_of_lightning_above_threshold\'\), expecting 1.")
        with self.assertRaisesRegex(ConstraintMismatchError, msg):
            self.plugin(CubeList([
                self.ltng_cube,
                self.precip_cube]))

    def test_no_lightning_cube(self):
        """Test that the method raises an error if the lightning cube is
        omitted from the cubelist"""
        msg = (r"Got 0 cubes for constraint Constraint\(name=\'rate_of_"
               r"lightning\'\), expecting 1.")
        with self.assertRaisesRegex(ConstraintMismatchError, msg):
            self.plugin(CubeList([
                self.fg_cube,
                self.precip_cube]))

    def test_no_precip_cube(self):
        """Test that the method raises an error if the precip cube is
        omitted from the cubelist"""
        msg = (r"Got 0 cubes for constraint Constraint\(name=\'probability_of_"
               r"lwe_precipitation_rate_above_threshold\'\), expecting 1.")
        with self.assertRaisesRegex(ConstraintMismatchError, msg):
            self.plugin(CubeList([
                self.fg_cube,
                self.ltng_cube]))

    def test_precip_has_no_thresholds(self):
        """Test that the method raises an error if the threshold coord is
        omitted from the precip_cube"""
        threshold_coord = find_threshold_coordinate(self.precip_cube)
        self.precip_cube.remove_coord(threshold_coord)
        msg = "No threshold coord found"
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            self.plugin(CubeList([
                self.fg_cube,
                self.ltng_cube,
                self.precip_cube]))

    def test_result_with_vii(self):
        """Test that the method returns the expected data when vii is
        present"""
        # Set precip_cube forecast period to be zero.
        self.precip_cube.coord('forecast_period').points = [0.]
        expected = self.set_up_vii_input_output()

        # No halo - we're only testing this method.
        # 2000m is the grid-length, so halo includes only one pixel.
        plugin = Plugin(2000.)
        result = plugin(CubeList([
            self.fg_cube,
            self.ltng_cube,
            self.precip_cube,
            self.vii_cube]))
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_result_with_vii_longfc(self):
        """Test that the method returns the expected data when vii is
        present and forecast time is 4 hours"""
        expected = self.set_up_vii_input_output()

        # test_vii_null with no precip will now return 0.0067
        expected.data[5, 5] = 0.0067

        # test_vii_small with no and light precip will now return zero
        expected.data[7, 5:7] = 0.

        # test_vii_large with no and light precip now return zero
        # and 0.25 for heavy precip
        expected.data[8, 5:8] = [0., 0., 0.25]
        # No halo - we're only testing this method.
        # 2000m is the grid-length, so halo includes only one pixel.
        plugin = Plugin(2000.)
        result = plugin(CubeList([
            self.fg_cube,
            self.ltng_cube,
            self.precip_cube,
            self.vii_cube]))
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, expected.data)


if __name__ == '__main__':
    unittest.main()
