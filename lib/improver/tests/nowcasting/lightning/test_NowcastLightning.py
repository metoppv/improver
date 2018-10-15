# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
from iris.util import squeeze
from iris.coords import DimCoord, CellMethod
from iris.cube import Cube, CubeList
from iris.tests import IrisTest
from iris.exceptions import CoordinateNotFoundError, ConstraintMismatchError
import numpy as np
import cf_units

from improver.nowcasting.lightning import NowcastLightning as Plugin
from improver.utilities.cube_checker import find_dimension_coordinate_mismatch
from improver.tests.nbhood.nbhood.test_BaseNeighbourhoodProcessing import (
    set_up_cube, set_up_cube_with_no_realizations)
from improver.tests.ensemble_calibration.ensemble_calibration.helper_functions\
    import add_forecast_reference_time_and_forecast_period


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
        """Create a cube with a single non-zero point like this:
        precipitation_amount / (kg m^-2)
        Dimension coordinates:
            realization: 1;
            time: 1;
            projection_y_coordinate: 3;
            projection_x_coordinate: 3;
        Auxiliary coordinates:
            forecast_period (on time coord): 4.0 hours
        Scalar coordinates:
            forecast_reference_time: 2015-11-23 03:00:00
            threshold: 0.5 mm hr-1
        Data:
            All points contain float(1.) except the
            zero point [0, 0, 1, 1] which is float(0.)
        """
        self.cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube())
        coord = DimCoord(0.5, long_name="threshold", units='mm hr^-1')
        self.cube.add_aux_coord(coord)
        self.cube.add_cell_method(CellMethod('mean', coords='realization'))
        self.plugin = Plugin()

    def test_basic(self):
        """Test that the method returns the expected cube type
        and that the metadata are as expected.
        We expect a new name, the threshold coord to be removed and
        cell methods to be discarded."""
        result = self.plugin._update_metadata(self.cube)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), "probability_of_lightning")
        msg = ("Expected to find exactly 1 threshold coordinate, but found "
               "none.")
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            result.coord('threshold')
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
        self.cube.remove_coord('threshold')
        msg = ("Expected to find exactly 1 threshold coordinate, but found no")
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            self.plugin._update_metadata(self.cube)


class Test__modify_first_guess(IrisTest):

    """Test the _modify_first_guess method."""

    def setUp(self):
        """Create cubes with a single zero prob(precip) point.
        The cubes look like this:
        precipitation_amount / (kg m^-2)
        Dimension coordinates:
            time: 1;
            projection_y_coordinate: 3;
            projection_x_coordinate: 3;
        Auxiliary coordinates:
            forecast_period (on time coord): 4.0 hours (simulates UM data)
        Scalar coordinates:
            forecast_reference_time: 2015-11-23 03:00:00
        Data:
        self.cube:
            Describes the nowcast fields to be calculated.
            forecast_period (on time coord): 0.0 hours (simulates nowcast data)
            All points contain float(1.) except the
            zero point [0, 1, 1] which is float(0.)
        self.fg_cube:
            All points contain float(1.)
        self.ltng_cube:
            forecast_period (on time coord): 0.0 hours (simulates nowcast data)
            All points contain float(1.)
        self.precip_cube:
            With extra coordinate of length(3) "threshold" containing
            points [0.5, 7., 35.] mm hr-1.
            All points contain float(1.) except the
            zero point [0, 0, 1, 1] which is float(0.)
            and [1:, 0, ...] which are float(0.)
        self.vii_cube:
            With extra coordinate of length(3) "threshold" containing
            points [0.5, 1., 2.] kg m^-2.
            forecast_period (on time coord): 0.0 hours (simulates nowcast data)
            Time and forecast_period dimensions "sqeezed" to be Scalar coords.
            All points contain float(0.)
        """
        self.cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube_with_no_realizations(zero_point_indices=((0, 1, 1),),
                                             num_grid_points=3), fp_point=0.0)
        self.fg_cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube_with_no_realizations(zero_point_indices=[],
                                             num_grid_points=3))
        self.ltng_cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube_with_no_realizations(zero_point_indices=[],
                                             num_grid_points=3),
            fp_point=0.0)
        self.precip_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_cube(num_realization_points=3,
                            zero_point_indices=((0, 1, 1),),
                            num_grid_points=3), fp_point=0.0))
        threshold_coord = self.precip_cube.coord('realization')
        threshold_coord.points = [0.5, 7.0, 35.0]
        threshold_coord.rename('threshold')
        threshold_coord.units = cf_units.Unit('mm hr-1')
        self.precip_cube.data[1:, 0, ...] = 0.
        # iris.util.queeze is applied here to demote the singular coord "time"
        # to a scalar coord.
        self.vii_cube = squeeze(
            add_forecast_reference_time_and_forecast_period(
                set_up_cube(num_realization_points=3,
                            zero_point_indices=[],
                            num_grid_points=3),
                fp_point=0.0))
        threshold_coord = self.vii_cube.coord('realization')
        threshold_coord.points = [0.5, 1.0, 2.0]
        threshold_coord.rename('threshold')
        threshold_coord.units = cf_units.Unit('kg m^-2')
        self.vii_cube.data = np.zeros_like(self.vii_cube.data)
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
        msg = ("No matching first-guess cube for")
        with self.assertRaisesRegex(ConstraintMismatchError, msg):
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
        expected.data[0, 1, 1] = 0.0067
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
        self.ltng_cube.data[0, 1, 1] = -1.
        self.fg_cube.data[0, 1, 1] = 0.
        # No halo - we're only testing this method.
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[0, 1, 1] = 0.9
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
        self.precip_cube.data[0, 0, 1, 1] = 0.1
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
        # Set precip data to 1. so it has a Null impact
        # Set prob(precip) data for lowest threshold to to 1., so it has a Null
        # impact when lightning is present.
        self.precip_cube.data[0, 0, 1, 1] = 1.
        # Set first-guess data zero point that will be increased
        self.fg_cube.data[0, 1, 1] = 0.
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
        self.precip_cube.data[0, 0, 1, 1] = 1.
        # Test the impact of the lightning-rate function.
        # A highish lightning value at one-hour lead time isn't high enough to
        # get to the high lightning category.
        self.ltng_cube.data[0, 1, 1] = 0.8
        self.cube.coord('forecast_period').points = [1.]  # hours
        # Set first-guess data zero point that will be increased
        self.fg_cube.data[0, 1, 1] = 0.
        # This time, lightning probability increases only to 0.25, not 1.
        expected.data[0, 1, 1] = 0.25
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
        self.precip_cube.data[0, 0, 1, 1] = 1.
        # Set first-guess data zero point that will be increased
        self.fg_cube.data[0, 1, 1] = 1.
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
        self.precip_cube.data[0, 0, 1, 1] = 1.
        # Set lightning data to zero to represent the data halo
        self.ltng_cube.data[0, 1, 1] = 0.
        # Set first-guess data zero point that will be increased
        self.fg_cube.data[0, 1, 1] = 0.
        # No halo - we're only testing this method.
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[0, 1, 1] = 0.25
        result = self.plugin._modify_first_guess(self.cube,
                                                 self.fg_cube,
                                                 self.ltng_cube,
                                                 self.precip_cube,
                                                 None)
        self.assertArrayAlmostEqual(result.data, expected.data)


class Test_apply_precip(IrisTest):

    """Test the apply_precip method."""

    def setUp(self):
        """Create cubes with a single zero prob(precip) point.
        The cubes look like this:
        precipitation_amount / (kg m^-2)
        Dimension coordinates:
            time: 1;
            projection_y_coordinate: 3;
            projection_x_coordinate: 3;
        Auxiliary coordinates:
            forecast_period (on time coord): 4.0 hours (simulates UM data)
        Scalar coordinates:
            forecast_reference_time: 2015-11-23 03:00:00
        Data:
        self.fg_cube:
            forecast_period (on time coord): 0.0 hours (simulates nowcast data)
            All points contain float(1.)
            Cube name is "probability_of_lightning".
        self.precip_cube:
            With extra coordinate of length(3) "threshold" containing
            points [0.5, 7., 35.] mm hr-1.
            All points contain float(1.) except the
            zero point [0, 0, 1, 1] which is float(0.)
            and [1:, 0, ...] which are float(0.)
            Cube name is "probability_of_precipitation".
            Cube has added attribute {'relative_to_threshold': 'above'}
        """
        self.fg_cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube_with_no_realizations(zero_point_indices=[],
                                             num_grid_points=3),
            fp_point=0.0)
        self.fg_cube.rename("probability_of_lightning")
        self.precip_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_cube(num_realization_points=3,
                            zero_point_indices=((0, 1, 1),),
                            num_grid_points=3)))
        threshold_coord = self.precip_cube.coord('realization')
        threshold_coord.points = [0.5, 7.0, 35.0]
        threshold_coord.rename('threshold')
        threshold_coord.units = cf_units.Unit('mm hr-1')
        self.precip_cube.rename("probability_of_precipitation")
        self.precip_cube.attributes.update({'relative_to_threshold': 'above'})
        self.precip_cube.data[1:, 0, ...] = 0.
        self.plugin = Plugin()

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
        self.precip_cube.coord('threshold').points = [0.5000000001, 7., 35.]
        self.plugin.apply_precip(self.fg_cube, self.precip_cube)

    def test_missing_threshold_low(self):
        """Test that the method raises an error if the precip_cube doesn't
        have a threshold coordinate for 0.5."""
        self.precip_cube.coord('threshold').points = [1.0, 7., 35.]
        msg = ("No matching any precip cube for")
        with self.assertRaisesRegex(ConstraintMismatchError, msg):
            self.plugin.apply_precip(self.fg_cube, self.precip_cube)

    def test_missing_threshold_mid(self):
        """Test that the method raises an error if the precip_cube doesn't
        have a threshold coordinate for 7.0."""
        self.precip_cube.coord('threshold').points = [0.5, 8., 35.]
        msg = ("No matching high precip cube for")
        with self.assertRaisesRegex(ConstraintMismatchError, msg):
            self.plugin.apply_precip(self.fg_cube, self.precip_cube)

    def test_missing_threshold_high(self):
        """Test that the method raises an error if the precip_cube doesn't
        have a threshold coordinate for 35.0."""
        self.precip_cube.coord('threshold').points = [0.5, 7., 20.]
        msg = ("No matching intense precip cube for")
        with self.assertRaisesRegex(ConstraintMismatchError, msg):
            self.plugin.apply_precip(self.fg_cube, self.precip_cube)

    def test_precip_zero(self):
        """Test that zero precip probs reduce high lightning risk a lot"""
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[0, 1, 1] = 0.0067
        result = self.plugin.apply_precip(self.fg_cube, self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_precip_small(self):
        """Test that small precip probs reduce high lightning risk a bit"""
        self.precip_cube.data[:, 0, 1, 1] = 0.
        self.precip_cube.data[0, 0, 1, 1] = 0.075
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[0, 1, 1] = 0.625
        result = self.plugin.apply_precip(self.fg_cube, self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_precip_light(self):
        """Test that high probs of light precip do not reduce high lightning
        risk"""
        self.precip_cube.data[:, 0, 1, 1] = 0.
        self.precip_cube.data[0, 0, 1, 1] = 0.8
        expected = self.fg_cube.copy()
        # expected.data contains all ones
        result = self.plugin.apply_precip(self.fg_cube, self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_precip_heavy(self):
        """Test that prob of heavy precip increases zero lightning risk"""
        self.precip_cube.data[0, 0, 1, 1] = 1.0
        self.precip_cube.data[1, 0, 1, 1] = 0.5
        # Set first-guess to zero
        self.fg_cube.data[0, 1, 1] = 0.0
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[0, 1, 1] = 0.25
        result = self.plugin.apply_precip(self.fg_cube, self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_precip_heavy_null(self):
        """Test that low prob of heavy precip does not increase
        low lightning risk"""
        self.precip_cube.data[0, 0, 1, 1] = 1.0
        self.precip_cube.data[1, 0, 1, 1] = 0.3
        # Set first-guess to zero
        self.fg_cube.data[0, 1, 1] = 0.1
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[0, 1, 1] = 0.1
        result = self.plugin.apply_precip(self.fg_cube, self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_precip_intense(self):
        """Test that prob of intense precip increases zero lightning risk"""
        expected = self.fg_cube.copy()
        # expected.data contains all ones
        self.precip_cube.data[0, 0, 1, 1] = 1.0
        self.precip_cube.data[1, 0, 1, 1] = 1.0
        self.precip_cube.data[2, 0, 1, 1] = 0.5
        # Set first-guess to zero
        self.fg_cube.data[0, 1, 1] = 0.0
        result = self.plugin.apply_precip(self.fg_cube, self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_precip_intense_null(self):
        """Test that low prob of intense precip does not increase
        low lightning risk"""
        self.precip_cube.data[0, 0, 1, 1] = 1.0
        self.precip_cube.data[1, 0, 1, 1] = 1.0
        self.precip_cube.data[2, 0, 1, 1] = 0.1
        # Set first-guess to zero
        self.fg_cube.data[0, 1, 1] = 0.1
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[0, 1, 1] = 0.25  # Heavy-precip result only
        result = self.plugin.apply_precip(self.fg_cube, self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)


class Test_apply_ice(IrisTest):

    """Test the apply_ice method."""

    def setUp(self):
        """Create cubes with a single zero prob(precip) point.
        The cubes look like this:
        precipitation_amount / (kg m^-2)
        Dimension coordinates:
            time: 1;
            projection_y_coordinate: 3;
            projection_x_coordinate: 3;
        Auxiliary coordinates:
            forecast_period (on time coord): 0.0 hours (simulates nowcast data)
        Scalar coordinates:
            forecast_reference_time: 2015-11-23 03:00:00
        Data:
        self.fg_cube:
            All points contain float(1.)
            Cube name is "probability_of_lightning".
        self.ice_cube:
            With extra coordinate of length(3) "threshold" containing
            points [0.5, 1., 2.] kg m^-2.
            Time and forecast_period dimensions "sqeezed" to be Scalar coords.
            All points contain float(0.)
            Cube name is "probability_of_vertical_integral_of_ice".
        """
        self.fg_cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube_with_no_realizations(zero_point_indices=[],
                                             num_grid_points=3),
            fp_point=0.0)
        self.fg_cube.rename("probability_of_lightning")
        self.ice_cube = squeeze(
            add_forecast_reference_time_and_forecast_period(
                set_up_cube(num_realization_points=3,
                            zero_point_indices=[],
                            num_grid_points=3),
                fp_point=0.0))
        threshold_coord = self.ice_cube.coord('realization')
        threshold_coord.points = [0.5, 1.0, 2.0]
        threshold_coord.rename('threshold')
        threshold_coord.units = cf_units.Unit('kg m^-2')
        self.ice_cube.data = np.zeros_like(self.ice_cube.data)
        self.ice_cube.rename("probability_of_vertical_integral_of_ice")
        self.plugin = Plugin()

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
        self.ice_cube.coord('threshold').points = [0.4, 1., 2.]
        msg = (r"No matching prob\(Ice\) cube for threshold 0.5")
        with self.assertRaisesRegex(ConstraintMismatchError, msg):
            self.plugin.apply_ice(self.fg_cube, self.ice_cube)

    def test_missing_threshold_mid(self):
        """Test that the method raises an error if the ice_cube doesn't
        have a threshold coordinate for 1.0."""
        self.ice_cube.coord('threshold').points = [0.5, 0.9, 2.]
        msg = (r"No matching prob\(Ice\) cube for threshold 1.")
        with self.assertRaisesRegex(ConstraintMismatchError, msg):
            self.plugin.apply_ice(self.fg_cube, self.ice_cube)

    def test_missing_threshold_high(self):
        """Test that the method raises an error if the ice_cube doesn't
        have a threshold coordinate for 2.0."""
        self.ice_cube.coord('threshold').points = [0.5, 1., 4.]
        msg = (r"No matching prob\(Ice\) cube for threshold 2.")
        with self.assertRaisesRegex(ConstraintMismatchError, msg):
            self.plugin.apply_ice(self.fg_cube, self.ice_cube)

    def test_ice_null(self):
        """Test that small VII probs do not increase moderate lightning risk"""
        self.ice_cube.data[:, 1, 1] = 0.
        self.ice_cube.data[0, 1, 1] = 0.5
        self.fg_cube.data[0, 1, 1] = 0.25
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[0, 1, 1] = 0.25
        result = self.plugin.apply_ice(self.fg_cube,
                                       self.ice_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_ice_zero(self):
        """Test that zero VII probs do not increase zero lightning risk"""
        self.ice_cube.data[:, 1, 1] = 0.
        self.fg_cube.data[0, 1, 1] = 0.
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[0, 1, 1] = 0.
        result = self.plugin.apply_ice(self.fg_cube,
                                       self.ice_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_ice_small(self):
        """Test that small VII probs do increase zero lightning risk"""
        self.ice_cube.data[:, 1, 1] = 0.
        self.ice_cube.data[0, 1, 1] = 0.5
        self.fg_cube.data[0, 1, 1] = 0.
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[0, 1, 1] = 0.05
        result = self.plugin.apply_ice(self.fg_cube,
                                       self.ice_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_ice_large(self):
        """Test that large VII probs do increase zero lightning risk"""
        self.ice_cube.data[:, 1, 1] = 1.
        self.fg_cube.data[0, 1, 1] = 0.
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[0, 1, 1] = 0.9
        result = self.plugin.apply_ice(self.fg_cube,
                                       self.ice_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_ice_large_with_fc(self):
        """Test that large VII probs do increase zero lightning risk when
        forecast lead time is non-zero (two forecast_period points)"""
        self.ice_cube.data[:, 1, 1] = 1.
        self.fg_cube.data[0, 1, 1] = 0.
        self.fg_cube.coord('forecast_period').points = [1.]  # hours
        fg_cube_next = self.fg_cube.copy()
        time_pt, = self.fg_cube.coord('time').points
        fg_cube_next.coord('time').points = [time_pt + 2.]  # hours
        fg_cube_next.coord('forecast_period').points = [3.]  # hours
        self.fg_cube = CubeList([squeeze(self.fg_cube),
                                 squeeze(fg_cube_next)]).merge_cube()
        expected = self.fg_cube.copy()
        # expected.data contains all ones except:
        expected.data[0, 1, 1] = 0.54
        expected.data[1, 1, 1] = 0.0
        result = self.plugin.apply_ice(self.fg_cube,
                                       self.ice_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)


class Test_process(IrisTest):

    """Test the nowcast lightning plugin."""

    def setUp(self):
        """Create cubes with a single zero prob(precip) point.
        The cubes look like this:
        precipitation_amount / (kg m^-2)
        Dimension coordinates:
            time: 1;
            projection_y_coordinate: 16;
            projection_x_coordinate: 16;
        Auxiliary coordinates:
            forecast_period (on time coord): 4.0 hours (simulates UM data)
        Scalar coordinates:
            forecast_reference_time: 2015-11-23 03:00:00
        Data:
        self.fg_cube:
            All points contain float(1.)
            Cube name is "probability_of_lightning".
        self.ltng_cube:
            forecast_period (on time coord): 0.0 hours (simulates nowcast data)
            All points contain float(1.)
            Cube name is "rate_of_lightning".
            Cube units are "min^-1".
        self.precip_cube:
            With extra coordinate of length(3) "threshold" containing
            points [0.5, 7., 35.] mm hr-1.
            All points contain float(1.) except the
            zero point [0, 0, 7, 7] which is float(0.)
            and [1:, 0, ...] which are float(0.)
            Cube name is "probability_of_precipitation".
            Cube has added attribute {'relative_to_threshold': 'above'}
        self.vii_cube:
            forecast_period (on time coord): 0.0 hours (simulates nowcast data)
            With extra coordinate of length(3) "threshold" containing
            points [0.5, 1., 2.] kg m^-2.
            forecast_period (on time coord): 0.0 hours (simulates nowcast data)
            Time and forecast_period dimensions "sqeezed" to be Scalar coords.
            All points contain float(0.)
            Cube name is "probability_of_vertical_integral_of_ice".
        """
        self.fg_cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube_with_no_realizations(zero_point_indices=[]))
        self.fg_cube.rename("probability_of_lightning")
        self.ltng_cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube_with_no_realizations(zero_point_indices=[]),
            fp_point=0.0)
        self.ltng_cube.rename("rate_of_lightning")
        self.ltng_cube.units = cf_units.Unit("min^-1")
        self.precip_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_cube(num_realization_points=3)))
        threshold_coord = self.precip_cube.coord('realization')
        threshold_coord.points = [0.5, 7.0, 35.0]
        threshold_coord.rename('threshold')
        threshold_coord.units = cf_units.Unit('mm hr-1')
        self.precip_cube.rename("probability_of_precipitation")
        self.precip_cube.attributes.update({'relative_to_threshold': 'above'})
        self.precip_cube.data[1:, 0, ...] = 0.
        self.vii_cube = squeeze(
            add_forecast_reference_time_and_forecast_period(
                set_up_cube(num_realization_points=3,
                            zero_point_indices=[]),
                fp_point=0.0))
        threshold_coord = self.vii_cube.coord('realization')
        threshold_coord.points = [0.5, 1.0, 2.0]
        threshold_coord.rename('threshold')
        threshold_coord.units = cf_units.Unit('kg m^-2')
        self.vii_cube.data = np.zeros_like(self.vii_cube.data)
        self.vii_cube.rename("probability_of_vertical_integral_of_ice")
        self.plugin = Plugin()

    def set_up_vii_input_output(self):
        """Used to modify setUp() to set up four standard VII tests."""

        # Repeat all tests relating to vii from Test__modify_first_guess
        expected = self.fg_cube.copy()
        # expected.data contains all ones except where modified below:

        # Set up precip_cube with increasing intensity along x-axis
        # y=5; no precip
        self.precip_cube.data[:, 0, 5:9, 5] = 0.
        # y=6; light precip
        self.precip_cube.data[0, 0, 5:9, 6] = 0.1
        self.precip_cube.data[1, 0:, 5:9, 6] = 0.
        # y=7; heavy precip
        self.precip_cube.data[:2, 0, 5:9, 7] = 1.
        self.precip_cube.data[2, 0, 5:9, 7] = 0.
        # y=8; intense precip
        self.precip_cube.data[:, 0, 5:9, 8] = 1.

        # test_vii_null - with lightning-halo
        self.vii_cube.data[:, 5, 5:9] = 0.
        self.vii_cube.data[0, 5, 5:9] = 0.5
        self.ltng_cube.data[0, 5, 5:9] = 0.
        self.fg_cube.data[0, 5, 5:9] = 0.
        expected.data[0, 5, 5:9] = [0.05, 0.25, 0.25, 1.]

        # test_vii_zero
        self.vii_cube.data[:, 6, 5:9] = 0.
        self.ltng_cube.data[0, 6, 5:9] = -1.
        self.fg_cube.data[0, 6, 5:9] = 0.
        expected.data[0, 6, 5:9] = [0., 0., 0.25, 1.]

        # test_vii_small
        # Set lightning data to -1 so it has a Null impact
        self.vii_cube.data[:, 7, 5:9] = 0.
        self.vii_cube.data[0, 7, 5:9] = 0.5
        self.ltng_cube.data[0, 7, 5:9] = -1.
        self.fg_cube.data[0, 7, 5:9] = 0.
        expected.data[0, 7, 5:9] = [0.05, 0.05, 0.25, 1.]

        # test_vii_large
        # Set lightning data to -1 so it has a Null impact
        self.vii_cube.data[:, 8, 5:9] = 1.
        self.ltng_cube.data[0, 8, 5:9] = -1.
        self.fg_cube.data[0, 8, 5:9] = 0.
        expected.data[0, 8, 5:9] = [0.9, 0.9, 0.9, 1.]
        return expected

    def test_basic(self):
        """Test that the method returns the expected cube type with coords"""
        result = self.plugin.process(CubeList([
            self.fg_cube,
            self.ltng_cube,
            self.precip_cube]))
        self.assertIsInstance(result, Cube)
        # We expect the threshold coordinate to have been removed.
        self.assertCountEqual(find_dimension_coordinate_mismatch(
                result, self.precip_cube), ['threshold'])
        self.assertTrue(result.name() == 'probability_of_lightning')

    def test_basic_with_vii(self):
        """Test that the method returns the expected cube type when vii is
        present"""
        result = self.plugin.process(CubeList([
            self.fg_cube,
            self.ltng_cube,
            self.precip_cube,
            self.vii_cube]))
        self.assertIsInstance(result, Cube)
        # We expect the threshold coordinate to have been removed.
        self.assertCountEqual(find_dimension_coordinate_mismatch(
                result, self.precip_cube), ['threshold'])
        self.assertTrue(result.name() == 'probability_of_lightning')

    def test_no_first_guess_cube(self):
        """Test that the method raises an error if the first_guess cube is
        omitted from the cubelist"""
        msg = (r"Got 0 cubes for constraint Constraint\(name=\'probability_of_"
               r"lightning\'\), expecting 1.")
        with self.assertRaisesRegex(ConstraintMismatchError, msg):
            self.plugin.process(CubeList([
                self.ltng_cube,
                self.precip_cube]))

    def test_no_lightning_cube(self):
        """Test that the method raises an error if the lightning cube is
        omitted from the cubelist"""
        msg = (r"Got 0 cubes for constraint Constraint\(name=\'rate_of_"
               r"lightning\'\), expecting 1.")
        with self.assertRaisesRegex(ConstraintMismatchError, msg):
            self.plugin.process(CubeList([
                self.fg_cube,
                self.precip_cube]))

    def test_no_precip_cube(self):
        """Test that the method raises an error if the precip cube is
        omitted from the cubelist"""
        msg = (r"Got 0 cubes for constraint Constraint\(name=\'probability_of_"
               r"precipitation\'\), expecting 1.")
        with self.assertRaisesRegex(ConstraintMismatchError, msg):
            self.plugin.process(CubeList([
                self.fg_cube,
                self.ltng_cube]))

    def test_precip_has_no_thresholds(self):
        """Test that the method raises an error if the threshold coord is
        omitted from the precip_cube"""
        self.precip_cube.remove_coord('threshold')
        msg = (r"Expected to find exactly 1 threshold coordinate, but found n")
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            self.plugin.process(CubeList([
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
        result = plugin.process(CubeList([
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
        expected.data[0, 5, 5] = 0.0067

        # test_vii_small with no and light precip will now return zero
        expected.data[0, 7, 5:7] = 0.

        # test_vii_large with no and light precip now return zero
        # and 0.25 for heavy precip
        expected.data[0, 8, 5:8] = [0., 0., 0.25]
        # No halo - we're only testing this method.
        # 2000m is the grid-length, so halo includes only one pixel.
        plugin = Plugin(2000.)
        result = plugin.process(CubeList([
            self.fg_cube,
            self.ltng_cube,
            self.precip_cube,
            self.vii_cube]))
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, expected.data)


if __name__ == '__main__':
    unittest.main()
