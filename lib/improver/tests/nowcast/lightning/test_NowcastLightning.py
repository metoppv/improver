# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from iris.tests import IrisTest
from iris.exceptions import CoordinateNotFoundError
import numpy as np
import cf_units

from improver.nowcast.lightning import NowcastLightning as Plugin
from improver.tests.nbhood.nbhood.test_BaseNeighbourhoodProcessing import (
    set_up_cube, set_up_cube_with_no_realizations)
from improver.tests.ensemble_calibration.ensemble_calibration.helper_functions\
    import add_forecast_reference_time_and_forecast_period


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        # Have to pass in a lambda to ensure two strings match the same
        # function address.
        set_lightning_thresholds = (lambda mins: mins, 0.)
        result = str(Plugin(
            lightning_thresholds=set_lightning_thresholds))
        msg = ("""<NowcastLightning: radius={radius}, debug={debug},
 lightning mapping (lightning rate in "min^-1"):
   upper: lightning rate {lthru} => min lightning prob {lprobu}
   lower: lightning rate {lthrl} => min lightning prob {lprobl}
With:
<ApplyPrecip:
 precipitation mapping:
   upper:  precip probability {precu} => max lightning prob {lprecu}
   middle: precip probability {precm} => max lightning prob {lprecm}
   lower:  precip probability {precl} => max lightning prob {lprecl}

   heavy:  prob(precip>7mm/hr)  {pphvy} => min lightning prob {lprobl}
   intense:prob(precip>35mm/hr) {ppint} => min lightning prob {lprobu}
>
<ApplyIce:
 VII (ice) mapping (kg/m2):
   upper:  VII {viiu} => max lightning prob {lviiu}
   middle: VII {viim} => max lightning prob {lviim}
   lower:  VII {viil} => max lightning prob {lviil}
>
>""".format(radius=10000., debug=False,
            lthru=set_lightning_thresholds[0], lthrl=0.,
            lprobu=1., lprobl=0.25,
            precu=0.1, precm=0.05, precl=0.0,
            lprecu=1., lprecm=0.2, lprecl=0.0067,
            pphvy=0.4, ppint=0.2,
            viiu=2.0, viim=1.0,
            viil=0.5,
            lviiu=0.9, lviim=0.5,
            lviil=0.1)
              )
        self.assertEqual(result, msg)


class Test__process_haloes(IrisTest):

    """Test the _process_haloes method."""

    def setUp(self):
        """Create a cube with a single non-zero point."""
        self.cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube())

    def test_basic(self):
        """Test that the method returns the expected cube type"""
        plugin = Plugin()
        result = plugin._process_haloes(self.cube)
        self.assertIsInstance(result, Cube)

    def test_input(self):
        """Test that the method does not modify the input cube."""
        plugin = Plugin()
        incube = self.cube.copy()
        plugin._process_haloes(incube)
        self.assertArrayAlmostEqual(incube.data, self.cube.data)

    def test_data(self):
        """Test that the method returns the expected data"""
        plugin = Plugin(3000.)
        expected = self.cube.data.copy()
        expected[0, 0, 6, :] = [1., 1., 1., 1., 1., 1., 11./12., 0.875,
                                11./12., 1., 1., 1., 1., 1., 1., 1.]
        expected[0, 0, 7, :] = [1., 1., 1., 1., 1., 1., 0.875, 5./6., 0.875,
                                1., 1., 1., 1., 1., 1., 1.]
        expected[0, 0, 8, :] = [1., 1., 1., 1., 1., 1., 11./12., 0.875,
                                11./12., 1., 1., 1., 1., 1., 1., 1.]
        result = plugin._process_haloes(self.cube)
        self.assertArrayAlmostEqual(result.data, expected)


class Test__update_meta(IrisTest):

    """Test the _update_meta method."""

    def setUp(self):
        """Create a cube with a single non-zero point."""
        self.cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube())
        coord = DimCoord(0.5, long_name="threshold", units='mm hr^-1')
        self.cube.add_aux_coord(coord)

    def test_basic(self):
        """Test that the method returns the expected cube type
        and that the metadata are as expected.
        We expect a new name, the threshold coord to be removed
        and an empty dictionary of attributes."""
        plugin = Plugin()
        self.cube.attributes = {'source': 'testing'}
        result = plugin._update_meta(self.cube)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), "lightning_probability")
        self.assertEqual(result.attributes, {})
        msg = "Expected to find exactly 1  coordinate, but found none."
        with self.assertRaisesRegexp(CoordinateNotFoundError, msg):
            result.coord('threshold')

    def test_input(self):
        """Test that the method does not modify the input cube data."""
        plugin = Plugin()
        incube = self.cube.copy()
        plugin._update_meta(incube)
        self.assertArrayAlmostEqual(incube.data, self.cube.data)


class Test__modify_first_guess(IrisTest):

    """Test the _modify_first_guess method."""

    def setUp(self):
        """Create cubes with a single zero prob(precip) point."""
        self.cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube_with_no_realizations(), fp_point=0.0)
        self.fg_cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube_with_no_realizations(zero_point_indices=[]))
        self.ltng_cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube_with_no_realizations(zero_point_indices=[]),
            fp_point=0.0)
        self.precip_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_cube(num_realization_points=3), fp_point=0.0))
        threshold_coord = self.precip_cube.coord('realization')
        threshold_coord.points = [0.5, 7.0, 35.0]
        threshold_coord.rename('threshold')
        threshold_coord.units = cf_units.Unit('kg m^-2')
        self.precip_cube.data[1:, 0, ...] = 0.
        # iris.util.queeze is applied here to demote the singular coord "time"
        # to a scalar coord.
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

    def test_basic(self):
        """Test that the method returns the expected cube type"""
        plugin = Plugin()
        result = plugin._modify_first_guess(self.cube,
                                            self.fg_cube,
                                            self.ltng_cube,
                                            self.precip_cube,
                                            None)
        self.assertIsInstance(result, Cube)

    def test_basic_with_vii(self):
        """Test that the method returns the expected cube type"""
        plugin = Plugin()
        result = plugin._modify_first_guess(self.cube,
                                            self.fg_cube,
                                            self.ltng_cube,
                                            self.precip_cube,
                                            self.vii_cube)
        self.assertIsInstance(result, Cube)

    def test_input(self):
        """Test that the method does not modify the input cube."""
        plugin = Plugin()
        cube_a = self.cube.copy()
        cube_b = self.fg_cube.copy()
        cube_c = self.ltng_cube.copy()
        cube_d = self.precip_cube.copy()
        plugin._modify_first_guess(cube_a, cube_b, cube_c, cube_d, None)
        self.assertArrayAlmostEqual(cube_a.data, self.cube.data)
        self.assertArrayAlmostEqual(cube_b.data, self.fg_cube.data)
        self.assertArrayAlmostEqual(cube_c.data, self.ltng_cube.data)
        self.assertArrayAlmostEqual(cube_d.data, self.precip_cube.data)

    def test_input_with_vii(self):
        """Test that the method does not modify the input cube."""
        plugin = Plugin()
        cube_a = self.cube.copy()
        cube_b = self.fg_cube.copy()
        cube_c = self.ltng_cube.copy()
        cube_d = self.precip_cube.copy()
        cube_e = self.vii_cube.copy()
        plugin._modify_first_guess(cube_a, cube_b, cube_c, cube_d, cube_e)
        self.assertArrayAlmostEqual(cube_a.data, self.cube.data)
        self.assertArrayAlmostEqual(cube_b.data, self.fg_cube.data)
        self.assertArrayAlmostEqual(cube_c.data, self.ltng_cube.data)
        self.assertArrayAlmostEqual(cube_d.data, self.precip_cube.data)
        self.assertArrayAlmostEqual(cube_e.data, self.vii_cube.data)

    def test_precip_zero(self):
        """Test that ApplyPrecip is being called"""
        # Set lightning data to zero so it has a Null impact
        self.ltng_cube.data = np.full_like(self.ltng_cube.data, -1.)
        # No halo - we're only testing this method.
        plugin = Plugin(0.)
        expected = set_up_cube_with_no_realizations()
        expected.data[0, 7, 7] = 0.0067
        result = plugin._modify_first_guess(self.cube,
                                            self.fg_cube,
                                            self.ltng_cube,
                                            self.precip_cube,
                                            None)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_vii_large(self):
        """Test that ApplyIce is being called"""
        # Set lightning data to zero so it has a Null impact
        self.vii_cube.data[:, 7, 7] = 1.
        self.ltng_cube.data[0, 7, 7] = -1.
        self.fg_cube.data[0, 7, 7] = 0.
        # No halo - we're only testing this method.
        plugin = Plugin()
        expected = set_up_cube_with_no_realizations()
        expected.data[0, 7, 7] = 0.9
        result = plugin._modify_first_guess(self.cube,
                                            self.fg_cube,
                                            self.ltng_cube,
                                            self.precip_cube,
                                            self.vii_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_null(self):
        """Test that large precip probs and -1 lrates have no impact"""
        # Set precip data to 0.1, at the top of the upper low range.
        self.precip_cube.data[0, 0, 7, 7] = 0.1
        # Set lightning data to -1 so it has a Null impact
        self.ltng_cube.data = self.ltng_cube.data * 0. - 1.
        # No halo - we're only testing this method.
        plugin = Plugin(0.)
        expected = set_up_cube_with_no_realizations(zero_point_indices=[])
        result = plugin._modify_first_guess(self.cube,
                                            self.fg_cube,
                                            self.ltng_cube,
                                            self.precip_cube,
                                            None)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_lrate_large(self):
        """Test that large lightning rates increase lightning risk"""
        # Set precip data to 1. so it has a Null impact
        self.precip_cube.data[0, 0, 7, 7] = 1.
        # Set first-guess data zero point to be increased
        self.fg_cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube_with_no_realizations())
        # No halo - we're only testing this method.
        plugin = Plugin(0.)
        expected = add_forecast_reference_time_and_forecast_period(
            set_up_cube_with_no_realizations())
        expected.data[0, 7, 7] = 1.
        result = plugin._modify_first_guess(self.cube,
                                            self.fg_cube,
                                            self.ltng_cube,
                                            self.precip_cube,
                                            None)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_lrate_halo(self):
        """Test that zero lightning rates increase lightning risk"""
        # Set precip data to 1. so it has a Null impact
        self.precip_cube.data[0, 0, 7, 7] = 1.
        # Set lightning data to zero to represent the data halo
        self.ltng_cube.data[0, 7, 7] = 0.
        # Set first-guess data zero point to be increased
        self.fg_cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube_with_no_realizations())
        # No halo - we're only testing this method.
        plugin = Plugin(0.)
        expected = set_up_cube_with_no_realizations()
        expected.data[0, 7, 7] = 0.25
        result = plugin._modify_first_guess(self.cube,
                                            self.fg_cube,
                                            self.ltng_cube,
                                            self.precip_cube,
                                            None)
        self.assertArrayAlmostEqual(result.data, expected.data)


class Test_process(IrisTest):

    """Test the nowcast lightning plugin."""

    def setUp(self):
        """Create a cube with a single non-zero point."""
        self.fg_cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube_with_no_realizations(zero_point_indices=[]))
        self.fg_cube.rename("probability_of_lightning")
        self.ltng_cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube_with_no_realizations(zero_point_indices=[]))
        self.ltng_cube.rename("rate_of_lightning")
        self.ltng_cube.units = cf_units.Unit("min^-1")
        self.precip_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_cube(num_realization_points=3), fp_point=0.0))
        threshold_coord = self.precip_cube.coord('realization')
        threshold_coord.points = [0.5, 7.0, 35.0]
        threshold_coord.rename('threshold')
        threshold_coord.units = cf_units.Unit('kg m^-2')
        self.precip_cube.rename("probability_of_precipitation")
        self.precip_cube.attributes.update({'relative_to_threshold': 'above'})
        self.precip_cube.data[1:, 0, ...] = 0.
        self.precip_cube.coord('forecast_period').points = [4.]
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

    def set_up_vii_input_output(self):
        """Used to set up four standard VII tests."""

        # Repeat all tests relating to vii from Test__modify_first_guess
        expected = set_up_cube_with_no_realizations()

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
        """Test that the method returns the expected cube type"""
        plugin = Plugin()
        result = plugin.process(CubeList([
            self.fg_cube,
            self.ltng_cube,
            self.precip_cube]))
        self.assertIsInstance(result, Cube)

    def test_basic_with_vii(self):
        """Test that the method returns the expected cube type when vii is
        present"""
        plugin = Plugin()
        result = plugin.process(CubeList([
            self.fg_cube,
            self.ltng_cube,
            self.precip_cube,
            self.vii_cube]))
        self.assertIsInstance(result, Cube)

    def test_result_with_vii(self):
        """Test that the method returns the expected data when vii is
        present"""
        # Set precip_cube forecast period to be zero.
        self.precip_cube.coord('forecast_period').points = [0.]
        expected = self.set_up_vii_input_output()

        # No halo - we're only testing this method.
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
