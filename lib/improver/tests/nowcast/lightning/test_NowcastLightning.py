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
"""Unit tests for the threshold.BasicThreshold plugin."""


import unittest

from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from iris.tests import IrisTest
import numpy as np
import StringIO
import sys

from improver.nowcast.lightning import NowcastLightning as Plugin
from improver.tests.nbhood.nbhood.test_BaseNeighbourhoodProcessing import (
    set_up_cube, set_up_cube_with_no_realizations)
from improver.tests.ensemble_calibration.ensemble_calibration.helper_functions\
    import add_forecast_reference_time_and_forecast_period


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(Plugin())
        msg = ('<NowcastLightning: radius=10000.0, debug=False>')
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

    def test_basic(self):
        """Test that the method returns the expected cube type
        and that the metadata are as expected.
        We expect a new name and an empty dictionary of attributes."""
        plugin = Plugin()
        self.cube.attributes = {'source': 'testing'}
        result = plugin._update_meta(self.cube)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), "lightning_probability")
        self.assertEqual(result.attributes, {})

    def test_input(self):
        """Test that the method does not modify the input cube."""
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
        self.precip_cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube_with_no_realizations(), fp_point=0.0)

    def test_basic(self):
        """Test that the method returns the expected cube type"""
        plugin = Plugin()
        result = plugin._modify_first_guess(self.cube,
                                            self.fg_cube,
                                            self.ltng_cube,
                                            self.precip_cube)
        self.assertIsInstance(result, Cube)

    def test_input(self):
        """Test that the method does not modify the input cube."""
        plugin = Plugin()
        cube_a = self.cube.copy()
        cube_b = self.fg_cube.copy()
        cube_c = self.ltng_cube.copy()
        cube_d = self.precip_cube.copy()
        plugin._modify_first_guess(cube_a, cube_b, cube_c, cube_d)
        self.assertArrayAlmostEqual(cube_a.data, self.cube.data)
        self.assertArrayAlmostEqual(cube_b.data, self.fg_cube.data)
        self.assertArrayAlmostEqual(cube_c.data, self.ltng_cube.data)
        self.assertArrayAlmostEqual(cube_d.data, self.precip_cube.data)

    def test_precip_zero(self):
        """Test that zero precip probs reduce lightning risk"""
        # Set lightning data to zero so it has a Null impact
        self.ltng_cube.data = self.ltng_cube.data * 0. - 1.
        # No halo - we're only testing this method.
        plugin = Plugin(0.)
        expected = set_up_cube_with_no_realizations()
        expected.data[0, 7, 7] = 0.0067
        result = plugin._modify_first_guess(self.cube,
                                            self.fg_cube,
                                            self.ltng_cube,
                                            self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_precip_small(self):
        """Test that small precip probs reduce lightning risk"""
        # Set precip data to 0.075, in the middle of the upper low range.
        self.precip_cube.data[0, 7, 7] = 0.075
        # Set lightning data to zero so it has a Null impact
        self.ltng_cube.data = self.ltng_cube.data * 0. - 1.
        # No halo - we're only testing this method.
        plugin = Plugin(0.)
        expected = set_up_cube_with_no_realizations()
        expected.data[0, 7, 7] = 0.6
        result = plugin._modify_first_guess(self.cube,
                                            self.fg_cube,
                                            self.ltng_cube,
                                            self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_null(self):
        """Test that large precip probs and -1 lrates have no impact"""
        # Set precip data to 0.1, at the top of the upper low range.
        self.precip_cube.data[0, 7, 7] = 0.1
        # Set lightning data to -1 so it has a Null impact
        self.ltng_cube.data = self.ltng_cube.data * 0. - 1.
        # No halo - we're only testing this method.
        plugin = Plugin(0.)
        expected = set_up_cube_with_no_realizations(zero_point_indices=[])
        result = plugin._modify_first_guess(self.cube,
                                            self.fg_cube,
                                            self.ltng_cube,
                                            self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_lrate_large(self):
        """Test that large lightning rates increase lightning risk"""
        # Set precip data to 1. so it has a Null impact
        self.precip_cube.data[0, 7, 7] = 1.
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
                                            self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_lrate_halo(self):
        """Test that zero lightning rates increase lightning risk"""
        # Set precip data to 1. so it has a Null impact
        self.precip_cube.data[0, 7, 7] = 1.
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
                                            self.precip_cube)
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
        self.precip_cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube_with_no_realizations())
        self.precip_cube.rename("probability_of_precipitation")
        self.precip_cube.attributes.update({'relative_to_threshold': 'above'})
        coord = DimCoord(0., long_name="threshold", units='mm hr^-1')
        self.precip_cube.add_aux_coord(coord)

    def test_basic(self):
        """Test that the method returns the expected cube type"""
        plugin = Plugin()
        result = plugin.process(CubeList([
            self.fg_cube,
            self.ltng_cube,
            self.precip_cube]))
        self.assertIsInstance(result, Cube)

if __name__ == '__main__':
    unittest.main()
