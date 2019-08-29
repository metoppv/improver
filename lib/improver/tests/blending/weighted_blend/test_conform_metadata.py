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
"""Module to test the conform_metadata function."""

import unittest
from datetime import datetime as dt

import iris
import numpy as np
from iris.coords import AuxCoord
from iris.tests import IrisTest

from improver.blending.weighted_blend import conform_metadata
from improver.tests.set_up_test_cubes import set_up_variable_cube
from improver.utilities.warnings_handler import ManageWarnings


class Test_conform_metadata(IrisTest):

    """Test the conform_metadata function."""

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def setUp(self):
        """Set up cubes for testing."""
        self.cube = set_up_variable_cube(
            np.full((3, 3), 275.15, dtype=np.float32),
            time=dt(2015, 11, 23, 7, 0),
            frt=dt(2015, 11, 23, 3, 0))

        # Cube with multiple times.
        cube2 = set_up_variable_cube(
            np.full((3, 3), 275.15, dtype=np.float32),
            time=dt(2015, 11, 23, 7, 0),
            frt=dt(2015, 11, 23, 4, 0))
        self.cube_orig = iris.cube.CubeList([self.cube, cube2]).merge_cube()

        # Cube without forecast_period.
        cube_orig_without_fp = self.cube_orig.copy()
        cube_orig_without_fp.remove_coord("forecast_period")
        self.cube_orig_without_fp = cube_orig_without_fp
        cube_without_fp = self.cube.copy()
        cube_without_fp.remove_coord("forecast_period")
        self.cube_without_fp = cube_without_fp

        # Cube with model_id and model configuration coordinates
        cube_orig_model = self.cube_orig.copy()
        cube_orig_model.add_aux_coord(
            AuxCoord(["Operational MOGREPS-UK Model Forecast",
                      "Operational UKV Model Forecast"], long_name="model_id"),
            data_dims=0)
        cube_orig_model.add_aux_coord(
            AuxCoord(["uk_ens", "uk_det"], long_name="model_configuration"),
            data_dims=0)
        self.cube_orig_model = cube_orig_model
        self.cube_model = cube_orig_model.collapsed(
            "forecast_reference_time", iris.analysis.MEAN)

        # Coordinate that is being blended.
        self.coord = "forecast_reference_time"

    def test_basic(self):
        """Test that conform_metadata returns a cube with a suitable title
        attribute."""
        result = conform_metadata(self.cube, self.cube_orig, self.coord)
        expected_attributes = {'title': 'IMPROVER Model Forecast'}
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertDictEqual(result.attributes, expected_attributes)

    def test_with_forecast_period(self):
        """Test that a cube is dealt with correctly, if the cube contains
        a forecast_reference_time and forecast_period coordinate."""
        result = conform_metadata(self.cube, self.cube_orig, self.coord)
        self.assertEqual(
            result.coord("forecast_reference_time").points,
            np.max(self.cube_orig.coord("forecast_reference_time").points))
        self.assertFalse(result.coord("forecast_reference_time").bounds)
        self.assertEqual(
            result.coord("forecast_period").points,
            np.min(self.cube_orig.coord("forecast_period").points))
        self.assertFalse(result.coord("forecast_period").bounds)

    def test_without_forecast_period(self):
        """Test that a cube is dealt with correctly, if the cube contains a
        forecast_reference_time coordinate but not a forecast_period."""
        result = conform_metadata(
            self.cube_without_fp, self.cube_orig_without_fp, self.coord)
        fp_coord = self.cube_orig.coord("forecast_period").copy()
        fp_coord.convert_units("seconds")
        self.assertEqual(
            result.coord("forecast_reference_time").points,
            np.max(self.cube_orig.coord("forecast_reference_time").points))
        self.assertFalse(result.coord("forecast_reference_time").bounds)
        self.assertEqual(
            result.coord("forecast_period").points,
            np.min(fp_coord.points))
        self.assertFalse(result.coord("forecast_period").bounds)

    def test_with_forecast_period_and_cycletime(self):
        """Test that a cube is dealt with correctly, if the cube contains
        a forecast_reference_time and forecast_period coordinate and a
        cycletime is specified."""
        expected_forecast_reference_time = np.array([1448258400])
        expected_forecast_period = np.array([3600])  # 1 hour.
        result = conform_metadata(
            self.cube, self.cube_orig, self.coord, cycletime="20151123T0600Z")
        self.assertArrayAlmostEqual(
            result.coord("forecast_reference_time").points,
            expected_forecast_reference_time)
        self.assertFalse(result.coord("forecast_reference_time").bounds)
        self.assertEqual(
            result.coord("forecast_period").points, expected_forecast_period)
        self.assertFalse(result.coord("forecast_period").bounds)

    def test_without_forecast_period_and_cycletime(self):
        """Test that a cube is dealt with correctly, if the cube contains a
        forecast_reference_time coordinate but not a forecast_period when a
        cycletime is specified. The same value for the forecast_period should
        be created compared to when the when the input cube has a forecast
        period coordinate."""
        expected_forecast_reference_time = np.array([1448258400])
        expected_forecast_period = np.array([3600])
        result = conform_metadata(
            self.cube_without_fp, self.cube_orig_without_fp, self.coord,
            cycletime="20151123T0600Z")
        self.assertEqual(
            result.coord("forecast_reference_time").points,
            expected_forecast_reference_time)
        self.assertFalse(result.coord("forecast_reference_time").bounds)
        self.assertEqual(
            result.coord("forecast_period").points, expected_forecast_period)
        self.assertFalse(result.coord("forecast_period").bounds)

    def test_with_model_coordinates(self):
        """Test that a cube is dealt with correctly, if the cube contains a
        model, model_id and model_configuration coordinate."""
        coord = "model_id"
        result = conform_metadata(self.cube_model, self.cube_orig_model, coord)
        self.assertFalse(result.coords("model_id"))
        self.assertFalse(result.coords("model_configuration"))

    def test_forecast_coordinate_bounds_removal(self):
        """Test that if a cube has bounds on the forecast period and reference
        time, that these are removed"""
        self.cube_orig.coord("forecast_period").bounds = np.array(
            [[x-0.5, x+0.5] for x in self.cube_orig.coord(
                "forecast_period").points])
        self.cube_orig.coord("forecast_reference_time").bounds = np.array(
            [[x-0.5, x+0.5] for x in self.cube_orig.coord(
                "forecast_reference_time").points])
        self.cube.coord("forecast_period").bounds = np.array(
            [[x-0.5, x+0.5] for x in self.cube.coord(
                "forecast_period").points])
        self.cube.coord("forecast_reference_time").bounds = np.array(
            [[x-0.5, x+0.5] for x in self.cube.coord(
                "forecast_reference_time").points])
        result = conform_metadata(
            self.cube, self.cube_orig, "forecast_reference_time")
        self.assertIsNone(result.coord("forecast_reference_time").bounds)
        self.assertIsNone(result.coord("forecast_period").bounds)


if __name__ == '__main__':
    unittest.main()
