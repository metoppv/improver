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
"""Module to test the conform_metadata function."""

import unittest

import numpy as np
import iris
from iris.tests import IrisTest
from iris.coords import AuxCoord

from improver.blending.weighted_blend import conform_metadata
from improver.tests.ensemble_calibration.ensemble_calibration.helper_functions\
    import add_forecast_reference_time_and_forecast_period, set_up_cube


class Test_conform_metadata(IrisTest):

    """Test the conform_metadata function."""

    def setUp(self):
        """Set up cubes for testing."""
        data = np.full((3, 1, 3, 3), 275.15, dtype=np.float)
        cube = add_forecast_reference_time_and_forecast_period(
            set_up_cube(data, "air_temperature", "Kelvin"))
        cube = cube[0]
        cube.remove_coord("realization")
        self.cube = cube

        # Cube with multiple times.
        data = np.full((3, 2, 3, 3), 275.15, dtype=np.float)
        temp_cube = set_up_cube(data, "air_temperature", "Kelvin", timesteps=2)
        fp_points = [3.0, 4.0]
        temp_cubes = iris.cube.CubeList([])
        for cube, fp_point in zip(temp_cube.slices_over("time"), fp_points):
            temp_cubes.append(add_forecast_reference_time_and_forecast_period(
                cube, time_point=cube.coord("time").points, fp_point=fp_point))
        cube_orig = temp_cubes.merge_cube()
        cube_orig.transpose([1, 0, 2, 3])
        cube_orig = cube_orig[0]
        cube_orig.remove_coord("realization")
        self.cube_orig = cube_orig

        # Cube without forecast_period.
        cube_orig_without_fp = cube_orig.copy()
        cube_orig_without_fp.remove_coord("forecast_period")
        self.cube_orig_without_fp = cube_orig_without_fp
        self.cube_without_fp = cube_orig_without_fp.copy().collapsed(
            "forecast_reference_time", iris.analysis.MEAN)

        # Cube with a model, model_id and model realization.
        cube_orig_model = cube_orig.copy()
        cube_orig_model.add_aux_coord(
            AuxCoord([1000, 1000], long_name="model"), data_dims=0)
        cube_orig_model.add_aux_coord(
            AuxCoord(["Operational MOGREPS-UK Model Forecast",
                      "Operational UKV Model Forecast"], long_name="model_id"),
            data_dims=0)
        cube_orig_model.add_aux_coord(
            AuxCoord([0, 1001], long_name="model_realization"), data_dims=0)
        cube_orig_model.add_aux_coord(
            AuxCoord([0, 1], long_name="realization"), data_dims=0)
        self.cube_orig_model = cube_orig_model
        self.cube_model = cube_orig_model.collapsed(
            "forecast_reference_time", iris.analysis.MEAN)

    def test_basic(self):
        """Test that conform_metadata returns a cube."""
        result = conform_metadata(self.cube, self.cube_orig)
        self.assertIsInstance(result, iris.cube.Cube)

    def test_with_forecast_reference_time_and_forecast_period(self):
        """Test that a cube is dealt with correctly, if the cube contains
        a forecast_reference_time and forecast_period coordinate."""
        result = conform_metadata(self.cube, self.cube_orig)
        self.assertEqual(
            result.coord("forecast_reference_time").points,
            np.max(self.cube_orig.coord("forecast_reference_time").points))
        self.assertFalse(result.coord("forecast_reference_time").bounds)
        self.assertEqual(
            result.coord("forecast_period").points,
            np.min(self.cube_orig.coord("forecast_period").points))
        self.assertFalse(result.coord("forecast_period").bounds)

    def test_with_forecast_reference_time_and_without_forecast_period(self):
        """Test that a cube is dealt with correctly, if the cube contains a
        forecast_reference_time coordinate but not a forecast_period."""
        result = conform_metadata(
            self.cube_without_fp, self.cube_orig_without_fp)
        self.assertEqual(
            result.coord("forecast_reference_time").points,
            np.max(self.cube_orig.coord("forecast_reference_time").points))
        self.assertFalse(result.coord("forecast_reference_time").bounds)
        self.assertEqual(
            result.coord("forecast_period").points,
            np.min(self.cube_orig.coord("forecast_period").points))
        self.assertFalse(result.coord("forecast_period").bounds)

    def test_with_model_model_id_and_model_realization(self):
        """Test that a cube is dealt with correctly, if the cube contains a
        model, model_id and model_realization coordinate."""
        result = conform_metadata(self.cube_model, self.cube_orig_model)
        self.assertFalse(result.coords("model_id"))
        self.assertFalse(result.coords("model_realization"))


if __name__ == '__main__':
    unittest.main()
