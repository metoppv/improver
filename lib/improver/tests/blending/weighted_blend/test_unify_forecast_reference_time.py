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
"""Module to test the unify_forecast_reference_times function."""

import unittest

import datetime
import numpy as np
import iris
from iris.tests import IrisTest

from improver.tests.blending.weights.helper_functions import (
    set_up_temperature_cube, add_model_id_and_model_configuration)
from improver.tests.ensemble_calibration.ensemble_calibration.helper_functions\
    import add_forecast_reference_time_and_forecast_period
from improver.tests.nbhood.nbhood.test_NeighbourhoodProcessing \
    import set_up_cube
from improver.blending.weighted_blend import unify_forecast_reference_time


class Test_unify_forecast_reference_time(IrisTest):

    """Test the unify_forecast_reference_time function."""

    def setUp(self):
        """Set up a UK deterministic cube for testing."""
        cube_uk_det = add_model_id_and_model_configuration(
            set_up_temperature_cube(timesteps=3), model_ids=[1000],
            model_configurations=["uk_det"], promote_to_new_axis=True)
        self.cube_uk_det = add_forecast_reference_time_and_forecast_period(
            cube_uk_det, time_point=[412233.0, 412235.0, 412237.0],
            fp_point=[6., 8., 10.])

    def test_cubelist_input(self):
        """Test when supplying a cubelist as input containing cubes
        representing UK deterministic and UK ensemble model configuration
        and unifying the forecast_reference_time, so that both model
        configurations have a common forecast_reference_time."""
        cube_uk_ens = add_model_id_and_model_configuration(
            set_up_temperature_cube(timesteps=3), model_ids=[2000],
            model_configurations=["uk_ens"], promote_to_new_axis=True)
        cube_uk_ens = add_forecast_reference_time_and_forecast_period(
            cube_uk_ens, time_point=[412231.0, 412233.0, 412235.0],
            fp_point=[5., 7., 9.])
        cubes = iris.cube.CubeList([self.cube_uk_det, cube_uk_ens])

        cycletime = datetime.datetime(2017, 1, 10, 6, 0)

        expected_uk_det = self.cube_uk_det.copy()
        frt_units = expected_uk_det.coord('forecast_reference_time').units
        frt_points = [frt_units.date2num(cycletime)]
        expected_uk_det.coord("forecast_reference_time").points = frt_points
        expected_uk_det.coord("forecast_period").points = (
            np.array([3., 5., 7.]))
        expected_uk_ens = cube_uk_ens.copy()
        expected_uk_ens.coord("forecast_reference_time").points = frt_points
        expected_uk_ens.coord("forecast_period").points = (
            np.array([1., 3., 5.]))
        expected = iris.cube.CubeList([expected_uk_det, expected_uk_ens])

        result = unify_forecast_reference_time(cubes, cycletime)

        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(result, expected)

    def test_cube_input(self):
        """Test when supplying a cube representing a UK deterministic model
        configuration only. This effectively updates the
        forecast_reference_time on the cube to the specified cycletime."""
        cycletime = datetime.datetime(2017, 1, 10, 6, 0)

        expected_uk_det = self.cube_uk_det.copy()
        frt_units = expected_uk_det.coord('forecast_reference_time').units
        frt_points = [frt_units.date2num(cycletime)]
        expected_uk_det.coord("forecast_reference_time").points = frt_points
        expected_uk_det.coord("forecast_period").points = (
            np.array([3., 5., 7.]))

        result = unify_forecast_reference_time(self.cube_uk_det, cycletime)

        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(result[0], expected_uk_det)

    def test_cube_input_no_forecast_period_coordinate(self):
        """Test when supplying a cube representing a UK deterministic model
        configuration only. This forces a forecast_period coordinate to be
        created from a forecast_reference_time coordinate and a time
        coordinate."""
        cycletime = datetime.datetime(2017, 1, 10, 6, 0)

        expected_uk_det = self.cube_uk_det.copy()
        frt_units = expected_uk_det.coord('forecast_reference_time').units
        frt_points = [frt_units.date2num(cycletime)]
        expected_uk_det.coord("forecast_reference_time").points = frt_points
        expected_uk_det.coord("forecast_period").points = (
            np.array([3., 5., 7.]))
        expected_uk_det.coord("forecast_period").convert_units("seconds")

        cube_uk_det = self.cube_uk_det.copy()
        cube_uk_det.remove_coord("forecast_period")

        result = unify_forecast_reference_time(cube_uk_det, cycletime)

        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(result[0], expected_uk_det)


if __name__ == '__main__':
    unittest.main()
