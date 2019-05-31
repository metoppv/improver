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
"""
Unit tests for the `ensemble_calibration.EnsembleCalibration` class.

"""
import imp
import unittest

import iris
from iris.tests import IrisTest
import numpy as np

from improver.ensemble_calibration.ensemble_calibration import (
    EnsembleCalibration as Plugin)
from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import _create_historic_forecasts, _create_truth
from improver.tests.set_up_test_cubes import set_up_variable_cube
from improver.utilities.warnings_handler import ManageWarnings

try:
    imp.find_module('statsmodels')
    STATSMODELS_FOUND = True
except ImportError:
    STATSMODELS_FOUND = False


IGNORED_MESSAGES = ["Collapsing a non-contiguous coordinate.",
                    "Not importing directory .*sphinxcontrib'",
                    "The pandas.core.datetools module is deprecated",
                    "numpy.dtype size changed",
                    "The statsmodels can not be imported",
                    "invalid escape sequence",
                    "can't resolve package from",
                    "Minimisation did not result in"
                    " convergence",
                    "\nThe final iteration resulted in a percentage "
                    "change that is greater than the"
                    " accepted threshold ",
                    "divide by zero encountered in true_divide",
                    "invalid value encountered in"]
WARNING_TYPES = [UserWarning, ImportWarning, FutureWarning, RuntimeWarning,
                 ImportWarning, DeprecationWarning, ImportWarning, UserWarning,
                 UserWarning, RuntimeWarning, RuntimeWarning]


class SetupCubes(IrisTest):

    """Set up cubes for testing."""

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def setUp(self):
        """Set up temperature and wind speed cubes for testing."""
        # Note: test_temperature_realizations_data_check produces ~0.5K
        # different results when the temperature forecast cube is float32
        # below. A bug?
        super().setUp()
        self.calibration_method = "ensemble model output_statistics"
        data = np.array([[[0.3, 1.1, 2.6],
                          [4.2, 5.3, 6.],
                          [7.1, 8.2, 9.]],
                         [[0.7, 2., 3],
                          [4.3, 5.6, 6.4],
                          [7., 8., 9.]],
                         [[2.1, 3., 3.],
                          [4.8, 5., 6.],
                          [7.9, 8., 8.9]]])
        data = data + 273.15
        data = data.astype(np.float32)
        self.current_temperature_forecast_cube = set_up_variable_cube(
            data, units="Kelvin", realizations=[0, 1, 2])

        self.historic_temperature_forecast_cube = (
            _create_historic_forecasts(self.current_temperature_forecast_cube))

        self.temperature_truth_cube = (
            _create_truth(self.current_temperature_forecast_cube))

        # Create a cube for testing wind speed.
        data = np.array([[[0.3, 1.1, 2.6],
                          [4.2, 5.3, 6.],
                          [7.1, 8.2, 9.]],
                         [[0.7, 2., 3],
                          [4.3, 5.6, 6.4],
                          [7., 8., 9.]],
                         [[2.1, 3., 3.],
                          [4.8, 5., 6.],
                          [7.9, 8., 8.9]]])
        data = data.astype(np.float32)
        self.current_wind_speed_forecast_cube = set_up_variable_cube(
            data, name="wind_speed", units="m s-1", realizations=[0, 1, 2])

        self.historic_wind_speed_forecast_cube = (
            _create_historic_forecasts(self.current_wind_speed_forecast_cube))

        self.wind_speed_truth_cube = (
            _create_truth(self.current_wind_speed_forecast_cube))


class SetupExpectedResults(IrisTest):

    """Set up expected results."""

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def setUp(self):
        """Set up temperature and wind speed cubes for testing."""
        super().setUp()
        self.expected_temperature_predictor_data = np.array(
            [[273.74304, 274.6559 , 275.41663],
             [276.84677, 277.63788, 278.39862],
             [279.49405, 280.16348, 280.98505]], dtype=np.float32)

        self.expected_temperature_variance_data = np.array(
            [[0.21317102, 0.21555844, 0.01273511],
             [0.02466668, 0.02148522, 0.01273511],
             [0.05807154, 0.00319121, 0.00080475]], dtype=np.float32)

        self.expected_wind_speed_predictor_data = np.array(
            [[0.45753962, 1.3974727, 2.1807506],
             [3.6533124, 4.4679203, 5.251199],
             [6.379119, 7.0684023, 7.914342]], dtype=np.float32)

        self.expected_wind_speed_variance_data = np.array(
            [[2.1278856 , 2.151705  , 0.12705675],
             [0.24615386, 0.2143944 , 0.12705675],
             [0.5796253 , 0.03177907, 0.0079598]], dtype=np.float32)

class Test_process_basic(SetupCubes):

    """Test the basic output from the process method."""

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic_temperature(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        with the desired length.
        The ensemble mean is the predictor.
        """
        distribution = "gaussian"
        plugin = Plugin(self.calibration_method, distribution)
        result = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic_temperature_realizations(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        with the desired length.
        The ensemble realizations is the predictor.
        """
        distribution = "gaussian"
        predictor_of_mean_flag = "realizations"
        plugin = Plugin(
            self.calibration_method, distribution,
            predictor_of_mean_flag=predictor_of_mean_flag)
        result = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic_wind_speed(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        with the desired length.
        The ensemble mean is the predictor.
        """
        distribution = "truncated gaussian"
        plugin = Plugin(self.calibration_method, distribution)
        result = plugin.process(
            self.current_wind_speed_forecast_cube,
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic_wind_speed_realizations(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        with the desired length.
        The ensemble realizations is the predictor.
        """
        distribution = "truncated gaussian"
        predictor_of_mean_flag = "realizations"
        plugin = Plugin(
            self.calibration_method, distribution,
            predictor_of_mean_flag=predictor_of_mean_flag)
        result = plugin.process(
            self.current_wind_speed_forecast_cube,
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_alternative_calibration_name(self):
        """
        Test that the plugin returns the calibrated predictor and the
        calibrated variance if an alternative name for the calibration
        is provided. The ensemble mean is the predictor.
        """
        calibration_method = "nonhomogeneous gaussian regression"
        distribution = "gaussian"
        plugin = Plugin(calibration_method, distribution)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertIsInstance(calibrated_predictor, iris.cube.Cube)
        self.assertIsInstance(calibrated_variance, iris.cube.Cube)

    def test_unknown_calibration_method(self):
        """
        Test that the plugin raises an error if an unknown calibration method
        is requested.
        The ensemble mean is the predictor.
        """
        calibration_method = "unknown"
        distribution = "gaussian"
        plugin = Plugin(calibration_method, distribution)
        msg = "unknown"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(
                self.current_temperature_forecast_cube,
                self.historic_temperature_forecast_cube,
                self.temperature_truth_cube)


class Test_process_check_data(SetupCubes, SetupExpectedResults):

    """Test the data with variance output from the process method."""

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_temperature_data_check(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of temperature cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance.
        The ensemble mean is the predictor.
        """
        distribution = "gaussian"
        plugin = Plugin(self.calibration_method, distribution)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertArrayAlmostEqual(
            calibrated_predictor.data,
            self.expected_temperature_predictor_data)
        self.assertArrayAlmostEqual(
            calibrated_variance.data,
            self.expected_temperature_variance_data)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_temperature_data_check_max_iterations(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of temperature cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance when the maximum number of iterations is specified.
        The ensemble mean is the predictor.
        """
        distribution = "gaussian"
        plugin = Plugin(self.calibration_method, distribution,
                        max_iterations=10000)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertArrayAlmostEqual(
            calibrated_predictor.data,
            self.expected_temperature_predictor_data)
        self.assertArrayAlmostEqual(
            calibrated_variance.data,
            self.expected_temperature_variance_data)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_wind_speed_data_check(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of wind_speed cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance.
        The ensemble mean is the predictor.
        """
        distribution = "truncated gaussian"
        plugin = Plugin(self.calibration_method, distribution)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_wind_speed_forecast_cube,
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)
        self.assertArrayAlmostEqual(
            calibrated_predictor.data,
            self.expected_wind_speed_predictor_data)
        self.assertArrayAlmostEqual(
            calibrated_variance.data,
            self.expected_wind_speed_variance_data)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_wind_speed_data_check_max_iterations(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of wind_speed cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance when the maximum number of iterations is specified.
        The ensemble mean is the predictor.
        """
        distribution = "truncated gaussian"
        plugin = Plugin(self.calibration_method, distribution,
                        max_iterations=10000)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_wind_speed_forecast_cube,
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)
        self.assertArrayAlmostEqual(
            calibrated_predictor.data,
            self.expected_wind_speed_predictor_data)
        self.assertArrayAlmostEqual(
            calibrated_variance.data,
            self.expected_wind_speed_variance_data)


@unittest.skipIf(
    STATSMODELS_FOUND is False, "statsmodels module not available.")
class Test_process_with_statsmodels(SetupCubes, SetupExpectedResults):

    """Additional tests for the process method when the statsmodels module
    is available. The statsmodels module is used for creating an initial
    guess for the coefficients that will be solved as part of the
    calibration."""

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def setUp(self):
        """Set up temperature and wind speed cubes for testing."""
        super().setUp()
        self.expected_specific_temperature_predictor_data = np.array(
            [[274.19583, 275.15402, 275.31442],
             [277.03394, 277.4055, 278.3714],
             [280.06595, 280.30804, 281.2208]], dtype=np.float32)

        self.expected_specific_temperature_variance_data = np.array(
            [[0.89734596, 0.90739626, 0.05357203],
             [0.10380029, 0.09040731, 0.05357203],
             [0.2444245, 0.01339514, 0.00334887]], dtype=np.float32)

        self.expected_specific_wind_speed_predictor_data = np.array(
            [[1.0959406, 2.043556, 2.2561445],
             [3.8869126, 4.308816, 5.229468],
             [6.7901444, 7.0756545, 7.9622397]], dtype=np.float32)

        self.expected_specific_wind_speed_variance_data = np.array(
            [[1.5079881, 1.5248687, 0.09002924],
             [0.17443168, 0.15192421, 0.09002924],
             [0.4107582, 0.0225073, 0.00562692]], dtype=np.float32)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_temperature_realizations_data_check(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of temperature cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance.
        The ensemble realizations is the predictor.
        """
        distribution = "gaussian"
        predictor_of_mean_flag = "realizations"
        plugin = Plugin(
            self.calibration_method, distribution,
            predictor_of_mean_flag=predictor_of_mean_flag,
            max_iterations=2000)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertArrayAlmostEqual(
            calibrated_predictor.data,
            self.expected_specific_temperature_predictor_data)
        self.assertArrayAlmostEqual(
            calibrated_variance.data,
            self.expected_specific_temperature_variance_data)
        self.assertArrayAlmostEqual(
            calibrated_predictor.data,
            self.expected_temperature_predictor_data, decimal=0)
        self.assertArrayAlmostEqual(
            calibrated_variance.data,
            self.expected_temperature_variance_data, decimal=0)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_wind_speed_realizations_data_check(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of wind_speed cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance.
        The ensemble realizations is the predictor.
        """
        distribution = "truncated gaussian"
        predictor_of_mean_flag = "realizations"
        plugin = Plugin(
            self.calibration_method, distribution,
            predictor_of_mean_flag=predictor_of_mean_flag,
            max_iterations=400)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_wind_speed_forecast_cube,
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)
        self.assertArrayAlmostEqual(
            calibrated_predictor.data,
            self.expected_specific_wind_speed_predictor_data)
        self.assertArrayAlmostEqual(
            calibrated_variance.data,
            self.expected_specific_wind_speed_variance_data)
        self.assertArrayAlmostEqual(
            calibrated_predictor.data,
            self.expected_wind_speed_predictor_data, decimal=0)
        self.assertArrayAlmostEqual(
            calibrated_variance.data,
            self.expected_wind_speed_variance_data, decimal=0)


@unittest.skipIf(
    STATSMODELS_FOUND is True, "statsmodels module is available.")
class Test_process_without_statsmodels(SetupCubes, SetupExpectedResults):

    """Additional tests for the process method when the statsmodels module
    is not available. A simple initial guess will be used for the
    coefficients that will be solved as part of the calibration."""
    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def setUp(self):
        """Set up temperature and wind speed cubes for testing."""
        super().setUp()
        self.expected_specific_temperature_predictor_data = np.array(
            [[273.6069, 274.5989, 275.27478],
             [276.86993, 277.62445, 278.48648],
             [279.79132, 280.4114, 281.3137]], dtype=np.float32)

        self.expected_specific_temperature_variance_data = np.array(
            [[1.2898316, 1.3042778, 0.07700347],
             [0.14920083, 0.12994996, 0.07700346],
             [0.351332, 0.0192538, 0.00481345]], dtype=np.float32)

        self.expected_specific_wind_speed_predictor_data = np.array(
            [[0.93635756, 1.6808116, 2.3318303],
             [3.820501, 4.3862634, 5.1313596],
             [6.44708, 6.968665, 7.713064]], dtype=np.float32)

        self.expected_specific_wind_speed_variance_data = np.array(
            [[1.5596627, 1.576899, 0.11182244],
             [0.19800353, 0.17502174, 0.11182244],
             [0.43931028, 0.04287757, 0.02564146]], dtype=np.float32)
    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_temperature_realizations_data_check(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of temperature cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance.
        The ensemble realizations is the predictor.
        """
        distribution = "gaussian"
        predictor_of_mean_flag = "realizations"
        plugin = Plugin(
            self.calibration_method, distribution,
            predictor_of_mean_flag=predictor_of_mean_flag,
            max_iterations=2000)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertArrayAlmostEqual(
            calibrated_predictor.data,
            self.expected_specific_temperature_predictor_data)
        self.assertArrayAlmostEqual(
            calibrated_variance.data,
            self.expected_specific_temperature_variance_data)
        self.assertArrayAlmostEqual(
            calibrated_predictor.data,
            self.expected_temperature_predictor_data, decimal=0)
        self.assertArrayAlmostEqual(
            calibrated_variance.data,
            self.expected_temperature_variance_data, decimal=0)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_wind_speed_realizations_data_check(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of wind_speed cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance. The ensemble realizations are the predictor.
        The choice to assert that the arrays are almost equal to only 2
        decimal places is to avoid sensitive tests that are overly
        dependent upon package versions and processor optimisation, in order
        to converge to a stable solution.
        """
        distribution = "truncated gaussian"
        predictor_of_mean_flag = "realizations"
        plugin = Plugin(
            self.calibration_method, distribution,
            predictor_of_mean_flag=predictor_of_mean_flag,
            max_iterations=2000)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_wind_speed_forecast_cube,
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)
        self.assertArrayAlmostEqual(
            calibrated_predictor.data,
            self.expected_specific_wind_speed_predictor_data)
        self.assertArrayAlmostEqual(
            calibrated_variance.data,
            self.expected_specific_wind_speed_variance_data)
        self.assertArrayAlmostEqual(
            calibrated_predictor.data,
            self.expected_wind_speed_predictor_data, decimal=0)
        self.assertArrayAlmostEqual(
            calibrated_variance.data,
            self.expected_wind_speed_variance_data, decimal=0)


if __name__ == '__main__':
    unittest.main()
