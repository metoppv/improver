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


class SetupCubesWithVariance(IrisTest):

    """Set up cubes for class."""

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def setUp(self):
        """Set up temperature and wind speed cubes for testing."""
        # Note: test_temperature_realizations_data_check produces ~0.5K
        # different results when the temperature forecast cube is float32
        # below. A bug?
        self.calibration_method = "ensemble model output_statistics"
        data = np.array([[[0., 1.1, 3.],
                          [4., 5.3, 6.],
                          [7., 8.2, 9.]],
                         [[1., 2., 3],
                          [4., 5.6, 6.],
                          [7., 8., 9.]],
                         [[2., 3., 3.],
                          [4.8, 5., 6.],
                          [7.9, 8., 9.]]])
        data = data + 273.15
        data = data.astype(np.float32)
        self.current_temperature_forecast_cube = set_up_variable_cube(
            data, units="Kelvin", realizations=[0, 1, 2])

        self.historic_temperature_forecast_cube = (
            _create_historic_forecasts(self.current_temperature_forecast_cube))

        self.temperature_truth_cube = (
            _create_truth(self.current_temperature_forecast_cube))

        self.expected_temperature_predictor_data = np.array(
            [[273.51123, 274.5292, 275.31207],
             [276.63153, 277.5993, 278.30295],
             [279.66272, 280.3389, 281.29385]], dtype=np.float32)

        self.expected_temperature_variance_data = np.array(
            [[5.9336913e-01, 5.3600669e-01, 5.1369886e-09],
             [1.2659121e-01, 5.3404309e-02, 5.1369886e-09],
             [1.6020750e-01, 7.9125594e-03, 5.1369886e-09]], dtype=np.float32)

        # Create a cube for testing wind speed.
        data = np.array([[[0., 1.1, 3.],
                          [4., 5.3, 6.],
                          [7., 8.2, 9.]],
                         [[1., 2., 3],
                          [4., 5.6, 6.],
                          [7., 8., 9.]],
                         [[2., 3., 3.],
                          [4.8, 5., 6.],
                          [7.9, 8., 9.]]])
        data = data.astype(np.float32)
        self.current_wind_speed_forecast_cube = set_up_variable_cube(
            data, name="wind_speed", units="m s-1", realizations=[0, 1, 2])

        self.historic_wind_speed_forecast_cube = (
            _create_historic_forecasts(self.current_wind_speed_forecast_cube))

        self.wind_speed_truth_cube = (
            _create_truth(self.current_wind_speed_forecast_cube))

        self.expected_wind_speed_predictor_data = np.array(
            [[0.99999917, 2.0333316, 2.9999974],
             [4.266663, 5.2999954, 5.9999948],
             [7.299994, 8.06666, 8.999992]], dtype=np.float32)

        self.expected_wind_speed_variance_data = np.array(
            [[1.1024998, 0.99592483, 0.],
             [0.23520009, 0.09922495, 0.],
             [0.297675, 0.01469997, 0.]], dtype=np.float32)


class Test_process_basic(SetupCubesWithVariance):

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


class Test_process_check_data(SetupCubesWithVariance):

    """Test the data with variance output from the process method."""

    def setUp(self):
        super().setUp()
        self.expected_temperature_predictor_data = np.array(
            [[273.7406, 274.67642, 275.55182],
             [276.69894, 277.63474, 278.26865],
             [279.44595, 280.14026, 280.9855]], dtype=np.float32)

        self.expected_temperature_variance_data = np.array(
            [[1., 0.9033276 , 0.],
             [0.2133431, 0.09000183, 0.],
             [0.26999635, 0.01333496, 0.]], dtype=np.float32)

        self.expected_wind_speed_predictor_data = np.array(
            [[0.59150004, 1.5272971, 2.4027202],
             [3.5498266, 4.4856234, 5.1195507],
             [6.2968435, 6.9911447, 7.836381]], dtype=np.float32)

        self.expected_wind_speed_variance_data = np.array(
            [[1.1024998, 0.99592483, 0.],
             [0.23520009, 0.09922495, 0.],
             [0.297675, 0.01469997, 0.]], dtype=np.float32)

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
        plugin = Plugin(self.calibration_method, distribution,
                        decimals=5)
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
class Test_process_with_statsmodels(SetupCubesWithVariance):

    """Additional tests for the process method when the statsmodels module
    is available. The statsmodels module is used for creating an initial
    guess for the coefficients that will be solved as part of the
    calibration."""

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
            max_iterations=400)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertArrayAlmostEqual(
            calibrated_predictor.data,
            self.expected_temperature_predictor_data, decimal=3)
        self.assertArrayAlmostEqual(
            calibrated_variance.data,
            self.expected_temperature_variance_data, decimal=3)

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
        expected_wind_speed_predictor_data = np.array(
            [[1.3333356, 2.0000021, 2.6666684],
             [3.3333352, 4.0000014, 4.666668],
             [5.3333344, 6.000001, 6.6666675]], dtype=np.float32)
        expected_wind_speed_variance_data = np.array(
            [[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]], dtype=np.float32)
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
            calibrated_predictor.data, expected_wind_speed_predictor_data,
            decimal=5)
        self.assertArrayAlmostEqual(
            calibrated_variance.data, expected_wind_speed_variance_data,
            decimal=5)


@unittest.skipIf(
    STATSMODELS_FOUND is True, "statsmodels module is available.")
class Test_process_without_statsmodels(SetupCubesWithVariance):

    """Additional tests for the process method when the statsmodels module
    is not available. A simple initial guess will be used for the
    coefficients that will be solved as part of the calibration."""

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
            self.expected_temperature_predictor_data, decimal=4)
        self.assertArrayAlmostEqual(
            calibrated_variance.data,
            self.expected_temperature_variance_data, decimal=4)

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
            self.expected_wind_speed_predictor_data, decimal=2)
        self.assertArrayAlmostEqual(
            calibrated_variance.data,
            self.expected_wind_speed_variance_data, decimal=2)


if __name__ == '__main__':
    unittest.main()
