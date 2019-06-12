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
    helper_functions import SetupCubes
from improver.utilities.warnings_handler import ManageWarnings

try:
    imp.find_module('statsmodels')
    STATSMODELS_FOUND = True
except ImportError:
    STATSMODELS_FOUND = False


IGNORED_MESSAGES = [
    "Collapsing a non-contiguous coordinate",  # Originating from Iris
    "The statsmodels can not be imported",
    "invalid escape sequence",  # Originating from statsmodels
    "can't resolve package from",  # Originating from statsmodels
    "Minimisation did not result in convergence",  # From calibration code
    "The final iteration resulted in",  # From calibration code
]
WARNING_TYPES = [
    UserWarning,
    ImportWarning,
    DeprecationWarning,
    ImportWarning,
    UserWarning,
    UserWarning,
]


class SetupExpectedResults(IrisTest):

    """Set up expected results."""

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def setUp(self):
        """Set up the expected results for the mean and variance,
        where either a temperature or wind speed cube has been provided
        as input."""
        super().setUp()
        self.expected_temperature_mean_data = np.array(
            [[273.74277, 274.65567, 275.41644],
             [276.84668, 277.63785, 278.39862],
             [279.49414, 280.16364, 280.98523]], dtype=np.float32)

        self.expected_temperature_variance_data = np.array(
            [[0.21338412, 0.21577403, 0.01273912],
             [0.02468313, 0.02149836, 0.01273912],
             [0.05812284, 0.00318527, 0.00079632]], dtype=np.float32)

        self.expected_wind_speed_mean_data = np.array(
            [[0.45730978, 1.3972956, 2.1806173],
             [3.6532617, 4.4679155, 5.2512374],
             [6.3792205, 7.068543, 7.9145303]], dtype=np.float32)

        self.expected_wind_speed_variance_data = np.array(
            [[2.128061, 2.1518826, 0.12704849],
             [0.24615654, 0.21439417, 0.12704849],
             [0.5796586, 0.03176204, 0.00794059]], dtype=np.float32)


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

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
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
            self.expected_temperature_mean_data)
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
            self.expected_temperature_mean_data)
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
            self.expected_wind_speed_mean_data)
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
            self.expected_wind_speed_mean_data)
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
            [[274.1963, 275.15448, 275.31494],
             [277.03445, 277.40604, 278.37195],
             [280.06644, 280.30856, 281.2213]], dtype=np.float32)

        self.expected_specific_temperature_variance_data = np.array(
            [[0.8972256, 0.9072746, 0.05356484],
             [0.10378637, 0.09039519, 0.05356484],
             [0.24439174, 0.01339334, 0.00334842]], dtype=np.float32)

        self.expected_specific_wind_speed_predictor_data = np.array(
            [[0.97038287, 1.7892897, 2.209787],
             [3.8476007, 4.288108, 5.145053],
             [6.708772, 7.083208, 7.9021144]], dtype=np.float32)

        self.expected_specific_wind_speed_variance_data = np.array(
            [[1.5944895, 1.6123377, 0.09524001],
             [0.18448117, 0.16068336, 0.09524001],
             [0.4343561, 0.02384708, 0.00599896]], dtype=np.float32)

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
            predictor_of_mean_flag=predictor_of_mean_flag)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertArrayAlmostEqual(
            calibrated_predictor.data,
            self.expected_specific_temperature_predictor_data, decimal=4)
        self.assertArrayAlmostEqual(
            calibrated_variance.data,
            self.expected_specific_temperature_variance_data)
        self.assertArrayAlmostEqual(
            calibrated_predictor.data,
            self.expected_temperature_mean_data, decimal=0)
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
            predictor_of_mean_flag=predictor_of_mean_flag)
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
            self.expected_wind_speed_mean_data, decimal=0)
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
            [[274.1325, 275.0439, 275.2852],
             [277.02405, 277.4005, 278.3493],
             [280.0655, 280.34113, 281.23315]], dtype=np.float32)

        self.expected_specific_temperature_variance_data = np.array(
            [[0.9980779, 1.0092506, 0.06006714],
             [0.11590514, 0.10101636, 0.06006713],
             [0.27223495, 0.01540309, 0.00423481]], dtype=np.float32)

        self.expected_specific_wind_speed_predictor_data = np.array(
            [[0.8932284, 1.6185861, 2.3541362],
             [3.8048353, 4.418258, 5.1315794],
             [6.3740964, 6.950509, 7.6758633]], dtype=np.float32)

        self.expected_specific_wind_speed_variance_data = np.array(
            [[1.6119667, 1.6299378, 0.10240279],
             [0.1922579, 0.16829637, 0.10240279],
             [0.4438519, 0.0305187, 0.01254779]], dtype=np.float32)

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
            predictor_of_mean_flag=predictor_of_mean_flag)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertArrayAlmostEqual(
            calibrated_predictor.data,
            self.expected_specific_temperature_predictor_data, decimal=4)
        self.assertArrayAlmostEqual(
            calibrated_variance.data,
            self.expected_specific_temperature_variance_data)
        self.assertArrayAlmostEqual(
            calibrated_predictor.data,
            self.expected_temperature_mean_data, decimal=0)
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
            predictor_of_mean_flag=predictor_of_mean_flag)
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
            self.expected_wind_speed_mean_data, decimal=0)
        self.assertArrayAlmostEqual(
            calibrated_variance.data,
            self.expected_wind_speed_variance_data, decimal=0)


if __name__ == '__main__':
    unittest.main()
