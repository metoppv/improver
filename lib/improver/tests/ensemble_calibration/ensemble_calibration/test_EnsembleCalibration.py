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
    helper_functions import SetupCubes, EnsembleCalibrationAssertions
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
        # Expected values for the ensemble mean when using
        # a Gaussian distribution using either the
        # ensemble mean or the realizations as the predictor for both
        # with and without the statsmodels module.
        self.expected_gaussian_mean_data = np.array(
            [[273.7428, 274.6557, 275.4164],
             [276.8467, 277.6379, 278.3986],
             [279.4941, 280.1636, 280.9852]], dtype=np.float32)

        self.expected_gaussian_realization_statsmodels = np.array(
            [[274.1949, 275.1531, 275.3136],
             [277.033, 277.4047, 278.3706],
             [280.065, 280.3072, 281.2199]], dtype=np.float32)

        self.expected_gaussian_realization_no_statsmodels = np.array(
            [[274.1325, 275.0439, 275.2852],
             [277.0241, 277.4005, 278.3493],
             [280.0655, 280.3411, 281.2332]], dtype=np.float32)

        # Expected values for the ensemble variance when using
        # a Gaussian distribution using either the
        # ensemble mean or the realizations as the predictor for both
        # with and without the statsmodels module.
        self.expected_gaussian_variance_data = np.array(
            [[0.2134, 0.2158, 0.0127],
             [0.0247, 0.0215, 0.0127],
             [0.0581, 0.0032, 0.0008]], dtype=np.float32)

        self.expected_gaussian_variance_statsmodels = np.array(
            [[0.8974, 0.9074, 0.0536],
             [0.1038, 0.0904, 0.0536],
             [0.2444, 0.0134, 0.0033]], dtype=np.float32)

        self.expected_gaussian_variance_no_statsmodels = np.array(
            [[0.9981, 1.0093, 0.0601],
             [0.1159, 0.1010, 0.0601],
             [0.2722, 0.0154, 0.0042]], dtype=np.float32)

        # Expected values for the ensemble mean when using
        # a truncated Gaussian distribution using either the
        # ensemble mean or the realizations as the predictor for both
        # with and without the statsmodels module.
        self.expected_truncated_gaussian_mean_data = np.array(
            [[0.4573, 1.3973, 2.1806],
             [3.6533, 4.4679, 5.2512],
             [6.3792, 7.0685, 7.9145]], dtype=np.float32)
        self.expected_truncated_gaussian_realization_statsmodels = (
            np.array(
                [[0.9704, 1.7893, 2.2098],
                 [3.8476, 4.2881, 5.1451],
                 [6.7088, 7.0832, 7.9021]], dtype=np.float32))
        self.expected_truncated_gaussian_realization_no_statsmodels = (
            np.array(
                [[0.8932, 1.6186, 2.3541],
                 [3.8048, 4.4183, 5.1316],
                 [6.3741, 6.9505, 7.6759]], dtype=np.float32))

        # Expected values for the ensemble variance when using
        # a truncated Gaussian distribution using either the
        # ensemble mean or the realizations as the predictor for both
        # with and without the statsmodels module.
        self.expected_truncated_gaussian_variance_data = np.array(
            [[2.1281, 2.1519, 0.1270],
             [0.2462, 0.2144, 0.1270],
             [0.5797, 0.0318, 0.0079]], dtype=np.float32)
        self.expected_truncated_gaussian_variance_statsmodels = (
            np.array(
                [[1.5945, 1.6123, 0.0952],
                 [0.1845, 0.1607, 0.0952],
                 [0.4344, 0.0238, 0.0060]], dtype=np.float32))
        self.expected_truncated_gaussian_variance_no_statsmodels = (
            np.array(
                [[1.6120, 1.6299, 0.1024],
                 [0.1923, 0.1683, 0.1024],
                 [0.4439, 0.0305, 0.0125]], dtype=np.float32))


class Test__init__(unittest.TestCase):

    """Test the __init__ method."""

    def test_raises_error(self):
        """Test an error is raised for an invalid distribution"""
        distribution = "biscuits"
        msg = "Given distribution biscuits not available. "
        with self.assertRaisesRegex(ValueError, msg):
            Plugin(distribution)


class Test_process_basic(SetupCubes):

    """Test the basic output from the process method."""

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic_gaussian(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        with the desired length. The ensemble mean is the predictor.
        """
        distribution = "gaussian"
        plugin = Plugin(distribution)
        result = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic_gaussian_realizations(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        with the desired length. The ensemble realizations is the predictor.
        """
        distribution = "gaussian"
        predictor_of_mean_flag = "realizations"
        plugin = Plugin(distribution,
                        predictor_of_mean_flag=predictor_of_mean_flag)
        result = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic_truncated_gaussian(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        with the desired length. The ensemble mean is the predictor.
        """
        distribution = "truncated gaussian"
        plugin = Plugin(distribution)
        result = plugin.process(
            self.current_wind_speed_forecast_cube,
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic_truncated_gaussian_realizations(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        with the desired length. The ensemble realizations is the predictor.
        """
        distribution = "truncated gaussian"
        predictor_of_mean_flag = "realizations"
        plugin = Plugin(distribution,
                        predictor_of_mean_flag=predictor_of_mean_flag)
        result = plugin.process(
            self.current_wind_speed_forecast_cube,
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


class Test_process_check_data(SetupCubes, SetupExpectedResults,
                              EnsembleCalibrationAssertions):

    """Test the data with variance output from the process method."""

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_gaussian_data_check(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of temperature cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance. The ensemble mean is the predictor.
        """
        distribution = "gaussian"
        plugin = Plugin(distribution)

        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_predictor.data,
            self.expected_gaussian_mean_data)
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_variance.data,
            self.expected_gaussian_variance_data)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_gaussian_data_check_max_iterations(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of temperature cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance when the maximum number of iterations is specified.
        The ensemble mean is the predictor.
        """
        distribution = "gaussian"
        plugin = Plugin(distribution,
                        max_iterations=10000)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_predictor.data,
            self.expected_gaussian_mean_data)
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_variance.data,
            self.expected_gaussian_variance_data)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_truncated_gaussian_data_check(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of wind_speed cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance. The ensemble mean is the predictor.
        """
        distribution = "truncated gaussian"
        plugin = Plugin(distribution)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_wind_speed_forecast_cube,
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_predictor.data,
            self.expected_truncated_gaussian_mean_data)
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_variance.data,
            self.expected_truncated_gaussian_variance_data)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_truncated_gaussian_data_check_max_iterations(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of wind_speed cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance when the maximum number of iterations is specified.
        The ensemble mean is the predictor.
        """
        distribution = "truncated gaussian"
        plugin = Plugin(distribution,
                        max_iterations=10000)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_wind_speed_forecast_cube,
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_predictor.data,
            self.expected_truncated_gaussian_mean_data)
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_variance.data,
            self.expected_truncated_gaussian_variance_data)


@unittest.skipIf(
    STATSMODELS_FOUND is False, "statsmodels module not available.")
class Test_process_with_statsmodels(SetupCubes, SetupExpectedResults,
                                    EnsembleCalibrationAssertions):

    """Additional tests for the process method when the statsmodels module
    is available. The statsmodels module is used for creating an initial
    guess for the coefficients that will be solved as part of the
    calibration."""

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_gaussian_realizations_data_check(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of temperature cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance. The ensemble realizations is the predictor.
        """
        distribution = "gaussian"
        predictor_of_mean_flag = "realizations"
        plugin = Plugin(
            distribution,
            predictor_of_mean_flag=predictor_of_mean_flag)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_predictor.data,
            self.expected_gaussian_realization_statsmodels)
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_variance.data,
            self.expected_gaussian_variance_statsmodels)
        # The assertions below are for comparison to the results
        # generated from using the ensemble mean as the predictor.
        # In this case, the expectation is for there to be broad agreement
        # between whether either the ensemble mean or the ensemble
        # realizations, but the results would not be expected to match exactly.
        self.assertArrayAlmostEqual(
            calibrated_predictor.data,
            self.expected_gaussian_mean_data, decimal=0)
        self.assertArrayAlmostEqual(
            calibrated_variance.data,
            self.expected_gaussian_variance_data, decimal=0)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_truncated_gaussian_realizations_data_check(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of wind_speed cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance. The ensemble realizations is the predictor.
        """
        distribution = "truncated gaussian"
        predictor_of_mean_flag = "realizations"
        plugin = Plugin(
            distribution,
            predictor_of_mean_flag=predictor_of_mean_flag)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_wind_speed_forecast_cube,
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_predictor.data,
            self.expected_truncated_gaussian_realization_statsmodels)
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_variance.data,
            self.expected_truncated_gaussian_variance_statsmodels)
        # The assertions below are for comparison to the results
        # generated from using the ensemble mean as the predictor.
        # In this case, the expectation is for there to be broad agreement
        # between whether either the ensemble mean or the ensemble
        # realizations, but the results would not be expected to match exactly.
        self.assertArrayAlmostEqual(
            calibrated_predictor.data,
            self.expected_truncated_gaussian_mean_data, decimal=0)
        self.assertArrayAlmostEqual(
            calibrated_variance.data,
            self.expected_truncated_gaussian_variance_data, decimal=0)


@unittest.skipIf(
    STATSMODELS_FOUND is True, "statsmodels module is available.")
class Test_process_without_statsmodels(SetupCubes, SetupExpectedResults,
                                       EnsembleCalibrationAssertions):

    """Additional tests for the process method when the statsmodels module
    is not available. A simple initial guess will be used for the
    coefficients that will be solved as part of the calibration."""

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_gaussian_realizations_data_check(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of temperature cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance. The ensemble realizations is the predictor.
        """
        distribution = "gaussian"
        predictor_of_mean_flag = "realizations"
        plugin = Plugin(
            distribution,
            predictor_of_mean_flag=predictor_of_mean_flag)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_predictor.data,
            self.expected_gaussian_realization_no_statsmodels)
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_variance.data,
            self.expected_gaussian_variance_no_statsmodels)
        # The assertions below are for comparison to the results
        # generated from using the ensemble mean as the predictor.
        # In this case, the expectation is for there to be broad agreement
        # between whether either the ensemble mean or the ensemble
        # realizations, but the results would not be expected to match exactly.
        self.assertArrayAlmostEqual(
            calibrated_predictor.data,
            self.expected_gaussian_mean_data, decimal=0)
        self.assertArrayAlmostEqual(
            calibrated_variance.data,
            self.expected_gaussian_variance_data, decimal=0)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_truncated_gaussian_realizations_data_check(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of wind_speed cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance. The ensemble realizations are the predictor.
        """
        distribution = "truncated gaussian"
        predictor_of_mean_flag = "realizations"
        plugin = Plugin(
            distribution,
            predictor_of_mean_flag=predictor_of_mean_flag)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_wind_speed_forecast_cube,
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_predictor.data,
            self.expected_truncated_gaussian_realization_no_statsmodels)
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_variance.data,
            self.expected_truncated_gaussian_variance_no_statsmodels)
        # The assertions below are for comparison to the results
        # generated from using the ensemble mean as the predictor.
        # In this case, the expectation is for there to be broad agreement
        # between whether either the ensemble mean or the ensemble
        # realizations, but the results would not be expected to match exactly.
        self.assertArrayAlmostEqual(
            calibrated_predictor.data,
            self.expected_truncated_gaussian_mean_data, decimal=0)
        self.assertArrayAlmostEqual(
            calibrated_variance.data,
            self.expected_truncated_gaussian_variance_data, decimal=0)


if __name__ == '__main__':
    unittest.main()
