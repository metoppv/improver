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
Unit tests for the
`ensemble_calibration.ApplyCoefficientsForEnsembleCalibration`
class.

"""
import unittest

import iris
import numpy as np
from iris.tests import IrisTest

from improver.ensemble_calibration.ensemble_calibration import (
    ApplyCoefficientsFromEnsembleCalibration as Plugin)
from improver.ensemble_calibration.ensemble_calibration import (
    EstimateCoefficientsForEnsembleCalibration)
from improver.tests.ensemble_calibration.ensemble_calibration. \
    helper_functions import SetupCubes, EnsembleCalibrationAssertions
from improver.tests.ensemble_calibration.ensemble_calibration.\
    test_EstimateCoefficientsForEnsembleCalibration import (
        create_coefficients_cube, SetupExpectedCoefficients)
from improver.tests.set_up_test_cubes import set_up_variable_cube
from improver.utilities.warnings_handler import ManageWarnings


class SetupCoefficientsCubes(SetupCubes, SetupExpectedCoefficients):

    """Set up coefficients cubes for testing."""

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "invalid escape sequence",
                          "can't resolve package from",
                          "The statsmodels can not be imported"],
        warning_types=[UserWarning, DeprecationWarning, ImportWarning,
                       ImportWarning])
    def setUp(self):
        """Set up coefficients cubes for when either the ensemble mean or the
        ensemble realizations have been used as the predictor. The coefficients
        have been constructed from the same underlying set of ensemble
        realizations, so application of these coefficients would be expected
        to give similar results. The values for the coefficients used to
        construct the coefficients cubes are taken from the
        SetupExpectedCoefficients class. These coefficients are the
        expected outputs from the tests to estimate the coefficients."""
        super().setUp()
        # Set up a coefficients cube when using the ensemble mean as the
        # predictor.
        current_cycle = "20171110T0000Z"
        estimator = (
            EstimateCoefficientsForEnsembleCalibration(
                "gaussian", current_cycle, desired_units="Celsius"))
        self.coeffs_from_mean = (
            estimator.create_coefficients_cube(
                self.expected_mean_predictor_gaussian,
                self.current_temperature_forecast_cube))

        # Set up a coefficients cube when using the ensemble realization as the
        # predictor and the coefficients have been estimated using statsmodels.
        estimator = (
            EstimateCoefficientsForEnsembleCalibration(
                "gaussian", current_cycle, desired_units="Celsius",
                predictor_of_mean_flag="realizations"))
        self.coeffs_from_statsmodels_realizations = (
            estimator.create_coefficients_cube(
                self.expected_realizations_gaussian_statsmodels,
                self.current_temperature_forecast_cube))

        # Set up a coefficients cube when using the ensemble realization as the
        # predictor and the coefficients have been estimated without using
        # statsmodels.
        estimator = (
            EstimateCoefficientsForEnsembleCalibration(
                "gaussian", current_cycle, desired_units="Celsius",
                predictor_of_mean_flag="realizations"))
        self.coeffs_from_no_statsmodels_realizations = (
            estimator.create_coefficients_cube(
                self.expected_realizations_gaussian_no_statsmodels,
                self.current_temperature_forecast_cube))


class Test__init__(IrisTest):

    """Test the __init__ method."""

    def setUp(self):
        """Set up test cubes."""
        data = np.ones([2, 2], dtype=np.float32)
        self.current_forecast = set_up_variable_cube(data)
        coeff_names = ["gamma", "delta", "alpha", "beta"]
        coeff_values = np.array([0, 1, 2, 3], np.int32)
        self.coefficients_cube, _, _ = create_coefficients_cube(
            self.current_forecast, coeff_names, coeff_values)

    def test_basic(self):
        """Test without specifying any keyword arguments."""
        plugin = Plugin(self.current_forecast, self.coefficients_cube)
        self.assertEqual(plugin.current_forecast, self.current_forecast)
        self.assertEqual(plugin.coefficients_cube, self.coefficients_cube)

    def test_with_kwargs(self):
        """Test without specifying any keyword arguments."""
        plugin = Plugin(self.current_forecast, self.coefficients_cube,
                        predictor_of_mean_flag="realizations")
        self.assertEqual(plugin.current_forecast, self.current_forecast)
        self.assertEqual(plugin.coefficients_cube, self.coefficients_cube)

    def test_mismatching_coordinates(self):
        """Test if there is a mismatch in the forecast_period coordinate."""
        self.current_forecast.coord("forecast_period").convert_units("hours")
        msg = "The forecast_period coordinate of the current forecast cube"
        with self.assertRaisesRegex(ValueError, msg):
            Plugin(self.current_forecast, self.coefficients_cube)

    def test_matching_domain(self):
        """Test whether the domain of the forecast and the domain of the
        coefficients cube matches."""
        current_forecast = self.current_forecast[0, :]
        msg = "The domain along the"
        with self.assertRaisesRegex(ValueError, msg):
            Plugin(current_forecast, self.coefficients_cube)


class Test__repr__(IrisTest):

    """Test the __repr__ method."""

    def setUp(self):
        """Set up test cubes."""
        data = np.ones([2, 2], dtype=np.float32)
        self.current_forecast = set_up_variable_cube(data)
        coeff_names = ["gamma", "delta", "alpha", "beta"]
        coeff_values = np.array([0, 1, 2, 3], np.int32)
        self.coefficients_cube, _, _ = create_coefficients_cube(
            self.current_forecast, coeff_names, coeff_values)

    def test_basic(self):
        """Test without specifying keyword arguments"""
        result = str(Plugin(self.current_forecast, self.coefficients_cube))
        msg = ("<ApplyCoefficientsFromEnsembleCalibration: "
               "current_forecast: air_temperature; "
               "coefficients_cube: emos_coefficients; "
               "predictor_of_mean_flag: mean>")
        self.assertEqual(result, msg)

    def test_with_kwargs(self):
        """Test when keyword arguments are specified."""
        result = str(Plugin(
            self.current_forecast, self.coefficients_cube,
            predictor_of_mean_flag="realizations"))
        msg = ("<ApplyCoefficientsFromEnsembleCalibration: "
               "current_forecast: air_temperature; "
               "coefficients_cube: emos_coefficients; "
               "predictor_of_mean_flag: realizations>")
        self.assertEqual(result, msg)


class Test_process(SetupCoefficientsCubes):

    """Test the process plugin."""

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic(self):
        """Test that the plugin returns a tuple."""
        cube = self.current_temperature_forecast_cube
        plugin = Plugin(cube, self.coeffs_from_mean)
        result = plugin.process()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_realizations(self):
        """
        Test that the plugin returns a tuple when using the ensemble
        realizations as the predictor of the mean.
        """
        cube = self.current_temperature_forecast_cube
        plugin = Plugin(cube, self.coeffs_from_statsmodels_realizations,
                        predictor_of_mean_flag="realizations")
        result = plugin.process()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_output_is_mean(self):
        """
        Test that the plugin returns a tuple containing cubes with a
        mean cell method.
        """
        cube = self.current_temperature_forecast_cube
        plugin = Plugin(cube, self.coeffs_from_mean)
        forecast_predictor, _ = plugin.process()
        for cell_method in forecast_predictor[0].cell_methods:
            self.assertEqual(cell_method.method, "mean")

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_output_is_variance(self):
        """
        Test that the plugin returns a tuple containing cubes with a
        variance cell method.
        """
        cube = self.current_temperature_forecast_cube
        plugin = Plugin(cube, self.coeffs_from_mean)
        _, forecast_variance = plugin.process()
        for cell_method in forecast_variance[0].cell_methods:
            self.assertEqual(cell_method.method, "variance")


class Test__apply_params(
        SetupCoefficientsCubes, EnsembleCalibrationAssertions):

    """Test the _apply_params plugin."""

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "invalid escape sequence",
                          "can't resolve package from",
                          "The statsmodels can not be imported"],
        warning_types=[UserWarning, DeprecationWarning, ImportWarning,
                       ImportWarning])
    def setUp(self):
        """Set up expected arrays for the calibrated ensemble mean and variance
        depending upon whether the ensemble mean or ensemble realizations have
        been used."""
        super().setUp()
        self.expected_calibrated_predictor_mean = (
            np.array([[273.7371, 274.6500, 275.4107],
                      [276.8409, 277.6321, 278.3928],
                      [279.4884, 280.1578, 280.9794]]))
        self.expected_calibrated_variance_mean = (
            np.array([[0.2134, 0.2158, 0.0127],
                      [0.0247, 0.0215, 0.0127],
                      [0.0581, 0.0032, 0.0008]]))
        self.expected_calibrated_predictor_statsmodels_realizations = (
            np.array([[274.2120, 275.1703, 275.3308],
                      [277.0504, 277.4221, 278.3881],
                      [280.0826, 280.3248, 281.2376]]))
        self.expected_calibrated_variance_statsmodels_realizations = (
            np.array([[0.8975, 0.9075, 0.0536],
                      [0.1038, 0.0904, 0.0536],
                      [0.2444, 0.0134, 0.0033]]))
        self.expected_calibrated_predictor_no_statsmodels_realizations = (
            np.array([[274.1428, 275.0543, 275.2956],
                      [277.0344, 277.4110, 278.3598],
                      [280.0760, 280.3517, 281.2437]]))
        self.expected_calibrated_variance_no_statsmodels_realizations = (
            np.array([[0.99803287, 1.0091798, 0.06006174],
                      [0.11588794, 0.10100815, 0.06006173],
                      [0.27221495, 0.01540077, 0.00423326]]))

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "invalid escape sequence"],
        warning_types=[UserWarning, DeprecationWarning])
    def test_basic(self):
        """Test that the plugin returns a tuple."""
        cube = self.current_temperature_forecast_cube

        predictor_cube = cube.collapsed("realization", iris.analysis.MEAN)
        variance_cube = cube.collapsed("realization", iris.analysis.VARIANCE)

        plugin = Plugin(cube, self.coeffs_from_mean)
        result = plugin._apply_params(predictor_cube, variance_cube)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "invalid escape sequence"],
        warning_types=[UserWarning, DeprecationWarning])
    def test_calibrated_predictor(self):
        """
        Test that the plugin returns the expected values for the calibrated
        ensemble mean when the ensemble mean is used as the predictor. Check
        that the calibrated mean is similar to when the ensemble realizations
        are used as the predictor.
        """
        cube = self.current_temperature_forecast_cube
        predictor_cube = cube.collapsed("realization", iris.analysis.MEAN)
        variance_cube = cube.collapsed("realization", iris.analysis.VARIANCE)

        plugin = Plugin(cube, self.coeffs_from_mean)
        forecast_predictor, _ = (
            plugin._apply_params(predictor_cube, variance_cube))
        self.assertCalibratedVariablesAlmostEqual(
            forecast_predictor.data, self.expected_calibrated_predictor_mean)
        self.assertArrayAlmostEqual(
            forecast_predictor.data,
            self.expected_calibrated_predictor_statsmodels_realizations,
            decimal=0)
        self.assertArrayAlmostEqual(
            forecast_predictor.data,
            self.expected_calibrated_predictor_no_statsmodels_realizations,
            decimal=0)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "invalid escape sequence"],
        warning_types=[UserWarning, DeprecationWarning])
    def test_calibrated_variance(self):
        """
        Test that the plugin returns the expected values for the calibrated
        ensemble variance when the ensemble mean is used as the predictor.
        Check that the calibrated variance is similar to when the ensemble
        realizations are used as the predictor.
        """
        cube = self.current_temperature_forecast_cube

        predictor_cube = cube.collapsed("realization", iris.analysis.MEAN)
        variance_cube = cube.collapsed("realization", iris.analysis.VARIANCE)

        plugin = Plugin(cube, self.coeffs_from_mean)
        _, forecast_variance = (
            plugin._apply_params(predictor_cube, variance_cube))
        self.assertCalibratedVariablesAlmostEqual(
            forecast_variance.data, self.expected_calibrated_variance_mean)
        self.assertArrayAlmostEqual(
            forecast_variance.data,
            self.expected_calibrated_variance_statsmodels_realizations,
            decimal=0)
        self.assertArrayAlmostEqual(
            forecast_variance.data,
            self.expected_calibrated_variance_no_statsmodels_realizations,
            decimal=0)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "invalid escape sequence"],
        warning_types=[UserWarning, DeprecationWarning])
    def test_calibrated_predictor_statsmodels_realizations(self):
        """
        Test that the plugin returns the expected values for the calibrated
        ensemble mean when the ensemble realizations are used as the predictor.
        The input coefficients have been generated using statsmodels. Check
        that the calibrated mean is similar to when the ensemble mean is used
        as the predictor.
        """
        cube = self.current_temperature_forecast_cube

        predictor_cube = cube.copy()
        variance_cube = cube.collapsed("realization", iris.analysis.VARIANCE)

        plugin = Plugin(cube, self.coeffs_from_statsmodels_realizations,
                        predictor_of_mean_flag="realizations")
        forecast_predictor, _ = plugin._apply_params(
            predictor_cube, variance_cube)
        self.assertCalibratedVariablesAlmostEqual(
            forecast_predictor.data,
            self.expected_calibrated_predictor_statsmodels_realizations)
        self.assertArrayAlmostEqual(
            forecast_predictor.data,
            self.expected_calibrated_predictor_mean, decimal=0)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "invalid escape sequence"],
        warning_types=[UserWarning, DeprecationWarning])
    def test_calibrated_variance_statsmodels_realizations(self):
        """
        Test that the plugin returns the expected values for the calibrated
        ensemble variance when the ensemble realizations are used as the
        predictor. The input coefficients have been generated using
        statsmodels. Check that the calibrated variance is similar to when the
        ensemble mean is used as the predictor.
        """
        cube = self.current_temperature_forecast_cube

        predictor_cube = cube.copy()
        variance_cube = cube.collapsed("realization", iris.analysis.VARIANCE)

        plugin = Plugin(cube, self.coeffs_from_statsmodels_realizations,
                        predictor_of_mean_flag="realizations")
        _, forecast_variance = plugin._apply_params(
            predictor_cube, variance_cube)
        self.assertCalibratedVariablesAlmostEqual(
            forecast_variance.data,
            self.expected_calibrated_variance_statsmodels_realizations)
        self.assertArrayAlmostEqual(
            forecast_variance.data,
            self.expected_calibrated_variance_mean, decimal=0)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "invalid escape sequence"],
        warning_types=[UserWarning, DeprecationWarning])
    def test_calibrated_predictor_no_statsmodels_realizations(self):
        """
        Test that the plugin returns the expected values for the calibrated
        ensemble mean when the ensemble realizations are used as the predictor.
        The input coefficients have been generated without statsmodels. Check
        that the calibrated mean is similar to when the ensemble mean is used
        as the predictor.
        """
        cube = self.current_temperature_forecast_cube

        predictor_cube = cube.copy()
        variance_cube = cube.collapsed("realization",
                                       iris.analysis.VARIANCE)

        plugin = Plugin(cube, self.coeffs_from_no_statsmodels_realizations,
                        predictor_of_mean_flag="realizations")
        forecast_predictor, _ = plugin._apply_params(
            predictor_cube, variance_cube)
        self.assertCalibratedVariablesAlmostEqual(
            forecast_predictor.data,
            self.expected_calibrated_predictor_no_statsmodels_realizations)
        self.assertArrayAlmostEqual(
            forecast_predictor.data,
            self.expected_calibrated_predictor_mean, decimal=0)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "invalid escape sequence"],
        warning_types=[UserWarning, DeprecationWarning])
    def test_calibrated_variance_no_statsmodels_realizations(self):
        """
        Test that the plugin returns the expected values for the calibrated
        ensemble variance when the ensemble realizations are used as the
        predictor. The input coefficients have been generated without
        statsmodels. Check that the calibrated variance is similar to when the
        ensemble mean is used as the predictor.
        """
        cube = self.current_temperature_forecast_cube

        predictor_cube = cube.copy()
        variance_cube = cube.collapsed("realization",
                                       iris.analysis.VARIANCE)

        plugin = Plugin(cube, self.coeffs_from_no_statsmodels_realizations,
                        predictor_of_mean_flag="realizations")
        _, forecast_variance = plugin._apply_params(
            predictor_cube, variance_cube)
        self.assertCalibratedVariablesAlmostEqual(
            forecast_variance.data,
            self.expected_calibrated_variance_no_statsmodels_realizations)
        self.assertArrayAlmostEqual(
            forecast_variance.data,
            self.expected_calibrated_variance_mean, decimal=0)


if __name__ == '__main__':
    unittest.main()
