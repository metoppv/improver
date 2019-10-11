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
from numpy.testing import assert_array_almost_equal
from iris.tests import IrisTest

from improver.ensemble_calibration.ensemble_calibration import (
    ApplyCoefficientsFromEnsembleCalibration as Plugin)
from improver.ensemble_calibration.ensemble_calibration import (
    EstimateCoefficientsForEnsembleCalibration)
from improver.tests.ensemble_calibration.ensemble_calibration. \
    helper_functions import SetupCubes, EnsembleCalibrationAssertions
from improver.tests.ensemble_calibration.ensemble_calibration.\
    test_EstimateCoefficientsForEnsembleCalibration import (
        SetupExpectedCoefficients)
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

        # Some expected data that are used in various tests.
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


class Test__init__(IrisTest):

    """Test the __init__ method."""

    def test_basic(self):
        """Test without specifying a predictor_of_mean_flag."""
        plugin = Plugin()
        self.assertEqual(plugin.predictor_of_mean_flag, "mean")

    def test_with_predictor_of_mean_flag(self):
        """Test specifying the predictor_of_mean_flag."""
        plugin = Plugin(predictor_of_mean_flag="realizations")
        self.assertEqual(plugin.predictor_of_mean_flag, "realizations")


class Test__repr__(IrisTest):

    """Test the __repr__ method."""

    def test_basic(self):
        """Test without the predictor_of_mean_flag."""
        result = str(Plugin())
        msg = ("<ApplyCoefficientsFromEnsembleCalibration: "
               "predictor_of_mean_flag: mean>")
        self.assertEqual(result, msg)

    def test_with_predictor_of_mean_flag(self):
        """Test specifying the predictor_of_mean_flag."""
        result = str(Plugin(predictor_of_mean_flag="realizations"))
        msg = ("<ApplyCoefficientsFromEnsembleCalibration: "
               "predictor_of_mean_flag: realizations>")
        self.assertEqual(result, msg)


class Test__spatial_domain_match(SetupCoefficientsCubes):

    """ Test the _spatial_domain_match method."""

    def setUp(self):
        super().setUp()
        self.plugin = Plugin()

    def test_matching(self):
        """Test case in which spatial domains match."""
        self.plugin.current_forecast = self.current_temperature_forecast_cube
        self.plugin.coefficients_cube = self.coeffs_from_mean
        self.plugin._spatial_domain_match()

    def test_unmatching_x_axis(self):
        """Test case in which the x-dimensions of the domains do not match."""
        self.current_temperature_forecast_cube.coord(axis='x').points = (
            self.current_temperature_forecast_cube.coord(axis='x').points * 2.)
        self.plugin.current_forecast = self.current_temperature_forecast_cube
        self.plugin.coefficients_cube = self.coeffs_from_mean
        msg = "The domain along the x axis given by the current forecast"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin._spatial_domain_match()

    def test_unmatching_y_axis(self):
        """Test case in which the y-dimensions of the domains do not match."""
        self.current_temperature_forecast_cube.coord(axis='y').points = (
            self.current_temperature_forecast_cube.coord(axis='y').points * 2.)
        self.plugin.current_forecast = self.current_temperature_forecast_cube
        self.plugin.coefficients_cube = self.coeffs_from_mean
        msg = "The domain along the y axis given by the current forecast"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin._spatial_domain_match()


class Test__get_calibrated_forecast_predictors_mean(
        SetupCoefficientsCubes, EnsembleCalibrationAssertions):

    """Test the _get_calibrated_forecast_predictors_mean method."""

    def setUp(self):
        """Setup cubes and sundries for testing calibration of mean."""
        super().setUp()
        self.optimised_coeffs = (
            dict(zip(self.coeffs_from_mean.coord("coefficient_name").points,
                     self.coeffs_from_mean.data)))
        self.plugin = Plugin()
        self.plugin.current_forecast = self.current_temperature_forecast_cube

        self.expected_calibrated_predictor_mean = (
            self.expected_calibrated_predictor_mean.flatten())

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic(self):
        """
        Test that the expected values are returned by this function.
        """
        expected_forecast_predictors = (
            self.current_temperature_forecast_cube.collapsed(
                "realization", iris.analysis.MEAN))
        predicted_mean, forecast_predictors = (
            self.plugin._get_calibrated_forecast_predictors_mean(
                self.optimised_coeffs))
        self.assertCalibratedVariablesAlmostEqual(
            predicted_mean, self.expected_calibrated_predictor_mean)
        self.assertCalibratedVariablesAlmostEqual(
            forecast_predictors.data, expected_forecast_predictors.data)


class Test__get_calibrated_forecast_predictors_realizations(
        SetupCoefficientsCubes, EnsembleCalibrationAssertions):

    """Test the _get_calibrated_forecast_predictors_realizations method."""

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def setUp(self):
        """Setup cubes and sundries for testing calibration of mean."""
        super().setUp()

        self.forecast_vars = self.current_temperature_forecast_cube.collapsed(
            "realization", iris.analysis.VARIANCE)

        self.plugin = Plugin()
        self.plugin.current_forecast = self.current_temperature_forecast_cube
        self.expected_calibrated_predictor_statsmodels_realizations = (
            self.expected_calibrated_predictor_statsmodels_realizations.
            flatten())
        self.expected_calibrated_predictor_no_statsmodels_realizations = (
            self.expected_calibrated_predictor_no_statsmodels_realizations.
            flatten())

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_with_statsmodels(self):
        """
        Test that the expected values are returned by this function when using
        statsmodels.
        """
        optimised_coeffs = dict(
            zip(self.coeffs_from_statsmodels_realizations.coord(
                    "coefficient_name").points,
                self.coeffs_from_statsmodels_realizations.data))
        expected_forecast_predictors = (
            self.current_temperature_forecast_cube.collapsed(
                "realization", iris.analysis.MEAN))
        predicted_mean, forecast_predictors = (
            self.plugin._get_calibrated_forecast_predictors_realizations(
                optimised_coeffs, self.forecast_vars))
        self.assertCalibratedVariablesAlmostEqual(
            predicted_mean,
            self.expected_calibrated_predictor_statsmodels_realizations)
        self.assertCalibratedVariablesAlmostEqual(
            forecast_predictors.data, expected_forecast_predictors.data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_without_statsmodels(self):
        """
        Test that the expected values are returned by this function when not
        using statsmodels.
        """
        optimised_coeffs = dict(
            zip(self.coeffs_from_no_statsmodels_realizations.coord(
                    "coefficient_name").points,
                self.coeffs_from_no_statsmodels_realizations.data))
        expected_forecast_predictors = (
            self.current_temperature_forecast_cube.collapsed(
                "realization", iris.analysis.MEAN))
        predicted_mean, forecast_predictors = (
            self.plugin._get_calibrated_forecast_predictors_realizations(
                optimised_coeffs, self.forecast_vars))
        self.assertCalibratedVariablesAlmostEqual(
            predicted_mean,
            self.expected_calibrated_predictor_no_statsmodels_realizations)
        self.assertCalibratedVariablesAlmostEqual(
            forecast_predictors.data, expected_forecast_predictors.data)


class Test_calibrate_forecast_data(
        SetupCoefficientsCubes, EnsembleCalibrationAssertions):

    """Test the calibrate_forecast_data method.

    Note that there are several cross comparisons of results between the tests.
    These overlap to ensure that if any one test is removed, the others still
    check that the behaviour is as expected."""

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def setUp(self):
        """Setup cubes and sundries for testing calibration."""
        super().setUp()

        self.forecast_vars = self.current_temperature_forecast_cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        self.forecast_predictor = (
            self.current_temperature_forecast_cube.collapsed(
                "realization", iris.analysis.MEAN))

        self.plugin = Plugin()

    def test_mean_as_predictor(self):
        """Test that the calibrated forecast using ensemble mean as the
        predictor returns the correct values. Check that the calibrated mean
        is similar to when the ensemble realizations are used as the predictor
        with and without statsmodels."""

        optimised_coeffs = dict(
            zip(self.coeffs_from_mean.coord("coefficient_name").points,
                self.coeffs_from_mean.data))
        predicted_mean = self.expected_calibrated_predictor_mean.flatten()

        calibrated_forecast_predictor, calibrated_forecast_var = (
            self.plugin.calibrate_forecast_data(
                optimised_coeffs, predicted_mean, self.forecast_predictor,
                self.forecast_vars))

        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_predictor.data,
            self.expected_calibrated_predictor_mean)
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_var.data,
            self.expected_calibrated_variance_mean)
        assert_array_almost_equal(
            calibrated_forecast_predictor.data,
            self.expected_calibrated_predictor_statsmodels_realizations,
            decimal=0)
        assert_array_almost_equal(
            calibrated_forecast_predictor.data,
            self.expected_calibrated_predictor_no_statsmodels_realizations,
            decimal=0)
        self.assertIsInstance(calibrated_forecast_predictor, iris.cube.Cube)
        self.assertIsInstance(calibrated_forecast_var, iris.cube.Cube)

    def test_realization_as_predictor_with_statsmodels(self):
        """Test that the calibrated forecast using realization as the
        predictor returns the correct values when using statsmodels. Check that
        the calibrated mean is similar to when not using statsmodels and to
        when the ensemble mean is used as the predictor."""

        optimised_coeffs = dict(
            zip(self.coeffs_from_statsmodels_realizations.coord(
                    "coefficient_name").points,
                self.coeffs_from_statsmodels_realizations.data))
        predicted_mean = (
            self.expected_calibrated_predictor_statsmodels_realizations.
            flatten())

        calibrated_forecast_predictor, calibrated_forecast_var = (
            self.plugin.calibrate_forecast_data(
                optimised_coeffs, predicted_mean, self.forecast_predictor,
                self.forecast_vars))

        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_predictor.data,
            self.expected_calibrated_predictor_statsmodels_realizations)
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_var.data,
            self.expected_calibrated_variance_statsmodels_realizations)
        assert_array_almost_equal(
            calibrated_forecast_predictor.data,
            self.expected_calibrated_predictor_mean,
            decimal=0)
        assert_array_almost_equal(
            calibrated_forecast_predictor.data,
            self.expected_calibrated_predictor_no_statsmodels_realizations,
            decimal=0)
        self.assertIsInstance(calibrated_forecast_predictor, iris.cube.Cube)
        self.assertIsInstance(calibrated_forecast_var, iris.cube.Cube)

    def test_realization_as_predictor_without_statsmodels(self):
        """Test that the calibrated forecast using realization as the
        predictor returns the correct values when not using statsmodels. Check
        that the calibrated mean is similar to when using statsmodels and to
        when the ensemble mean is used as the predictor."""

        optimised_coeffs = dict(
            zip(self.coeffs_from_no_statsmodels_realizations.coord(
                    "coefficient_name").points,
                self.coeffs_from_no_statsmodels_realizations.data))
        predicted_mean = (
            self.expected_calibrated_predictor_no_statsmodels_realizations.
            flatten())

        calibrated_forecast_predictor, calibrated_forecast_var = (
            self.plugin.calibrate_forecast_data(
                optimised_coeffs, predicted_mean, self.forecast_predictor,
                self.forecast_vars))
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_predictor.data,
            self.expected_calibrated_predictor_no_statsmodels_realizations)
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_var.data,
            self.expected_calibrated_variance_no_statsmodels_realizations)
        assert_array_almost_equal(
            calibrated_forecast_predictor.data,
            self.expected_calibrated_predictor_mean,
            decimal=0)
        assert_array_almost_equal(
            calibrated_forecast_predictor.data,
            self.expected_calibrated_predictor_statsmodels_realizations,
            decimal=0)
        self.assertIsInstance(calibrated_forecast_predictor, iris.cube.Cube)
        self.assertIsInstance(calibrated_forecast_var, iris.cube.Cube)


class Test_process(SetupCoefficientsCubes, EnsembleCalibrationAssertions):

    """Test the process plugin."""

    def setUp(self):
        """Setup cubes and sundries for testing calibration."""
        super().setUp()
        self.plugin = Plugin()

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_variable_setting(self):
        """Test that the cubes passed into the plugin are allocated to
        plugin variables appropriately."""

        _, _ = self.plugin.process(self.current_temperature_forecast_cube,
                                   self.coeffs_from_mean)
        self.assertEqual(self.current_temperature_forecast_cube,
                         self.plugin.current_forecast)
        self.assertEqual(self.coeffs_from_mean,
                         self.plugin.coefficients_cube)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_end_to_end(self):
        """An example end-to-end calculation. This repeats the test elements
        above but all grouped together."""
        calibrated_forecast_predictor, calibrated_forecast_var = (
            self.plugin.process(self.current_temperature_forecast_cube,
                                self.coeffs_from_mean))

        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_predictor.data,
            self.expected_calibrated_predictor_mean)
        self.assertCalibratedVariablesAlmostEqual(
            calibrated_forecast_var.data,
            self.expected_calibrated_variance_mean)
        self.assertEqual(calibrated_forecast_predictor.dtype, np.float32)


if __name__ == '__main__':
    unittest.main()
