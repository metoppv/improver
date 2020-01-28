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
`ensemble_calibration.ContinuousRankedProbabilityScoreMinimisers`
class.

"""
import unittest

import iris
import numpy as np
from iris.tests import IrisTest

from improver.calibration.ensemble_calibration import \
    ContinuousRankedProbabilityScoreMinimisers as Plugin
from improver.calibration.utilities import convert_cube_data_to_2d
from improver.utilities.warnings_handler import ManageWarnings

from .helper_functions import EnsembleCalibrationAssertions, SetupCubes


class Test__repr__(IrisTest):

    """Test the __repr__ method."""

    def test_basic(self):
        """A simple tests for the __repr__ method."""
        result = str(Plugin())
        msg = ("<ContinuousRankedProbabilityScoreMinimisers: "
               "minimisation_dict: {'gaussian': 'calculate_normal_crps', "
               "'truncated_gaussian': 'calculate_truncated_normal_crps'}; "
               "tolerance: 0.01; max_iterations: 1000>")
        self.assertEqual(result, msg)

    def test_update_kwargs(self):
        """A test to update the available keyword argument."""
        result = str(Plugin(tolerance=10, max_iterations=10))
        msg = ("<ContinuousRankedProbabilityScoreMinimisers: "
               "minimisation_dict: {'gaussian': 'calculate_normal_crps', "
               "'truncated_gaussian': 'calculate_truncated_normal_crps'}; "
               "tolerance: 10; max_iterations: 10>")
        self.assertEqual(result, msg)


class SetupInputs(IrisTest):

    """Set up inputs for testing."""

    def setUp(self):
        """Set up inputs for testing."""
        super().setUp()
        self.sqrt_pi = np.sqrt(np.pi).astype(np.float64)

        self.initial_guess_for_mean = np.array([0, 1, 0, 1], dtype=np.float64)
        self.initial_guess_for_realization = (
            np.array([0, 1, 0, np.sqrt(1/3.), np.sqrt(1/3.), np.sqrt(1/3.)],
                     dtype=np.float64))


class SetupGaussianInputs(SetupInputs, SetupCubes):

    """Create a class for setting up cubes for testing."""

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."],
        warning_types=[UserWarning])
    def setUp(self):
        """Set up expected inputs."""
        super().setUp()
        # Set up cubes and associated data arrays for temperature.
        self.forecast_predictor_mean = (
            self.historic_temperature_forecast_cube.collapsed(
                "realization", iris.analysis.MEAN))
        self.forecast_predictor_realizations = (
            self.historic_temperature_forecast_cube.copy())
        self.forecast_variance = (
            self.historic_temperature_forecast_cube.collapsed(
                "realization", iris.analysis.VARIANCE))
        self.truth = (
            self.historic_temperature_forecast_cube.collapsed(
                "realization", iris.analysis.MAX))
        self.forecast_predictor_data = (
            self.forecast_predictor_mean.data.flatten().astype(
                np.float64))
        self.forecast_predictor_data_realizations = (
            convert_cube_data_to_2d(
                self.historic_temperature_forecast_cube.copy()
            ).astype(np.float64))
        self.forecast_variance_data = (
            self.forecast_variance.data.flatten().astype(
                np.float64))
        self.truth_data = self.truth.data.flatten().astype(
            np.float64)


class Test_calculate_normal_crps(SetupGaussianInputs):

    """
    Test minimising the CRPS for a gaussian distribution.
    Either the ensemble mean or the individual ensemble realizations are
    used as the predictors.
    """
    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_mean_predictor(self):
        """
        Test that the plugin returns a numpy float value with the
        mean as the predictor. The result indicates the minimum value for the
        CRPS that was achieved by the minimisation.
        """
        predictor = "mean"

        plugin = Plugin()
        result = plugin.calculate_normal_crps(
            self.initial_guess_for_mean, self.forecast_predictor_data,
            self.truth_data, self.forecast_variance_data, self.sqrt_pi,
            predictor)

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, 0.2609063)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_realizations_predictor(self):
        """
        Test that the plugin returns a numpy float value with the ensemble
        realizations as the predictor. The result indicates the minimum value
        for the CRPS that was achieved by the minimisation.
        """
        predictor = "realizations"

        plugin = Plugin()
        result = plugin.calculate_normal_crps(
            self.initial_guess_for_realization,
            self.forecast_predictor_data_realizations, self.truth_data,
            self.forecast_variance_data, self.sqrt_pi,
            predictor)

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, 0.2609061)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "invalid value encountered in"],
        warning_types=[UserWarning, RuntimeWarning])
    def test_basic_mean_predictor_bad_value(self):
        """
        Test that the plugin returns a numpy float64 value
        and that the value matches the BAD_VALUE, when the appropriate
        condition is found. The ensemble mean is the predictor.
        The initial guess is specifically set to float32 precision for the
        purpose for generating the BAD_VALUE for the unit test.
        """
        initial_guess = np.array([1e65, 1e65, 1e65, 1e65], dtype=np.float32)

        predictor = "mean"

        plugin = Plugin()
        result = plugin.calculate_normal_crps(
            initial_guess, self.forecast_predictor_data, self.truth_data,
            self.forecast_variance_data, self.sqrt_pi, predictor)

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, plugin.BAD_VALUE)


class Test_process_gaussian_distribution(
        SetupGaussianInputs, EnsembleCalibrationAssertions):

    """
    Test minimising the CRPS for a gaussian distribution.
    Either the ensemble mean or the individual ensemble realizations are used
    as the predictors.
    """
    def setUp(self):
        """Set up expected output.
        The coefficients are in the order [gamma, delta, alpha, beta].
        """
        super().setUp()
        self.tolerance = 1e-4
        self.plugin = Plugin(tolerance=self.tolerance)
        self.expected_mean_coefficients = (
            [0.0023, 0.8070, -0.0008, 1.0009])
        self.expected_realizations_coefficients = (
            [-0.1373, 0.1141, 0.0409, 0.414, 0.2056, 0.8871])

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "Minimisation did not result in convergence",
                          "divide by zero encountered in"],
        warning_types=[UserWarning, UserWarning, RuntimeWarning])
    def test_basic_mean_predictor(self):
        """
        Test that the plugin returns a numpy array with the expected
        coefficients. The ensemble mean is the predictor.
        """
        predictor = "mean"
        distribution = "gaussian"
        result = self.plugin.process(
            self.initial_guess_for_mean, self.forecast_predictor_mean,
            self.truth, self.forecast_variance, predictor,
            distribution)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_mean_coefficients)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "Minimisation did not result in convergence",
                          "divide by zero encountered in",
                          "invalid value encountered in"],
        warning_types=[UserWarning, UserWarning, RuntimeWarning,
                       RuntimeWarning])
    def test_basic_realizations_predictor(self):
        """
        Test that the plugin returns a numpy array with the expected
        coefficients. The ensemble realizations are the predictor.
        """
        predictor = "realizations"
        distribution = "gaussian"
        result = self.plugin.process(
            self.initial_guess_for_realization,
            self.forecast_predictor_realizations, self.truth,
            self.forecast_variance, predictor, distribution)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_realizations_coefficients)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_mean_predictor_keyerror(self):
        """
        Test that the minimisation has resulted in a KeyError, if the
        distribution that has been requested was not within the dictionary
        containing the minimisation functions.
        """
        predictor = "mean"
        distribution = "foo"

        msg = "Distribution requested"
        with self.assertRaisesRegex(KeyError, msg):
            self.plugin.process(
                self.initial_guess_for_mean, self.forecast_predictor_mean,
                self.truth, self.forecast_variance,
                predictor, distribution)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "Minimisation did not result in convergence",
                          "divide by zero encountered in"],
        warning_types=[UserWarning, UserWarning, RuntimeWarning])
    def test_mean_predictor_max_iterations(self):
        """
        Test that the plugin returns a list of coefficients
        equal to specific values, when the ensemble mean is the predictor
        assuming a normal distribution and the value specified for the
        max_iterations is overridden. The coefficients are calculated by
        minimising the CRPS and using a set default value for the
        initial guess.
        """
        predictor = "mean"
        max_iterations = 400
        distribution = "gaussian"

        plugin = Plugin(
            tolerance=self.tolerance, max_iterations=max_iterations)
        result = plugin.process(
            self.initial_guess_for_mean, self.forecast_predictor_mean,
            self.truth, self.forecast_variance,
            predictor, distribution)
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_mean_coefficients)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "Minimisation did not result in convergence",
                          "divide by zero encountered in",
                          "invalid value encountered in"],
        warning_types=[UserWarning, UserWarning, RuntimeWarning,
                       RuntimeWarning])
    def test_realizations_predictor_max_iterations(self):
        """
        Test that the plugin returns a list of coefficients
        equal to specific values, when the ensemble realizations are the
        predictor assuming a truncated normal distribution and the value
        specified for the MAX_ITERATIONS is overridden. The coefficients are
        calculated by minimising the CRPS and using a set default value for
        the initial guess.
        """
        predictor = "realizations"
        max_iterations = 1000
        distribution = "gaussian"

        plugin = Plugin(
            tolerance=self.tolerance, max_iterations=max_iterations)
        result = plugin.process(
            self.initial_guess_for_realization,
            self.forecast_predictor_realizations, self.truth,
            self.forecast_variance, predictor, distribution)
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_realizations_coefficients)

    @ManageWarnings(
        record=True,
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_catch_warnings(self, warning_list=None):
        """
        Test that a warning is generated if the minimisation
        does not result in a convergence. The ensemble mean is the predictor.
        """
        predictor = "mean"
        distribution = "gaussian"

        plugin = Plugin(tolerance=self.tolerance, max_iterations=10)
        plugin.process(
            self.initial_guess_for_mean, self.forecast_predictor_mean,
            self.truth, self.forecast_variance, predictor,
            distribution)
        warning_msg = "Minimisation did not result in convergence after"
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))

    @ManageWarnings(
        record=True,
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_catch_warnings_percentage_change(self, warning_list=None):
        """
        Test that two warnings are generated if the minimisation
        does not result in a convergence. The first warning reports a that
        the minimisation did not result in convergence, whilst the second
        warning reports that the percentage change in the final iteration was
        greater than the tolerated value. The ensemble mean is the predictor.
        """
        initial_guess = np.array([5000, 1, 0, 1], dtype=np.float64)
        predictor = "mean"
        distribution = "gaussian"

        plugin = Plugin(tolerance=self.tolerance, max_iterations=5)
        plugin.process(
            initial_guess, self.forecast_predictor_mean, self.truth,
            self.forecast_variance, predictor, distribution)
        warning_msg_min = "Minimisation did not result in convergence after"
        warning_msg_iter = "The final iteration resulted in a percentage "
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg_min in str(item)
                            for item in warning_list))
        self.assertTrue(any(warning_msg_iter in str(item)
                            for item in warning_list))


class SetupTruncatedGaussianInputs(SetupInputs, SetupCubes):

    """Create a class for setting up cubes for testing."""

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."],
        warning_types=[UserWarning])
    def setUp(self):
        """Set up expected inputs."""
        super().setUp()
        # Set up cubes and associated data arrays for wind speed.
        self.forecast_predictor_mean = (
            self.historic_wind_speed_forecast_cube.collapsed(
                "realization", iris.analysis.MEAN))
        self.forecast_predictor_realizations = (
            self.historic_wind_speed_forecast_cube.copy())
        self.forecast_variance = (
            self.historic_wind_speed_forecast_cube.collapsed(
                "realization", iris.analysis.VARIANCE))
        self.truth = (
            self.historic_wind_speed_forecast_cube.collapsed(
                "realization", iris.analysis.MAX))
        self.forecast_predictor_data = (
            self.forecast_predictor_mean.data.flatten().astype(
                np.float64))
        self.forecast_predictor_data_realizations = (
            convert_cube_data_to_2d(
                self.historic_wind_speed_forecast_cube.copy()
            ).astype(np.float64))
        self.forecast_variance_data = (
            self.forecast_variance.data.flatten().astype(
                np.float64))
        self.truth_data = self.truth.data.flatten().astype(np.float64)


class Test_calculate_truncated_normal_crps(SetupTruncatedGaussianInputs):

    """
    Test minimising the crps for a truncated gaussian distribution.
    Either the ensemble mean or the individual ensemble realizations are used
    as the predictors.
    """
    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_mean_predictor(self):
        """
        Test that the plugin returns a numpy float value. The ensemble mean
        is the predictor. The result indicates the minimum value
        for the CRPS that was achieved by the minimisation.
        """
        predictor = "mean"

        plugin = Plugin()
        result = plugin.calculate_truncated_normal_crps(
            self.initial_guess_for_mean, self.forecast_predictor_data,
            self.truth_data, self.forecast_variance_data, self.sqrt_pi,
            predictor)

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, 0.1670168)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_realizations_predictor(self):
        """
        Test that the plugin returns a numpy float value. The ensemble
        realizations are the predictor. The result indicates the minimum value
        for the CRPS that was achieved by the minimisation.
        """
        predictor = "realizations"

        plugin = Plugin()
        result = plugin.calculate_truncated_normal_crps(
            self.initial_guess_for_realization,
            self.forecast_predictor_data_realizations, self.truth_data,
            self.forecast_variance_data, self.sqrt_pi, predictor)

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, 0.1670167)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "invalid value encountered in"],
        warning_types=[UserWarning, RuntimeWarning])
    def test_basic_mean_predictor_bad_value(self):
        """
        Test that the plugin returns a numpy float64 value
        and that the value matches the BAD_VALUE, when the appropriate
        condition is found. The ensemble mean is the predictor. The initial
        guess is specifically set to float32 precision for the purpose for
        generating the BAD_VALUE for the unit test.
        """
        initial_guess = np.array([1e65, 1e65, 1e65, 1e65], dtype=np.float32)

        predictor = "mean"

        plugin = Plugin()
        result = plugin.calculate_truncated_normal_crps(
            initial_guess, self.forecast_predictor_data, self.truth_data,
            self.forecast_variance_data, self.sqrt_pi, predictor)

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, plugin.BAD_VALUE)


class Test_process_truncated_gaussian_distribution(
        SetupTruncatedGaussianInputs, EnsembleCalibrationAssertions):

    """
    Test minimising the CRPS for a truncated gaussian distribution.
    Either the ensemble mean or the individual ensemble realizations are used
    as the predictors.
    """
    def setUp(self):
        """Set up expected output."""
        super().setUp()
        self.tolerance = 1e-4
        self.plugin = Plugin(tolerance=self.tolerance)
        self.expected_mean_coefficients = (
            [0.0459, 0.6047, 0.3965, 0.958])
        self.expected_realizations_coefficients = (
            [0.0265, 0.2175, 0.2692, 0.0126, 0.5965, 0.7952])

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "The final iteration resulted in",
                          "invalid value encountered in",
                          "divide by zero encountered in"],
        warning_types=[UserWarning, UserWarning, RuntimeWarning,
                       RuntimeWarning])
    def test_basic_mean_predictor(self):
        """
        Test that the plugin returns a numpy array. The ensemble mean
        is the predictor.
        """
        predictor = "mean"
        distribution = "truncated_gaussian"

        result = self.plugin.process(
            self.initial_guess_for_mean, self.forecast_predictor_mean,
            self.truth, self.forecast_variance, predictor, distribution)
        self.assertIsInstance(result, np.ndarray)
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_mean_coefficients)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "Minimisation did not result in convergence",
                          "invalid value encountered in",
                          "divide by zero encountered in"],
        warning_types=[UserWarning, UserWarning, RuntimeWarning,
                       RuntimeWarning])
    def test_basic_realizations_predictor(self):
        """
        Test that the plugin returns a numpy array with the expected
        coefficients. The ensemble realizations are the predictor.
        """
        predictor = "realizations"
        distribution = "truncated_gaussian"

        result = self.plugin.process(
            self.initial_guess_for_realization,
            self.forecast_predictor_realizations, self.truth,
            self.forecast_variance, predictor, distribution)
        self.assertIsInstance(result, np.ndarray)
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_realizations_coefficients)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_mean_predictor_keyerror(self):
        """
        Test that an exception is raised when the distribution requested is
        not an available option when the predictor is the
        ensemble mean.
        """
        predictor = "mean"
        distribution = "foo"

        msg = "Distribution requested"
        with self.assertRaisesRegex(KeyError, msg):
            self.plugin.process(
                self.initial_guess_for_mean, self.forecast_predictor_mean,
                self.truth, self.forecast_variance, predictor, distribution)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "Minimisation did not result in convergence",
                          "The final iteration resulted in",
                          "invalid value encountered in",
                          "divide by zero encountered in"],
        warning_types=[UserWarning, UserWarning, UserWarning,
                       RuntimeWarning, RuntimeWarning])
    def test_mean_predictor_max_iterations(self):
        """
        Test that the plugin returns a list of coefficients
        equal to specific values, when the ensemble mean is the predictor
        assuming a truncated normal distribution and the value specified
        for the max_iterations is overridden. The coefficients are
        calculated by minimising the CRPS and using a set default value for
        the initial guess.
        """
        predictor = "mean"
        max_iterations = 400
        distribution = "truncated_gaussian"

        plugin = Plugin(
            tolerance=self.tolerance, max_iterations=max_iterations)
        result = plugin.process(
            self.initial_guess_for_mean, self.forecast_predictor_mean,
            self.truth, self.forecast_variance, predictor, distribution)
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_mean_coefficients)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "Minimisation did not result in convergence",
                          "invalid value encountered in",
                          "divide by zero encountered in"],
        warning_types=[UserWarning, UserWarning, RuntimeWarning,
                       RuntimeWarning])
    def test_realizations_predictor_max_iterations(self):
        """
        Test that the plugin returns a list of coefficients
        equal to specific values, when the ensemble realizations are the
        predictor assuming a truncated normal distribution and the value
        specified for the max_iterations is overridden. The coefficients are
        calculated by minimising the CRPS and using a set default value for
        the initial guess.
        """
        predictor = "realizations"
        max_iterations = 1000
        distribution = "truncated_gaussian"

        plugin = Plugin(
            tolerance=self.tolerance, max_iterations=max_iterations)
        result = plugin.process(
            self.initial_guess_for_realization,
            self.forecast_predictor_realizations, self.truth,
            self.forecast_variance, predictor, distribution)
        self.assertEMOSCoefficientsAlmostEqual(
            result, self.expected_realizations_coefficients)

    @ManageWarnings(
        record=True,
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_catch_warnings(self, warning_list=None):
        """
        Test that a warning is generated if the minimisation
        does not result in a convergence. The ensemble mean is the predictor.
        """
        predictor = "mean"
        distribution = "truncated_gaussian"

        plugin = Plugin(tolerance=self.tolerance, max_iterations=10)
        plugin.process(
            self.initial_guess_for_mean, self.forecast_predictor_mean,
            self.truth, self.forecast_variance, predictor, distribution)
        warning_msg = "Minimisation did not result in convergence after"
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))

    @ManageWarnings(
        record=True,
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_catch_warnings_percentage_change(self, warning_list=None):
        """
        Test that two warnings are generated if the minimisation
        does not result in a convergence. The first warning reports a that
        the minimisation did not result in convergence, whilst the second
        warning reports that the percentage change in the final iteration was
        greater than the tolerated value.
        The ensemble mean is the predictor.
        """
        initial_guess = np.array([5000, 1, 0, 1], dtype=np.float64)

        predictor = "mean"
        distribution = "truncated_gaussian"

        plugin = Plugin(tolerance=self.tolerance, max_iterations=5)

        plugin.process(
            initial_guess, self.forecast_predictor_mean,
            self.truth, self.forecast_variance, predictor, distribution)
        warning_msg_min = "Minimisation did not result in convergence after"
        warning_msg_iter = "The final iteration resulted in a percentage "
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg_min in str(item)
                            for item in warning_list))
        self.assertTrue(any(warning_msg_iter in str(item)
                            for item in warning_list))


if __name__ == '__main__':
    unittest.main()
