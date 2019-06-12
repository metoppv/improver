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
from iris.tests import IrisTest
import numpy as np

from improver.ensemble_calibration.ensemble_calibration import (
    ContinuousRankedProbabilityScoreMinimisers as Plugin)
from improver.ensemble_calibration.ensemble_calibration_utilities import (
    convert_cube_data_to_2d)
from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import set_up_temperature_cube, set_up_wind_speed_cube
from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import SetupCubes
from improver.utilities.warnings_handler import ManageWarnings


class Test__repr__(IrisTest):

    """Test the __repr__ method."""

    def test_basic(self):
        """A simple tests for the __repr__ method."""
        result = str(Plugin())
        msg = ("<ContinuousRankedProbabilityScoreMinimisers: "
               "minimisation_dict: {'gaussian': 'normal_crps_minimiser', "
               "'truncated gaussian': 'truncated_normal_crps_minimiser'}; "
               "max_iterations: 1000>")
        self.assertEqual(result, msg)

    def test_update_max_iterations(self):
        """A test to update the max_iterations
        keyword argument."""
        result = str(Plugin(max_iterations=10))
        msg = ("<ContinuousRankedProbabilityScoreMinimisers: "
               "minimisation_dict: {'gaussian': 'normal_crps_minimiser', "
               "'truncated gaussian': 'truncated_normal_crps_minimiser'}; "
               "max_iterations: 10>")
        self.assertEqual(result, msg)


class Test_normal_crps_minimiser(IrisTest):

    """
    Test minimising the CRPS for a normal distribution.
    Either the ensemble mean or the individual ensemble realizations are
    used as the predictors.
    """
    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_mean_predictor(self):
        """
        Test that the plugin returns a numpy float value with
        mean as predictor.
        """
        initial_guess = [5, 1, 0, 1]
        initial_guess = np.array(initial_guess, dtype=np.float32)
        cube = set_up_temperature_cube()

        forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        truth = cube.collapsed("realization", iris.analysis.MAX)

        forecast_predictor_data = (
            forecast_predictor.data.flatten().astype(np.float32))
        forecast_variance_data = (
            forecast_variance.data.flatten().astype(np.float32))
        truth_data = truth.data.flatten().astype(np.float32)

        sqrt_pi = np.sqrt(np.pi).astype(np.float32)

        predictor_of_mean_flag = "mean"

        plugin = Plugin()
        result = plugin.normal_crps_minimiser(
            initial_guess, forecast_predictor_data, truth_data,
            forecast_variance_data, sqrt_pi, predictor_of_mean_flag)

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, 16.607763767419634)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_realizations_predictor(self):
        """
        Test that the plugin returns a numpy float array with ensemble
        realizations as predictor.
        """
        initial_guess = [5, 1, 0, 1, 1, 1]
        initial_guess = np.array(initial_guess, dtype=np.float32)
        cube = set_up_temperature_cube()

        forecast_predictor = cube.copy()
        forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        truth = cube.collapsed("realization", iris.analysis.MAX)

        forecast_predictor_data = (
            convert_cube_data_to_2d(
                forecast_predictor).astype(np.float32))
        forecast_variance_data = (
            forecast_variance.data.flatten().astype(np.float32))
        truth_data = truth.data.flatten().astype(np.float32)

        sqrt_pi = np.sqrt(np.pi).astype(np.float32)

        predictor_of_mean_flag = "realizations"

        plugin = Plugin()
        result = plugin.normal_crps_minimiser(
            initial_guess, forecast_predictor_data, truth_data,
            forecast_variance_data, sqrt_pi, predictor_of_mean_flag)

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, 4886.9467779764836)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_mean_predictor_bad_value(self):
        """
        Test that the plugin returns a numpy float64 value
        and that the value matches the BAD_VALUE, when the appropriate
        condition is found.
        The ensemble mean is the predictor.
        """
        initial_guess = [1e65, 1e65, 1e65, 1e65]
        initial_guess = np.array(initial_guess, dtype=np.float32)
        cube = set_up_temperature_cube()

        forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        truth = cube.collapsed("realization", iris.analysis.MAX)

        forecast_predictor_data = (
            forecast_predictor.data.flatten().astype(np.float32))
        forecast_variance_data = (
            forecast_variance.data.flatten().astype(np.float32))
        truth_data = truth.data.flatten().astype(np.float32)

        sqrt_pi = np.sqrt(np.pi).astype(np.float32)

        predictor_of_mean_flag = "mean"

        plugin = Plugin()
        result = plugin.normal_crps_minimiser(
            initial_guess, forecast_predictor_data, truth_data,
            forecast_variance_data, sqrt_pi, predictor_of_mean_flag)

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, plugin.BAD_VALUE)


class Test_truncated_normal_crps_minimiser(IrisTest):

    """
    Test minimising the crps for a truncated normal distribution.
    Either the ensemble mean or the individual ensemble realizations are used
    as the predictors.
    """
    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_mean_predictor(self):
        """
        Test that the plugin returns a numpy float value.
        The ensemble mean is the predictor.
        """
        initial_guess = [5, 1, 0, 1]
        initial_guess = np.array(initial_guess, dtype=np.float32)
        cube = set_up_wind_speed_cube()

        forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        truth = cube.collapsed("realization", iris.analysis.MAX)

        forecast_predictor_data = (
            forecast_predictor.data.flatten().astype(np.float32))
        forecast_variance_data = (
            forecast_variance.data.flatten().astype(np.float32))
        truth_data = truth.data.flatten().astype(np.float32)

        sqrt_pi = np.sqrt(np.pi).astype(np.float32)

        predictor_of_mean_flag = "mean"

        plugin = Plugin()
        result = plugin.truncated_normal_crps_minimiser(
            initial_guess, forecast_predictor_data, truth_data,
            forecast_variance_data, sqrt_pi, predictor_of_mean_flag)

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, 13.182782882390779)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_realizations_predictor(self):
        """
        Test that the plugin returns a numpy array.
        The ensemble realizations are the predictor.
        """
        initial_guess = [5, 1, 0, 1, 1, 1]
        initial_guess = np.array(initial_guess, dtype=np.float32)
        cube = set_up_wind_speed_cube()

        forecast_predictor = cube.copy()
        forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        truth = cube.collapsed("realization", iris.analysis.MAX)

        forecast_predictor_data = (
            convert_cube_data_to_2d(
                forecast_predictor).astype(np.float32))
        forecast_variance_data = (
            forecast_variance.data.flatten().astype(np.float32))
        truth_data = truth.data.flatten().astype(np.float32)

        sqrt_pi = np.sqrt(np.pi).astype(np.float32)

        predictor_of_mean_flag = "realizations"

        plugin = Plugin()
        result = plugin.truncated_normal_crps_minimiser(
            initial_guess, forecast_predictor_data, truth_data,
            forecast_variance_data, sqrt_pi, predictor_of_mean_flag)

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, 533.48760954557883)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_mean_predictor_bad_value(self):
        """
        Test that the plugin returns a numpy float64 value
        and that the value matches the BAD_VALUE, when the appropriate
        condition is found.
        The ensemble mean is the predictor.
        """
        initial_guess = [1e65, 1e65, 1e65, 1e65]
        initial_guess = np.array(initial_guess, dtype=np.float32)
        cube = set_up_wind_speed_cube()

        forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        truth = cube.collapsed("realization", iris.analysis.MAX)

        forecast_predictor_data = (
            forecast_predictor.data.flatten().astype(np.float32))
        forecast_variance_data = (
            forecast_variance.data.flatten().astype(np.float32))
        truth_data = truth.data.flatten().astype(np.float32)

        sqrt_pi = np.sqrt(np.pi).astype(np.float32)

        predictor_of_mean_flag = "mean"

        plugin = Plugin()
        result = plugin.truncated_normal_crps_minimiser(
            initial_guess, forecast_predictor_data, truth_data,
            forecast_variance_data, sqrt_pi, predictor_of_mean_flag)

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, plugin.BAD_VALUE)


class SetupInputs(SetupCubes):

    """Create a class for setting up cubes for testing."""

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."],
        warning_types=[UserWarning])
    def setUp(self):
        """Set up expected output."""
        super().setUp()
        self.temperature_forecast_predictor_mean = (
            self.historic_temperature_forecast_cube.collapsed(
                "realization", iris.analysis.MEAN))
        self.temperature_forecast_predictor_realizations = (
            self.historic_temperature_forecast_cube.copy())
        self.temperature_forecast_variance = (
            self.historic_temperature_forecast_cube.collapsed(
                "realization", iris.analysis.VARIANCE))

        # Create a cube for testing wind speed.
        self.wind_speed_forecast_predictor_mean = (
            self.historic_wind_speed_forecast_cube.collapsed(
                "realization", iris.analysis.MEAN))
        self.wind_speed_forecast_predictor_realizations = (
            self.historic_wind_speed_forecast_cube.copy())
        self.wind_speed_forecast_variance = (
            self.historic_wind_speed_forecast_cube.collapsed(
                "realization", iris.analysis.VARIANCE))

        self.initial_guess_for_mean = np.array([0, 1, 0, 1], dtype=np.float32)
        self.initial_guess_for_realization = (
            np.array([0, 1, 0, np.sqrt(1/3.), np.sqrt(1/3.), np.sqrt(1/3.)],
                     dtype=np.float32))


class Test_crps_minimiser_wrapper_gaussian_distribution(SetupInputs):

    """
    Test minimising the CRPS for a normal distribution.
    Either the ensemble mean or the individual ensemble realizations are used
    as the predictors.
    """
    def setUp(self):
        """Set up expected output."""
        super().setUp()
        self.expected_mean_coefficients = (
            [-0.000235, 0.797650, 0.000423, 0.997330])
        self.expected_realizations_coefficients = (
            [0.0226,  1.0567, -0.0039,  0.3432,  0.2542,  0.9026])

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "Minimisation did not result in convergence",
                          "divide by zero encountered in"],
        warning_types=[UserWarning, UserWarning, RuntimeWarning])
    def test_basic_mean_predictor(self):
        """
        Test that the plugin returns a numpy float value.
        The ensemble mean is the predictor.
        """
        predictor_of_mean_flag = "mean"
        distribution = "gaussian"
        plugin = Plugin()
        result = plugin.crps_minimiser_wrapper(
            self.initial_guess_for_mean,
            self.temperature_forecast_predictor_mean,
            self.temperature_truth_cube,  self.temperature_forecast_variance,
            predictor_of_mean_flag, distribution)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        self.assertArrayAlmostEqual(
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
        Test that the plugin returns a numpy array.
        The ensemble realizations are the predictor.
        """
        predictor_of_mean_flag = "realizations"
        distribution = "gaussian"
        plugin = Plugin()
        result = plugin.crps_minimiser_wrapper(
            self.initial_guess_for_realization,
            self.temperature_forecast_predictor_realizations,
            self.temperature_truth_cube,  self.temperature_forecast_variance,
            predictor_of_mean_flag, distribution)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        self.assertArrayAlmostEqual(
            result, self.expected_realizations_coefficients, decimal=4)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_mean_predictor_keyerror(self):
        """
        Test that the minimisation has resulted in a KeyError, if the
        distribution that has been requested was not within the dictionary
        containing the minimisation functions.
        """
        predictor_of_mean_flag = "mean"
        distribution = "foo"

        plugin = Plugin()
        msg = "Distribution requested"
        with self.assertRaisesRegex(KeyError, msg):
            plugin.crps_minimiser_wrapper(
                self.initial_guess_for_mean,
                self.temperature_forecast_predictor_mean,
                self.temperature_truth_cube,
                self.temperature_forecast_variance,
                predictor_of_mean_flag, distribution)

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
        predictor_of_mean_flag = "mean"
        max_iterations = 400
        distribution = "gaussian"

        plugin = Plugin(max_iterations=max_iterations)
        result = plugin.crps_minimiser_wrapper(
            self.initial_guess_for_mean,
            self.temperature_forecast_predictor_mean,
            self.temperature_truth_cube,  self.temperature_forecast_variance,
            predictor_of_mean_flag, distribution)
        self.assertArrayAlmostEqual(
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
        specified for the MAX_ITERATIONS is overriden. The coefficients are
        calculated by minimising the CRPS and using a set default value for
        the initial guess.
        """
        predictor_of_mean_flag = "realizations"
        max_iterations = 1000
        distribution = "gaussian"

        plugin = Plugin(max_iterations=max_iterations)
        result = plugin.crps_minimiser_wrapper(
            self.initial_guess_for_realization,
            self.temperature_forecast_predictor_realizations,
            self.temperature_truth_cube,  self.temperature_forecast_variance,
            predictor_of_mean_flag, distribution)
        self.assertArrayAlmostEqual(
            result, self.expected_realizations_coefficients, decimal=4)

    @ManageWarnings(
        record=True,
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_catch_warnings(self, warning_list=None):
        """
        Test that a warning is generated if the minimisation
        does not result in a convergence.
        The ensemble mean is the predictor.
        """
        predictor_of_mean_flag = "mean"
        distribution = "gaussian"

        plugin = Plugin(max_iterations=10)
        plugin.crps_minimiser_wrapper(
            self.initial_guess_for_mean,
            self.temperature_forecast_predictor_mean,
            self.temperature_truth_cube,  self.temperature_forecast_variance,
            predictor_of_mean_flag, distribution)
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
        initial_guess = np.array([5000, 1, 0, 1], dtype=np.float32)
        predictor_of_mean_flag = "mean"
        distribution = "gaussian"

        plugin = Plugin(max_iterations=5)
        plugin.crps_minimiser_wrapper(
            initial_guess,
            self.temperature_forecast_predictor_mean,
            self.temperature_truth_cube,  self.temperature_forecast_variance,
            predictor_of_mean_flag, distribution)
        warning_msg_min = "Minimisation did not result in convergence after"
        warning_msg_iter = "The final iteration resulted in a percentage "
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg_min in str(item)
                            for item in warning_list))
        self.assertTrue(any(warning_msg_iter in str(item)
                            for item in warning_list))


class Test_crps_minimiser_wrapper_truncated_gaussian_distribution(SetupInputs):

    """
    Test minimising the CRPS for a normal distribution.
    Either the ensemble mean or the individual ensemble realizations are used
    as the predictors.
    """
    def setUp(self):
        """Set up expected output."""
        super().setUp()
        self.expected_mean_coefficients = (
            [0.000005, 1.543387, -0.514061, 0.939994])
        self.expected_realizations_coefficients = (
            [0.080978, 1.34056, -0.031015, 0.700256, -0.003556, 0.608326])

    """Test minimising the CRPS for a truncated_normal distribution."""
    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "The final iteration resulted in",
                          "invalid value encountered in",
                          "divide by zero encountered in"],
        warning_types=[UserWarning, UserWarning, RuntimeWarning,
                       RuntimeWarning])
    def test_basic_mean_predictor(self):
        """
        Test that the plugin returns a numpy float value.
        The ensemble mean is the predictor.
        """
        predictor_of_mean_flag = "mean"
        distribution = "truncated gaussian"

        plugin = Plugin()
        result = plugin.crps_minimiser_wrapper(
            self.initial_guess_for_mean,
            self.wind_speed_forecast_predictor_mean,
            self.wind_speed_truth_cube,  self.wind_speed_forecast_variance,
            predictor_of_mean_flag, distribution)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, self.expected_mean_coefficients)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "Minimisation did not result in convergence",
                          "invalid value encountered in",
                          "divide by zero encountered in"],
        warning_types=[UserWarning, UserWarning, RuntimeWarning,
                       RuntimeWarning])
    def test_basic_realizations_predictor(self):
        """Test that the plugin returns a numpy array."""
        predictor_of_mean_flag = "realizations"
        distribution = "truncated gaussian"

        plugin = Plugin()
        result = plugin.crps_minimiser_wrapper(
            self.initial_guess_for_realization,
            self.wind_speed_forecast_predictor_realizations,
            self.wind_speed_truth_cube,  self.wind_speed_forecast_variance,
            predictor_of_mean_flag, distribution)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(
            result, self.expected_realizations_coefficients)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_mean_predictor_keyerror(self):
        """
        Test that the minimisation has resulted in a successful convergence,
        and that the object returned is an OptimizeResult object, when the
        ensemble mean is the predictor.
        """
        predictor_of_mean_flag = "mean"
        distribution = "foo"

        plugin = Plugin()
        msg = "Distribution requested"
        with self.assertRaisesRegex(KeyError, msg):
            plugin.crps_minimiser_wrapper(
                self.initial_guess_for_mean,
                self.wind_speed_forecast_predictor_mean,
                self.wind_speed_truth_cube, self.wind_speed_forecast_variance,
                predictor_of_mean_flag, distribution)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_realizations_predictor_keyerror(self):
        """
        Test that the minimisation has resulted in a successful convergence,
        and that the object returned is an OptimizeResult object, when the
        ensemble realizations are the predictor.
        """
        predictor_of_mean_flag = "realizations"
        distribution = "foo"

        plugin = Plugin()
        msg = "Distribution requested"
        with self.assertRaisesRegex(KeyError, msg):
            plugin.crps_minimiser_wrapper(
                self.initial_guess_for_realization,
                self.wind_speed_forecast_predictor_realizations,
                self.wind_speed_truth_cube, self.wind_speed_forecast_variance,
                predictor_of_mean_flag, distribution)

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
        predictor_of_mean_flag = "mean"
        max_iterations = 400
        distribution = "truncated gaussian"

        plugin = Plugin(max_iterations=max_iterations)
        result = plugin.crps_minimiser_wrapper(
            self.initial_guess_for_mean,
            self.wind_speed_forecast_predictor_mean,
            self.wind_speed_truth_cube, self.wind_speed_forecast_variance,
            predictor_of_mean_flag, distribution)
        self.assertArrayAlmostEqual(result, self.expected_mean_coefficients)

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
        predictor_of_mean_flag = "realizations"
        max_iterations = 1000
        distribution = "truncated gaussian"

        plugin = Plugin(max_iterations=max_iterations)
        result = plugin.crps_minimiser_wrapper(
            self.initial_guess_for_realization,
            self.wind_speed_forecast_predictor_realizations,
            self.wind_speed_truth_cube,  self.wind_speed_forecast_variance,
            predictor_of_mean_flag, distribution)
        self.assertArrayAlmostEqual(
            result, self.expected_realizations_coefficients)

    @ManageWarnings(
        record=True,
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_catch_warnings(self, warning_list=None):
        """
        Test that a warning is generated if the minimisation
        does not result in a convergence.
        The ensemble mean is the predictor.
        """
        predictor_of_mean_flag = "mean"
        distribution = "truncated gaussian"

        plugin = Plugin(max_iterations=10)
        plugin.crps_minimiser_wrapper(
            self.initial_guess_for_mean,
            self.wind_speed_forecast_predictor_mean,
            self.wind_speed_truth_cube, self.wind_speed_forecast_variance,
            predictor_of_mean_flag, distribution)
        warning_msg = "Minimisation did not result in convergence after"
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))

    @ManageWarnings(
        record=True,
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_catch_warnings_percentage_change(
            self,
            warning_list=None):
        """
        Test that two warnings are generated if the minimisation
        does not result in a convergence. The first warning reports a that
        the minimisation did not result in convergence, whilst the second
        warning reports that the percentage change in the final iteration was
        greater than the tolerated value.
        The ensemble mean is the predictor.
        """
        initial_guess = np.array([5000, 1, 0, 1], dtype=np.float32)

        predictor_of_mean_flag = "mean"
        distribution = "truncated gaussian"

        plugin = Plugin(max_iterations=5)

        plugin.crps_minimiser_wrapper(
            initial_guess, self.wind_speed_forecast_predictor_mean,
            self.wind_speed_truth_cube, self.wind_speed_forecast_variance,
            predictor_of_mean_flag, distribution)
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
