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
from improver.utilities.warnings_handler import ManageWarnings


class Test_normal_crps_minimiser(IrisTest):

    """
    Test minimising the CRPS for a normal distribution.
    Either the ensemble mean or the individual ensemble members are used as
    the predictors.
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
        self.assertAlmostEqual(result, 16.6076833546)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_members_predictor(self):
        """
        Test that the plugin returns a numpy float array with ensemble members
        as predictor.
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

        predictor_of_mean_flag = "members"

        plugin = Plugin()
        result = plugin.normal_crps_minimiser(
            initial_guess, forecast_predictor_data, truth_data,
            forecast_variance_data, sqrt_pi, predictor_of_mean_flag)

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, 4886.94724835)

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
    Either the ensemble mean or the individual ensemble members are used as
    the predictors.
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
        self.assertAlmostEqual(result, 13.1827829517)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_members_predictor(self):
        """
        Test that the plugin returns a numpy array.
        The ensemble members are the predictor.
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

        predictor_of_mean_flag = "members"

        plugin = Plugin()
        result = plugin.truncated_normal_crps_minimiser(
            initial_guess, forecast_predictor_data, truth_data,
            forecast_variance_data, sqrt_pi, predictor_of_mean_flag)

        self.assertIsInstance(result, np.float64)
        self.assertAlmostEqual(result, 533.487612959)

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


class Test_crps_minimiser_wrapper(IrisTest):

    """
    Test minimising the CRPS for a normal distribution.
    Either the ensemble mean or the individual ensemble members are used as
    the predictors.
    """
    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "Minimisation did not result in convergence"])
    def test_basic_normal_mean_predictor(self):
        """
        Test that the plugin returns a numpy float value.
        The ensemble mean is the predictor.
        """
        initial_guess = [5, 1, 0, 1]
        initial_guess = np.array(initial_guess, dtype=np.float32)
        cube = set_up_temperature_cube()

        forecast_predictor = cube.collapsed("realization", iris.analysis.MEAN)
        forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        truth = cube.collapsed("realization", iris.analysis.MAX)

        predictor_of_mean_flag = "mean"

        plugin = Plugin()
        distribution = "gaussian"
        result = plugin.crps_minimiser_wrapper(
            initial_guess, forecast_predictor, truth, forecast_variance,
            predictor_of_mean_flag, distribution)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(
            result, [-0.08169791, -0.09784413, 0.00822535, 1.00956199])

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "Minimisation did not result in convergence"])
    def test_basic_normal_members_predictor(self):
        """
        Test that the plugin returns a numpy array.
        The ensemble members are the predictor.
        """
        initial_guess = [5, 1, 0, 1, 1, 1]
        initial_guess = np.array(initial_guess, dtype=np.float32)
        cube = set_up_temperature_cube()

        forecast_predictor = cube.copy()
        forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        truth = cube.collapsed("realization", iris.analysis.MAX)

        predictor_of_mean_flag = "members"

        plugin = Plugin()
        distribution = "gaussian"
        result = plugin.crps_minimiser_wrapper(
            initial_guess, forecast_predictor, truth, forecast_variance,
            predictor_of_mean_flag, distribution)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(
            result, [6.24021609e+00, 1.35694934e+00, 1.84642787e-03,
                     5.55444682e-01, 5.04367388e-01, 6.68575194e-01])

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_normal_mean_predictor_keyerror(self):
        """
        Test that the minimisation has resulted in a KeyError, if the
        distribution that has been requested was not within the dictionary
        containing the minimisation functions.
        """
        initial_guess = [
            -8.70808509e-06, 7.23255721e-06, 2.66662740e+00, 1.00000012e+00]
        initial_guess = np.array(initial_guess, dtype=np.float32)
        cube = set_up_temperature_cube()

        forecast_predictor = cube.collapsed("realization", iris.analysis.MEAN)
        forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        truth = cube.collapsed("realization", iris.analysis.MAX)

        predictor_of_mean_flag = "mean"

        plugin = Plugin()
        distribution = "foo"
        msg = "Distribution requested"
        with self.assertRaisesRegex(KeyError, msg):
            plugin.crps_minimiser_wrapper(
                initial_guess, forecast_predictor, truth, forecast_variance,
                predictor_of_mean_flag, distribution)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "Minimisation did not result in convergence"])
    def test_normal_mean_predictor_max_iterations(self):
        """
        Test that the plugin returns a list of coefficients
        equal to specific values, when the ensemble mean is the predictor
        assuming a normal distribution and the value specified for the
        MAX_ITERATIONS is overriden. The coefficients are calculated by
        minimising the CRPS and using a set default value for the
        initial guess.
        """
        initial_guess = [5, 1, 0, 1]
        initial_guess = np.array(initial_guess, dtype=np.float32)
        cube = set_up_temperature_cube()

        forecast_predictor = cube.collapsed("realization", iris.analysis.MEAN)
        forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        truth = cube.collapsed("realization", iris.analysis.MAX)

        predictor_of_mean_flag = "mean"

        plugin = Plugin()
        plugin.MAX_ITERATIONS = 400
        distribution = "gaussian"
        result = plugin.crps_minimiser_wrapper(
            initial_guess, forecast_predictor, truth, forecast_variance,
            predictor_of_mean_flag, distribution)
        self.assertArrayAlmostEqual(
            result, [-0.303343, -0.022553, 0.008502, 1.009565])

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "Minimisation did not result in convergence"])
    def test_normal_members_predictor_max_iterations(self):
        """
        Test that the plugin returns a list of coefficients
        equal to specific values, when the ensemble members are the predictor
        assuming a truncated normal distribution and the value specified
        for the MAX_ITERATIONS is overriden. The coefficients are
        calculated by minimising the CRPS and using a set default value for
        the initial guess.
        """
        initial_guess = [5, 1, 0, 1, 1, 1]
        initial_guess = np.array(initial_guess, dtype=np.float32)
        cube = set_up_temperature_cube()

        forecast_predictor = cube.copy()
        forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        truth = cube.collapsed("realization", iris.analysis.MAX)

        predictor_of_mean_flag = "members"

        plugin = Plugin()
        plugin.MAX_ITERATIONS = 400
        distribution = "gaussian"
        result = plugin.crps_minimiser_wrapper(
            initial_guess, forecast_predictor, truth, forecast_variance,
            predictor_of_mean_flag, distribution)
        self.assertArrayAlmostEqual(
            result, [5.375955e+00, 1.457850e+00, 2.566869e-03,
                     1.934232e-01, 5.540603e-01, 8.115994e-01])

    @ManageWarnings(
        record=True,
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_normal_catch_warnings(self, warning_list=None):
        """
        Test that a warning is generated if the minimisation
        does not result in a convergence.
        The ensemble mean is the predictor.
        """
        initial_guess = [5, 1, 0, 1]
        initial_guess = np.array(initial_guess, dtype=np.float32)
        cube = set_up_temperature_cube()

        forecast_predictor = cube.collapsed("realization", iris.analysis.MEAN)
        forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        truth = cube.collapsed("realization", iris.analysis.MAX)

        predictor_of_mean_flag = "mean"

        plugin = Plugin()
        distribution = "gaussian"

        result = plugin.crps_minimiser_wrapper(
            initial_guess, forecast_predictor, truth, forecast_variance,
            predictor_of_mean_flag, distribution)
        self.assertTrue(len(warning_list) == 1)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue("Minimisation did not result in convergence after"
                        in str(warning_list[0]))

    @ManageWarnings(
        record=True,
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_normal_catch_warnings_percentage_change(self, warning_list=None):
        """
        Test that two warnings are generated if the minimisation
        does not result in a convergence. The first warning reports a that
        the minimisation did not result in convergence, whilst the second
        warning reports that the percentage change in the final iteration was
        greater than the tolerated value.
        The ensemble mean is the predictor.
        """
        initial_guess = [500, 100, 0, 100]
        initial_guess = np.array(initial_guess, dtype=np.float32)
        cube = set_up_temperature_cube()

        forecast_predictor = cube.collapsed("realization", iris.analysis.MEAN)
        forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        truth = cube.collapsed("realization", iris.analysis.MAX)

        predictor_of_mean_flag = "mean"

        plugin = Plugin()
        distribution = "gaussian"
        result = plugin.crps_minimiser_wrapper(
            initial_guess, forecast_predictor, truth, forecast_variance,
            predictor_of_mean_flag, distribution)
        self.assertTrue(len(warning_list) == 2)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue("Minimisation did not result in convergence after"
                        in str(warning_list[0]))
        self.assertTrue("The final iteration resulted in a percentage "
                        "change" in str(warning_list[1]))

    """Test minimising the CRPS for a truncated_normal distribution."""
    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "Minimisation did not result in convergence"])
    def test_basic_truncated_normal_mean_predictor(self):
        """
        Test that the plugin returns a numpy float value.
        The ensemble mean is the predictor.
        """
        initial_guess = [5, 1, 0, 1]
        initial_guess = np.array(initial_guess, dtype=np.float32)
        cube = set_up_temperature_cube()

        forecast_predictor = cube.collapsed("realization", iris.analysis.MEAN)
        forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        truth = cube.collapsed("realization", iris.analysis.MAX)

        predictor_of_mean_flag = "mean"

        plugin = Plugin()
        distribution = "truncated gaussian"
        result = plugin.crps_minimiser_wrapper(
            initial_guess, forecast_predictor, truth, forecast_variance,
            predictor_of_mean_flag, distribution)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(
            result, [-0.08169791, -0.09784413, 0.00822535, 1.00956199])

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "Minimisation did not result in convergence"])
    def test_basic_truncated_normal_members_predictor(self):
        """Test that the plugin returns a numpy array."""
        initial_guess = [5, 1, 0, 1, 1, 1]
        initial_guess = np.array(initial_guess, dtype=np.float32)
        cube = set_up_temperature_cube()

        forecast_predictor = cube.copy()
        forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        truth = cube.collapsed("realization", iris.analysis.MAX)

        predictor_of_mean_flag = "members"

        plugin = Plugin()
        distribution = "truncated gaussian"
        result = plugin.crps_minimiser_wrapper(
            initial_guess, forecast_predictor, truth, forecast_variance,
            predictor_of_mean_flag, distribution)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(
            result, [6.24021609e+00, 1.35694934e+00, 1.84642787e-03,
                     5.55444682e-01, 5.04367388e-01, 6.68575194e-01])

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_truncated_normal_mean_predictor_keyerror(self):
        """
        Test that the minimisation has resulted in a successful convergence,
        and that the object returned is an OptimizeResult object, when the
        ensemble mean is the predictor.
        """
        initial_guess = [
            -8.70808509e-06, 7.23255721e-06, 2.66662740e+00, 1.00000012e+00]
        initial_guess = np.array(initial_guess, dtype=np.float32)
        cube = set_up_temperature_cube()

        forecast_predictor = cube.collapsed("realization", iris.analysis.MEAN)
        forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        truth = cube.collapsed("realization", iris.analysis.MAX)

        predictor_of_mean_flag = "mean"

        plugin = Plugin()
        distribution = "foo"
        msg = "Distribution requested"
        with self.assertRaisesRegex(KeyError, msg):
            plugin.crps_minimiser_wrapper(
                initial_guess, forecast_predictor, truth, forecast_variance,
                predictor_of_mean_flag, distribution)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_truncated_normal_members_predictor_keyerror(self):
        """
        Test that the minimisation has resulted in a successful convergence,
        and that the object returned is an OptimizeResult object, when the
        ensemble members are the predictor.
        """
        initial_guess = [
            -8.70808509e-06, 7.23255721e-06, 2.66662740e+00, 1.00000012e+00]
        initial_guess = np.array(initial_guess, dtype=np.float32)
        cube = set_up_temperature_cube()

        forecast_predictor = cube.collapsed("realization", iris.analysis.MEAN)
        forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        truth = cube.collapsed("realization", iris.analysis.MAX)

        predictor_of_mean_flag = "members"

        plugin = Plugin()
        distribution = "foo"
        msg = "Distribution requested"
        with self.assertRaisesRegex(KeyError, msg):
            plugin.crps_minimiser_wrapper(
                initial_guess, forecast_predictor, truth, forecast_variance,
                predictor_of_mean_flag, distribution)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "Minimisation did not result in convergence"])
    def test_truncated_normal_mean_predictor_max_iterations(self):
        """
        Test that the plugin returns a list of coefficients
        equal to specific values, when the ensemble mean is the predictor
        assuming a truncated normal distribution and the value specified
        for the MAX_ITERATIONS is overriden. The coefficients are
        calculated by minimising the CRPS and using a set default value for
        the initial guess.
        """
        initial_guess = [5, 1, 0, 1]
        initial_guess = np.array(initial_guess, dtype=np.float32)
        cube = set_up_temperature_cube()

        forecast_predictor = cube.collapsed("realization", iris.analysis.MEAN)
        forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        truth = cube.collapsed("realization", iris.analysis.MAX)

        predictor_of_mean_flag = "mean"

        plugin = Plugin()
        plugin.MAX_ITERATIONS = 400
        distribution = "truncated gaussian"
        result = plugin.crps_minimiser_wrapper(
            initial_guess, forecast_predictor, truth, forecast_variance,
            predictor_of_mean_flag, distribution)
        self.assertArrayAlmostEqual(
            result, [-0.303343, -0.022553, 0.008502, 1.009565])

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "Minimisation did not result in convergence"])
    def test_truncated_normal_members_predictor_max_iterations(self):
        """
        Test that the plugin returns a list of coefficients
        equal to specific values, when the ensemble members are the predictor
        assuming a truncated normal distribution and the value specified
        for the MAX_ITERATIONS is overriden. The coefficients are
        calculated by minimising the CRPS and using a set default value for
        the initial guess.
        """
        initial_guess = [5, 1, 0, 1, 1, 1]
        initial_guess = np.array(initial_guess, dtype=np.float32)
        cube = set_up_temperature_cube()

        forecast_predictor = cube.copy()
        forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        truth = cube.collapsed("realization", iris.analysis.MAX)

        predictor_of_mean_flag = "members"

        plugin = Plugin()
        plugin.MAX_ITERATIONS = 400
        distribution = "truncated gaussian"
        result = plugin.crps_minimiser_wrapper(
            initial_guess, forecast_predictor, truth, forecast_variance,
            predictor_of_mean_flag, distribution)
        self.assertArrayAlmostEqual(
            result, [5.375955, 1.45785, 0.002567,
                     0.193423, 0.55406, 0.811599])

    @ManageWarnings(
        record=True,
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_truncated_normal_catch_warnings(self, warning_list=None):
        """
        Test that a warning is generated if the minimisation
        does not result in a convergence.
        The ensemble mean is the predictor.
        """
        initial_guess = [5, 1, 0, 1]
        initial_guess = np.array(initial_guess, dtype=np.float32)
        cube = set_up_temperature_cube()

        forecast_predictor = cube.collapsed("realization", iris.analysis.MEAN)
        forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        truth = cube.collapsed("realization", iris.analysis.MAX)

        predictor_of_mean_flag = "mean"

        plugin = Plugin()
        distribution = "truncated gaussian"
        result = plugin.crps_minimiser_wrapper(
            initial_guess, forecast_predictor, truth, forecast_variance,
            predictor_of_mean_flag, distribution)
        self.assertTrue(len(warning_list) == 1)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue("Minimisation did not result in convergence after"
                        in str(warning_list[0]))

    @ManageWarnings(
        record=True,
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_truncated_normal_catch_warnings_percentage_change(
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
        initial_guess = [5000, 1, 0, 1]
        initial_guess = np.array(initial_guess, dtype=np.float32)
        cube = set_up_temperature_cube()

        forecast_predictor = cube.collapsed("realization", iris.analysis.MEAN)
        forecast_variance = cube.collapsed(
            "realization", iris.analysis.VARIANCE)
        truth = cube.collapsed("realization", iris.analysis.MAX)

        predictor_of_mean_flag = "mean"

        plugin = Plugin()
        distribution = "truncated gaussian"
        result = plugin.crps_minimiser_wrapper(
            initial_guess, forecast_predictor, truth, forecast_variance,
            predictor_of_mean_flag, distribution)
        self.assertTrue(len(warning_list) == 2)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue("Minimisation did not result in convergence after"
                        in str(warning_list[0]))
        self.assertTrue("The final iteration resulted in a percentage "
                        "change" in str(warning_list[1]))


if __name__ == '__main__':
    unittest.main()
