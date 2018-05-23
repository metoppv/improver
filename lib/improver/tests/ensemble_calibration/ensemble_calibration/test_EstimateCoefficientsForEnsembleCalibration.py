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
`ensemble_calibration.EstimateCoefficientsForEnsembleCalibration`
class.

"""
import unittest

import iris
from iris.cube import CubeList
from iris.tests import IrisTest
import numpy as np

from improver.ensemble_calibration.ensemble_calibration import (
    EstimateCoefficientsForEnsembleCalibration as Plugin)
from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import (set_up_temperature_cube, set_up_wind_speed_cube,
                             add_forecast_reference_time_and_forecast_period,
                             _create_historic_forecasts, _create_truth)
from improver.utilities.warnings_handler import ManageWarnings


IGNORED_MESSAGES = ["Collapsing a non-contiguous coordinate.",
                    "Not importing directory .*sphinxcontrib'",
                    "The pandas.core.datetools module is deprecated",
                    "numpy.dtype size changed",
                    "The statsmodels can not be imported",
                    "invalid escape sequence",
                    "can't resolve package from",
                    "Collapsing a non-contiguous coordinate.",
                    "Minimisation did not result in"
                    " convergence",
                    "\nThe final iteration resulted in a percentage "
                    "change that is greater than the"
                    " accepted threshold "]
WARNING_TYPES = [UserWarning, ImportWarning, FutureWarning, RuntimeWarning,
                 ImportWarning, DeprecationWarning, ImportWarning, UserWarning,
                 UserWarning, UserWarning]


class Test__init__(IrisTest):

    """Test the initialisation of the class."""

    def setUp(self):
        """Set up cube for testing."""
        self.cube = set_up_temperature_cube()

    @ManageWarnings(
        record=True,
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_statsmodels_mean(self, warning_list=None):
        """
        Test that the plugin raises no warnings if the statsmodels module
        is not found for when the predictor is the ensemble mean.
        """
        import imp
        try:
            statsmodels_found = imp.find_module('statsmodels')
            statsmodels_found = True
        except ImportError:
            statsmodels_found = False

        cube = self.cube

        historic_forecasts = CubeList([])
        for index in [1.0, 2.0, 3.0, 4.0, 5.0]:
            temp_cube = cube.copy()
            temp_cube.coord("time").points = (
                temp_cube.coord("time").points - index)
            historic_forecasts.append(temp_cube)
        historic_forecasts.concatenate_cube()

        current_forecast_predictor = cube
        truth = cube.collapsed("realization", iris.analysis.MAX)
        distribution = "gaussian"
        desired_units = "degreesC"
        predictor_of_mean_flag = "mean"
        no_of_members = 3
        estimate_coefficients_from_linear_model_flag = True

        if not statsmodels_found:
            plugin = Plugin(distribution, desired_units,
                            predictor_of_mean_flag=predictor_of_mean_flag)
            self.assertTrue(len(warning_list) == 0)

    @ManageWarnings(
        record=True,
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "Not importing directory .*sphinxcontrib'",
                          "The pandas.core.datetools module is deprecated",
                          "numpy.dtype size changed",
                          "invalid escape sequence",
                          "can't resolve package from",
                          "Collapsing a non-contiguous coordinate.",
                          "Minimisation did not result in"
                          " convergence",
                          "\nThe final iteration resulted in a percentage "
                          "change that is greater than the"
                          " accepted threshold "],
        warning_types=[UserWarning, ImportWarning, FutureWarning,
                       RuntimeWarning, DeprecationWarning, ImportWarning,
                       UserWarning, UserWarning, UserWarning])
    def test_statsmodels_members(self, warning_list=None):
        """
        Test that the plugin raises the desired warning if the statsmodels
        module is not found for when the predictor is the ensemble members.
        """
        import imp
        try:
            statsmodels_found = imp.find_module('statsmodels')
            statsmodels_found = True
        except ImportError:
            statsmodels_found = False

        cube = self.cube

        historic_forecasts = CubeList([])
        for index in [1.0, 2.0, 3.0, 4.0, 5.0]:
            temp_cube = cube.copy()
            temp_cube.coord("time").points = (
                temp_cube.coord("time").points - index)
            historic_forecasts.append(temp_cube)
        historic_forecasts.concatenate_cube()

        current_forecast_predictor = cube
        truth = cube.collapsed("realization", iris.analysis.MAX)
        distribution = "gaussian"
        desired_units = "degreesC"
        predictor_of_mean_flag = "members"
        no_of_members = 3
        estimate_coefficients_from_linear_model_flag = True

        if not statsmodels_found:
            plugin = Plugin(distribution, desired_units,
                            predictor_of_mean_flag=predictor_of_mean_flag)
            self.assertTrue(len(warning_list) == 1)
            self.assertTrue(any(item.category == ImportWarning
                                for item in warning_list))
            self.assertTrue("The statsmodels can not be imported"
                            in str(warning_list[0]))


class Test_compute_initial_guess(IrisTest):

    """Test the compute_initial_guess plugin."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic_mean_predictor(self):
        """
        Test that the plugin returns a list containing the initial guess
        for the calibration coefficients, when the ensemble mean is used
        as the predictor.
        """
        cube = self.cube

        current_forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        truth = cube.collapsed("realization", iris.analysis.MAX)
        distribution = "gaussian"
        desired_units = "degreesC"
        predictor_of_mean_flag = "mean"
        estimate_coefficients_from_linear_model_flag = False

        plugin = Plugin(distribution, desired_units)
        result = plugin.compute_initial_guess(
            truth, current_forecast_predictor, predictor_of_mean_flag,
            estimate_coefficients_from_linear_model_flag)
        self.assertIsInstance(result, list)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic_members_predictor(self):
        """
        Test that the plugin returns a list containing the initial guess
        for the calibration coefficients, when the individual ensemble members
        are used as predictors.
        """
        cube = self.cube

        current_forecast_predictor = cube.copy()
        truth = cube.collapsed("realization", iris.analysis.MAX)
        distribution = "gaussian"
        desired_units = "degreesC"
        predictor_of_mean_flag = "members"
        no_of_members = 3
        estimate_coefficients_from_linear_model_flag = False

        plugin = Plugin(distribution, desired_units)
        result = plugin.compute_initial_guess(
            truth, current_forecast_predictor, predictor_of_mean_flag,
            estimate_coefficients_from_linear_model_flag,
            no_of_members=no_of_members)
        self.assertIsInstance(result, list)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic_mean_predictor_value_check(self):
        """
        Test that the plugin returns the expected values for the initial guess
        for the calibration coefficients, when the ensemble mean is used
        as the predictor. As coefficients are not estimated using a
        linear model, the default values for the initial guess are used.
        """
        data = [1, 1, 0, 1]

        cube = self.cube

        current_forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        truth = cube.collapsed("realization", iris.analysis.MAX)
        distribution = "gaussian"
        desired_units = "degreesC"
        predictor_of_mean_flag = "mean"
        estimate_coefficients_from_linear_model_flag = False

        plugin = Plugin(distribution, desired_units)
        result = plugin.compute_initial_guess(
            truth, current_forecast_predictor, predictor_of_mean_flag,
            estimate_coefficients_from_linear_model_flag)
        self.assertArrayAlmostEqual(result, data)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic_members_predictor_value_check(self):
        """
        Test that the plugin returns the expected values for the initial guess
        for the calibration coefficients, when the individual ensemble members
        are used as predictors. As coefficients are not estimated using a
        linear model, the default values for the initial guess are used.
        """
        data = [1, 1, 0, 1, 1, 1]
        cube = self.cube

        current_forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        truth = cube.collapsed("realization", iris.analysis.MAX)
        distribution = "gaussian"
        desired_units = "degreesC"
        predictor_of_mean_flag = "members"
        no_of_members = 3
        estimate_coefficients_from_linear_model_flag = False

        plugin = Plugin(distribution, desired_units)
        result = plugin.compute_initial_guess(
            truth, current_forecast_predictor, predictor_of_mean_flag,
            estimate_coefficients_from_linear_model_flag,
            no_of_members=no_of_members)
        self.assertArrayAlmostEqual(result, data)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_mean_predictor_estimate_coefficients(self):
        """
        Test that the plugin returns the expected values for the initial guess
        for the calibration coefficients, when the ensemble mean is used
        as the predictor. The coefficients are estimated using a linear model.
        """
        data = [1, 1, 2.66666667, 1]

        cube = self.cube

        current_forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        truth = cube.collapsed("realization", iris.analysis.MAX)
        distribution = "gaussian"
        desired_units = "degreesC"
        predictor_of_mean_flag = "mean"
        estimate_coefficients_from_linear_model_flag = True

        plugin = Plugin(distribution, desired_units)
        result = plugin.compute_initial_guess(
            truth, current_forecast_predictor, predictor_of_mean_flag,
            estimate_coefficients_from_linear_model_flag)

        self.assertArrayAlmostEqual(result, data)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_members_predictor_estimate_coefficients(self):
        """
        Test that the plugin returns the expected values for the initial guess
        for the calibration coefficients, when the ensemble mean is used
        as the predictor. The coefficients are estimated using a linear model.
        """
        import imp
        try:
            statsmodels_found = imp.find_module('statsmodels')
            statsmodels_found = True
        except ImportError:
            statsmodels_found = False

        if statsmodels_found:
            data = [1., 1., 0.13559322, -0.11864407,
                    0.42372881, 0.69491525]
        else:
            data = [1, 1, 0, 1, 1, 1]

        cube = self.cube

        current_forecast_predictor = cube
        truth = cube.collapsed("realization", iris.analysis.MAX)
        distribution = "gaussian"
        desired_units = "degreesC"
        predictor_of_mean_flag = "members"
        no_of_members = 3
        estimate_coefficients_from_linear_model_flag = True

        plugin = Plugin(distribution, desired_units)
        result = plugin.compute_initial_guess(
            truth, current_forecast_predictor, predictor_of_mean_flag,
            estimate_coefficients_from_linear_model_flag,
            no_of_members=no_of_members)
        self.assertArrayAlmostEqual(result, data)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_mean_predictor_estimate_coefficients_nans(self):
        """
        Test that the plugin returns the expected values for the initial guess
        for the calibration coefficients, when the ensemble mean is used
        as the predictor, when one value from the input data is set to NaN.
        The coefficients are estimated using a linear model.
        """
        data = [1, 1, 2.66666667, 1]

        cube = self.cube

        current_forecast_predictor = cube.collapsed(
            "realization", iris.analysis.MEAN)
        current_forecast_predictor.data = (
            current_forecast_predictor.data.filled())
        truth = cube.collapsed("realization", iris.analysis.MAX)
        distribution = "gaussian"
        desired_units = "degreesC"
        predictor_of_mean_flag = "mean"
        estimate_coefficients_from_linear_model_flag = True

        current_forecast_predictor.data[0][0][0] = np.nan

        plugin = Plugin(distribution, desired_units)
        result = plugin.compute_initial_guess(
            truth, current_forecast_predictor, predictor_of_mean_flag,
            estimate_coefficients_from_linear_model_flag)

        self.assertArrayAlmostEqual(result, data)


class Test_estimate_coefficients_for_ngr(IrisTest):

    """Test the estimate_coefficients_for_ngr plugin."""

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def setUp(self):
        """Set up multiple cubes for testing."""
        self.current_temperature_forecast_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_temperature_cube()))

        self.historic_temperature_forecast_cube = (
            _create_historic_forecasts(self.current_temperature_forecast_cube))

        self.temperature_truth_cube = (
            _create_truth(self.current_temperature_forecast_cube))

        self.current_wind_speed_forecast_cube = (
            add_forecast_reference_time_and_forecast_period(
                set_up_wind_speed_cube()))

        self.historic_wind_speed_forecast_cube = (
            _create_historic_forecasts(self.current_wind_speed_forecast_cube))

        self.wind_speed_truth_cube = (
            _create_truth(self.current_wind_speed_forecast_cube))

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic(self):
        """Ensure that the optimised_coeffs are returned as a dictionary,
           and the coefficient names are returned as a list."""
        current_forecast = self.current_temperature_forecast_cube

        historic_forecasts = self.historic_temperature_forecast_cube

        truth = self.temperature_truth_cube

        distribution = "gaussian"
        desired_units = "degreesC"

        plugin = Plugin(distribution, desired_units)
        result = plugin.estimate_coefficients_for_ngr(
            current_forecast, historic_forecasts, truth)
        optimised_coeffs, coeff_names = result
        self.assertIsInstance(optimised_coeffs, dict)
        self.assertIsInstance(coeff_names, list)
        for key in list(optimised_coeffs.keys()):
            self.assertEqual(
                len(optimised_coeffs[key]), len(coeff_names))

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficient_values_for_gaussian_distribution(self):
        """
        Ensure that the values generated within optimised_coeffs match the
        expected values, and the coefficient names also match
        expected values.
        """

        data = [4.55819380e-06, -8.02401974e-09,
                1.66667055e+00, 1.00000011e+00]

        current_forecast = self.current_temperature_forecast_cube

        historic_forecasts = self.historic_temperature_forecast_cube

        truth = self.temperature_truth_cube

        distribution = "gaussian"
        desired_units = "degreesC"

        plugin = Plugin(distribution, desired_units)
        result = plugin.estimate_coefficients_for_ngr(
            current_forecast, historic_forecasts, truth)
        optimised_coeffs, coeff_names = result

        for key in list(optimised_coeffs.keys()):
            self.assertArrayAlmostEqual(optimised_coeffs[key], data)
        self.assertListEqual(coeff_names, ["gamma", "delta", "a", "beta"])

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficient_values_for_truncated_gaussian_distribution(self):
        """
        Ensure that the values generated within optimised_coeffs match the
        expected values, and the coefficient names also match
        expected values.
        """
        data = [3.16843498e-06, -5.34489037e-06,
                9.99985648e-01, 1.00000028e+00]

        current_forecast = self.current_wind_speed_forecast_cube

        historic_forecasts = self.historic_wind_speed_forecast_cube

        truth = self.wind_speed_truth_cube

        distribution = "truncated gaussian"
        desired_units = "m s^-1"

        plugin = Plugin(distribution, desired_units)
        result = plugin.estimate_coefficients_for_ngr(
            current_forecast, historic_forecasts, truth)
        optimised_coeffs, coeff_names = result

        for key in list(optimised_coeffs.keys()):
            self.assertArrayAlmostEqual(optimised_coeffs[key], data)
        self.assertListEqual(coeff_names, ["gamma", "delta", "a", "beta"])

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficient_values_for_gaussian_distribution_members(self):
        """
        Ensure that the values generated within optimised_coeffs match the
        expected values, and the coefficient names also match
        expected values.
        """
        import imp
        try:
            statsmodels_found = imp.find_module('statsmodels')
            statsmodels_found = True
        except ImportError:
            statsmodels_found = False

        if statsmodels_found:
            data = [0.23710627, 0.0037429, 0.10456126, 0.10277997, 0.66682032,
                    0.7364042]
        else:
            data = [4.30804737e-02, 1.39042785e+00, 8.99047025e-04,
                    2.02661310e-01, 9.27197381e-01, 3.17407626e-01]

        current_forecast = self.current_temperature_forecast_cube

        historic_forecasts = self.historic_temperature_forecast_cube

        truth = self.temperature_truth_cube

        distribution = "gaussian"
        desired_units = "degreesC"
        predictor_of_mean_flag = "members"

        plugin = Plugin(distribution, desired_units,
                        predictor_of_mean_flag=predictor_of_mean_flag)
        result = plugin.estimate_coefficients_for_ngr(
            current_forecast, historic_forecasts, truth)
        optimised_coeffs, coeff_names = result

        for key in list(optimised_coeffs.keys()):
            self.assertArrayAlmostEqual(optimised_coeffs[key], data)
        self.assertListEqual(coeff_names, ["gamma", "delta", "a", "beta"])

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficient_values_for_truncated_gaussian_distribution_mem(self):
        """
        Ensure that the values generated within optimised_coeffs match the
        expected values, and the coefficient names also match
        expected values.
        """
        import imp
        try:
            statsmodels_found = imp.find_module('statsmodels')
            statsmodels_found = True
        except ImportError:
            statsmodels_found = False

        if statsmodels_found:
            data = [0.11821805, -0.00474737, 0.17631301, 0.17178835,
                    0.66749225, 0.72287342]
        else:
            data = [2.05550997, 0.10577237, 0.00028531,
                    0.53208837, 0.67233013, 0.53704241]

        current_forecast = self.current_wind_speed_forecast_cube

        historic_forecasts = self.historic_wind_speed_forecast_cube

        truth = self.wind_speed_truth_cube

        distribution = "truncated gaussian"
        desired_units = "m s^-1"
        predictor_of_mean_flag = "members"

        plugin = Plugin(distribution, desired_units,
                        predictor_of_mean_flag=predictor_of_mean_flag)
        result = plugin.estimate_coefficients_for_ngr(
            current_forecast, historic_forecasts, truth)
        optimised_coeffs, coeff_names = result
        for key in list(optimised_coeffs.keys()):
            self.assertArrayAlmostEqual(optimised_coeffs[key], data)
        self.assertListEqual(coeff_names, ["gamma", "delta", "a", "beta"])

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficient_values_for_fake_distribution(self):
        """
        Ensure the appropriate error is raised if the minimisation function
        requested is not available.
        """
        current_forecast = self.current_temperature_forecast_cube

        historic_forecasts = self.historic_temperature_forecast_cube

        truth = self.temperature_truth_cube

        distribution = "fake"
        desired_units = "degreesC"

        plugin = Plugin(distribution, desired_units)
        msg = "Distribution requested"
        with self.assertRaisesRegex(KeyError, msg):
            plugin.estimate_coefficients_for_ngr(
                current_forecast, historic_forecasts, truth)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_truth_unit_conversion(self):
        """
        Ensure the expected optimised coefficients are generated, even if the
        input truth cube has different units.
        """
        data = [4.55819380e-06, -8.02401974e-09,
                1.66667055e+00, 1.00000011e+00]

        current_forecast = self.current_temperature_forecast_cube

        historic_forecasts = self.historic_temperature_forecast_cube

        truth = self.temperature_truth_cube

        truth.convert_units("Fahrenheit")

        distribution = "gaussian"
        desired_units = "degreesC"

        plugin = Plugin(distribution, desired_units)
        result = plugin.estimate_coefficients_for_ngr(
            current_forecast, historic_forecasts, truth)
        optimised_coeffs = result[0]

        for key in list(optimised_coeffs.keys()):
            self.assertArrayAlmostEqual(optimised_coeffs[key], data)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_historic_forecast_unit_conversion(self):
        """
        Ensure the expected optimised coefficients are generated, even if the
        input historic forecast cube has different units.
        """
        data = [4.55819380e-06, -8.02401974e-09,
                1.66667055e+00, 1.00000011e+00]

        current_forecast = self.current_temperature_forecast_cube

        historic_forecasts = self.historic_temperature_forecast_cube

        historic_forecasts.convert_units("Fahrenheit")

        truth = self.temperature_truth_cube

        distribution = "gaussian"
        desired_units = "degreesC"

        plugin = Plugin(distribution, desired_units)
        result = plugin.estimate_coefficients_for_ngr(
            current_forecast, historic_forecasts, truth)
        optimised_coeffs = result[0]

        for key in list(optimised_coeffs.keys()):
            self.assertArrayAlmostEqual(optimised_coeffs[key], data)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_current_forecast_unit_conversion(self):
        """
        Ensure the expected optimised coefficients are generated, even if the
        input current forecast cube has different units.
        """
        data = [4.55819380e-06, -8.02401974e-09,
                1.66667055e+00, 1.00000011e+00]

        current_forecast = self.current_temperature_forecast_cube

        current_forecast.convert_units("Fahrenheit")

        historic_forecasts = self.historic_temperature_forecast_cube

        truth = self.temperature_truth_cube

        distribution = "gaussian"
        desired_units = "degreesC"

        plugin = Plugin(distribution, desired_units)
        result = plugin.estimate_coefficients_for_ngr(
            current_forecast, historic_forecasts, truth)
        optimised_coeffs = result[0]

        for key in list(optimised_coeffs.keys()):
            self.assertArrayAlmostEqual(optimised_coeffs[key], data)

    def test_truth_data_is_none(self):
        """
        Ensure that a ValueError with the expected text is generated,
        if the input data is None, rather than a cube.
        """
        current_forecast = self.current_temperature_forecast_cube

        historic_forecasts = self.historic_temperature_forecast_cube

        truth = iris.cube.CubeList([None])

        distribution = "gaussian"
        desired_units = "degreesC"

        plugin = Plugin(distribution, desired_units)
        msg = "The input data within the"
        with self.assertRaisesRegex(TypeError, msg):
            plugin.estimate_coefficients_for_ngr(
                current_forecast, historic_forecasts, truth)

    @ManageWarnings(
        ignored_messages=["Insufficient input data present"])
    def test_historic_forecast_is_empty_cubelist(self):
        """
        Ensure that the expected empty list for the optimised coefficients
        and the expected coefficient names are generated, if the input
        cubelist for the historic_forecasts is empty.
        """
        desired_coeff_names = ["gamma", "delta", "a", "beta"]

        current_forecast = self.current_temperature_forecast_cube

        historic_forecasts = iris.cube.CubeList([])

        truth = self.temperature_truth_cube

        distribution = "gaussian"
        desired_units = "degreesC"

        plugin = Plugin(distribution, desired_units)

        result = plugin.estimate_coefficients_for_ngr(
            current_forecast, historic_forecasts, truth)
        optimised_coeffs, coeff_names = result
        self.assertFalse(optimised_coeffs)
        self.assertCountEqual(coeff_names, desired_coeff_names)

    @ManageWarnings(record=True)
    def test_current_forecast_cubes_is_fake_catch_warning(self,
                                                          warning_list=None):
        """
        Ensure that a ValueError with the expected text is generated,
        if the input data is None, rather than a cube.
        """
        current_forecast = "fake"

        historic_forecasts = self.historic_temperature_forecast_cube

        truth = self.temperature_truth_cube

        distribution = "gaussian"
        desired_units = "degreesC"

        plugin = Plugin(distribution, desired_units)

        plugin.estimate_coefficients_for_ngr(
            current_forecast, historic_forecasts, truth)
        self.assertTrue(len(warning_list) == 1)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue("is not a Cube or CubeList"
                        in str(warning_list[0]))

    @ManageWarnings(record=True)
    def test_historic_forecast_cubes_is_fake_catch_warning(self,
                                                           warning_list=None):
        """
        Ensure that a ValueError with the expected text is generated,
        if the input data is None, rather than a cube.
        """
        current_forecast = self.current_temperature_forecast_cube

        historic_forecasts = "fake"

        truth = self.temperature_truth_cube

        distribution = "gaussian"
        desired_units = "degreesC"

        plugin = Plugin(distribution, desired_units)

        plugin.estimate_coefficients_for_ngr(
            current_forecast, historic_forecasts, truth)
        self.assertTrue(len(warning_list) == 1)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue("is not a Cube or CubeList"
                        in str(warning_list[0]))

    @ManageWarnings(record=True)
    def test_truth_data_is_fake_catch_warning(self, warning_list=None):
        """
        Ensure that a ValueError with the expected text is generated,
        if the input data is None, rather than a cube.
        """
        current_forecast = self.current_temperature_forecast_cube

        historic_forecasts = self.historic_temperature_forecast_cube

        truth = "fake"

        distribution = "gaussian"
        desired_units = "degreesC"

        plugin = Plugin(distribution, desired_units)

        plugin.estimate_coefficients_for_ngr(
            current_forecast, historic_forecasts, truth)
        self.assertTrue(len(warning_list) == 1)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue("is not a Cube or CubeList"
                        in str(warning_list[0]))

    @ManageWarnings(record=True)
    def test_current_forecast_cubes_len_zero_catch_warning(self,
                                                           warning_list=None):
        """
        Ensure that a ValueError with the expected text is generated,
        if the input data is None, rather than a cube.
        """
        current_forecast = iris.cube.CubeList([])

        historic_forecasts = self.historic_temperature_forecast_cube

        truth = self.temperature_truth_cube

        distribution = "gaussian"
        desired_units = "degreesC"

        plugin = Plugin(distribution, desired_units)

        plugin.estimate_coefficients_for_ngr(
            current_forecast, historic_forecasts, truth)
        self.assertTrue(len(warning_list) == 1)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue("Insufficient input data present to estimate "
                        "coefficients using NGR." in str(warning_list[0]))

    @ManageWarnings(record=True)
    def test_historic_forecast_cubes_len_zero_catch_warning(self,
                                                            warning_list=None):
        """
        Ensure that a ValueError with the expected text is generated,
        if the input data is None, rather than a cube.
        """
        current_forecast = self.current_temperature_forecast_cube

        historic_forecasts = iris.cube.CubeList([])

        truth = self.temperature_truth_cube

        distribution = "gaussian"
        desired_units = "degreesC"

        plugin = Plugin(distribution, desired_units)

        plugin.estimate_coefficients_for_ngr(
            current_forecast, historic_forecasts, truth)
        self.assertTrue(len(warning_list) == 1)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue("Insufficient input data present to estimate "
                        "coefficients using NGR." in str(warning_list[0]))

    @ManageWarnings(record=True)
    def test_truth_data_length_zero_catch_warning(self, warning_list=None):
        """
        Ensure that a ValueError with the expected text is generated,
        if the input data is None, rather than a cube.
        """
        current_forecast = self.current_temperature_forecast_cube

        historic_forecasts = self.historic_temperature_forecast_cube

        truth = iris.cube.CubeList([])

        distribution = "gaussian"
        desired_units = "degreesC"

        plugin = Plugin(distribution, desired_units)

        plugin.estimate_coefficients_for_ngr(
            current_forecast, historic_forecasts, truth)
        self.assertTrue(len(warning_list) == 1)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue("Insufficient input data present to estimate "
                        "coefficients using NGR." in str(warning_list[0]))

    @ManageWarnings(record=True)
    def test_truth_data_has_wrong_time_catch_warning(self, warning_list=None):
        """
        Ensure that a ValueError with the expected text is generated,
        if the input data is None, rather than a cube.
        """
        current_forecast = self.current_temperature_forecast_cube

        historic_forecasts = self.historic_temperature_forecast_cube

        truth = self.temperature_truth_cube
        truth.coord("forecast_reference_time").points += 10

        distribution = "gaussian"
        desired_units = "degreesC"

        plugin = Plugin(distribution, desired_units)

        plugin.estimate_coefficients_for_ngr(
            current_forecast, historic_forecasts, truth)
        self.assertTrue(len(warning_list) == 1)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue("Unable to calibrate for the time points"
                        in str(warning_list[0]))


if __name__ == '__main__':
    unittest.main()
