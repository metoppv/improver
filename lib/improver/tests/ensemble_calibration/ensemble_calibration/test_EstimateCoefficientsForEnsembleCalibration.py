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
`ensemble_calibration.EstimateCoefficientsForEnsembleCalibration`
class.

"""
import imp
import unittest

import iris
from iris.cube import CubeList
from iris.tests import IrisTest
import numpy as np

from improver.ensemble_calibration.ensemble_calibration import (
    EstimateCoefficientsForEnsembleCalibration as Plugin)
from improver.ensemble_calibration.ensemble_calibration import (
    ContinuousRankedProbabilityScoreMinimisers)
from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import (set_up_temperature_cube,
                             _create_historic_forecasts, _create_truth)
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
                    "Collapsing a non-contiguous coordinate.",
                    "Minimisation did not result in"
                    " convergence",
                    "\nThe final iteration resulted in a percentage "
                    "change that is greater than the"
                    " accepted threshold ",
                    "divide by zero encountered in true_divide",
                    "invalid value encountered in",
                    ]
WARNING_TYPES = [UserWarning, ImportWarning, FutureWarning, RuntimeWarning,
                 ImportWarning, DeprecationWarning, ImportWarning, UserWarning,
                 UserWarning, UserWarning, RuntimeWarning, RuntimeWarning]


class Test__init__(IrisTest):

    """Test the initialisation of the class."""

    def setUp(self):
        """Set up cube for testing."""
        self.cube = set_up_temperature_cube()
        self.distribution = "gaussian"
        self.desired_units = "degreesC"

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coeff_names(self):
        """Test that the plugin instance defines the expected
        coefficient names."""
        expected = ["gamma", "delta", "alpha", "beta"]
        predictor_of_mean_flag = "mean"
        max_iterations = 10
        plugin = Plugin(self.distribution, self.desired_units,
                        predictor_of_mean_flag=predictor_of_mean_flag,
                        max_iterations=max_iterations)
        self.assertEqual(plugin.coeff_names, expected)

    @ManageWarnings(
        record=True,
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_statsmodels_mean(self, warning_list=None):
        """
        Test that the plugin raises no warnings if the statsmodels module
        is not found for when the predictor is the ensemble mean.
        """
        cube = self.cube

        historic_forecasts = CubeList([])
        for index in [1.0, 2.0, 3.0, 4.0, 5.0]:
            temp_cube = cube.copy()
            temp_cube.coord("time").points = (
                temp_cube.coord("time").points - index)
            historic_forecasts.append(temp_cube)
        historic_forecasts.concatenate_cube()

        predictor_of_mean_flag = "mean"

        if not STATSMODELS_FOUND:
            Plugin(self.distribution, self.desired_units,
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
    def test_statsmodels_realizations(self, warning_list=None):
        """
        Test that the plugin raises the desired warning if the statsmodels
        module is not found for when the predictor is the ensemble
        realizations.
        """
        cube = self.cube

        historic_forecasts = CubeList([])
        for index in [1.0, 2.0, 3.0, 4.0, 5.0]:
            temp_cube = cube.copy()
            temp_cube.coord("time").points = (
                temp_cube.coord("time").points - index)
            historic_forecasts.append(temp_cube)
        historic_forecasts.concatenate_cube()

        predictor_of_mean_flag = "realizations"

        if not STATSMODELS_FOUND:
            Plugin(self.distribution, self.desired_units,
                   predictor_of_mean_flag=predictor_of_mean_flag)
            warning_msg = "The statsmodels can not be imported"
            self.assertTrue(any(item.category == ImportWarning
                                for item in warning_list))
            self.assertTrue(any(warning_msg in str(item)
                                for item in warning_list))


class Test__repr__(IrisTest):

    """Test the __repr__ method."""

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def setUp(self):
        """Set up values for tests."""
        self.distribution = "gaussian"
        self.current_cycle = "20171110T0000Z"
        self.minimiser = repr(ContinuousRankedProbabilityScoreMinimisers())
        self.coeff_names = ["gamma", "delta", "alpha", "beta"]

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic(self):
        """Test without specifying keyword arguments"""
        result = str(Plugin(self.distribution, self.current_cycle))
        msg = ("<EstimateCoefficientsForEnsembleCalibration: "
               "distribution: gaussian; "
               "current_cycle: 20171110T0000Z; "
               "desired_units: None; "
               "predictor_of_mean_flag: mean; "
               "minimiser: <class 'improver.ensemble_calibration."
               "ensemble_calibration."
               "ContinuousRankedProbabilityScoreMinimisers'>; "
               "coeff_names: ['gamma', 'delta', 'alpha', 'beta'];"
               "max_iterations: 1000>")
        self.assertEqual(result, msg)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_with_kwargs(self):
        """Test when keyword arguments are specified."""
        result = str(Plugin(
            self.distribution, self.current_cycle,
            desired_units="Kelvin", predictor_of_mean_flag="realizations",
            max_iterations=10))
        msg = ("<EstimateCoefficientsForEnsembleCalibration: "
               "distribution: gaussian; "
               "current_cycle: 20171110T0000Z; "
               "desired_units: Kelvin; "
               "predictor_of_mean_flag: realizations; "
               "minimiser: <class 'improver.ensemble_calibration."
               "ensemble_calibration."
               "ContinuousRankedProbabilityScoreMinimisers'>; "
               "coeff_names: ['gamma', 'delta', 'alpha', 'beta'];"
               "max_iterations: 10>")
        self.assertEqual(result, msg)


class Test_create_coefficients_cube(IrisTest):

    """Test the create_coefficients_cube method."""

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def setUp(self):
        """Set up the plugin and cubes for testing."""
        data = np.ones((3, 3), dtype=np.float32)
        self.historic_forecast = (
            _create_historic_forecasts(set_up_variable_cube(
                data, standard_grid_metadata="uk_det")))
        data_with_realizations = np.ones((3, 3, 3), dtype=np.float32)
        self.historic_forecast_with_realizations = (
            _create_historic_forecasts(set_up_variable_cube(
                data_with_realizations, realizations=[0, 1, 2],
                standard_grid_metadata="uk_det")))
        self.optimised_coeffs = np.array([0, 1, 2, 3], np.int32)
        coeff_names = ["gamma", "delta", "alpha", "beta"]

        coefficient_index = iris.coords.DimCoord(
            self.optimised_coeffs, long_name="coefficient_index", units="1")
        dim_coords_and_dims = [(coefficient_index, 0)]

        coefficient_name = iris.coords.AuxCoord(
            coeff_names, long_name="coefficient_name", units="no_unit")

        time_point = (
            np.max(self.historic_forecast.coord("time").points) + 60*60*24)
        time_coord = self.historic_forecast.coord("time").copy(time_point)

        frt_orig_coord = (
            self.historic_forecast.coord("forecast_reference_time"))
        frt_point = np.max(frt_orig_coord.points) + 60*60*24
        frt_coord = frt_orig_coord.copy(frt_point)

        aux_coords_and_dims = [
            (coefficient_name, 0), (time_coord, None), (frt_coord, None),
            (self.historic_forecast[-1].coord("forecast_period"), None)]

        attributes = {"mosg__model_configuration": "uk_det",
                      "diagnostic_standard_name": "air_temperature"}

        self.expected = iris.cube.Cube(
            self.optimised_coeffs, long_name="emos_coefficients", units="1",
            dim_coords_and_dims=dim_coords_and_dims,
            aux_coords_and_dims=aux_coords_and_dims, attributes=attributes)

        self.distribution = "gaussian"
        self.current_cycle = "20171110T0000Z"
        self.desired_units = "degreesC"
        self.predictor_of_mean_flag = "mean"
        self.plugin = (
            Plugin(distribution=self.distribution,
                   current_cycle=self.current_cycle,
                   desired_units=self.desired_units,
                   predictor_of_mean_flag=self.predictor_of_mean_flag))

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficients_from_mean(self):
        """Test that the expected coefficient cube is returned when the
        ensemble mean is used as the predictor."""
        expected_coeff_names = ["gamma", "delta", "alpha", "beta"]
        result = self.plugin.create_coefficients_cube(
            self.optimised_coeffs, self.historic_forecast)
        self.assertEqual(result, self.expected)
        self.assertEqual(
            self.plugin.coeff_names, expected_coeff_names)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficients_from_realizations(self):
        """Test that the expected coefficient cube is returned when the
        ensemble realizations are used as the predictor."""
        expected_coeff_names = (
            ["gamma", "delta", "alpha", "beta0", "beta1", "beta2"])
        predictor_of_mean_flag = "realizations"
        optimised_coeffs = [0, 1, 2, 3, 4, 5]

        # Set up an expected cube.
        coefficient_index = iris.coords.DimCoord(
            optimised_coeffs, long_name="coefficient_index", units="1")
        dim_coords_and_dims = [(coefficient_index, 0)]

        coefficient_name = iris.coords.AuxCoord(
            expected_coeff_names, long_name="coefficient_name",
            units="no_unit")

        time_point = (
            np.max(self.historic_forecast.coord("time").points) + 60*60*24)
        time_coord = self.historic_forecast.coord("time").copy(time_point)

        frt_orig_coord = (
            self.historic_forecast.coord("forecast_reference_time"))
        frt_point = np.max(frt_orig_coord.points) + 60*60*24
        frt_coord = frt_orig_coord.copy(frt_point)

        aux_coords_and_dims = [
            (coefficient_name, 0), (time_coord, None), (frt_coord, None),
            (self.historic_forecast[-1].coord("forecast_period"), None)]

        attributes = {"mosg__model_configuration": "uk_det",
                      "diagnostic_standard_name": "air_temperature"}

        expected = iris.cube.Cube(
            optimised_coeffs, long_name="emos_coefficients", units="1",
            dim_coords_and_dims=dim_coords_and_dims,
            aux_coords_and_dims=aux_coords_and_dims, attributes=attributes)

        plugin = Plugin(distribution=self.distribution,
                        current_cycle=self.current_cycle,
                        desired_units=self.desired_units,
                        predictor_of_mean_flag=predictor_of_mean_flag)
        result = plugin.create_coefficients_cube(
            optimised_coeffs, self.historic_forecast_with_realizations)
        self.assertEqual(result, expected)
        self.assertArrayEqual(
            result.coord("coefficient_name").points, expected_coeff_names)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficients_from_mean_non_standard_units(self):
        """Test that the expected coefficient cube is returned when the
        historic forecast units are non-standard."""
        expected_coeff_names = ["gamma", "delta", "alpha", "beta"]
        self.historic_forecast.coord("forecast_period").convert_units("hours")
        self.expected.coord("forecast_period").convert_units("hours")
        result = self.plugin.create_coefficients_cube(
            self.optimised_coeffs, self.historic_forecast)
        self.assertEqual(result, self.expected)
        self.assertEqual(
            self.plugin.coeff_names, expected_coeff_names)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_forecast_period_coordinate_not_present(self):
        """Test that the coefficients cube is created correctly when the
        forecast_period coordinate is not present within the input cube."""
        self.expected.remove_coord("forecast_period")
        self.expected.remove_coord("time")
        self.historic_forecast.remove_coord("forecast_period")
        result = self.plugin.create_coefficients_cube(
            self.optimised_coeffs, self.historic_forecast)
        self.assertEqual(result, self.expected)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_model_configuration_not_present(self):
        """Test that the coefficients cube is created correctly when a
        model_configuration coordinate is not present within the input cube."""
        self.expected.attributes.pop("mosg__model_configuration")
        self.historic_forecast.attributes.pop("mosg__model_configuration")
        result = self.plugin.create_coefficients_cube(
            self.optimised_coeffs, self.historic_forecast)
        self.assertEqual(result, self.expected)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_mismatching_number_of_coefficients(self):
        """Test that an exception is raised if the number of coefficients
        provided for creating the coefficients cube is not equal to the
        number of coefficient names."""
        distribution = "truncated_gaussian"
        desired_units = "Fahrenheit"
        predictor_of_mean_flag = "realizations"
        optimised_coeffs = [1, 2, 3, 4, 5]
        plugin = Plugin(distribution=distribution,
                        current_cycle=self.current_cycle,
                        desired_units=desired_units,
                        predictor_of_mean_flag=predictor_of_mean_flag)
        msg = "The number of coefficients in"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.create_coefficients_cube(
                optimised_coeffs, self.historic_forecast_with_realizations)


class Test_compute_initial_guess(IrisTest):

    """Test the compute_initial_guess plugin."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.distribution = "gaussian"
        self.desired_units = "degreesC"
        self.predictor_of_mean_flag = "mean"
        data = np.array([[[0., 1., 2.],
                          [3., 4., 5.],
                          [6., 7., 8.]],
                         [[1., 2., 3],
                          [4., 5., 6.],
                          [7., 8., 9.]],
                         [[2., 3., 4.],
                          [5., 6., 7.],
                          [8., 9., 10.]]])
        data = data + 273.15
        data = data.astype(np.float32)
        self.cube = set_up_variable_cube(
            data, units="Kelvin", realizations=[0, 1, 2])

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic_mean_predictor(self):
        """
        Test that the plugin returns a list containing the initial guess
        for the calibration coefficients, when the ensemble mean is used
        as the predictor.
        """
        current_forecast_predictor = self.cube.collapsed(
            "realization", iris.analysis.MEAN)
        truth = self.cube.collapsed("realization", iris.analysis.MAX)

        estimate_coefficients_from_linear_model_flag = False

        plugin = Plugin(self.distribution, self.desired_units)
        result = plugin.compute_initial_guess(
            truth, current_forecast_predictor, self.predictor_of_mean_flag,
            estimate_coefficients_from_linear_model_flag)
        self.assertIsInstance(result, np.ndarray)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic_realizations_predictor(self):
        """
        Test that the plugin returns a list containing the initial guess
        for the calibration coefficients, when the individual ensemble
        realizations are used as predictors.
        """
        current_forecast_predictor = self.cube.copy()
        truth = self.cube.collapsed("realization", iris.analysis.MAX)

        predictor_of_mean_flag = "realizations"
        no_of_realizations = 3
        estimate_coefficients_from_linear_model_flag = False

        plugin = Plugin(self.distribution, self.desired_units)
        result = plugin.compute_initial_guess(
            truth, current_forecast_predictor, predictor_of_mean_flag,
            estimate_coefficients_from_linear_model_flag,
            no_of_realizations=no_of_realizations)
        self.assertIsInstance(result, np.ndarray)

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

        current_forecast_predictor = self.cube.collapsed(
            "realization", iris.analysis.MEAN)
        truth = self.cube.collapsed("realization", iris.analysis.MAX)

        estimate_coefficients_from_linear_model_flag = False

        plugin = Plugin(self.distribution, self.desired_units)
        result = plugin.compute_initial_guess(
            truth, current_forecast_predictor, self.predictor_of_mean_flag,
            estimate_coefficients_from_linear_model_flag)
        self.assertArrayAlmostEqual(result, data)

    @unittest.skipIf(
        STATSMODELS_FOUND is True, "statsmodels module is available.")
    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic_realizations_predictor_value_check(self):
        """
        Test that the plugin returns the expected values for the initial guess
        for the calibration coefficients, when the individual ensemble
        realizations are used as predictors. As coefficients are not estimated
        using a linear model, the default values for the initial guess
        are used.
        """
        no_of_realizations = 3
        data = [1, 1, 0,
                np.sqrt(1./no_of_realizations), np.sqrt(1./no_of_realizations),
                np.sqrt(1./no_of_realizations)]

        current_forecast_predictor = self.cube.collapsed(
            "realization", iris.analysis.MEAN)
        truth = self.cube.collapsed("realization", iris.analysis.MAX)
        predictor_of_mean_flag = "realizations"
        no_of_realizations = 3
        estimate_coefficients_from_linear_model_flag = False

        plugin = Plugin(self.distribution, self.desired_units)
        result = plugin.compute_initial_guess(
            truth, current_forecast_predictor, predictor_of_mean_flag,
            estimate_coefficients_from_linear_model_flag,
            no_of_realizations=no_of_realizations)
        self.assertArrayAlmostEqual(result, data)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_mean_predictor_estimate_coefficients(self):
        """
        Test that the plugin returns the expected values for the initial guess
        for the calibration coefficients, when the ensemble mean is used
        as the predictor. The coefficients are estimated using a linear model.
        """
        data = np.array([0., 1., 1., 1.], dtype=np.float32)

        current_forecast_predictor = self.cube.collapsed(
            "realization", iris.analysis.MEAN)
        truth = self.cube.collapsed("realization", iris.analysis.MAX)
        estimate_coefficients_from_linear_model_flag = True

        plugin = Plugin(self.distribution, self.desired_units)
        result = plugin.compute_initial_guess(
            truth, current_forecast_predictor, self.predictor_of_mean_flag,
            estimate_coefficients_from_linear_model_flag)

        self.assertArrayAlmostEqual(result, data, decimal=5)

    @unittest.skipIf(
        STATSMODELS_FOUND is False, "statsmodels module not available.")
    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_realizations_predictor_estimate_coefficients(self):
        """
        Test that the plugin returns the expected values for the initial guess
        for the calibration coefficients, when the ensemble mean is used
        as the predictor. The coefficients are estimated using a linear model.
        """
        no_of_realizations = 3

        data = [1., 1., 0.333333, 0., 0.333333, 0.666667]

        current_forecast_predictor = self.cube
        truth = self.cube.collapsed("realization", iris.analysis.MAX)
        predictor_of_mean_flag = "realizations"
        estimate_coefficients_from_linear_model_flag = True

        plugin = Plugin(self.distribution, self.desired_units)
        result = plugin.compute_initial_guess(
            truth, current_forecast_predictor, predictor_of_mean_flag,
            estimate_coefficients_from_linear_model_flag,
            no_of_realizations=no_of_realizations)
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
        data = np.array([0., 1., 1., 1.], dtype=np.float32)

        current_forecast_predictor = self.cube.collapsed(
            "realization", iris.analysis.MEAN)
        current_forecast_predictor.data = (
            current_forecast_predictor.data.filled())
        truth = self.cube.collapsed("realization", iris.analysis.MAX)

        estimate_coefficients_from_linear_model_flag = True

        current_forecast_predictor.data[0][0] = np.nan

        plugin = Plugin(self.distribution, self.desired_units)
        result = plugin.compute_initial_guess(
            truth, current_forecast_predictor, self.predictor_of_mean_flag,
            estimate_coefficients_from_linear_model_flag)
        self.assertArrayAlmostEqual(result, data)


class Test_estimate_coefficients_for_ngr(IrisTest):

    """Test the estimate_coefficients_for_ngr plugin."""

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def setUp(self):
        """Set up multiple cubes for testing."""
        self.current_cycle = "20171110T0000Z"
        self.distribution = "gaussian"
        self.desired_units = "degreesC"
        data = np.array([[[1., 2., 3.],
                          [4., 5., 6.],
                          [7., 8., 9.]],
                         [[1., 2., 3],
                          [4., 5., 6.],
                          [7., 8., 9.]],
                         [[1., 2., 3.],
                          [4., 5., 6.],
                          [7., 8., 9.]]])
        data = data + 273.15
        data = data.astype(np.float32)
        self.current_temperature_forecast_cube = set_up_variable_cube(
            data, units="Kelvin", realizations=[0, 1, 2])

        self.historic_temperature_forecast_cube = (
            _create_historic_forecasts(self.current_temperature_forecast_cube))

        self.temperature_truth_cube = (
            _create_truth(self.current_temperature_forecast_cube))

        # Create a cube for testing wind speed.
        data = np.array([[[1., 2., 3.],
                          [4., 5., 6.],
                          [7., 8., 9.]],
                         [[1., 2., 3],
                          [4., 5., 6.],
                          [7., 8., 9.]],
                         [[1., 2., 3.],
                          [4., 5., 6.],
                          [7., 8., 9.]]])
        data = data.astype(np.float32)
        self.current_wind_speed_forecast_cube = set_up_variable_cube(
            data, name="wind_speed", units="m s-1", realizations=[0, 1, 2])

        self.historic_wind_speed_forecast_cube = (
            _create_historic_forecasts(self.current_wind_speed_forecast_cube))

        self.wind_speed_truth_cube = (
            _create_truth(self.current_wind_speed_forecast_cube))

        self.coeff_names = ["gamma", "delta", "alpha", "beta"]

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic(self):
        """Ensure that the optimised_coefficients are returned as a cube,
        with the expected number of coefficients."""
        plugin = Plugin(self.distribution, self.current_cycle)
        result = plugin.estimate_coefficients_for_ngr(
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(len(result.data), len(self.coeff_names))

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficient_values_for_gaussian_distribution(self):
        """Ensure that the values for the optimised_coefficients match the
        expected values, and the coefficient names also match
        expected values for a Gaussian distribution."""
        data = [-0., 0.971547, -1.,  1.]

        plugin = Plugin(
            self.distribution, self.current_cycle,
            desired_units=self.desired_units)
        result = plugin.estimate_coefficients_for_ngr(
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)

        self.assertArrayAlmostEqual(result.data, data)
        self.assertArrayEqual(
            result.coord("coefficient_name").points, self.coeff_names)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficient_values_for_gaussian_distribution_max_iterations(
            self):
        """Ensure that the values for the optimised_coefficients match the
        expected values, and the coefficient names also match
        expected values for a Gaussian distribution."""
        data = [2.5e-04, 1., -1.,  1.]

        max_iterations = 10

        plugin = Plugin(
            self.distribution, self.current_cycle,
            desired_units=self.desired_units,
            max_iterations=max_iterations)
        result = plugin.estimate_coefficients_for_ngr(
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)

        self.assertArrayAlmostEqual(result.data, data)
        self.assertArrayEqual(
            result.coord("coefficient_name").points, self.coeff_names)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficient_values_for_truncated_gaussian_distribution(self):
        """Ensure that the values for the optimised_coefficients match the
        expected values, and the coefficient names also match
        expected values for a truncated Gaussian distribution."""
        data = [0., 1., -1., 1.]

        distribution = "truncated gaussian"

        plugin = Plugin(distribution, self.current_cycle)
        result = plugin.estimate_coefficients_for_ngr(
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)

        self.assertArrayAlmostEqual(result.data, data)
        self.assertArrayEqual(
            result.coord("coefficient_name").points, self.coeff_names)

    @unittest.skipIf(
        STATSMODELS_FOUND is False, "statsmodels module not available.")
    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficients_gaussian_realizations_statsmodels(self):
        """Ensure that the values for the optimised_coefficients match the
        expected values, and the coefficient names also match
        expected values for a Gaussian distribution where the
        realizations are used as the predictor of the mean."""
        data = [0.00001, -0.25516, -1., 0.66631, 0.55059, 0.50287]

        predictor_of_mean_flag = "realizations"
        expected_coeff_names = (
            ['gamma', 'delta', 'alpha', 'beta0', 'beta1', 'beta2'])

        plugin = Plugin(self.distribution, self.current_cycle,
                        desired_units=self.desired_units,
                        predictor_of_mean_flag=predictor_of_mean_flag)
        result = plugin.estimate_coefficients_for_ngr(
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertArrayAlmostEqual(result.data, data, decimal=5)
        self.assertArrayEqual(
            result.coord("coefficient_name").points, expected_coeff_names)

    @unittest.skipIf(
        STATSMODELS_FOUND is True, "statsmodels module is available.")
    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficients_gaussian_realizations_no_statsmodels(self):
        """Ensure that the values for the optimised_coefficients match the
        expected values, and the coefficient names also match
        expected values for a Gaussian distribution where the
        realizations are used as the predictor of the mean."""
        data = np.array([0.0450917,  123.35383, -0.943704, -0.747887,
                         -0.0180841, -0.655514], dtype=np.float32)

        predictor_of_mean_flag = "realizations"
        expected_coeff_names = (
            ['gamma', 'delta', 'alpha', 'beta0', 'beta1', 'beta2'])

        plugin = Plugin(self.distribution, self.current_cycle,
                        desired_units=self.desired_units,
                        predictor_of_mean_flag=predictor_of_mean_flag)
        result = plugin.estimate_coefficients_for_ngr(
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertArrayAlmostEqual(result.data, data, decimal=5)
        self.assertArrayEqual(
            result.coord("coefficient_name").points, expected_coeff_names)

    @unittest.skipIf(
        STATSMODELS_FOUND is False, "statsmodels module not available.")
    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficients_truncated_gaussian_realizations_statsmodels(self):
        """Ensure that the values for the optimised_coefficients match the
        expected values, and the coefficient names also match
        expected values for a truncated Gaussian distribution where the
        realizations are used as the predictor of the mean."""
        data = [-0., 2.907516, 0.666669, 0.774827, -0.040465, 0.254308]

        distribution = "truncated gaussian"
        predictor_of_mean_flag = "realizations"
        expected_coeff_names = (
            ['gamma', 'delta', 'alpha', 'beta0', 'beta1', 'beta2'])

        plugin = Plugin(distribution, self.current_cycle,
                        predictor_of_mean_flag=predictor_of_mean_flag)
        result = plugin.estimate_coefficients_for_ngr(
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)

        self.assertArrayAlmostEqual(result.data, data)
        self.assertArrayEqual(
            result.coord("coefficient_name").points, expected_coeff_names)

    @unittest.skipIf(
        STATSMODELS_FOUND is True, "statsmodels module is available.")
    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficients_truncated_gaussian_realizations_no_statsmodels(self):
        """Ensure that the values for the optimised_coefficients match the
        expected values, and the coefficient names also match
        expected values for a truncated Gaussian distribution where the
        realizations are used as the predictor of the mean."""
        data = [1.660938, 3.388195, 0.013228, -0.514865, -0.38911, 0.586623]

        distribution = "truncated gaussian"
        predictor_of_mean_flag = "realizations"
        expected_coeff_names = (
            ['gamma', 'delta', 'alpha', 'beta0', 'beta1', 'beta2'])

        plugin = Plugin(distribution, self.current_cycle,
                        predictor_of_mean_flag=predictor_of_mean_flag)
        result = plugin.estimate_coefficients_for_ngr(
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)

        self.assertArrayAlmostEqual(result.data, data)
        self.assertArrayEqual(
            result.coord("coefficient_name").points, expected_coeff_names)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficient_values_for_fake_distribution(self):
        """Ensure the appropriate error is raised if the minimisation function
        requested is not available."""
        distribution = "fake"

        plugin = Plugin(distribution, self.current_cycle)
        msg = "Distribution requested"
        with self.assertRaisesRegex(KeyError, msg):
            plugin.estimate_coefficients_for_ngr(
                self.historic_temperature_forecast_cube,
                self.temperature_truth_cube)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_truth_unit_conversion(self):
        """Ensure the expected optimised coefficients are generated,
        even if the input truth cube has different units."""
        data = [-0., 0.97151, -1.,  1.]

        truth = self.temperature_truth_cube

        truth.convert_units("Fahrenheit")

        plugin = Plugin(
            self.distribution, self.current_cycle,
            desired_units=self.desired_units)
        result = plugin.estimate_coefficients_for_ngr(
            self.historic_temperature_forecast_cube, truth)

        self.assertArrayAlmostEqual(result.data, data, decimal=5)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_historic_forecast_unit_conversion(self):
        """Ensure the expected optimised coefficients are generated,
        even if the input historic forecast cube has different units."""
        data = [-0., 0.96858, -1.,  1.]

        historic_forecast = self.historic_temperature_forecast_cube

        historic_forecast.convert_units("Fahrenheit")

        plugin = Plugin(
            self.distribution, self.current_cycle,
            desired_units=self.desired_units)
        result = plugin.estimate_coefficients_for_ngr(
            historic_forecast, self.temperature_truth_cube)

        self.assertArrayAlmostEqual(result.data, data, decimal=5)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_non_matching_units(self):
        """Test that an exception is raised if the historic forecasts and truth
        have non matching units."""
        historic_forecast = self.historic_temperature_forecast_cube

        historic_forecast.convert_units("Fahrenheit")

        plugin = Plugin(self.distribution, self.current_cycle)

        msg = "The historic forecast units"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.estimate_coefficients_for_ngr(
                historic_forecast, self.temperature_truth_cube)


if __name__ == '__main__':
    unittest.main()
