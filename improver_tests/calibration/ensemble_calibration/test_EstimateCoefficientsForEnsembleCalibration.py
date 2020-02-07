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
import datetime
import imp
import unittest

import iris
import numpy as np
from iris.tests import IrisTest

from improver.calibration.ensemble_calibration import (
    ContinuousRankedProbabilityScoreMinimisers)
from improver.calibration.ensemble_calibration import (
    EstimateCoefficientsForEnsembleCalibration as Plugin)
from improver.utilities.warnings_handler import ManageWarnings

from ...set_up_test_cubes import set_up_variable_cube
from .helper_functions import (
    EnsembleCalibrationAssertions, SetupCubes, _create_historic_forecasts)

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
    "Invalid value encountered in",  # From calculating percentage change in
                                     # calibration code
]
WARNING_TYPES = [
    UserWarning,
    ImportWarning,
    DeprecationWarning,
    ImportWarning,
    UserWarning,
    UserWarning,
    RuntimeWarning
]


def create_coefficients_cube(template_cube, coeff_names, coeff_values):
    """Create a cube containing EMOS coefficients.

    Args:
        template_cube (iris.cube.Cube):
            Cube containing information about the time,
            forecast_reference_time, forecast_period, x coordinate and
            y coordinate that will be used within the EMOS coefficient cube.
        coeff_names (list):
            The names of the EMOS coefficients. These names will be used to
            construct the coefficient_name coordinate.
        coeff_values (list):
            The values of the coefficients. These values will be used as the
            cube data.

    Returns:
        (tuple): tuple containing:
            **result** (iris.cube.Cube) - The resulting EMOS
                coefficients cube.
            **x_coord** (iris.coords.DimCoord): The x coordinate
                appropriate for describing the domain that the EMOS
                coefficients cube is valid for.
            **y_coord** (iris.coords.DimCoord): The y coordinate
                appropriate for describing the domain that the EMOS
                coefficients cube is valid for.

    """
    coefficient_index = iris.coords.DimCoord(
        np.arange(len(coeff_names)), long_name="coefficient_index",
        units="1")
    dim_coords_and_dims = [(coefficient_index, 0)]

    coefficient_name = iris.coords.AuxCoord(
        coeff_names, long_name="coefficient_name", units="no_unit")

    time_point = np.min(template_cube.coord("time").points)
    time_coord = template_cube.coord("time").copy(time_point)

    frt_orig_coord = (
        template_cube.coord("forecast_reference_time"))
    frt_point = np.min(frt_orig_coord.points)
    frt_coord = frt_orig_coord.copy(frt_point)

    x_point = np.median(template_cube.coord(axis="x").points)
    x_bounds = [template_cube.coord(axis="x").points[0],
                template_cube.coord(axis="x").points[-1]]
    x_coord = template_cube.coord(axis="x").copy(
        points=x_point, bounds=x_bounds)

    y_point = np.median(template_cube.coord(axis="y").points)
    y_bounds = [template_cube.coord(axis="y").points[0],
                template_cube.coord(axis="y").points[-1]]
    y_coord = template_cube.coord(axis="y").copy(
        points=y_point, bounds=y_bounds)

    aux_coords_and_dims = [
        (coefficient_name, 0), (time_coord, None), (frt_coord, None),
        (template_cube[-1].coord("forecast_period"), None),
        (x_coord, None), (y_coord, None)]

    attributes = {"mosg__model_configuration": "uk_det",
                  "diagnostic_standard_name": "air_temperature"}

    result = iris.cube.Cube(
        coeff_values, long_name="emos_coefficients", units="1",
        dim_coords_and_dims=dim_coords_and_dims,
        aux_coords_and_dims=aux_coords_and_dims, attributes=attributes)
    return result, x_coord, y_coord


class SetupExpectedCoefficients(IrisTest):

    """Expected coefficients generated by EMOS using the test data."""

    def setUp(self):
        """Set up expected coefficients."""
        super().setUp()
        # The expected coefficients for temperature in Kelvin.
        self.expected_mean_predictor_gaussian = (
            np.array([0.0013, 0.4675, 25.4302, 0.9058], dtype=np.float32))
        # The expected coefficients for wind speed in m s^-1.
        self.expected_mean_predictor_truncated_gaussian = (
            np.array([-0.0025, 1.5457, -0.5185, 0.9408], dtype=np.float32))

        self.expected_realizations_gaussian_statsmodels = (
            np.array([-0.0003, 1.0022, -0.2838, -0.0774, 0.3892, 0.9167],
                     dtype=np.float32))
        self.expected_realizations_gaussian_no_statsmodels = (
            np.array([0.0001, 1.0227, -0., 0.5785, 0.578, 0.5733],
                     dtype=np.float32))
        self.expected_realizations_truncated_gaussian_statsmodels = (
            np.array([0.0003, 1.2571, -0.606, -0.0623, 0.3786, 0.9014],
                     dtype=np.float32))
        self.expected_realizations_truncated_gaussian_no_statsmodels = (
            np.array([0.0007, 1.356, -0.0015, 0.7171, -0.0089, 0.585],
                     dtype=np.float32))


class Test__init__(SetupCubes):

    """Test the initialisation of the class."""

    def setUp(self):
        """Set up variables for testing."""
        self.distribution = "gaussian"
        self.desired_units = "degreesC"

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coeff_names(self):
        """Test that the plugin instance defines the expected
        coefficient names."""
        expected = ["gamma", "delta", "alpha", "beta"]
        predictor = "mean"
        tolerance = 10
        max_iterations = 10
        plugin = Plugin(self.distribution, self.desired_units,
                        predictor=predictor, tolerance=tolerance,
                        max_iterations=max_iterations)
        self.assertEqual(plugin.coeff_names, expected)

    def test_invalid_distribution(self):
        """Test an error is raised for an invalid distribution"""
        distribution = "biscuits"
        msg = "Given distribution biscuits not available. "
        with self.assertRaisesRegex(ValueError, msg):
            Plugin(distribution, self.desired_units)

    @unittest.skipIf(
        STATSMODELS_FOUND is True, "statsmodels module is available.")
    @ManageWarnings(
        record=True,
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_statsmodels_mean(self, warning_list=None):
        """
        Test that the plugin raises no warnings if the statsmodels module
        is not found for when the predictor is the ensemble mean.
        """
        predictor = "mean"
        statsmodels_warning = "The statsmodels can not be imported"

        Plugin(self.distribution, self.desired_units, predictor=predictor)
        self.assertNotIn(statsmodels_warning, warning_list)

    @unittest.skipIf(
        STATSMODELS_FOUND is True, "statsmodels module is available.")
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
        predictor = "realizations"

        Plugin(self.distribution, self.desired_units, predictor=predictor)
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
               "predictor: mean; "
               "minimiser: <class 'improver.calibration.ensemble_calibration."
               "ContinuousRankedProbabilityScoreMinimisers'>; "
               "coeff_names: ['gamma', 'delta', 'alpha', 'beta']; "
               "tolerance: 0.01; "
               "max_iterations: 1000>")
        self.assertEqual(result, msg)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_with_kwargs(self):
        """Test when keyword arguments are specified."""
        result = str(Plugin(
            self.distribution, self.current_cycle,
            desired_units="Kelvin", predictor="realizations",
            tolerance=10, max_iterations=10))
        msg = ("<EstimateCoefficientsForEnsembleCalibration: "
               "distribution: gaussian; "
               "current_cycle: 20171110T0000Z; "
               "desired_units: Kelvin; "
               "predictor: realizations; "
               "minimiser: <class 'improver.calibration.ensemble_calibration."
               "ContinuousRankedProbabilityScoreMinimisers'>; "
               "coeff_names: ['gamma', 'delta', 'alpha', 'beta']; "
               "tolerance: 10; "
               "max_iterations: 10>")
        self.assertEqual(result, msg)


class Test_create_coefficients_cube(IrisTest):

    """Test the create_coefficients_cube method."""

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def setUp(self):
        """Set up the plugin and cubes for testing."""
        frt_dt = datetime.datetime(2017, 11, 10, 0, 0)
        time_dt = datetime.datetime(2017, 11, 10, 4, 0)
        data = np.ones((3, 3), dtype=np.float32)
        self.historic_forecast = (
            _create_historic_forecasts(
                data, time_dt, frt_dt, standard_grid_metadata="uk_det"
            ).merge_cube())
        data_with_realizations = np.ones((3, 3, 3), dtype=np.float32)
        self.historic_forecast_with_realizations = (
            _create_historic_forecasts(
                data_with_realizations, time_dt, frt_dt,
                standard_grid_metadata="uk_det",
                realizations=[0, 1, 2]).merge_cube())
        self.optimised_coeffs = np.array([0, 1, 2, 3], np.int32)

        coeff_names = ["gamma", "delta", "alpha", "beta"]
        self.expected, self.x_coord, self.y_coord = create_coefficients_cube(
            self.historic_forecast, coeff_names, self.optimised_coeffs)

        self.distribution = "gaussian"
        self.current_cycle = "20171110T0000Z"
        self.desired_units = "degreesC"
        self.predictor = "mean"
        self.plugin = (
            Plugin(distribution=self.distribution,
                   current_cycle=self.current_cycle,
                   desired_units=self.desired_units,
                   predictor=self.predictor))

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
        predictor = "realizations"
        optimised_coeffs = [0, 1, 2, 3, 4, 5]

        # Set up an expected cube.
        coefficient_index = iris.coords.DimCoord(
            optimised_coeffs, long_name="coefficient_index", units="1")
        dim_coords_and_dims = [(coefficient_index, 0)]

        coefficient_name = iris.coords.AuxCoord(
            expected_coeff_names, long_name="coefficient_name",
            units="no_unit")

        time_point = (
            np.min(self.historic_forecast.coord("time").points))
        time_coord = self.historic_forecast.coord("time").copy(time_point)

        frt_orig_coord = (
            self.historic_forecast.coord("forecast_reference_time"))
        frt_point = np.min(frt_orig_coord.points)
        frt_coord = frt_orig_coord.copy(frt_point)

        aux_coords_and_dims = [
            (coefficient_name, 0), (time_coord, None), (frt_coord, None),
            (self.historic_forecast[-1].coord("forecast_period"), None),
            (self.x_coord, None), (self.y_coord, None)]

        attributes = {"mosg__model_configuration": "uk_det",
                      "diagnostic_standard_name": "air_temperature"}

        expected = iris.cube.Cube(
            optimised_coeffs, long_name="emos_coefficients", units="1",
            dim_coords_and_dims=dim_coords_and_dims,
            aux_coords_and_dims=aux_coords_and_dims, attributes=attributes)

        plugin = Plugin(distribution=self.distribution,
                        current_cycle=self.current_cycle,
                        desired_units=self.desired_units,
                        predictor=predictor)
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
        predictor = "realizations"
        optimised_coeffs = [1, 2, 3, 4, 5]
        plugin = Plugin(distribution=distribution,
                        current_cycle=self.current_cycle,
                        desired_units=desired_units,
                        predictor=predictor)
        msg = "The number of coefficients in"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.create_coefficients_cube(
                optimised_coeffs, self.historic_forecast_with_realizations)


class Test_compute_initial_guess(IrisTest):

    """Test the compute_initial_guess method."""

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def setUp(self):
        """
        Use temperature cube to test with. Also set up versions with a masked
        halo surrounding the original data.
        Set up expected results for different situations.
        """
        self.distribution = "gaussian"
        self.desired_units = "degreesC"
        self.predictor = "mean"
        self.no_of_realizations = 3
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
        cube = set_up_variable_cube(
            data, units="Kelvin", realizations=[0, 1, 2])

        self.current_forecast_predictor_mean = cube.collapsed(
            "realization", iris.analysis.MEAN)
        self.current_forecast_predictor_realizations = cube.copy()
        self.truth = cube.collapsed("realization", iris.analysis.MAX)
        # Set up a version of the same cube but with a masked halo surrounding
        # the original data.
        data = np.array([[[np.nan, np.nan, np.nan, np.nan, np.nan],
                          [np.nan, 0., 1., 2., np.nan],
                          [np.nan, 3., 4., 5., np.nan],
                          [np.nan, 6., 7., 8., np.nan],
                          [np.nan, np.nan, np.nan, np.nan, np.nan]],
                         [[np.nan, np.nan, np.nan, np.nan, np.nan],
                          [np.nan, 1., 2., 3, np.nan],
                          [np.nan, 4., 5., 6., np.nan],
                          [np.nan, 7., 8., 9., np.nan],
                          [np.nan, np.nan, np.nan, np.nan, np.nan]],
                         [[np.nan, np.nan, np.nan, np.nan, np.nan],
                          [np.nan, 2., 3., 4., np.nan],
                          [np.nan, 5., 6., 7., np.nan],
                          [np.nan, 8., 9., 10., np.nan],
                          [np.nan, np.nan, np.nan, np.nan, np.nan]]])
        data = data + 273.15
        data = data.astype(np.float32)
        data = np.ma.masked_invalid(data)
        cube = set_up_variable_cube(
            data, units="Kelvin", realizations=[0, 1, 2])

        # Note that when numpy collapses masked arrays it modifies the data
        # type so we convert it back to float32.
        self.current_forecast_predictor_mean_masked_halo = cube.collapsed(
            "realization", iris.analysis.MEAN)
        self.current_forecast_predictor_mean_masked_halo.data = (
            self.current_forecast_predictor_mean_masked_halo.data.astype(
                np.float32))
        self.current_forecast_predictor_realizations_masked_halo = cube.copy()
        self.truth_masked_halo = cube.collapsed(
            "realization", iris.analysis.MAX)
        self.truth_masked_halo.data = self.truth_masked_halo.data.astype(
            np.float32)

        # Set up expected results:
        # Set up results for the case where the
        # estimate_coefficients_from_linear_model_flag is False
        self.expected_mean_predictor_no_linear_model = np.array(
            [0, 1, 0, 1], dtype=np.float32)
        self.expected_realizations_predictor_no_linear_model = np.array(
            [0, 1, 0,
             np.sqrt(1./self.no_of_realizations),
             np.sqrt(1./self.no_of_realizations),
             np.sqrt(1./self.no_of_realizations)], dtype=np.float32)
        # Set up results for the case where the
        # estimate_coefficients_from_linear_model_flag is True
        self.expected_mean_predictor_with_linear_model = np.array(
            [0., 1., 1., 1.], dtype=np.float32)
        self.expected_realizations_predictor_with_linear_model = np.array(
            [0., 1., 0.333333, 0., 0.333333, 0.666667], dtype=np.float32)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic_mean_predictor(self):
        """
        Test that the plugin returns a list containing the initial guess
        for the calibration coefficients, when the ensemble mean is used
        as the predictor.
        """
        estimate_coefficients_from_linear_model_flag = False

        plugin = Plugin(self.distribution, self.desired_units)
        result = plugin.compute_initial_guess(
            self.truth, self.current_forecast_predictor_mean,
            self.predictor, estimate_coefficients_from_linear_model_flag)
        self.assertIsInstance(result, np.ndarray)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic_realizations_predictor(self):
        """
        Test that the plugin returns a list containing the initial guess
        for the calibration coefficients, when the individual ensemble
        realizations are used as predictors.
        """
        predictor = "realizations"
        estimate_coefficients_from_linear_model_flag = False

        plugin = Plugin(self.distribution, self.desired_units)
        result = plugin.compute_initial_guess(
            self.truth, self.current_forecast_predictor_realizations,
            predictor, estimate_coefficients_from_linear_model_flag,
            no_of_realizations=self.no_of_realizations)
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
        estimate_coefficients_from_linear_model_flag = False

        plugin = Plugin(self.distribution, self.desired_units)
        result = plugin.compute_initial_guess(
            self.truth, self.current_forecast_predictor_mean,
            self.predictor, estimate_coefficients_from_linear_model_flag)
        self.assertArrayAlmostEqual(
            result, self.expected_mean_predictor_no_linear_model)

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
        predictor = "realizations"
        estimate_coefficients_from_linear_model_flag = False

        plugin = Plugin(self.distribution, self.desired_units)
        result = plugin.compute_initial_guess(
            self.truth, self.current_forecast_predictor_realizations,
            predictor, estimate_coefficients_from_linear_model_flag,
            no_of_realizations=self.no_of_realizations)
        self.assertArrayAlmostEqual(
            result, self.expected_realizations_predictor_no_linear_model)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_mean_predictor_estimate_coefficients(self):
        """
        Test that the plugin returns the expected values for the initial guess
        for the calibration coefficients, when the ensemble mean is used
        as the predictor. The coefficients are estimated using a linear model,
        where there is an offset of one between the truth and the forecast
        during the training period. Therefore, in this case the result of the
        linear regression is a gradient of 1 and an intercept of 1.
        """
        estimate_coefficients_from_linear_model_flag = True

        plugin = Plugin(self.distribution, self.desired_units)
        result = plugin.compute_initial_guess(
            self.truth, self.current_forecast_predictor_mean,
            self.predictor, estimate_coefficients_from_linear_model_flag)

        self.assertArrayAlmostEqual(
            self.expected_mean_predictor_with_linear_model, result)

    @unittest.skipIf(
        STATSMODELS_FOUND is False, "statsmodels module not available.")
    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_realizations_predictor_estimate_coefficients(self):
        """
        Test that the plugin returns the expected values for the initial guess
        for the calibration coefficients, when the ensemble mean is used
        as the predictor. The coefficients are estimated using a linear model.
        In this case, the result of the linear regression is for an intercept
        of 0.333333 with different weights for the realizations because
        some of the realizations are closer to the truth, in this instance.
        """
        predictor = "realizations"
        estimate_coefficients_from_linear_model_flag = True

        plugin = Plugin(self.distribution, self.desired_units)
        result = plugin.compute_initial_guess(
            self.truth, self.current_forecast_predictor_realizations,
            predictor, estimate_coefficients_from_linear_model_flag,
            no_of_realizations=self.no_of_realizations)
        self.assertArrayAlmostEqual(
            self.expected_realizations_predictor_with_linear_model, result)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_mean_predictor_estimate_coefficients_masked_halo(self):
        """
        Test that the plugin returns the expected values for the initial guess
        for the calibration coefficients, when the ensemble mean is used
        as the predictor. The coefficients are estimated using a linear model,
        where there is an offset of one between the truth and the forecast
        during the training period. Therefore, in this case the result of the
        linear regression is a gradient of 1 and an intercept of 1. In this
        case the original data has been surrounded by a halo of masked nans,
        which gives the same coefficients as the original data.
        """
        estimate_coefficients_from_linear_model_flag = True

        plugin = Plugin(self.distribution, self.desired_units)
        result = plugin.compute_initial_guess(
            self.truth_masked_halo,
            self.current_forecast_predictor_mean_masked_halo,
            self.predictor, estimate_coefficients_from_linear_model_flag)

        self.assertArrayAlmostEqual(
            self.expected_mean_predictor_with_linear_model, result)

    @unittest.skipIf(
        STATSMODELS_FOUND is False, "statsmodels module not available.")
    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_realizations_predictor_estimate_coefficients_masked_halo(self):
        """
        Test that the plugin returns the expected values for the initial guess
        for the calibration coefficients, when the ensemble mean is used
        as the predictor. The coefficients are estimated using a linear model.
        In this case, the result of the linear regression is for an intercept
        of 0.333333 with different weights for the realizations because
        some of the realizations are closer to the truth, in this instance. In
        this case the original data has been surrounded by a halo of masked
        nans, which gives the same coefficients as the original data.
        """
        predictor = "realizations"
        estimate_coefficients_from_linear_model_flag = True

        plugin = Plugin(self.distribution, self.desired_units)
        result = plugin.compute_initial_guess(
            self.truth_masked_halo,
            self.current_forecast_predictor_realizations_masked_halo,
            predictor, estimate_coefficients_from_linear_model_flag,
            no_of_realizations=self.no_of_realizations)
        self.assertArrayAlmostEqual(
            self.expected_realizations_predictor_with_linear_model, result)


class Test_mask_cube(SetupCubes):
    """Test the mask_cube method"""

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def setUp(self):
        """Set up additional cube for land-sea mask."""
        super().setUp()
        mask_data = np.array([[0, 1, 0],
                              [0, 1, 1],
                              [1, 1, 0]],
                             dtype=np.int32)
        self.mask_cube = set_up_variable_cube(
            mask_data, name="land_binary_mask", units="1")
        self.plugin = Plugin("gaussian", "20171110T0000Z")
        # Copy a few slices of the temperature truth cube to test on.
        self.cube3D = self.temperature_truth_cube[0:2, ...].copy()

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic(self):
        """Test that a simple cube is masked in the correct way."""
        expected_result = np.array(
            [[[np.nan, 273.15, np.nan],
              [np.nan, 275.75, 276.55],
              [278.05, 278.35, np.nan]],
             [[np.nan, 273.15, np.nan],
              [np.nan, 275.75, 276.55],
              [278.05, 278.35, np.nan]]], dtype=np.float32)
        expected_result = np.ma.masked_invalid(expected_result)
        self.plugin.mask_cube(self.cube3D, self.mask_cube)
        self.assertArrayAlmostEqual(
            expected_result.data, self.cube3D.data.data)
        self.assertArrayEqual(
            np.ma.getmask(expected_result),
            np.ma.getmask(self.cube3D.data))

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic_2D_input_cube(self):
        """Test that a simple 2D cube is masked in the correct way."""
        cube2D = self.cube3D[0].copy()
        expected_result = np.array(
            [[np.nan, 273.15, np.nan],
             [np.nan, 275.75, 276.55],
             [278.05, 278.35, np.nan]], dtype=np.float32)
        expected_result = np.ma.masked_invalid(expected_result)
        self.plugin.mask_cube(cube2D, self.mask_cube)
        self.assertArrayAlmostEqual(
            expected_result.data, cube2D.data.data)
        self.assertArrayEqual(
            np.ma.getmask(expected_result),
            np.ma.getmask(cube2D.data))

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_fail_mismatched_arrays(self):
        """Test that an error is raised when input have incompatible shapes."""
        cube_mismatched = self.cube3D[..., 0].copy()
        msg = "Cube and landsea_mask shapes are not compatible."
        with self.assertRaisesRegex(IndexError, msg):
            self.plugin.mask_cube(cube_mismatched, self.mask_cube)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_fail_transposed_input(self):
        """Test that an error is raised when the input cube is transposed"""
        self.cube3D.transpose((2, 0, 1))
        msg = "Cube and landsea_mask shapes are not compatible."
        with self.assertRaisesRegex(IndexError, msg):
            self.plugin.mask_cube(self.cube3D, self.mask_cube)


class Test_process(SetupCubes, EnsembleCalibrationAssertions,
                   SetupExpectedCoefficients):

    """Test the process method"""

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def setUp(self):
        """Set up multiple cubes for testing."""
        super().setUp()
        self.current_cycle = "20171110T0000Z"
        self.distribution = "gaussian"

        self.coeff_names = ["gamma", "delta", "alpha", "beta"]
        self.coeff_names_realizations = (
            ['gamma', 'delta', 'alpha', 'beta0', 'beta1', 'beta2'])

        landsea_data = np.array([[0, 0, 0, 0, 0],
                                 [0, 1, 1, 1, 0],
                                 [0, 1, 1, 1, 0],
                                 [0, 1, 1, 1, 0],
                                 [0, 0, 0, 0, 0]], dtype=np.int32)
        self.landsea_cube = set_up_variable_cube(
            landsea_data, name="land_binary_mask", units="1")

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic(self):
        """Ensure that the optimised_coefficients are returned as a cube,
        with the expected number of coefficients."""
        plugin = Plugin(self.distribution, self.current_cycle)
        result = plugin.process(
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(len(result.data), len(self.coeff_names))

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficient_values_for_gaussian_distribution(self):
        """Ensure that the values for the optimised_coefficients match the
        expected values, and the coefficient names also match
        expected values for a Gaussian distribution. In this case,
        a linear least-squares regression is used to construct the initial
        guess."""
        plugin = Plugin(self.distribution, self.current_cycle)
        result = plugin.process(
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)

        self.assertEMOSCoefficientsAlmostEqual(
            result.data, self.expected_mean_predictor_gaussian)
        self.assertArrayEqual(
            result.coord("coefficient_name").points, self.coeff_names)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficient_values_for_gaussian_distribution_landsea_mask(self):
        """Ensure that the values for the optimised_coefficients match the
        expected values, and the coefficient names also match
        expected values for a Gaussian distribution. In this case,
        a linear least-squares regression is used to construct the initial
        guess. The original data is surrounded by a halo that is masked
        out by the landsea_mask, giving the same results as the original data.
        """
        plugin = Plugin(self.distribution, self.current_cycle)
        result = plugin.process(
            self.historic_temperature_forecast_cube_halo,
            self.temperature_truth_cube_halo,
            landsea_mask=self.landsea_cube)

        self.assertEMOSCoefficientsAlmostEqual(
            result.data, self.expected_mean_predictor_gaussian)
        self.assertArrayEqual(
            result.coord("coefficient_name").points, self.coeff_names)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficient_values_for_gaussian_distribution_mismatching_inputs(
            self):
        """Test that the values for the optimised coefficients match the
        expected values, and the coefficient names also match
        expected values for a Gaussian distribution for when the historic
        forecasts and truths input having some mismatches in validity time.
        """
        expected = [0.0041, 0.4885, 23.4593, 0.9128]
        partial_historic_forecasts = (
            self.historic_forecasts[:2] +
            self.historic_forecasts[3:]).merge_cube()
        partial_truth = self.truth[1:].merge_cube()
        plugin = Plugin(self.distribution, self.current_cycle)
        result = plugin.process(partial_historic_forecasts, partial_truth)

        self.assertEMOSCoefficientsAlmostEqual(result.data, expected)
        self.assertArrayEqual(
            result.coord("coefficient_name").points, self.coeff_names)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficients_gaussian_distribution_default_initial_guess(self):
        """Ensure that the values for the optimised_coefficients match the
        expected values, and the coefficient names also match
        expected values for a Gaussian distribution, where the
        default values for the initial guess are used, rather than using a
        linear least-squares regression to construct an initial guess.
        Reducing the value for the tolerance would result in the coefficients
        more closely matching the coefficients created when using a linear
        least-squares regression to construct the initial guess."""
        expected = [0.0001, 1.0374, -0.0001, 0.9974]
        plugin = Plugin(self.distribution, self.current_cycle)
        plugin.ESTIMATE_COEFFICIENTS_FROM_LINEAR_MODEL_FLAG = False
        result = plugin.process(
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)

        self.assertEMOSCoefficientsAlmostEqual(result.data, expected)
        self.assertArrayEqual(
            result.coord("coefficient_name").points, self.coeff_names)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficient_values_for_gaussian_distribution_max_iterations(
            self):
        """Ensure that the values for the optimised_coefficients match the
        expected values, and the coefficient names also match
        expected values for a Gaussian distribution, when the max_iterations
        argument is specified."""
        max_iterations = 800

        plugin = Plugin(
            self.distribution, self.current_cycle,
            max_iterations=max_iterations)
        result = plugin.process(
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)

        self.assertEMOSCoefficientsAlmostEqual(
            result.data, self.expected_mean_predictor_gaussian)
        self.assertArrayEqual(
            result.coord("coefficient_name").points, self.coeff_names)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficient_values_for_truncated_gaussian_distribution(self):
        """Ensure that the values for the optimised_coefficients match the
        expected values, and the coefficient names also match
        expected values for a truncated Gaussian distribution. In this case,
        a linear least-squares regression is used to construct the initial
        guess."""
        distribution = "truncated_gaussian"

        plugin = Plugin(distribution, self.current_cycle)
        result = plugin.process(
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)

        self.assertEMOSCoefficientsAlmostEqual(
            result.data, self.expected_mean_predictor_truncated_gaussian)
        self.assertArrayEqual(
            result.coord("coefficient_name").points, self.coeff_names)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficient_values_for_truncated_gaussian_distribution_mask(self):
        """Ensure that the values for the optimised_coefficients match the
        expected values, and the coefficient names also match
        expected values for a truncated Gaussian distribution. In this case,
        a linear least-squares regression is used to construct the initial
        guess.The original data is surrounded by a halo that is masked
        out by the land-sea mask, giving the same results as the original data.
        """
        distribution = "truncated_gaussian"

        plugin = Plugin(distribution, self.current_cycle)
        result = plugin.process(
            self.historic_wind_speed_forecast_cube_halo,
            self.wind_speed_truth_cube_halo,
            landsea_mask=self.landsea_cube)

        self.assertEMOSCoefficientsAlmostEqual(
            result.data, self.expected_mean_predictor_truncated_gaussian)
        self.assertArrayEqual(
            result.coord("coefficient_name").points, self.coeff_names)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficients_truncated_gaussian_default_initial_guess(self):
        """Ensure that the values for the optimised_coefficients match the
        expected values, and the coefficient names also match
        expected values for a truncated Gaussian distribution, where the
        default values for the initial guess are used, rather than using a
        linear least-squares regression to construct an initial guess.
        Reducing the value for the tolerance would result in the coefficients
        more closely matching the coefficients created when using a linear
        least-squares regression to construct the initial guess."""
        expected = [-0.0013, 1.3785, -0.0002, 0.8557]
        distribution = "truncated_gaussian"

        plugin = Plugin(distribution, self.current_cycle)
        plugin.ESTIMATE_COEFFICIENTS_FROM_LINEAR_MODEL_FLAG = False
        result = plugin.process(
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)

        self.assertEMOSCoefficientsAlmostEqual(result.data, expected)
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
        realizations are used as the predictor."""
        predictor = "realizations"

        plugin = Plugin(self.distribution, self.current_cycle,
                        predictor=predictor)
        result = plugin.process(
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)

        self.assertEMOSCoefficientsAlmostEqual(
            result.data, self.expected_realizations_gaussian_statsmodels)
        self.assertArrayEqual(
            result.coord("coefficient_name").points,
            self.coeff_names_realizations)

    @unittest.skipIf(
        STATSMODELS_FOUND is True, "statsmodels module is available.")
    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficients_gaussian_realizations_no_statsmodels(self):
        """Ensure that the values for the optimised_coefficients match the
        expected values, and the coefficient names also match
        expected values for a Gaussian distribution where the
        realizations are used as the predictor.
        """
        predictor = "realizations"

        plugin = Plugin(self.distribution, self.current_cycle,
                        predictor=predictor)
        result = plugin.process(
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)

        self.assertEMOSCoefficientsAlmostEqual(
            result.data, self.expected_realizations_gaussian_no_statsmodels)
        self.assertArrayEqual(
            result.coord("coefficient_name").points,
            self.coeff_names_realizations)

    @unittest.skipIf(
        STATSMODELS_FOUND is False, "statsmodels module not available.")
    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficients_truncated_gaussian_realizations_statsmodels(self):
        """Ensure that the values for the optimised_coefficients match the
        expected values, and the coefficient names also match
        expected values for a truncated Gaussian distribution where the
        realizations are used as the predictor."""
        distribution = "truncated_gaussian"
        predictor = "realizations"

        plugin = Plugin(distribution, self.current_cycle,
                        predictor=predictor)
        result = plugin.process(
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)
        self.assertEMOSCoefficientsAlmostEqual(
            result.data,
            self.expected_realizations_truncated_gaussian_statsmodels)
        self.assertArrayEqual(
            result.coord("coefficient_name").points,
            self.coeff_names_realizations)

    @unittest.skipIf(
        STATSMODELS_FOUND is True, "statsmodels module is available.")
    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_coefficients_truncated_gaussian_realizations_no_statsmodels(self):
        """Ensure that the values for the optimised_coefficients match the
        expected values, and the coefficient names also match
        expected values for a truncated Gaussian distribution where the
        realizations are used as the predictor."""
        distribution = "truncated_gaussian"
        predictor = "realizations"

        plugin = Plugin(distribution, self.current_cycle, predictor=predictor)
        result = plugin.process(
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)

        self.assertEMOSCoefficientsAlmostEqual(
            result.data,
            self.expected_realizations_truncated_gaussian_no_statsmodels)
        self.assertArrayEqual(
            result.coord("coefficient_name").points,
            self.coeff_names_realizations)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_truth_unit_conversion(self):
        """Ensure the expected optimised coefficients are generated,
        even if the input truth cube has different units."""
        self.temperature_truth_cube.convert_units("Fahrenheit")
        desired_units = "Kelvin"

        plugin = Plugin(
            self.distribution, self.current_cycle, desired_units=desired_units)
        result = plugin.process(
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)

        self.assertEMOSCoefficientsAlmostEqual(
            result.data, self.expected_mean_predictor_gaussian)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_historic_forecast_unit_conversion(self):
        """Ensure the expected optimised coefficients are generated,
        even if the input historic forecast cube has different units."""
        self.historic_temperature_forecast_cube.convert_units("Fahrenheit")
        desired_units = "Kelvin"

        plugin = Plugin(
            self.distribution, self.current_cycle, desired_units=desired_units)
        result = plugin.process(
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)

        self.assertEMOSCoefficientsAlmostEqual(
            result.data, self.expected_mean_predictor_gaussian)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_non_matching_units(self):
        """Test that an exception is raised if the historic forecasts and truth
        have non matching units."""
        self.historic_temperature_forecast_cube.convert_units("Fahrenheit")

        plugin = Plugin(self.distribution, self.current_cycle)

        msg = "The historic forecast units"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(
                self.historic_temperature_forecast_cube,
                self.temperature_truth_cube)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_missing_cube(self):
        """Test that an exception is raised if either of the  historic
        forecasts or truth were missing."""
        self.historic_temperature_forecast_cube.convert_units("Fahrenheit")

        plugin = Plugin(self.distribution, self.current_cycle)

        msg = ".*cubes must be provided"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(
                self.historic_temperature_forecast_cube,
                None)


if __name__ == '__main__':
    unittest.main()
