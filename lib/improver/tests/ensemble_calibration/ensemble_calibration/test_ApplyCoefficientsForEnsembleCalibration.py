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
from improver.tests.ensemble_calibration.ensemble_calibration.\
    test_EstimateCoefficientsForEnsembleCalibration import (
        create_coefficients_cube)
from improver.tests.set_up_test_cubes import set_up_variable_cube
from improver.utilities.warnings_handler import ManageWarnings


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


class Test_process(IrisTest):

    """Test the process plugin."""

    @ManageWarnings(
        ignored_messages=["The statsmodels can not be imported"],
        warning_types=[ImportWarning])
    def setUp(self):
        """Use temperature cube to test with."""
        data = np.ones((3, 3, 3), dtype=np.float32)
        self.current_temperature_forecast_cube = set_up_variable_cube(
            data, realizations=[0, 1, 2])

        optimised_coeffs = [4.55819380e-06, -8.02401974e-09,
                            1.66667055e+00, 1.00000011e+00]
        current_cycle = "20171110T0000Z"
        estimator = (
            EstimateCoefficientsForEnsembleCalibration(
                "gaussian", current_cycle, desired_units="Celsius"))
        self.coeffs_from_mean = (
            estimator.create_coefficients_cube(
                optimised_coeffs, self.current_temperature_forecast_cube))

        optimised_coeffs = np.array([
            4.55819380e-06, -8.02401974e-09, 1.66667055e+00, 1.00000011e+00,
            1.00000011e+00, 1.00000011e+00])
        estimator = (
            EstimateCoefficientsForEnsembleCalibration(
                "gaussian", current_cycle, desired_units="Celsius",
                predictor_of_mean_flag="realizations"))
        self.coeffs_from_realizations = (
            estimator.create_coefficients_cube(
                optimised_coeffs, self.current_temperature_forecast_cube))

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
        plugin = Plugin(cube, self.coeffs_from_realizations,
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


class Test__apply_params(IrisTest):

    """Test the _apply_params plugin."""

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "invalid escape sequence",
                          "can't resolve package from",
                          "The statsmodels can not be imported"],
        warning_types=[UserWarning, DeprecationWarning, ImportWarning,
                       ImportWarning])
    def setUp(self):
        """Use temperature cube to test with."""
        data = (np.tile(np.linspace(-45.0, 45.0, 9), 3).reshape(3, 3, 3) +
                273.15)
        data[0] -= 2
        data[1] += 2
        data[2] += 4
        data = data.astype(np.float32)
        self.current_temperature_forecast_cube = set_up_variable_cube(
            data, units="Kelvin", realizations=[0, 1, 2])

        optimised_coeffs = [4.55819380e-06, -8.02401974e-09,
                            1.66667055e+00, 1.00000011e+00]
        current_cycle = "20171110T0000Z"
        estimator = (
            EstimateCoefficientsForEnsembleCalibration(
                "gaussian", current_cycle, desired_units="Celsius"))
        self.coeffs_from_mean = (
            estimator.create_coefficients_cube(
                optimised_coeffs, self.current_temperature_forecast_cube))

        optimised_coeffs = np.array([5, 1, 0, 0.57, 0.6, 0.6])
        estimator = (
            EstimateCoefficientsForEnsembleCalibration(
                "gaussian", current_cycle, desired_units="Celsius",
                predictor_of_mean_flag="realizations"))
        self.coeffs_from_realizations = (
            estimator.create_coefficients_cube(
                optimised_coeffs, self.current_temperature_forecast_cube))

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
        Test that the plugin returns values for the calibrated predictor (the
        calibrated mean), which match the expected values.
        """
        data = np.array(
            [[231.15001794, 242.40001917, 253.65002041],
             [264.90000639, 276.15000763, 287.40000887],
             [298.6500101, 309.90001134, 321.15001258]]
        )
        cube = self.current_temperature_forecast_cube
        predictor_cube = cube.collapsed("realization", iris.analysis.MEAN)
        variance_cube = cube.collapsed("realization", iris.analysis.VARIANCE)

        plugin = Plugin(cube, self.coeffs_from_mean)
        forecast_predictor, _ = (
            plugin._apply_params(predictor_cube, variance_cube))
        self.assertArrayAlmostEqual(forecast_predictor.data, data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "invalid escape sequence"],
        warning_types=[UserWarning, DeprecationWarning])
    def test_calibrated_variance(self):
        """
        Test that the plugin returns values for the calibrated variance,
        which match the expected values.
        """
        data = np.array([[2.07777316e-11, 2.07777316e-11, 2.07777316e-11],
                         [2.07777316e-11, 2.07777316e-11, 2.07777316e-11],
                         [2.07777316e-11, 2.07777316e-11, 2.07777316e-11]])

        cube = self.current_temperature_forecast_cube

        predictor_cube = cube.collapsed("realization", iris.analysis.MEAN)
        variance_cube = cube.collapsed("realization", iris.analysis.VARIANCE)

        plugin = Plugin(cube, self.coeffs_from_mean)
        _, forecast_variance = (
            plugin._apply_params(predictor_cube, variance_cube))
        self.assertArrayAlmostEqual(forecast_variance.data, data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "invalid escape sequence"],
        warning_types=[UserWarning, DeprecationWarning])
    def test_calibrated_predictor_realizations(self):
        """
        Test that the plugin returns values for the calibrated forecasts,
        which match the expected values when the individual ensemble
        realizations are used as the predictor.
        """
        data = np.array([[239.904142, 251.659267, 263.414393],
                         [275.169518, 286.92465, 298.67975],
                         [310.43488, 322.19, 333.94516]],
                        dtype=np.float32)

        cube = self.current_temperature_forecast_cube

        predictor_cube = cube.copy()
        variance_cube = cube.collapsed("realization", iris.analysis.VARIANCE)

        plugin = Plugin(cube, self.coeffs_from_realizations,
                        predictor_of_mean_flag="realizations")
        forecast_predictor, _ = plugin._apply_params(
            predictor_cube, variance_cube)
        self.assertArrayAlmostEqual(forecast_predictor.data, data,
                                    decimal=4)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate.",
                          "invalid escape sequence"],
        warning_types=[UserWarning, DeprecationWarning])
    def test_calibrated_variance_realizations(self):
        """
        Test that the plugin returns values for the calibrated forecasts,
        which match the expected values when the individual ensemble
        realizations are used as the predictor.
        """
        data = np.array([[34.333333, 34.333333, 34.333333],
                         [34.333333, 34.333333, 34.333333],
                         [34.333333, 34.333333, 34.333333]])

        cube = self.current_temperature_forecast_cube

        predictor_cube = cube.copy()
        variance_cube = cube.collapsed("realization", iris.analysis.VARIANCE)

        plugin = Plugin(cube, self.coeffs_from_realizations,
                        predictor_of_mean_flag="realizations")
        _, forecast_variance = plugin._apply_params(
            predictor_cube, variance_cube)
        self.assertArrayAlmostEqual(forecast_variance.data, data,
                                    decimal=4)


if __name__ == '__main__':
    unittest.main()
