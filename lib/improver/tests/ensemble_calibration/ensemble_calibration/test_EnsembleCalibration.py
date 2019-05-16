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
import unittest

import iris
from iris.tests import IrisTest
import numpy as np

from improver.ensemble_calibration.ensemble_calibration import (
    EnsembleCalibration as Plugin)
from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import _create_historic_forecasts, _create_truth
from improver.tests.set_up_test_cubes import set_up_variable_cube
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


class Test_process(IrisTest):

    """Test the process plugin."""

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def setUp(self):
        """Set up temperature and wind speed cubes for testing."""
        # Note: test_temperature_realizations_data_check produces ~0.5K
        # different results when the temperature forecast cube is float32
        # below. A bug?
        data = (np.tile(np.linspace(-45.0, 45.0, 9), 3).reshape(3, 3, 3) +
                273.15)
        data[0] -= 2
        data[1] += 2
        data[2] += 4
        data = data.astype(np.float32)
        self.current_temperature_forecast_cube = set_up_variable_cube(
            data, units="Kelvin", realizations=[0, 1, 2])

        self.historic_temperature_forecast_cube = (
            _create_historic_forecasts(self.current_temperature_forecast_cube))

        self.temperature_truth_cube = (
            _create_truth(self.current_temperature_forecast_cube))

        data = np.tile(np.linspace(0, 60, 9), 3).reshape(3, 3, 3)
        data[1] += 2
        data[2] += 4
        data = data.astype(np.float32)
        self.current_wind_speed_forecast_cube = set_up_variable_cube(
            data, name="wind_speed", units="m s-1", realizations=[0, 1, 2])

        self.historic_wind_speed_forecast_cube = (
            _create_historic_forecasts(self.current_wind_speed_forecast_cube))

        self.wind_speed_truth_cube = (
            _create_truth(self.current_wind_speed_forecast_cube))

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic_temperature(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        with the desired length.
        The ensemble mean is the predictor.
        """
        calibration_method = "ensemble model output statistics"
        distribution = "gaussian"
        desired_units = "degreesC"
        plugin = Plugin(calibration_method, distribution, desired_units)
        result = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic_temperature_realizations(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        with the desired length.
        The ensemble realizations is the predictor.
        """
        calibration_method = "ensemble model output statistics"
        distribution = "gaussian"
        desired_units = "degreesC"
        predictor_of_mean_flag = "realizations"
        plugin = Plugin(
            calibration_method, distribution, desired_units,
            predictor_of_mean_flag=predictor_of_mean_flag)
        result = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic_wind_speed(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        with the desired length.
        The ensemble mean is the predictor.
        """
        calibration_method = "ensemble model output_statistics"
        distribution = "truncated gaussian"
        desired_units = "m s^-1"
        plugin = Plugin(calibration_method, distribution, desired_units)
        result = plugin.process(
            self.current_wind_speed_forecast_cube,
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_basic_wind_speed_realizations(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        with the desired length.
        The ensemble realizations is the predictor.
        """
        calibration_method = "ensemble model output_statistics"
        distribution = "truncated gaussian"
        desired_units = "m s^-1"
        predictor_of_mean_flag = "realizations"
        plugin = Plugin(
            calibration_method, distribution, desired_units,
            predictor_of_mean_flag=predictor_of_mean_flag)
        result = plugin.process(
            self.current_wind_speed_forecast_cube,
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_temperature_data_check(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of temperature cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance.
        The ensemble mean is the predictor.
        """
        predictor_data = np.array(
            [[231.15002, 242.40002, 253.65002],
             [264.9, 276.15, 287.4],
             [298.65, 309.9, 321.15]], dtype=np.float32)
        variance_data = np.array(
            [[2.07777316e-11, 2.07777316e-11, 2.07777316e-11],
             [2.07777316e-11, 2.07777316e-11, 2.07777316e-11],
             [2.07777316e-11, 2.07777316e-11, 2.07777316e-11]])
        calibration_method = "ensemble model output_statistics"
        distribution = "gaussian"
        desired_units = "degreesC"
        plugin = Plugin(calibration_method, distribution, desired_units)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertArrayAlmostEqual(calibrated_predictor.data, predictor_data)
        self.assertArrayAlmostEqual(calibrated_variance.data, variance_data)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_temperature_data_check_max_iterations(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of temperature cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance when the maximum number of iterations is specified.
        The ensemble mean is the predictor.
        """
        predictor_data = np.array(
            [[231.15004, 242.38078, 253.61153],
             [264.84225, 276.073, 287.3037],
             [298.53445, 309.7652, 320.99594]], dtype=np.float32)
        variance_data = np.array(
            [[2.9209013, 2.9209013, 2.9209013],
             [2.9209008, 2.9209008, 2.9209008],
             [2.9209008, 2.9209008, 2.9209008]])
        calibration_method = "ensemble model output_statistics"
        distribution = "gaussian"
        desired_units = "degreesC"
        plugin = Plugin(calibration_method, distribution, desired_units,
                        max_iterations=10)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertArrayAlmostEqual(calibrated_predictor.data,
                                    predictor_data)
        self.assertArrayAlmostEqual(calibrated_variance.data,
                                    variance_data)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_temperature_realizations_data_check(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of temperature cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance.
        The ensemble realizations is the predictor.
        """
        import imp
        try:
            imp.find_module('statsmodels')
            statsmodels_found = True
            import statsmodels.api as sm
            self.sm = sm
        except ImportError:
            statsmodels_found = False
        if statsmodels_found:
            predictor_data = np.array(
                [[231.1493, 242.3992, 253.6492],
                 [264.8991, 276.149, 287.399],
                 [298.649, 309.8989, 321.1488]],
                dtype=np.float32
            )
            variance_data = np.array(
                [[0.000001, 0.000001, 0.000001],
                 [0.000001, 0.000001, 0.000001],
                 [0.000001, 0.000001, 0.000001]],
                dtype=np.float32
            )
        else:
            predictor_data = np.array(
                [[231.46936, 242.74925, 254.02914],
                 [265.30902, 276.5889, 287.8688],
                 [299.14868, 310.42856, 321.70844]],
                dtype=np.float32
            )
            variance_data = np.array(
                [[0.89248854, 0.89248854, 0.89248854],
                 [0.89248854, 0.89248854, 0.89248854],
                 [0.89248854, 0.89248854, 0.89248854]],
                dtype=np.float32
            )
        calibration_method = "ensemble model output_statistics"
        distribution = "gaussian"
        desired_units = "degreesC"
        predictor_of_mean_flag = "realizations"
        plugin = Plugin(
            calibration_method, distribution, desired_units,
            predictor_of_mean_flag=predictor_of_mean_flag)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertArrayAlmostEqual(calibrated_predictor.data, predictor_data,
                                    decimal=4)
        self.assertArrayAlmostEqual(calibrated_variance.data, variance_data,
                                    decimal=4)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_wind_speed_data_check(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of wind_speed cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance.
        The ensemble mean is the predictor.
        """
        predictor_data = np.array(
            [[2.9999862, 10.499988, 17.999989],
             [25.49999, 32.999992, 40.499992],
             [47.999996, 55.499996, 63.]], dtype=np.float32)
        variance_data = np.array(
            [[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]],
            dtype=np.float32
        )
        calibration_method = "ensemble model output_statistics"
        distribution = "truncated gaussian"
        desired_units = "m s^-1"
        plugin = Plugin(calibration_method, distribution, desired_units)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_wind_speed_forecast_cube,
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)
        self.assertArrayAlmostEqual(calibrated_predictor.data, predictor_data)
        self.assertArrayAlmostEqual(calibrated_variance.data, variance_data)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_wind_speed_data_check_max_iterations(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of wind_speed cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance when the maximum number of iterations is specified.
        The ensemble mean is the predictor.
        """
        predictor_data = np.array(
            [[3.1675692, 10.681027, 18.194487],
             [25.707945, 33.2214, 40.73486],
             [48.248318, 55.761776, 63.275234]], dtype=np.float32)
        variance_data = np.array(
            [[2.8555098, 2.8555098, 2.8555098],
             [2.8555098, 2.8555098, 2.8555098],
             [2.8555098, 2.8555098, 2.8555098]],
            dtype=np.float32
        )
        calibration_method = "ensemble model output_statistics"
        distribution = "truncated gaussian"
        desired_units = "m s^-1"
        plugin = Plugin(calibration_method, distribution, desired_units,
                        max_iterations=10)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_wind_speed_forecast_cube,
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)
        self.assertArrayAlmostEqual(calibrated_predictor.data,
                                    predictor_data)
        self.assertArrayAlmostEqual(calibrated_variance.data,
                                    variance_data)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_wind_speed_realizations_data_check(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of wind_speed cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance.
        The ensemble realizations is the predictor.
        """
        import imp
        try:
            imp.find_module('statsmodels')
            statsmodels_found = True
            import statsmodels.api as sm
            self.sm = sm
        except ImportError:
            statsmodels_found = False
        if statsmodels_found:
            predictor_data = np.array(
                [[3.15758874, 10.63961216, 18.12163557],
                 [25.60365899, 33.08568241, 40.56770583],
                 [48.04972924, 55.53175266, 63.01377608]])
            variance_data = np.array(
                [[0.01406566, 0.01406566, 0.01406566],
                 [0.01406566, 0.01406566, 0.01406566],
                 [0.01406566, 0.01406566, 0.01406566]])
        else:
            predictor_data = np.array(
                [[2.7344577, 10.298571, 17.862682],
                 [25.426796, 32.99091, 40.55502],
                 [48.119133, 55.683247, 63.24736]])
            variance_data = np.array(
                [[0.03198197, 0.03198197, 0.03198197],
                 [0.03198197, 0.03198197, 0.03198197],
                 [0.03198197, 0.03198197, 0.03198197]])
        calibration_method = "ensemble model output_statistics"
        distribution = "truncated gaussian"
        desired_units = "m s^-1"
        predictor_of_mean_flag = "realizations"
        plugin = Plugin(
            calibration_method, distribution, desired_units,
            predictor_of_mean_flag=predictor_of_mean_flag)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_wind_speed_forecast_cube,
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)
        self.assertArrayAlmostEqual(calibrated_predictor.data, predictor_data,
                                    decimal=4)
        self.assertArrayAlmostEqual(calibrated_variance.data, variance_data)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_alternative_calibration_name(self):
        """
        Test that the plugin returns the calibrated predictor and the
        calibrated variance if an alternative name for the calibration
        is provided. The ensemble mean is the predictor.
        """
        calibration_method = "nonhomogeneous gaussian regression"
        distribution = "gaussian"
        desired_units = "degreesC"
        plugin = Plugin(calibration_method, distribution, desired_units)
        calibrated_predictor, calibrated_variance = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertIsInstance(calibrated_predictor, iris.cube.Cube)
        self.assertIsInstance(calibrated_variance, iris.cube.Cube)

    def test_unknown_calibration_method(self):
        """
        Test that the plugin raises an error if an unknown calibration method
        is requested.
        The ensemble mean is the predictor.
        """
        calibration_method = "unknown"
        distribution = "gaussian"
        desired_units = "degreesC"
        plugin = Plugin(calibration_method, distribution, desired_units)
        msg = "unknown"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(
                self.current_temperature_forecast_cube,
                self.historic_temperature_forecast_cube,
                self.temperature_truth_cube)


if __name__ == '__main__':
    unittest.main()
