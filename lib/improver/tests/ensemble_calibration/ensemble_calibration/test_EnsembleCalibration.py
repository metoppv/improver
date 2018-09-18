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
Unit tests for the `ensemble_calibration.EnsembleCalibration` class.

"""
import unittest

from iris.cube import CubeList
from iris.tests import IrisTest
import numpy as np

from improver.ensemble_calibration.ensemble_calibration import (
    EnsembleCalibration as Plugin)
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


class Test_process(IrisTest):

    """Test the process plugin."""

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def setUp(self):
        """Set up temperature and wind speed cubes for testing."""
        # Note: test_temperature_realizations_data_check produces ~0.5K
        # different results when the temperature forecast cube is float32
        # below. A bug?
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
        self.assertIsInstance(result, CubeList)
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
        self.assertIsInstance(result, CubeList)
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
        self.assertIsInstance(result, CubeList)
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
        self.assertIsInstance(result, CubeList)
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
            [[231.150024, 242.400024, 253.650024],
             [264.899994, 276.149994, 287.399994],
             [298.650024, 309.900024, 321.150024]],
            dtype=np.float32
        )
        variance_data = np.array(
            [[2.07777316e-11, 2.07777316e-11, 2.07777316e-11],
             [2.07777316e-11, 2.07777316e-11, 2.07777316e-11],
             [2.07777316e-11, 2.07777316e-11, 2.07777316e-11]])
        calibration_method = "ensemble model output_statistics"
        distribution = "gaussian"
        desired_units = "degreesC"
        plugin = Plugin(calibration_method, distribution, desired_units)
        result = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertArrayAlmostEqual(result[0][0].data, predictor_data)
        self.assertArrayAlmostEqual(result[1][0].data, variance_data)

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
            statsmodels_found = imp.find_module('statsmodels')
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
                [[230.53659896, 241.80363361, 253.07066826],
                 [264.33770292, 275.60473757, 286.87177222],
                 [298.13880687, 309.40584153, 320.67287618]],
                dtype=np.float32
            )
            variance_data = np.array(
                [[18.04589231, 18.04589231, 18.04589231],
                 [18.04589231, 18.04589231, 18.04589231],
                 [18.04589231, 18.04589231, 18.04589231]],
                dtype=np.float32
            )
        calibration_method = "ensemble model output_statistics"
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
        self.assertArrayAlmostEqual(result[0][0].data, predictor_data,
                                    decimal=4)
        self.assertArrayAlmostEqual(result[1][0].data, variance_data,
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
            [[2.9999862, 10.49998827, 17.99999034],
             [25.4999924, 32.99999447, 40.49999654],
             [47.99999861, 55.50000068, 63.00000275]],
            dtype=np.float32
        )
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
        result = plugin.process(
            self.current_wind_speed_forecast_cube,
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)
        self.assertArrayAlmostEqual(result[0][0].data, predictor_data)
        self.assertArrayAlmostEqual(result[1][0].data, variance_data)

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
            statsmodels_found = imp.find_module('statsmodels')
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
                [[2.05799912, 9.73470204, 17.41140496],
                 [25.08810788, 32.7648108, 40.44151372],
                 [48.11821664, 55.79491955, 63.47162247]])
            variance_data = np.array(
                [[4.26987243, 4.26987243, 4.26987243],
                 [4.26987243, 4.26987243, 4.26987243],
                 [4.26987243, 4.26987243, 4.26987243]])
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
        self.assertArrayAlmostEqual(result[0][0].data, predictor_data,
                                    decimal=4)
        self.assertArrayAlmostEqual(result[1][0].data, variance_data)

    @ManageWarnings(
        ignored_messages=IGNORED_MESSAGES, warning_types=WARNING_TYPES)
    def test_alternative_calibration_name(self):
        """
        Test that the plugin returns an iris.cube.CubeList.
        The ensemble mean is the predictor.
        """
        calibration_method = "nonhomogeneous gaussian regression"
        distribution = "gaussian"
        desired_units = "degreesC"
        plugin = Plugin(calibration_method, distribution, desired_units)
        result = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertIsInstance(result, CubeList)

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
