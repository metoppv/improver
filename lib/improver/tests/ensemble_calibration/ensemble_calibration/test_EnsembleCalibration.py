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
    def test_basic_temperature_members(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        with the desired length.
        The ensemble members is the predictor.
        """
        calibration_method = "ensemble model output statistics"
        distribution = "gaussian"
        desired_units = "degreesC"
        predictor_of_mean_flag = "members"
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
    def test_basic_wind_speed_members(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        with the desired length.
        The ensemble members is the predictor.
        """
        calibration_method = "ensemble model output_statistics"
        distribution = "truncated gaussian"
        desired_units = "m s^-1"
        predictor_of_mean_flag = "members"
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
            [[231.15002892, 242.40003015, 253.65003137],
             [264.9000326, 276.15003383, 287.40003505],
             [298.65003628, 309.90003751, 321.15003874]])
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
    def test_temperature_members_data_check(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of temperature cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance.
        The ensemble members is the predictor.
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
                [[230.72248097, 241.94440325, 253.16632553],
                 [264.38824782, 275.6101701, 286.83209238],
                 [298.05401466, 309.27593695, 320.49785923]])
            variance_data = np.array(
                [[0.05635014, 0.05635014, 0.05635014],
                 [0.05635014, 0.05635014, 0.05635014],
                 [0.05635014, 0.05635014, 0.05635014]])
        else:
            predictor_data = np.array(
                [[230.53659896, 241.80363361, 253.07066826],
                 [264.33770292, 275.60473757, 286.87177222],
                 [298.13880687, 309.40584153, 320.67287618]])
            variance_data = np.array(
                [[18.04589231, 18.04589231, 18.04589231],
                 [18.04589231, 18.04589231, 18.04589231],
                 [18.04589231, 18.04589231, 18.04589231]])
        calibration_method = "ensemble model output_statistics"
        distribution = "gaussian"
        desired_units = "degreesC"
        predictor_of_mean_flag = "members"
        plugin = Plugin(
            calibration_method, distribution, desired_units,
            predictor_of_mean_flag=predictor_of_mean_flag)
        result = plugin.process(
            self.current_temperature_forecast_cube,
            self.historic_temperature_forecast_cube,
            self.temperature_truth_cube)
        self.assertArrayAlmostEqual(result[0][0].data, predictor_data)
        self.assertArrayAlmostEqual(result[1][0].data, variance_data)

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
             [47.99999861, 55.50000068, 63.00000275]])
        variance_data = np.array(
            [[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]])
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
    def test_wind_speed_members_data_check(self):
        """
        Test that the plugin returns an iris.cube.CubeList
        of wind_speed cubes with the expected data, where the plugin
        returns a cubelist of, firstly, the predictor and, secondly the
        variance.
        The ensemble members is the predictor.
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
        predictor_of_mean_flag = "members"
        plugin = Plugin(
            calibration_method, distribution, desired_units,
            predictor_of_mean_flag=predictor_of_mean_flag)
        result = plugin.process(
            self.current_wind_speed_forecast_cube,
            self.historic_wind_speed_forecast_cube,
            self.wind_speed_truth_cube)
        self.assertArrayAlmostEqual(result[0][0].data, predictor_data)
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
