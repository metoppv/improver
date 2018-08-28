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
"""Unit tests for the ChooseWeightsLinearFromDict plugin."""


import unittest

import iris
from iris.tests import IrisTest
import numpy as np

from improver.blending.weights import ChooseWeightsLinearFromDict
from improver.tests.blending.weights.helper_functions import (
    set_up_temperature_cube, set_up_basic_model_config_cube,
    add_model_id_and_model_configuration, add_height)
from improver.tests.ensemble_calibration.ensemble_calibration.helper_functions\
    import add_forecast_reference_time_and_forecast_period


class Test__init__(IrisTest):
    """Test the __init__ method."""

    def test_with_config_dict(self):
        """Test the class is initialised correctly."""
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                  "weights": [0, 1, 1, 0],
                                  "units": "hours"}}
        weights_coord_name = "weights"
        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name, config_dict)
        self.assertEqual(weighting_coord_name, plugin.weighting_coord_name)
        self.assertEqual(config_coord_name, plugin.config_coord_name)
        self.assertEqual(config_dict, plugin.config_dict)
        self.assertEqual(weights_coord_name, "weights")


class Test__repr__(IrisTest):
    """Test the __repr__ method."""

    def test_basic(self):
        """Test the repr function formats the arguments correctly"""
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                  "weights": [0, 1, 1, 0],
                                  "units": "hours"}}
        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name, config_dict)
        result = str(plugin)
        expected = ("<ChooseWeightsLinearFromDict "
                    "config_dict = {}, "
                    "weighting_coord_name = forecast_period, "
                    "config_coord_name = model_configuration, "
                    "weights_coord_name = weights>".format(str(config_dict)))
        self.assertEqual(result, expected)

    def test_alternative_name(self):
        """Test the repr function formats the arguments correctly when using
        an alternative name for the weights_coord_name."""
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                  "alternative_name": [0, 1, 1, 0],
                                  "units": "hours"}}
        weights_coord_name = "alternative_name"
        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name, config_dict,
            weights_coord_name=weights_coord_name)
        result = str(plugin)
        expected = ("<ChooseWeightsLinearFromDict "
                    "config_dict = {}, "
                    "weighting_coord_name = forecast_period, "
                    "config_coord_name = model_configuration, "
                    "weights_coord_name = alternative_name>".format(
                        str(config_dict)))
        self.assertEqual(result, expected)


class Test__check_config_dict(IrisTest):
    """Test the _check_config_dict method. """

    def test_dictionary_key_mismatch(self):
        """Test whether there is a mismatch in the dictionary keys. As
        _check_config_dict is called within the initialisation,
        _check_config_dict is not called directly."""
        config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                  "weights": [0, 1, 0],
                                  "units": "hours"}}
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"

        msg = ('These items in the configuration dictionary')
        with self.assertRaisesRegex(ValueError, msg):
            ChooseWeightsLinearFromDict(
                weighting_coord_name, config_coord_name, config_dict)

    def test_dictionary_key_match(self):
        """Test that an exception is not raised when the dictionary keys
        match in length as expected."""
        config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                  "weights": [0, 1, 1, 0],
                                  "units": "hours"}}
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name, config_dict)
        result = plugin._check_config_dict()
        self.assertIsNone(result)


class Test__arrange_interpolation_inputs(IrisTest):
    """Test the _arrange_interpolation_inputs method. """

    def test_basic(self):
        """Test that the values for the source_points, target_points,
        source_weights, axis and fill_value are as expected."""
        expected_source_points = [7, 12, 48, 54]
        expected_target_points = [6., 7., 8.]
        expected_source_weights = [0, 1, 1, 0]
        expected_fill_value = (0, 0)

        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                  "weights": [0, 1, 1, 0],
                                  "units": "hours"}}

        cube = set_up_basic_model_config_cube()

        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name, config_dict)
        source_points, target_points, source_weights, fill_value = (
            plugin._arrange_interpolation_inputs(cube))
        self.assertArrayAlmostEqual(source_points, expected_source_points)
        self.assertArrayAlmostEqual(target_points, expected_target_points)
        self.assertArrayAlmostEqual(source_weights, expected_source_weights)
        self.assertEqual(fill_value[0], expected_fill_value[0])
        self.assertEqual(fill_value[1], expected_fill_value[1])

    def test_unit_conversion(self):
        """Test that the values for the source_points, target_points,
        source_weights, axis and fill_value are as expected when a unit
        conversion has been required."""
        expected_source_points = [7, 12, 48, 54]
        expected_target_points = [6., 7., 8.]
        expected_source_weights = [0, 1, 1, 0]
        expected_fill_value = (0, 0)

        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        config_dict = {"uk_det": {"forecast_period": [420, 720, 2880, 3240],
                                  "weights": [0, 1, 1, 0],
                                  "units": "minutes"}}

        cube = set_up_basic_model_config_cube()

        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name, config_dict)
        source_points, target_points, source_weights, fill_value = (
            plugin._arrange_interpolation_inputs(cube))
        self.assertArrayAlmostEqual(source_points, expected_source_points)
        self.assertArrayAlmostEqual(target_points, expected_target_points)
        self.assertArrayAlmostEqual(source_weights, expected_source_weights)
        self.assertEqual(fill_value[0], expected_fill_value[0])
        self.assertEqual(fill_value[1], expected_fill_value[1])


class Test__create_new_weights_cube(IrisTest):
    """Test the _create_new_weights_cube method."""

    def setUp(self):
        """Set up configuration dictionary."""
        self.config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                       "weights": [0, 1, 1, 0],
                                       "units": "hours"}}

    def test_basic(self):
        """Test a new weights cube is created as intended, with the desired
        cube name."""
        cube = set_up_basic_model_config_cube()
        weights = np.array([0., 0., 0.2])

        expected_weights = np.array([[[[0., 0.],
                                       [0., 0.]],
                                      [[0., 0.],
                                       [0., 0.]],
                                      [[0.2, 0.2],
                                       [0.2, 0.2]]]])

        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name, self.config_dict)
        new_weights_cube = plugin._create_new_weights_cube(cube, weights)
        self.assertArrayAlmostEqual(expected_weights, new_weights_cube.data)
        self.assertEqual(new_weights_cube.name(), "weights")

    def test_alternative_name(self):
        """Test a new weights cube is created as intended, with the desired
        cube name when an alternative weights_coord_name is specified."""
        cube = set_up_basic_model_config_cube()
        weights = np.array([0., 0., 0.2])

        expected_weights = np.array([[[[0., 0.],
                                       [0., 0.]],
                                      [[0., 0.],
                                       [0., 0.]],
                                      [[0.2, 0.2],
                                       [0.2, 0.2]]]])
        config_dict = self.config_dict
        config_dict["uk_det"]["alternative_name"] = (
            config_dict["uk_det"].pop("weights"))
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name, self.config_dict,
            weights_coord_name="alternative_name")
        new_weights_cube = plugin._create_new_weights_cube(cube, weights)
        self.assertArrayAlmostEqual(expected_weights, new_weights_cube.data)
        self.assertEqual(new_weights_cube.name(), "alternative_name")


class Test__interpolate_to_create_weights(IrisTest):
    """Test the _interpolate_to_create_weights method."""

    def setUp(self):
        """Set up configuration dictionary."""
        config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                  "weights": [0, 1, 1, 0],
                                  "units": "hours"}}
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        self.plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name, config_dict)

    def test_below_range(self):
        """Test that interpolation works as intended when the forecast period
        required for the interpolation output is below the range specified
        within the inputs."""
        cube = set_up_basic_model_config_cube()

        expected_weights = np.array([[[[0., 0.],
                                       [0., 0.]],
                                      [[0., 0.],
                                       [0., 0.]],
                                      [[0.2, 0.2],
                                       [0.2, 0.2]]]])

        new_weights_cube = self.plugin._interpolate_to_create_weights(cube)
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data, expected_weights)
        self.assertEqual(new_weights_cube.name(), "weights")

    def test_within_range(self):
        """Test that interpolation works as intended when the forecast period
        required for the interpolation output is within the range specified
        within the inputs."""
        cube = add_model_id_and_model_configuration(
            set_up_temperature_cube(timesteps=3), model_ids=[1000],
            model_configurations=["uk_det"])
        cube = add_forecast_reference_time_and_forecast_period(
            cube, time_point=[402299.0, 402300.0, 402301.0],
            fp_point=[11., 12., 13.])

        expected_weights = np.array([[[[0.8, 0.8],
                                       [0.8, 0.8]],
                                      [[1., 1.],
                                       [1., 1.]],
                                      [[1., 1.],
                                       [1., 1.]]]])

        new_weights_cube = self.plugin._interpolate_to_create_weights(cube)
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data, expected_weights)
        self.assertEqual(new_weights_cube.name(), "weights")

    def test_above_range(self):
        """Test that interpolation works as intended when the forecast period
        required for the interpolation output is above the range specified
        within the inputs."""
        cube = add_model_id_and_model_configuration(
            set_up_temperature_cube(timesteps=3), model_ids=[1000],
            model_configurations=["uk_det"])
        cube = add_forecast_reference_time_and_forecast_period(
            cube, time_point=[402294.0, 402295.0, 402296.0],
            fp_point=[53., 54., 55.])

        expected_weights = np.array([[[[0.166667, 0.166667],
                                       [0.166667, 0.166667]],
                                      [[0., 0.],
                                       [0., 0.]],
                                      [[0., 0.],
                                       [0., 0.]]]])

        new_weights_cube = self.plugin._interpolate_to_create_weights(cube)
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data, expected_weights)
        self.assertEqual(new_weights_cube.name(), "weights")


class Test_process(IrisTest):
    """Test the process plugin. """

    def test_forecast_period_and_model_configuration(self):
        """Test when forecast_period is the weighting_coord_name and
        model_configuration is the config_coord_name. This demonstrates
        blending in one model and blending out another model with forecast
        period."""
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"

        cube = add_model_id_and_model_configuration(
            set_up_temperature_cube(timesteps=3), promote_to_new_axis=True)
        cube = add_forecast_reference_time_and_forecast_period(
            cube, time_point=[402294.0, 402295.0, 402296.0],
            fp_point=[6., 7., 8.])

        expected_weights = np.array([[[[[1., 1.],
                                        [1., 1.]],
                                       [[1., 1.],
                                        [1., 1.]],
                                       [[0.8, 0.8],
                                        [0.8, 0.8]]]],
                                     [[[[0., 0.],
                                        [0., 0.]],
                                       [[0., 0.],
                                        [0., 0.]],
                                       [[0.2, 0.2],
                                        [0.2, 0.2]]]]])

        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        config_dict = {"uk_det": {"forecast_period": [7, 12],
                                  "weights": [1, 0],
                                  "units": "hours"},
                       "uk_ens": {"forecast_period": [7, 12, 48, 54],
                                  "weights": [0, 1, 1, 0],
                                  "units": "hours"}}
        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name, config_dict)
        result = plugin.process(cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.name(), "weights")

    def test_forecast_period_and_model_configuration_three_models(self):
        """Test when forecast_period is the weighting_coord_name and
        model_configuration is the config_coord_name. This demonstrates
        blending in one model and blending out another model with forecast
        period."""
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"

        cube = add_model_id_and_model_configuration(
            set_up_temperature_cube(timesteps=3), model_ids=[1000, 2000, 3000],
            model_configurations=["uk_det", "uk_ens", "gl_ens"],
            promote_to_new_axis=True)
        cube = add_forecast_reference_time_and_forecast_period(
            cube, time_point=[402294.0, 402295.0, 402296.0],
            fp_point=[6., 7., 8.])

        expected_weights = np.array([[[[[1., 1.],
                                        [1., 1.]],
                                       [[1., 1.],
                                        [1., 1.]],
                                       [[0.66666667, 0.66666667],
                                        [0.66666667, 0.66666667]]]],
                                     [[[[0., 0.],
                                        [0., 0.]],
                                       [[0., 0.],
                                        [0., 0.]],
                                       [[0.16666667, 0.16666667],
                                        [0.16666667, 0.16666667]]]],
                                     [[[[0., 0.],
                                        [0., 0.]],
                                       [[0., 0.],
                                        [0., 0.]],
                                       [[0.16666667, 0.16666667],
                                        [0.16666667, 0.16666667]]]]])

        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        config_dict = {"uk_det": {"forecast_period": [7, 12],
                                  "weights": [1, 0],
                                  "units": "hours"},
                       "uk_ens": {"forecast_period": [7, 12, 48, 54],
                                  "weights": [0, 1, 1, 0],
                                  "units": "hours"},
                       "gl_ens": {"forecast_period": [7, 12, 48, 54],
                                  "weights": [0, 1, 1, 1],
                                  "units": "hours"}}
        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name, config_dict)
        result = plugin.process(cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.name(), "weights")

    def test_height_and_model_configuration(self):
        """Test when forecast_period is the weighting_coord_name and
        model_configuration is the config_coord_name. This demonstrates
        blending in one model and blending out another model with forecast
        period."""
        weighting_coord_name = "height"
        config_coord_name = "model_configuration"

        cube = add_model_id_and_model_configuration(
            set_up_temperature_cube(timesteps=1), promote_to_new_axis=True)
        heights = [10., 20.]
        cube = add_height(cube, heights)

        expected_weights = np.array([[[[[[1., 1.],
                                         [1., 1.]]]],
                                      [[[[0.5, 0.5],
                                         [0.5, 0.5]]]]],
                                     [[[[[0., 0.],
                                         [0., 0.]]]],
                                      [[[[0.5, 0.5],
                                         [0.5, 0.5]]]]]])

        config_dict = {"uk_det": {"height": [15, 25],
                                  "weights": [1, 0],
                                  "units": "m"},
                       "uk_ens": {"height": [15, 25],
                                  "weights": [0, 1],
                                  "units": "m"}}
        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name, config_dict)
        result = plugin.process(cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.name(), "weights")

    def test_height_and_realization(self):
        """Test when forecast_period is the weighting_coord_name and
        model_configuration is the config_coord_name. This demonstrates
        blending in one model and blending out another model with forecast
        period."""
        weighting_coord_name = "height"
        config_coord_name = "realization"

        data = np.zeros([2, 1, 2, 2]) + 273.15
        cube = set_up_temperature_cube(
            data=data, timesteps=1, realizations=[0, 1])
        heights = [10., 20.]
        cube = add_height(cube, heights)

        expected_weights = np.array([[[[[1., 1.],
                                        [1., 1.]]],
                                      [[[0.5, 0.5],
                                        [0.5, 0.5]]]],
                                     [[[[0., 0.],
                                        [0., 0.]]],
                                      [[[0.5, 0.5],
                                        [0.5, 0.5]]]]])

        weighting_coord_name = "height"
        config_coord_name = "realization"
        config_dict = {0: {"height": [15, 25],
                           "weights": [1, 0],
                           "units": "m"},
                       1: {"height": [15, 25],
                           "weights": [0, 1],
                           "units": "m"}}
        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name, config_dict)
        result = plugin.process(cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.name(), "weights")


if __name__ == '__main__':
    unittest.main()
