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
"""Unit tests for the weights.WeightsLinear plugin."""


import unittest

import iris
from iris.coords import AuxCoord
from iris.tests import IrisTest
import numpy as np

from improver.blending.weights import ChooseWeightsLinearFromDict
from improver.tests.blending.weights.helper_functions import (
    set_up_temperature_cube, set_up_basic_model_config_cube,
    set_up_weights_cube, set_up_basic_weights_cube,
    add_model_id_and_model_configuration, add_height)
from improver.tests.ensemble_calibration.ensemble_calibration.helper_functions\
    import add_forecast_reference_time_and_forecast_period


class Test__init__(IrisTest):
    """Test the __init__ method."""

    def test_with_config_dict(self):
        """Test the class is initialised correctly, if the config_dict
           argument is supplied."""
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                  "weights": [0, 1, 1, 0],
                                  "units": "hours"}}
        weights_coord_name = "weights"
        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name, config_dict=config_dict)
        self.assertEqual(weighting_coord_name, plugin.weighting_coord_name)
        self.assertEqual(config_coord_name, plugin.config_coord_name)
        self.assertEqual(config_dict, plugin.config_dict)
        self.assertEqual(weights_coord_name, "weights")


class Test__repr__(IrisTest):
    """Test the __repr__ method."""

    def test_basic(self):
        pass


class Test__check_config_dict(IrisTest):
    """Test the linear weights function. """

    def test_dictionary_key_mismatch(self):
        """Test that the function returns an array of weights. """
        config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                  "weights": [0, 1, 0],
                                  "units": "hours"}}
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name, config_dict=config_dict)

        msg = ('These items in the configuration dictionary')
        with self.assertRaisesRegex(ValueError, msg):
            result = plugin._check_config_dict()

    def test_dictionary_key_match(self):
        """Test that the function returns an array of weights. """
        config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                  "weights": [0, 1, 1, 0],
                                  "units": "hours"}}
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name, config_dict=config_dict)
        result = plugin._check_config_dict()
        self.assertIsNone(result)


class Test__arrange_interpolation_inputs(IrisTest):
    """Test the linear weights function. """

    def test_basic(self):
        """Test that the function returns an array of weights. """
        expected_source_points = [7, 12, 48, 54]
        expected_target_points = [6., 7., 8.]
        expected_associated_data = [0, 1, 1, 0]
        expected_fill_value = (0, 0)

        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                  "weights": [0, 1, 1, 0],
                                  "units": "hours"}}

        cube = set_up_basic_model_config_cube()

        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name, config_dict=config_dict)
        source_points, target_points, associated_data, fill_value = (
            plugin._arrange_interpolation_inputs(cube))
        self.assertArrayAlmostEqual(source_points, expected_source_points)
        self.assertArrayAlmostEqual(target_points, expected_target_points)
        self.assertArrayAlmostEqual(associated_data, expected_associated_data)
        self.assertEqual(fill_value[0], expected_fill_value[0])
        self.assertEqual(fill_value[1], expected_fill_value[1])

    def test_unit_conversion(self):
        """Test that the function returns an array of weights. """
        expected_source_points = [7, 12, 48, 54]
        expected_target_points = [6., 7., 8.]
        expected_associated_data = [0, 1, 1, 0]
        expected_fill_value = (0, 0)

        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        config_dict = {"uk_det": {"forecast_period": [420, 720, 2880, 3240],
                                  "weights": [0, 1, 1, 0],
                                  "units": "minutes"}}

        cube = set_up_basic_model_config_cube()

        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name, config_dict=config_dict)
        source_points, target_points, associated_data, fill_value = (
            plugin._arrange_interpolation_inputs(cube))
        self.assertArrayAlmostEqual(source_points, expected_source_points)
        self.assertArrayAlmostEqual(target_points, expected_target_points)
        self.assertArrayAlmostEqual(associated_data, expected_associated_data)
        self.assertEqual(fill_value[0], expected_fill_value[0])
        self.assertEqual(fill_value[1], expected_fill_value[1])


class Test_interpolate_to_find_weights(IrisTest):
    """Test the linear weights function. """

    def test_1d_array(self):
        """Test that the function returns an array of weights. """
        expected_weights = (
            np.array([0., 0.5, 1., 1., 1., 0.5, 0.]))
        source_points = np.array([0, 2, 4, 6])
        target_points = np.arange(0, 7)
        associated_data = np.array([0, 1, 1, 0])
        axis = 0
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name)
        weights = plugin.interpolate_to_find_weights(
            source_points, target_points, associated_data, axis=axis)
        self.assertArrayAlmostEqual(weights, expected_weights)

    def test_2d_array_same_weights(self):
        """Test that the function returns an array of weights. """
        expected_weights = (
            np.array([[0., 0.5, 1., 1., 1., 0.5, 0.],
                      [0., 0.5, 1., 1., 1., 0.5, 0.]]))
        source_points = np.array([0, 2, 4, 6])
        target_points = np.arange(0, 7)
        associated_data = np.array([[0, 1, 1, 0], [0, 1, 1, 0]])
        axis = 1
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name)
        weights = plugin.interpolate_to_find_weights(
            source_points, target_points, associated_data, axis=axis)
        self.assertArrayAlmostEqual(weights, expected_weights)

    def test_2d_array_different_weights(self):
        """Test that the function returns an array of weights. """
        expected_weights = (
            np.array([[1., 1., 1., 0.5, 0., 0., 0.],
                      [0., 0., 0., 0.5, 1., 0.5, 0.]]))
        source_points = np.array([0, 2, 4, 6])
        target_points = np.arange(0, 7)
        associated_data = np.array([[1, 1, 0, 0], [0, 0, 1, 0]])
        axis = 1
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name)
        weights = plugin.interpolate_to_find_weights(
            source_points, target_points, associated_data, axis=axis)
        self.assertArrayAlmostEqual(weights, expected_weights)

    def test_3d_array(self):
        """Test that the function returns an array of weights. """
        expected_weights = (
            np.array([[[1., 1., 1., 0.5, 0., 0., 0.],
                       [0., 0., 0., 0.5, 1., 0.5, 0.],
                       [1., 0.5, 0., 0.5, 1., 0.5, 0.]]]))
        source_points = np.array([0, 2, 4, 6])
        target_points = np.arange(0, 7)
        associated_data = np.array([[[1, 1, 0, 0], [0, 0, 1, 0], [1, 0, 1, 0]]])
        axis = 2
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name)
        weights = plugin.interpolate_to_find_weights(
            source_points, target_points, associated_data, axis)
        self.assertArrayAlmostEqual(weights, expected_weights)


class Test__create_new_weights_cube(IrisTest):
    """Test the linear weights function. """

    def test_without_weights_cube(self):

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
            weighting_coord_name, config_coord_name)
        new_weights_cube = plugin._create_new_weights_cube(cube, weights)
        self.assertArrayAlmostEqual(expected_weights, new_weights_cube.data)
        self.assertEqual(new_weights_cube.name(), "weights")

    def test_without_weights_cube_alternative_name(self):

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
            weighting_coord_name, config_coord_name,
            weights_coord_name="alternative_name")
        new_weights_cube = plugin._create_new_weights_cube(cube, weights)
        self.assertArrayAlmostEqual(expected_weights, new_weights_cube.data)
        self.assertEqual(new_weights_cube.name(), "alternative_name")


class Test__interpolate_to_create_weights(IrisTest):
    """Test the linear weights function. """

    def test_below_range(self):
        """Test that the function returns an array of weights. """
        cube = set_up_basic_model_config_cube()
        weights_cube = set_up_basic_weights_cube()

        expected_weights = np.array([[[[0., 0.],
                                       [0., 0.]],
                                      [[0., 0.],
                                       [0., 0.]],
                                      [[0.2, 0.2],
                                       [0.2, 0.2]]]])

        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name)
        new_weights_cube = plugin._interpolate_to_create_weights(cube)
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data, expected_weights)
        self.assertAlmostEqual(new_weights_cube.metadata,
                               weights_cube.metadata)

    def test_within_range(self):
        cube = add_model_id_and_model_configuration(
            set_up_temperature_cube(timesteps=3), model_ids=[1000],
            model_configurations=["uk_det"])
        cube = add_forecast_reference_time_and_forecast_period(
            cube, time_point=[402299.0, 402300.0, 402301.0],
            fp_point=[11., 12., 13.])
        weights_cube = set_up_basic_weights_cube()

        expected_weights = np.array([[[[0.8, 0.8],
                                       [0.8, 0.8]],
                                      [[1., 1.],
                                       [1., 1.]],
                                      [[1., 1.],
                                       [1., 1.]]]])

        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name)
        new_weights_cube = plugin._interpolate_to_create_weights(cube)
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data, expected_weights)
        self.assertAlmostEqual(new_weights_cube.metadata,
                               weights_cube.metadata)

    def test_above_range(self):
        cube = add_model_id_and_model_configuration(
            set_up_temperature_cube(timesteps=3), model_ids=[1000],
            model_configurations=["uk_det"])
        cube = add_forecast_reference_time_and_forecast_period(
            cube, time_point=[402294.0, 402295.0, 402296.0],
            fp_point=[53., 54., 55.])
        weights_cube = set_up_basic_weights_cube()

        expected_weights = np.array([[[[0.166667, 0.166667],
                                       [0.166667, 0.166667]],
                                      [[0., 0.],
                                       [0., 0.]],
                                      [[0., 0.],
                                       [0., 0.]]]])

        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name)
        new_weights_cube = plugin._interpolate_to_create_weights(cube)
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data, expected_weights)
        self.assertAlmostEqual(new_weights_cube.metadata,
                               weights_cube.metadata)

    def test_spatial_varying_weights(self):
        cube = add_model_id_and_model_configuration(
            set_up_temperature_cube(timesteps=3), model_ids=[1000],
            model_configurations=["uk_det"])
        cube = add_forecast_reference_time_and_forecast_period(
            cube, time_point=[402294.0, 402295.0, 402296.0],
            fp_point=[9., 15., 21.])

        expected_weights = np.array([[[[1., 0.],
                                       [0.5, 0.5]],
                                      [[0.5, 0.5],
                                       [0.5, 1.]],
                                      [[0., 0.5],
                                       [0., 1.]]]])

        weights_cube = set_up_weights_cube(timesteps=4)
        weights_cube = add_forecast_reference_time_and_forecast_period(
            weights_cube, time_point=[402294.0, 402300.0, 402306.0, 402312.0],
            fp_point=[6., 12., 18., 24.])
        weights_cube.data = np.array([[[[1., 0.],
                                        [0., 0.]],
                                       [[1., 0.],
                                        [1., 1.]],
                                       [[0., 1.],
                                        [0., 1.]],
                                       [[0., 0.],
                                        [0., 1.]]]])

        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinearFromDict(
            weighting_coord_name, config_coord_name)
        new_weights_cube = plugin._interpolate_to_create_weights(cube)
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data, expected_weights)
        self.assertAlmostEqual(new_weights_cube.metadata,
                               weights_cube.metadata)


class Test_process(IrisTest):
    """Test the Default Linear Weights plugin. """

    def test_forecast_period_and_model_configuration_with_dict(self):
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
            weighting_coord_name, config_coord_name, config_dict=config_dict)
        result = plugin.process(cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.name(), "weights")


if __name__ == '__main__':
    unittest.main()
