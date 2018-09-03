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
"""Unit tests for the ChooseWeightsLinear plugin."""

import unittest

import iris
from iris.tests import IrisTest
import numpy as np

from improver.blending.weights import ChooseWeightsLinear
from improver.tests.blending.weights.helper_functions import (
    set_up_temperature_cube, set_up_basic_model_config_cube,
    set_up_weights_cube, set_up_basic_weights_cube,
    add_model_id_and_model_configuration, add_height)
from improver.tests.ensemble_calibration.ensemble_calibration.helper_functions\
    import add_forecast_reference_time_and_forecast_period


class Test__init__(IrisTest):
    """Test the __init__ method"""

    def setUp(self):
        """Set up some initialisation arguments"""
        self.weighting_coord_name = "forecast_period"
        self.config_coord_name = "model_configuration"
        self.config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                       "weights": [0, 1, 1, 0],
                                       "units": "hours"}}

    def test_basic(self):
        """Test default initialisation from cubes"""
        plugin = ChooseWeightsLinear(self.weighting_coord_name)
        self.assertEqual(
            plugin.weighting_coord_name, self.weighting_coord_name)
        self.assertEqual(plugin.config_coord_name, self.config_coord_name)
        self.assertFalse(plugin.use_dict)
        self.assertIsNone(plugin.config_dict)
        self.assertIsNone(plugin.weights_key_name)

    def test_config_coord_name(self):
        """Test different config coord name"""
        plugin = ChooseWeightsLinear(self.weighting_coord_name, "height")
        self.assertEqual(
            plugin.weighting_coord_name, self.weighting_coord_name)
        self.assertEqual(plugin.config_coord_name, "height")

    def test_with_config_dict(self):
        """Test initialisation from dict"""
        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, use_dict=True,
            config_dict=self.config_dict)
        self.assertEqual(
            plugin.weighting_coord_name, self.weighting_coord_name)
        self.assertEqual(plugin.config_coord_name, self.config_coord_name)
        self.assertTrue(plugin.use_dict)
        self.assertEqual(plugin.config_dict, self.config_dict)
        self.assertEqual(plugin.weights_key_name, "weights")

    def test_weights_key_name(self):
        """Test initialisation from dict with alternative weights_key_name"""
        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, use_dict=True,
            config_dict=self.config_dict, weights_key_name="forecast_period")
        self.assertEqual(plugin.weights_key_name, "forecast_period")


class Test__repr__(IrisTest):
    """Test the __repr__ method"""

    def test_cubes(self):
        """Test with default initialisation from cubes"""
        weighting_coord_name = "forecast_period"
        plugin = ChooseWeightsLinear(weighting_coord_name)
        expected_result = (
            "<ChooseWeightsLinear(): weighting_coord_name = forecast_period, "
            "config_coord_name = model_configuration, use_dict = False, "
            "config_dict = None, weights_key_name = None>")
        self.assertEqual(str(plugin), expected_result)

    def test_dict(self):
        """Test with configuration dictionary"""
        weighting_coord_name = "forecast_period"
        config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                  "weights": [0, 1, 1, 0],
                                  "units": "hours"}}
        plugin = ChooseWeightsLinear(
            weighting_coord_name, use_dict=True, config_dict=config_dict)
        expected_result = (
            "<ChooseWeightsLinear(): weighting_coord_name = forecast_period, "
            "config_coord_name = model_configuration, use_dict = True, "
            "config_dict = {'uk_det': {'forecast_period': [7, 12, 48, 54], "
            "'weights': [0, 1, 1, 0], 'units': 'hours'}}, weights_key_name "
            "= weights>")
        self.assertEqual(str(plugin), expected_result)


class Test__check_config_dict(IrisTest):
    """Test the _check_config_dict method."""

    def setUp(self):
        """Set up some plugin inputs"""
        self.config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                       "weights": [0, 1, 1, 0],
                                       "units": "hours"}}
        self.weighting_coord_name = "forecast_period"

    def test_dictionary_key_mismatch(self):
        """Test whether there is a mismatch in the dictionary keys. As
        _check_config_dict is called within the initialisation,
        _check_config_dict is not called directly."""
        self.config_dict["uk_det"]["weights"] = [0, 1, 0]
        msg = ('These items in the configuration dictionary')
        with self.assertRaisesRegex(ValueError, msg):
            _ = ChooseWeightsLinear(
                self.weighting_coord_name, use_dict=True,
                config_dict=self.config_dict)

    def test_dictionary_key_match(self):
        """Test that an exception is not raised when the dictionary keys
        match in length as expected."""
        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, use_dict=True,
            config_dict=self.config_dict)
        result = plugin._check_config_dict()
        self.assertIsNone(result)

    def test_error_weighting_coord_not_in_dict(self):
        """Test that an exception is raised when the required weighting_coord
        is not in the configuration dictionary"""
        with self.assertRaises(KeyError):
            _ = ChooseWeightsLinear(
                "height", use_dict=True, config_dict=self.config_dict)


class Test__get_interpolation_inputs_from_cube(IrisTest):
    """Test the _get_interpolation_inputs_from_cube method."""

    def test_basic(self):
        """Test that the values for the source_points, target_points,
        source_weights, axis and fill_value are as expected."""
        expected_source_points = [7, 12, 48, 54]
        expected_target_points = [6., 7., 8.]
        expected_source_weights = np.array([[[[0., 0.],
                                              [0., 0.]],
                                             [[1., 1.],
                                              [1., 1.]],
                                             [[1., 1.],
                                              [1., 1.]],
                                             [[0., 0.],
                                              [0., 0.]]]])
        expected_axis = 1
        expected_fill_value = (np.array([[[0., 0.],
                                          [0., 0.]]]),
                               np.array([[[0., 0.],
                                          [0., 0.]]]))

        cube = set_up_basic_model_config_cube()
        weights_cube = set_up_basic_weights_cube()

        plugin = ChooseWeightsLinear("forecast_period")
        source_points, target_points, source_weights, axis, fill_value = (
            plugin._get_interpolation_inputs_from_cube(cube, weights_cube))
        self.assertArrayAlmostEqual(source_points, expected_source_points)
        self.assertArrayAlmostEqual(target_points, expected_target_points)
        self.assertArrayAlmostEqual(source_weights, expected_source_weights)
        self.assertEqual(axis, expected_axis)
        self.assertArrayAlmostEqual(fill_value[0], expected_fill_value[0])
        self.assertArrayAlmostEqual(fill_value[1], expected_fill_value[1])


class Test__get_interpolation_inputs_from_dict(IrisTest):
    """Test the _get_interpolation_inputs_from_dict method."""

    def setUp(self):
        """Set up some plugin inputs"""
        self.expected_source_points = [7, 12, 48, 54]
        self.expected_target_points = [6., 7., 8.]
        self.expected_source_weights = [0, 1, 1, 0]
        self.expected_fill_value = (0, 0)

        self.cube = set_up_basic_model_config_cube()
        self.weighting_coord_name = "forecast_period"

    def test_basic(self):
        """Test that the values for the source_points, target_points,
        source_weights, axis and fill_value are as expected."""
        config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                  "weights": [0, 1, 1, 0],
                                  "units": "hours"}}

        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, use_dict=True, config_dict=config_dict)

        source_points, target_points, source_weights, fill_value = (
            plugin._get_interpolation_inputs_from_dict(self.cube))

        self.assertArrayAlmostEqual(source_points, self.expected_source_points)
        self.assertArrayAlmostEqual(target_points, self.expected_target_points)
        self.assertArrayAlmostEqual(
            source_weights, self.expected_source_weights)
        self.assertEqual(fill_value[0], self.expected_fill_value[0])
        self.assertEqual(fill_value[1], self.expected_fill_value[1])

    def test_unit_conversion(self):
        """Test that the values for the source_points, target_points,
        source_weights, axis and fill_value are as expected when a unit
        conversion has been required."""
        config_dict = {"uk_det": {"forecast_period": [420, 720, 2880, 3240],
                                  "weights": [0, 1, 1, 0],
                                  "units": "minutes"}}

        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, use_dict=True, config_dict=config_dict)

        source_points, target_points, source_weights, fill_value = (
            plugin._get_interpolation_inputs_from_dict(self.cube))

        self.assertArrayAlmostEqual(source_points, self.expected_source_points)
        self.assertArrayAlmostEqual(target_points, self.expected_target_points)
        self.assertArrayAlmostEqual(
            source_weights, self.expected_source_weights)
        self.assertEqual(fill_value[0], self.expected_fill_value[0])
        self.assertEqual(fill_value[1], self.expected_fill_value[1])


class Test__interpolate_to_find_weights(IrisTest):
    """Test the _interpolate_to_find_weights method."""

    def setUp(self):
        """Set up plugin instance"""
        self.plugin = ChooseWeightsLinear("forecast_period")

    def test_1d_array(self):
        """Test that the interpolation produces the expected result for a
        1d input array."""
        expected_weights = (
            np.array([0., 0.5, 1., 1., 1., 0.5, 0.]))
        source_points = np.array([0, 2, 4, 6])
        target_points = np.arange(0, 7)
        source_weights = np.array([0, 1, 1, 0])
        weights = self.plugin._interpolate_to_find_weights(
            source_points, target_points, source_weights, axis=0)
        self.assertArrayAlmostEqual(weights, expected_weights)

    def test_2d_array_same_weights(self):
        """Test that the interpolation produces the expected result for a
        2d input array, where the two each of the input dimensions have the
        same weights within the input numpy array."""
        expected_weights = (
            np.array([[0., 0.5, 1., 1., 1., 0.5, 0.],
                      [0., 0.5, 1., 1., 1., 0.5, 0.]]))
        source_points = np.array([0, 2, 4, 6])
        target_points = np.arange(0, 7)
        source_weights = np.array([[0, 1, 1, 0], [0, 1, 1, 0]])
        weights = self.plugin._interpolate_to_find_weights(
            source_points, target_points, source_weights, axis=1)
        self.assertArrayAlmostEqual(weights, expected_weights)

    def test_2d_array_different_weights(self):
        """Test that the interpolation produces the expected result for a
        2d input array, where the two each of the input dimensions have
        different weights within the input numpy array."""
        expected_weights = (
            np.array([[1., 1., 1., 0.5, 0., 0., 0.],
                      [0., 0., 0., 0.5, 1., 0.5, 0.]]))
        source_points = np.array([0, 2, 4, 6])
        target_points = np.arange(0, 7)
        source_weights = np.array([[1, 1, 0, 0], [0, 0, 1, 0]])
        weights = self.plugin._interpolate_to_find_weights(
            source_points, target_points, source_weights, axis=1)
        self.assertArrayAlmostEqual(weights, expected_weights)

    def test_3d_array(self):
        """Test that the interpolation produces the expected result for a
        3d input array."""
        expected_weights = (
            np.array([[[1., 1., 1., 0.5, 0., 0., 0.],
                       [0., 0., 0., 0.5, 1., 0.5, 0.],
                       [1., 0.5, 0., 0.5, 1., 0.5, 0.]]]))
        source_points = np.array([0, 2, 4, 6])
        target_points = np.arange(0, 7)
        source_weights = (
            np.array([[[1, 1, 0, 0], [0, 0, 1, 0], [1, 0, 1, 0]]]))
        weights = self.plugin._interpolate_to_find_weights(
            source_points, target_points, source_weights, axis=2)
        self.assertArrayAlmostEqual(weights, expected_weights)


class Test__create_coord_and_dims_list(IrisTest):
    """Test the _create_coord_and_dims_list method."""

    def setUp(self):
        """Set up some plugin inputs"""
        self.cube = set_up_basic_model_config_cube()
        self.weights_cube = set_up_basic_weights_cube()
        self.weighting_coord_name = "forecast_period"
        self.plugin = ChooseWeightsLinear(self.weighting_coord_name)

    def test_dim_coords(self):
        """Test that the expected list of coordinates is returned when the
        dimension coordinates are checked."""
        expected_coord_list = [(self.weights_cube.coord("realization"), 0),
                               (self.cube.coord("time"), 1),
                               (self.weights_cube.coord("latitude"), 2),
                               (self.weights_cube.coord("longitude"), 3)]

        new_coord_list = self.plugin._create_coord_and_dims_list(
            self.weights_cube, self.cube, self.weights_cube.dim_coords,
            self.weighting_coord_name)

        self.assertEqual(new_coord_list, expected_coord_list)

    def test_aux_coords(self):
        """Test that the expected list of coordinates is returned when the
        dimension coordinates are checked."""
        expected_coord_list = [
            (self.weights_cube.coord("forecast_reference_time"), None),
            (self.weights_cube.coord("model_configuration"), None),
            (self.weights_cube.coord("model_id"), None),
            (self.cube.coord("forecast_period"), 1)]

        new_coord_list = self.plugin._create_coord_and_dims_list(
            self.weights_cube, self.cube, self.weights_cube.aux_coords,
            self.weighting_coord_name)

        self.assertEqual(new_coord_list, expected_coord_list)


class Test__create_new_weights_cube(IrisTest):
    """Test the _create_new_weights_cube function. """

    def setUp(self):
        """Set up some plugin inputs"""
        self.cube = set_up_basic_model_config_cube()
        self.weighting_coord_name = "forecast_period"
        self.config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                       "weights": [0, 1, 1, 0],
                                       "units": "hours"}}
        self.weights = np.array([0., 0., 0.2])

    def test_with_weights_cube(self):
        """Test that the the expected cube containg the new weights is
        returned."""
        weights_cube = set_up_basic_weights_cube()
        weights = np.array([[[[0., 0.],
                              [0., 0.]],
                             [[0., 0.],
                              [0., 0.]],
                             [[0.2, 0.2],
                              [0.2, 0.2]]]])

        expected_weights = np.array([[[[0., 0.],
                                       [0., 0.]],
                                      [[0., 0.],
                                       [0., 0.]],
                                      [[0.2, 0.2],
                                       [0.2, 0.2]]]])

        plugin = ChooseWeightsLinear(self.weighting_coord_name)
        new_weights_cube = plugin._create_new_weights_cube(
            self.cube, weights, weights_cube)
        self.assertArrayAlmostEqual(expected_weights, new_weights_cube.data)
        self.assertEqual(weights_cube.metadata, new_weights_cube.metadata)

    def test_with_dict(self):
        """Test a new weights cube is created as intended, with the desired
        cube name."""
        expected_weights = np.array([[[[0., 0.],
                                       [0., 0.]],
                                      [[0., 0.],
                                       [0., 0.]],
                                      [[0.2, 0.2],
                                       [0.2, 0.2]]]])

        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, use_dict=True,
            config_dict=self.config_dict)
        new_weights_cube = plugin._create_new_weights_cube(
            self.cube, self.weights)
        self.assertArrayAlmostEqual(expected_weights, new_weights_cube.data)
        self.assertEqual(new_weights_cube.name(), "weights")

    def test_with_dict_alternative_name(self):
        """Test a new weights cube is created as intended, with the desired
        cube name when an alternative weights_key_name is specified."""
        cube = set_up_basic_model_config_cube()
        expected_weights = np.array([[[[0., 0.],
                                       [0., 0.]],
                                      [[0., 0.],
                                       [0., 0.]],
                                      [[0.2, 0.2],
                                       [0.2, 0.2]]]])

        self.config_dict["uk_det"]["alternative_name"] = (
            self.config_dict["uk_det"].pop("weights"))
        weighting_coord_name = "forecast_period"
        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, use_dict=True,
            config_dict=self.config_dict, weights_key_name="alternative_name")
        new_weights_cube = (
            plugin._create_new_weights_cube(self.cube, self.weights))
        self.assertArrayAlmostEqual(expected_weights, new_weights_cube.data)
        self.assertEqual(new_weights_cube.name(), "alternative_name")


class Test__calculate_weights(IrisTest):
    """Test the _calculate_weights method"""

    def setUp(self):
        """Set up some cubes and plugins to work with"""
        self.temp_cube = add_model_id_and_model_configuration(
            set_up_temperature_cube(timesteps=3), model_ids=[1000],
            model_configurations=["uk_det"])

        self.weights_cube = set_up_basic_weights_cube()

        config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                  "weights": [0, 1, 1, 0],
                                  "units": "hours"}}
        weighting_coord_name = "forecast_period"

        self.plugin_cubes = ChooseWeightsLinear(weighting_coord_name)
        self.plugin_dict = ChooseWeightsLinear(
            weighting_coord_name, use_dict=True, config_dict=config_dict)

        self.expected_weights_below_range = np.array(
            [[[[0., 0.], [0., 0.]],
              [[0., 0.], [0., 0.]],
              [[0.2, 0.2], [0.2, 0.2]]]])

        self.expected_weights_within_range = np.array(
            [[[[0.8, 0.8], [0.8, 0.8]],
              [[1., 1.], [1., 1.]],
              [[1., 1.], [1., 1.]]]])

        self.expected_weights_above_range = np.array(
            [[[[0.166667, 0.166667], [0.166667, 0.166667]],
              [[0., 0.], [0., 0.]],
              [[0., 0.], [0., 0.]]]])

    def test_below_range_cubes(self):
        """Test that interpolation works as intended when the forecast period
        required for the interpolation output is below the range specified
        within the inputs."""
        cube = set_up_basic_model_config_cube()
        new_weights_cube = (
            self.plugin_cubes._calculate_weights(
                cube, self.weights_cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data,
                                    self.expected_weights_below_range)
        self.assertAlmostEqual(new_weights_cube.metadata,
                               self.weights_cube.metadata)

    def test_below_range_dict(self):
        """Test that interpolation works as intended when the forecast period
        required for the interpolation output is below the range specified
        within the inputs."""
        cube = set_up_basic_model_config_cube()
        new_weights_cube = (
            self.plugin_dict._calculate_weights(cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data,
                                    self.expected_weights_below_range)
        self.assertEqual(new_weights_cube.name(), "weights")

    def test_within_range_cubes(self):
        """Test that interpolation works as intended when the forecast period
        required for the interpolation output is within the range specified
        within the inputs."""
        cube = add_forecast_reference_time_and_forecast_period(
            self.temp_cube, time_point=[402299.0, 402300.0, 402301.0],
            fp_point=[11., 12., 13.])
        new_weights_cube = (
            self.plugin_cubes._calculate_weights(
                cube, self.weights_cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data,
                                    self.expected_weights_within_range)
        self.assertAlmostEqual(new_weights_cube.metadata,
                               self.weights_cube.metadata)

    def test_within_range_dict(self):
        """Test that interpolation works as intended when the forecast period
        required for the interpolation output is within the range specified
        within the inputs."""
        cube = add_forecast_reference_time_and_forecast_period(
            self.temp_cube, time_point=[402299.0, 402300.0, 402301.0],
            fp_point=[11., 12., 13.])
        new_weights_cube = (
            self.plugin_dict._calculate_weights(cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data,
                                    self.expected_weights_within_range)
        self.assertEqual(new_weights_cube.name(), "weights")

    def test_above_range_cubes(self):
        """Test that interpolation works as intended when the forecast period
        required for the interpolation output is above the range specified
        within the inputs."""
        cube = add_forecast_reference_time_and_forecast_period(
            self.temp_cube, time_point=[412280.0, 412281.0, 412282.0],
            fp_point=[53., 54., 55.])
        new_weights_cube = (
            self.plugin_cubes._calculate_weights(
                cube, self.weights_cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data,
                                    self.expected_weights_above_range)
        self.assertAlmostEqual(new_weights_cube.metadata,
                               self.weights_cube.metadata)

    def test_above_range_dict(self):
        """Test that interpolation works as intended when the forecast period
        required for the interpolation output is above the range specified
        within the inputs."""
        cube = add_forecast_reference_time_and_forecast_period(
            self.temp_cube, time_point=[402294.0, 402295.0, 402296.0],
            fp_point=[53., 54., 55.])
        new_weights_cube = (
            self.plugin_dict._calculate_weights(cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data,
                                    self.expected_weights_above_range)
        self.assertEqual(new_weights_cube.name(), "weights")

    def test_spatial_varying_weights(self):
        """Test that interpolation works as intended when the weights vary
        spatially within the input cube."""
        cube = add_forecast_reference_time_and_forecast_period(
            self.temp_cube, time_point=[412280.0, 412281.0, 412282.0],
            fp_point=[9., 15., 21.])

        expected_weights = np.array([[[[1., 0.],
                                       [0.5, 0.5]],
                                      [[0.5, 0.5],
                                       [0.5, 1.]],
                                      [[0., 0.5],
                                       [0., 1.]]]])

        weights_cube = set_up_weights_cube(timesteps=4)
        weights_cube = add_forecast_reference_time_and_forecast_period(
            weights_cube, time_point=[412233.0, 412239.0, 412245.0, 412251.0],
            fp_point=[6., 12., 18., 24.])
        weights_cube.data = np.array([[[[1., 0.],
                                        [0., 0.]],
                                       [[1., 0.],
                                        [1., 1.]],
                                       [[0., 1.],
                                        [0., 1.]],
                                       [[0., 0.],
                                        [0., 1.]]]])

        new_weights_cube = (
            self.plugin_cubes._calculate_weights(
                cube, weights_cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data, expected_weights)
        self.assertAlmostEqual(new_weights_cube.metadata,
                               weights_cube.metadata)


class Test_process(IrisTest):
    """Test the process method"""

    def setUp(self):
        """Set up some cubes and plugin inputs"""
        self.weighting_coord_name = "forecast_period"
        self.plugin_cube = ChooseWeightsLinear(self.weighting_coord_name)
        self.config_dict_fp = {"uk_det": {"forecast_period": [7, 12],
                                          "weights": [1, 0],
                                          "units": "hours"},
                               "uk_ens": {"forecast_period": [7, 12, 48, 54],
                                          "weights": [0, 1, 1, 0],
                                          "units": "hours"}}

    def test_forecast_period_and_model_configuration_cube(self):
        """Test when forecast_period is the weighting_coord_name. This
        demonstrates blending models whose relative weights differ with
        forecast period."""
        cube = add_model_id_and_model_configuration(
            set_up_temperature_cube(timesteps=3), promote_to_new_axis=True)
        cube = add_forecast_reference_time_and_forecast_period(
            cube, time_point=[412235.0, 412247.0, 412278.0],
            fp_point=[8., 20., 51.])

        weights_cube_uk_det = set_up_weights_cube(timesteps=2)
        weights_cube_uk_det = add_forecast_reference_time_and_forecast_period(
            weights_cube_uk_det, time_point=[412234.0, 412239.0],
            fp_point=[7., 12.])
        weights_cube_uk_det.data[:, 0] = np.ones([1, 2, 2])
        weights_cube_uk_det = add_model_id_and_model_configuration(
            weights_cube_uk_det, model_ids=[1000],
            model_configurations=["uk_det"])

        weights_cube_uk_ens = set_up_weights_cube(timesteps=4)
        weights_cube_uk_ens = add_forecast_reference_time_and_forecast_period(
            weights_cube_uk_ens,
            time_point=[412234.0, 412239.0, 412275.0, 412281.0],
            fp_point=[7., 12., 48., 54.])
        weights_cube_uk_ens.data[:, 1:3] = np.ones([1, 2, 2, 2])
        weights_cube_uk_ens = add_model_id_and_model_configuration(
            weights_cube_uk_ens, model_ids=[2000],
            model_configurations=["uk_ens"])

        weights_cubes = (
            iris.cube.CubeList([weights_cube_uk_det, weights_cube_uk_ens]))

        expected_weights = np.array([[[[[0.8, 0.8],
                                        [0.8, 0.8]],
                                       [[0., 0.],
                                        [0., 0.]],
                                       [[0., 0.],
                                        [0., 0.]]]],
                                     [[[[0.2, 0.2],
                                        [0.2, 0.2]],
                                       [[1., 1.],
                                        [1., 1.]],
                                       [[1., 1.],
                                        [1., 1.]]]]])

        result = self.plugin_cube.process(cube, weights_cubes)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.metadata,
                               weights_cubes[0].metadata)

    def test_forecast_period_and_model_configuration_dict(self):
        """Test blending models over forecast_period with a configuration
        dictionary."""
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

        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, use_dict=True,
            config_dict=self.config_dict_fp)
        result = plugin.process(cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.name(), "weights")

    def test_forecast_period_and_model_configuration_three_models_cube(self):
        """Test blending three models with relative weights varying along the
        forecast_period coordinate."""
        cube = add_model_id_and_model_configuration(
            set_up_temperature_cube(timesteps=3), model_ids=[1000, 2000, 3000],
            model_configurations=["uk_det", "uk_ens", "gl_ens"],
            promote_to_new_axis=True)
        cube = add_forecast_reference_time_and_forecast_period(
            cube, time_point=[412235.0, 412247.0, 412278.0],
            fp_point=[8., 20., 51.])

        weights_cube_uk_det = set_up_weights_cube(timesteps=2)
        weights_cube_uk_det = add_forecast_reference_time_and_forecast_period(
            weights_cube_uk_det, time_point=[412234.0, 412239.0],
            fp_point=[7., 12.])
        weights_cube_uk_det.data[:, 0] = np.ones([1, 2, 2])
        weights_cube_uk_det = add_model_id_and_model_configuration(
            weights_cube_uk_det, model_ids=[1000],
            model_configurations=["uk_det"])

        weights_cube_uk_ens = set_up_weights_cube(timesteps=4)
        weights_cube_uk_ens = add_forecast_reference_time_and_forecast_period(
            weights_cube_uk_ens,
            time_point=[412234.0, 412239.0, 412275.0, 412281.0],
            fp_point=[7., 12., 48., 54.])
        weights_cube_uk_ens.data[:, 1:3] = np.ones([1, 2, 2, 2])
        weights_cube_uk_ens = add_model_id_and_model_configuration(
            weights_cube_uk_ens, model_ids=[2000],
            model_configurations=["uk_ens"])

        weights_cube_gl_ens = set_up_weights_cube(timesteps=2)
        weights_cube_gl_ens = add_forecast_reference_time_and_forecast_period(
            weights_cube_gl_ens,
            time_point=[412275.0, 412281.0],
            fp_point=[48., 54.])
        weights_cube_gl_ens.data[:, 1] = np.ones([1, 2, 2])
        weights_cube_gl_ens = add_model_id_and_model_configuration(
            weights_cube_gl_ens, model_ids=[3000],
            model_configurations=["gl_ens"])

        weights_cubes = (
            iris.cube.CubeList([weights_cube_uk_det, weights_cube_uk_ens,
                                weights_cube_gl_ens]))

        expected_weights = np.array([[[[[0.8, 0.8],
                                        [0.8, 0.8]],
                                       [[0., 0.],
                                        [0., 0.]],
                                       [[0., 0.],
                                        [0., 0.]]]],
                                     [[[[0.2, 0.2],
                                        [0.2, 0.2]],
                                       [[1., 1.],
                                        [1., 1.]],
                                       [[0.5, 0.5],
                                        [0.5, 0.5]]]],
                                     [[[[0., 0.],
                                        [0., 0.]],
                                       [[0., 0.],
                                        [0., 0.]],
                                       [[0.5, 0.5],
                                        [0.5, 0.5]]]]])

        result = self.plugin_cube.process(cube, weights_cubes)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.metadata,
                               weights_cubes[0].metadata)

    def test_forecast_period_and_model_configuration_three_models_dict(self):
        """Test blending three models over forecast period with a
        configuration dictionary."""
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

        self.config_dict_fp["gl_ens"] = {"forecast_period": [7, 12, 48, 54],
                                         "weights": [0, 1, 1, 1],
                                         "units": "hours"}

        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, use_dict=True,
            config_dict=self.config_dict_fp)
        result = plugin.process(cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.name(), "weights")

    def test_height_and_realization_cube(self):
        """Test when height is the weighting_coord_name and realization is the
        config_coord_name. This demonstrates blending in one member and
        blending out another member with height."""
        data = np.zeros([2, 1, 2, 2]) + 273.15
        cube = set_up_temperature_cube(
            data=data, timesteps=1, realizations=[0, 1])
        heights = [10., 20.]
        cube = add_height(cube, heights)

        data = np.zeros([1, 1, 2, 2])
        weights_cube_uk_det = (
            set_up_weights_cube(data=data, timesteps=1, realizations=[0]))
        heights = [15., 25.]
        weights_cube_uk_det = add_height(weights_cube_uk_det, heights)
        weights_cube_uk_det.data[0] = np.ones([1, 2, 2])
        for cube_slice in weights_cube_uk_det.slices_over("realization"):
            weights_cube_uk_det = cube_slice
            break

        weights_cube_uk_ens = (
            set_up_weights_cube(data=data, timesteps=1, realizations=[1]))
        heights = [15., 25.]
        weights_cube_uk_ens = add_height(weights_cube_uk_ens, heights)
        weights_cube_uk_ens.data[1] = np.ones([1, 2, 2])
        for cube_slice in weights_cube_uk_ens.slices_over("realization"):
            weights_cube_uk_ens = cube_slice
            break

        weights_cubes = (
            iris.cube.CubeList([weights_cube_uk_det, weights_cube_uk_ens]))

        expected_weights = np.array([[[[[1., 1.],
                                        [1., 1.]]],
                                      [[[0.5, 0.5],
                                        [0.5, 0.5]]]],
                                     [[[[0., 0.],
                                        [0., 0.]]],
                                      [[[0.5, 0.5],
                                        [0.5, 0.5]]]]])

        plugin = ChooseWeightsLinear("height", config_coord_name="realization")
        result = plugin.process(cube, weights_cubes)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.metadata,
                               weights_cubes[0].metadata)

    def test_height_and_realization_dict(self):
        """Test blending members with a configuration dictionary."""
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

        config_dict = {0: {"height": [15, 25],
                           "weights": [1, 0],
                           "units": "m"},
                       1: {"height": [15, 25],
                           "weights": [0, 1],
                           "units": "m"}}
        plugin = ChooseWeightsLinear(
            "height", config_coord_name="realization", use_dict=True,
            config_dict=config_dict)
        result = plugin.process(cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.name(), "weights")

    def test_exception_if_weights_incorrect(self):
        """Test that an exception is raised when the length of the
        weights_cubes cubelist differs from the number of points along the
        config_coord_name dimension within the input cube."""
        cube = set_up_basic_model_config_cube()
        weights_cubes = iris.cube.CubeList([])
        plugin = ChooseWeightsLinear(self.weighting_coord_name)
        msg = ('The coordinate used to configure the weights')
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(cube, weights_cubes)


if __name__ == '__main__':
    unittest.main()
