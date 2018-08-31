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
        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, self.config_coord_name)
        self.assertEqual(
            plugin.weighting_coord_name, self.weighting_coord_name)
        self.assertEqual(plugin.config_coord_name, self.config_coord_name)
        self.assertEqual(plugin.config_source, "cubes")
        self.assertIsNone(plugin.config_dict)
        self.assertIsNone(plugin.weights_coord_name)

    def test_with_config_dict(self):
        """Test initialisation from dict"""
        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, self.config_coord_name,
            config_dict=self.config_dict, dict_or_cubes="dict")
        self.assertEqual(
            plugin.weighting_coord_name, self.weighting_coord_name)
        self.assertEqual(plugin.config_coord_name, self.config_coord_name)
        self.assertEqual(plugin.config_source, "dict")
        self.assertEqual(plugin.config_dict, self.config_dict)
        self.assertEqual(plugin.weights_coord_name, "weights")

    def test_weights_coord_name(self):
        """Test initialisation from dict with alternative weights_coord_name"""
        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, self.config_coord_name,
            config_dict=self.config_dict, dict_or_cubes="dict",
            weights_coord_name="forecast_period")
        self.assertEqual(
            plugin.weighting_coord_name, self.weighting_coord_name)
        self.assertEqual(plugin.config_coord_name, self.config_coord_name)
        self.assertEqual(plugin.config_source, "dict")
        self.assertEqual(plugin.config_dict, self.config_dict)
        self.assertEqual(plugin.weights_coord_name, "forecast_period")

    def test_error_invalid_config_source(self):
        """Test invalid dict_or_cubes value raises error"""
        msg = ('Configuration source argument "dict_or_cubes" '
               'must be "dict" or "cubes", found kittens')
        with self.assertRaisesRegex(ValueError, msg):
            _ = ChooseWeightsLinear(
                self.weighting_coord_name, self.config_coord_name,
                dict_or_cubes="kittens")


class Test__repr__(IrisTest):
    """Test the __repr__ method"""
    pass  # TODO


class Test__check_config_dict(IrisTest):
    """Test the _check_config_dict method."""

    def setUp(self):
        """Set up some plugin inputs"""
        self.config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                       "weights": [0, 1, 1, 0],
                                       "units": "hours"}}
        self.weighting_coord_name = "forecast_period"
        self.config_coord_name = "model_configuration"

    def test_dictionary_key_mismatch(self):
        """Test whether there is a mismatch in the dictionary keys. As
        _check_config_dict is called within the initialisation,
        _check_config_dict is not called directly."""
        self.config_dict["uk_det"]["weights"] = [0, 1, 0]
        msg = ('These items in the configuration dictionary')
        with self.assertRaisesRegex(ValueError, msg):
            _ = ChooseWeightsLinear(
                self.weighting_coord_name, self.config_coord_name,
                dict_or_cubes="dict", config_dict=self.config_dict)

    def test_dictionary_key_match(self):
        """Test that an exception is not raised when the dictionary keys
        match in length as expected."""
        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, self.config_coord_name,
            dict_or_cubes="dict", config_dict=self.config_dict)
        result = plugin._check_config_dict()
        self.assertIsNone(result)

    def test_error_weighting_coord_not_in_dict(self):
        """Test that an exception is raised when the required weighting_coord
        is not in the configuration dictionary"""
        weighting_coord_name = "height"
        with self.assertRaises(KeyError):
            _ = ChooseWeightsLinear(
                weighting_coord_name, self.config_coord_name,
                dict_or_cubes="dict", config_dict=self.config_dict)


class Test__check_weights_cubes(IrisTest):
    """Test the _check_weights_cubes method."""

    def setUp(self):
        """Set up some plugin inputs"""
        self.cube = set_up_basic_model_config_cube()
        self.weighting_coord_name = "forecast_period"
        self.config_coord_name = "model_configuration"

    def test_exception_not_raised(self):
        """An exception is not raised when the length of the weights_cubes
        cubelist is the same as the number of points along the
        config_coord_name dimension within the input cube."""
        weights_cubes = iris.cube.CubeList([set_up_basic_weights_cube()])
        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, self.config_coord_name)
        self.assertIsNone(
            plugin._check_weights_cubes(self.cube, weights_cubes))

    def test_exception_raised(self):
        """An exception is raised when the length of the weights_cubes
        cubelist is different to the number of points along the
        config_coord_name dimension within the input cube."""
        weights_cubes = iris.cube.CubeList([])
        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, self.config_coord_name)
        msg = ('The coordinate used to configure the weights')
        with self.assertRaisesRegex(ValueError, msg):
            plugin._check_weights_cubes(self.cube, weights_cubes)


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

        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"

        cube = set_up_basic_model_config_cube()
        weights_cube = set_up_basic_weights_cube()

        plugin = ChooseWeightsLinear(
            weighting_coord_name, config_coord_name)
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
        self.config_coord_name = "model_configuration"

    def test_basic(self):
        """Test that the values for the source_points, target_points,
        source_weights, axis and fill_value are as expected."""
        config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                  "weights": [0, 1, 1, 0],
                                  "units": "hours"}}

        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, self.config_coord_name,
            dict_or_cubes="dict", config_dict=config_dict)

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
            self.weighting_coord_name, self.config_coord_name,
            dict_or_cubes="dict", config_dict=config_dict)

        source_points, target_points, source_weights, fill_value = (
            plugin._get_interpolation_inputs_from_dict(self.cube))

        self.assertArrayAlmostEqual(source_points, self.expected_source_points)
        self.assertArrayAlmostEqual(target_points, self.expected_target_points)
        self.assertArrayAlmostEqual(
            source_weights, self.expected_source_weights)
        self.assertEqual(fill_value[0], self.expected_fill_value[0])
        self.assertEqual(fill_value[1], self.expected_fill_value[1])


class Test__create_coord_and_dims_list(IrisTest):
    """Test the _create_coord_and_dims_list method."""

    def setUp(self):
        """Set up some plugin inputs"""
        self.cube = set_up_basic_model_config_cube()
        self.weights_cube = set_up_basic_weights_cube()
        self.weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        self.plugin = ChooseWeightsLinear(
            self.weighting_coord_name, config_coord_name)

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
        self.config_coord_name = "model_configuration"
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

        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, self.config_coord_name)
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
            self.weighting_coord_name, self.config_coord_name,
            dict_or_cubes="dict", config_dict=self.config_dict)
        new_weights_cube = plugin._create_new_weights_cube(
            self.cube, self.weights)
        self.assertArrayAlmostEqual(expected_weights, new_weights_cube.data)
        self.assertEqual(new_weights_cube.name(), "weights")

    def test_with_dict_alternative_name(self):
        """Test a new weights cube is created as intended, with the desired
        cube name when an alternative weights_coord_name is specified."""
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
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, self.config_coord_name,
            dict_or_cubes="dict", config_dict=self.config_dict,
            weights_coord_name="alternative_name")
        new_weights_cube = (
            plugin._create_new_weights_cube(self.cube, self.weights))
        self.assertArrayAlmostEqual(expected_weights, new_weights_cube.data)
        self.assertEqual(new_weights_cube.name(), "alternative_name")


class Test__interpolate_to_create_weights(IrisTest):
    """Test the _interpolate_to_create_weights method"""
    # TODO this class can be refactored / shortened further

    def setUp(self):
        """Set up some cubes and plugins to work with"""
        self.weights_cube = set_up_basic_weights_cube()

        config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                  "weights": [0, 1, 1, 0],
                                  "units": "hours"}}
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"

        self.plugin_cubes = ChooseWeightsLinear(
            weighting_coord_name, config_coord_name)

        self.plugin_dict = ChooseWeightsLinear(
            weighting_coord_name, config_coord_name,
            dict_or_cubes="dict", config_dict=config_dict)

    def test_below_range_cubes(self):
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

        new_weights_cube = (
            self.plugin_cubes._interpolate_to_create_weights(
                cube, self.weights_cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data, expected_weights)
        self.assertAlmostEqual(new_weights_cube.metadata,
                               self.weights_cube.metadata)

    def test_below_range_dict(self):
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

        new_weights_cube = (
            self.plugin_dict._interpolate_to_create_weights(cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data, expected_weights)
        self.assertEqual(new_weights_cube.name(), "weights")

    def test_within_range_cubes(self):
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

        new_weights_cube = (
            self.plugin_cubes._interpolate_to_create_weights(
                cube, self.weights_cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data, expected_weights)
        self.assertAlmostEqual(new_weights_cube.metadata,
                               self.weights_cube.metadata)

    def test_within_range_dict(self):
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

        new_weights_cube = (
            self.plugin_dict._interpolate_to_create_weights(cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data, expected_weights)
        self.assertEqual(new_weights_cube.name(), "weights")

    def test_above_range_cubes(self):
        """Test that interpolation works as intended when the forecast period
        required for the interpolation output is above the range specified
        within the inputs."""
        cube = add_model_id_and_model_configuration(
            set_up_temperature_cube(timesteps=3), model_ids=[1000],
            model_configurations=["uk_det"])
        cube = add_forecast_reference_time_and_forecast_period(
            cube, time_point=[412280.0, 412281.0, 412282.0],
            fp_point=[53., 54., 55.])

        expected_weights = np.array([[[[0.166667, 0.166667],
                                       [0.166667, 0.166667]],
                                      [[0., 0.],
                                       [0., 0.]],
                                      [[0., 0.],
                                       [0., 0.]]]])

        new_weights_cube = (
            self.plugin_cubes._interpolate_to_create_weights(
                cube, self.weights_cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data, expected_weights)
        self.assertAlmostEqual(new_weights_cube.metadata,
                               self.weights_cube.metadata)

    def test_above_range_dict(self):
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

        new_weights_cube = (
            self.plugin_dict._interpolate_to_create_weights(cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data, expected_weights)
        self.assertEqual(new_weights_cube.name(), "weights")

    def test_spatial_varying_weights(self):
        """Test that interpolation works as intended when the weights vary
        spatially within the input cube."""
        cube = add_model_id_and_model_configuration(
            set_up_temperature_cube(timesteps=3), model_ids=[1000],
            model_configurations=["uk_det"])
        cube = add_forecast_reference_time_and_forecast_period(
            cube, time_point=[412280.0, 412281.0, 412282.0],
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
            self.plugin_cubes._interpolate_to_create_weights(
                cube, weights_cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data, expected_weights)
        self.assertAlmostEqual(new_weights_cube.metadata,
                               weights_cube.metadata)


class Test_process(IrisTest):
    """Test the process method"""
    pass  # TODO


if __name__ == '__main__':
    unittest.main()
