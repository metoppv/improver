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

from improver.blending.weights import ChooseWeightsLinear
from improver.tests.blending.weights.helper_functions import (
    set_up_precipitation_cube)
from improver.tests.ensemble_calibration.ensemble_calibration.helper_functions\
    import (add_forecast_reference_time_and_forecast_period, set_up_cube)


def set_up_temperature_cube(data=None, timesteps=3, realizations=None):
    """Create a cube with metadata and values suitable for air temperature."""
    if realizations is None:
        realizations = [0]
    if data is None:
        data = np.zeros([1, timesteps, 2, 2]) + 273.15
    temp_range = np.arange(-2, 30, 2)
    for timestep in np.arange(timesteps):
        data[0, timestep] -= temp_range[timestep]
    cube = set_up_cube(data, standard_name="air_temperature", units="K",
                       realizations=realizations, timesteps=timesteps,
                       y_dimension_length=2, x_dimension_length=2)
    return cube


#def set_up_model_config_cube(
        #cube, model_ids=[1000, 2000],
        #model_configurations=["uk_det", "uk_ens"], promote_to_new_axis=False):
    #cubelist = iris.cube.CubeList([])
    #for model_id, model_configuration in zip(model_ids, model_configurations):
        #cube_copy = cube.copy()
        #model_id_coord = iris.coords.AuxCoord(
            #model_id, long_name='model_id')
        #cube_copy.add_aux_coord(model_id_coord)
        #model_config_coord = iris.coords.AuxCoord(
            #model_configuration, long_name='model_configuration')
        #if promote_to_new_axis:
            #cube_copy = iris.util.new_axis(cube_copy, "model_id")
            #index = cube_copy.coord_dims("model_id")[0]
            #cube_copy.add_aux_coord(model_config_coord, data_dims=index)
        #else:
            #cube_copy.add_aux_coord(model_config_coord)
        #cubelist.append(cube_copy)
    #return cubelist.merge_cube()


def set_up_basic_model_config_cube():
    cube = add_model_id_and_model_configuration(
        set_up_temperature_cube(timesteps=3), model_ids=[1000],
        model_configurations=["uk_det"])
    cube = add_forecast_reference_time_and_forecast_period(
        cube, time_point=[402294.0, 402295.0, 402296.0],
        fp_point=[6., 7., 8.])
    return cube


def set_up_weights_cube(data=None, timesteps=3, realizations=None):
    if realizations is None:
        realizations = [0]
    if data is None:
        data = np.zeros([1, timesteps, 2, 2])
    cube = set_up_cube(data, long_name="weights",
                       realizations=realizations, timesteps=timesteps,
                       y_dimension_length=2, x_dimension_length=2)
    return cube


def set_up_basic_weights_cube(
        model_ids=[1000], model_configurations=["uk_det"],
        promote_to_new_axis=False, concatenate=True):
    weights_cube = set_up_weights_cube(timesteps=4)
    weights_cube = add_forecast_reference_time_and_forecast_period(
        weights_cube, time_point=[402295.0, 402300.0, 402336.0, 402342.0],
        fp_point=[7., 12., 48., 54.])
    weights_cube.data[:,1:3] = np.ones([1, 2, 2, 2])
    weights_cube = add_model_id_and_model_configuration(
        weights_cube, model_ids=model_ids,
        model_configurations=model_configurations,
        promote_to_new_axis=promote_to_new_axis, concatenate=concatenate)
    return weights_cube


def add_model_id_and_model_configuration(
        cube, model_ids=[1000, 2000],
        model_configurations=["uk_det", "uk_ens"], promote_to_new_axis=False,
        concatenate=True):
    cubelist = iris.cube.CubeList([])
    for model_id, model_configuration in zip(model_ids, model_configurations):
        cube_copy = cube.copy()
        model_id_coord = iris.coords.AuxCoord(
            model_id, long_name='model_id')
        cube_copy.add_aux_coord(model_id_coord)
        model_config_coord = iris.coords.AuxCoord(
            model_configuration, long_name='model_configuration')
        if promote_to_new_axis:
            cube_copy = iris.util.new_axis(cube_copy, "model_id")
            index = cube_copy.coord_dims("model_id")[0]
            cube_copy.add_aux_coord(model_config_coord, data_dims=index)
        else:
            cube_copy.add_aux_coord(model_config_coord)
        cubelist.append(cube_copy)
    if concatenate:
        result = cubelist.concatenate_cube()
    else:
        result = cubelist
    return result


def add_height(cube, heights):
    cubelist = iris.cube.CubeList([])
    for height in heights:
        cube_copy = cube.copy()
        height_coord = iris.coords.AuxCoord(height, long_name='height')
        cube_copy.add_aux_coord(height_coord)
        cubelist.append(cube_copy)
    return cubelist.merge_cube()


#model_id_coord = iris.coords.AuxCoord(
    #1000, long_name='model_id')
#weights_cube.add_aux_coord(model_id_coord)
#model_config_coord = iris.coords.AuxCoord(
    #"uk_det", long_name='model_configuration')
#weights_cube.add_aux_coord(model_config_coord)


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
        plugin = ChooseWeightsLinear(weighting_coord_name, config_coord_name,
                                     config_dict=config_dict)
        self.assertEqual(weighting_coord_name, plugin.weighting_coord_name)
        self.assertEqual(config_coord_name, plugin.config_coord_name)
        self.assertEqual(config_dict, plugin.config_dict)
        self.assertEqual(weights_coord_name, "weights")

    def test_without_config_dict(self):
        """Test the class is initialised correctly, if the config_dict
           argument is supplied."""
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinear(weighting_coord_name, config_coord_name)
        self.assertEqual(weighting_coord_name, plugin.weighting_coord_name)
        self.assertEqual(config_coord_name, plugin.config_coord_name)
        self.assertEqual(None, plugin.config_dict)


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
        plugin = ChooseWeightsLinear(weighting_coord_name, config_coord_name,
                                     config_dict=config_dict)

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
        plugin = ChooseWeightsLinear(weighting_coord_name, config_coord_name,
                            config_dict=config_dict)
        result = plugin._check_config_dict()
        self.assertIsNone(result)


class Test__interpolation_inputs_from_cube(IrisTest):
    """Test the linear weights function. """

    def test_basic(self):
        """Test that the function returns an array of weights. """
        expected_source_points = [7, 12, 48, 54]
        expected_target_points = [6., 7., 8.]
        expected_associated_data = np.array([[[[0., 0.],
                                               [0., 0.]],
                                              [[1., 1.],
                                               [1., 1.]],
                                              [[1., 1.],
                                               [1., 1.]],
                                              [[0., 0.],
                                               [0., 0.]]]])
        expected_axis = 1
        expected_fill_value = (np.array([[[ 0.,  0.],
                                          [ 0.,  0.]]]),
                               np.array([[[ 0.,  0.],
                                          [ 0.,  0.]]]))

        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"

        cube = set_up_basic_model_config_cube()
        weights_cube = set_up_basic_weights_cube()

        plugin = ChooseWeightsLinear(weighting_coord_name, config_coord_name)
        source_points, target_points, associated_data, axis, fill_value = (
            plugin._interpolation_inputs_from_cube(cube, weights_cube))
        print("fill_value = ", repr(fill_value))
        self.assertArrayAlmostEqual(source_points, expected_source_points)
        self.assertArrayAlmostEqual(target_points, expected_target_points)
        self.assertArrayAlmostEqual(associated_data, expected_associated_data)
        self.assertEqual(axis, expected_axis)
        self.assertArrayAlmostEqual(fill_value[0], expected_fill_value[0])
        self.assertArrayAlmostEqual(fill_value[1], expected_fill_value[1])


class Test__interpolation_inputs_from_dict(IrisTest):
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

        plugin = ChooseWeightsLinear(
            weighting_coord_name, config_coord_name, config_dict=config_dict)
        source_points, target_points, associated_data, fill_value = (
            plugin._interpolation_inputs_from_dict(cube))
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

        plugin = ChooseWeightsLinear(
            weighting_coord_name, config_coord_name, config_dict=config_dict)
        source_points, target_points, associated_data, fill_value = (
            plugin._interpolation_inputs_from_dict(cube))
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
        plugin = ChooseWeightsLinear(weighting_coord_name, config_coord_name)
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
        plugin = ChooseWeightsLinear(weighting_coord_name, config_coord_name)
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
        plugin = ChooseWeightsLinear(weighting_coord_name, config_coord_name)
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
        plugin = ChooseWeightsLinear(weighting_coord_name, config_coord_name)
        weights = plugin.interpolate_to_find_weights(
            source_points, target_points, associated_data, axis)
        self.assertArrayAlmostEqual(weights, expected_weights)


class Test__create_new_weights_cube(IrisTest):
    """Test the linear weights function. """

    def test_with_weights_cube(self):
        """Test that the function returns an array of weights. """
        cube = set_up_basic_model_config_cube()
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

        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinear(weighting_coord_name, config_coord_name)
        new_weights_cube = plugin._create_new_weights_cube(
            cube, weights, weights_cube=weights_cube)
        self.assertArrayAlmostEqual(expected_weights, new_weights_cube.data)
        self.assertEqual(weights_cube.metadata, new_weights_cube.metadata)

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
        plugin = ChooseWeightsLinear(weighting_coord_name, config_coord_name)
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
        plugin = ChooseWeightsLinear(weighting_coord_name, config_coord_name, weights_coord_name="alternative_name")
        new_weights_cube = plugin._create_new_weights_cube(cube, weights)
        self.assertArrayAlmostEqual(expected_weights, new_weights_cube.data)
        self.assertEqual(new_weights_cube.name(), "alternative_name")


class Test__interpolate_using_dict(IrisTest):
    """Test the linear weights function. """

    def test_basic(self):
        """Test that the function returns an array of weights. """
        config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                  "weights": [0, 1, 1, 0],
                                  "units": "hours"}}
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"

        cube = set_up_basic_model_config_cube()

        expected_weights = np.array([[[[ 0. ,  0. ],
                                       [ 0. ,  0. ]],
                                      [[ 0. ,  0. ],
                                       [ 0. ,  0. ]],
                                      [[ 0.2,  0.2],
                                       [ 0.2,  0.2]]]])

        plugin = ChooseWeightsLinear(
            weighting_coord_name, config_coord_name, config_dict=config_dict)
        new_weights_cube = plugin._interpolate_using_dict(cube)
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data, expected_weights)
        self.assertEqual(new_weights_cube.name(), "weights")


class Test__interpolate_using_cube(IrisTest):
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
        plugin = ChooseWeightsLinear(weighting_coord_name, config_coord_name)
        new_weights_cube = plugin._interpolate_using_cube(cube, weights_cube)
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
        plugin = ChooseWeightsLinear(weighting_coord_name, config_coord_name)
        new_weights_cube = plugin._interpolate_using_cube(cube, weights_cube)
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
        plugin = ChooseWeightsLinear(weighting_coord_name, config_coord_name)
        new_weights_cube = plugin._interpolate_using_cube(cube, weights_cube)
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
        plugin = ChooseWeightsLinear(weighting_coord_name, config_coord_name)
        new_weights_cube = plugin._interpolate_using_cube(cube, weights_cube)
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data, expected_weights)
        self.assertAlmostEqual(new_weights_cube.metadata,
                               weights_cube.metadata)


class Test_process(IrisTest):
    """Test the Default Linear Weights plugin. """

    def test_config_dict_and_weights_cube_specified(self):
        """Test that the plugin returns an array of weights. """
        config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                  "weights": [0, 1, 1, 0],
                                  "units": "hours"}}
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"

        cube = add_model_id_and_model_configuration(
            set_up_temperature_cube(timesteps=3), promote_to_new_axis=True)
        weights_cube = set_up_basic_weights_cube()

        plugin = ChooseWeightsLinear(
            weighting_coord_name, config_coord_name, config_dict=config_dict)

        msg = ('A configuration dictionary and a weights cube')
        with self.assertRaisesRegex(ValueError, msg):
            result = plugin.process(cube, weights_cubes=weights_cube)

    def test_config_dict_and_weights_cube_not_specified(self):
        config_dict = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                                  "weights": [0, 1, 1, 0],
                                  "units": "hours"}}
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"

        cube = add_model_id_and_model_configuration(
            set_up_temperature_cube(timesteps=3), promote_to_new_axis=True)

        plugin = ChooseWeightsLinear(
            weighting_coord_name, config_coord_name)

        msg = ('Either the configuration dictionary or the weights cube')
        with self.assertRaisesRegex(ValueError, msg):
            result = plugin.process(cube)

    def test_forecast_period_and_model_configuration_with_cube(self):
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"

        cube = add_model_id_and_model_configuration(
            set_up_temperature_cube(timesteps=3), promote_to_new_axis=True)
        cube = add_forecast_reference_time_and_forecast_period(
            cube, time_point=[402294.0, 402295.0, 402296.0],
            fp_point=[6., 7., 8.])

        weights_cube_uk_det = set_up_weights_cube(timesteps=2)
        weights_cube_uk_det = add_forecast_reference_time_and_forecast_period(
            weights_cube_uk_det, time_point=[402295.0, 402300.0],
            fp_point=[7., 12.])
        weights_cube_uk_det.data[:,0] = np.ones([1, 2, 2])
        weights_cube_uk_det = add_model_id_and_model_configuration(
            weights_cube_uk_det, model_ids=[1000],
            model_configurations=["uk_det"])

        weights_cube_uk_ens = set_up_weights_cube(timesteps=4)
        weights_cube_uk_ens = add_forecast_reference_time_and_forecast_period(
            weights_cube_uk_ens,
            time_point=[402295.0, 402300.0, 402336.0, 402342.0],
            fp_point=[7., 12., 48., 54.])
        weights_cube_uk_ens.data[:,1:3] = np.ones([1, 2, 2, 2])
        weights_cube_uk_ens = add_model_id_and_model_configuration(
            weights_cube_uk_ens, model_ids=[2000],
            model_configurations=["uk_ens"])

        weights_cubes = (
            iris.cube.CubeList([weights_cube_uk_det, weights_cube_uk_ens]))

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

        plugin = ChooseWeightsLinear(weighting_coord_name, config_coord_name)
        result = plugin.process(cube, weights_cubes=weights_cubes)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.metadata,
                               weights_cubes[0].metadata)

    def test_forecast_period_and_model_configuration_with_dict(self):
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"

        cube = add_model_id_and_model_configuration(set_up_temperature_cube(timesteps=3), promote_to_new_axis=True)
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
        plugin = ChooseWeightsLinear(
            weighting_coord_name, config_coord_name, config_dict=config_dict)
        result = plugin.process(cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.name(), "weights")

    def test_height_and_model_configuration(self):
        weighting_coord_name = "height"
        config_coord_name = "model_configuration"

        cube = add_model_id_and_model_configuration(
            set_up_temperature_cube(timesteps=1), promote_to_new_axis=True)
        heights = [10., 20.]
        cube = add_height(cube, heights)

        weights_cube_uk_det = set_up_weights_cube(timesteps=1)
        weights_cube_uk_det = add_model_id_and_model_configuration(
            weights_cube_uk_det, model_ids=[1000],
            model_configurations=["uk_det"])
        heights = [15., 25.]
        weights_cube_uk_det = add_height(weights_cube_uk_det, heights)
        weights_cube_uk_det.data[0] = np.ones([1, 2, 2])

        weights_cube_uk_ens = set_up_weights_cube(timesteps=1)
        weights_cube_uk_ens = add_model_id_and_model_configuration(
            weights_cube_uk_ens, model_ids=[2000],
            model_configurations=["uk_ens"])
        heights = [15., 25.]
        weights_cube_uk_ens = add_height(weights_cube_uk_ens, heights)
        weights_cube_uk_ens.data[1] = np.ones([1, 2, 2])

        weights_cubes = (
            iris.cube.CubeList([weights_cube_uk_det, weights_cube_uk_ens]))

        expected_weights = np.array([[[[[[1., 1.],
                                         [1., 1.]]]],
                                      [[[[0.5, 0.5],
                                         [0.5, 0.5]]]]],
                                     [[[[[0., 0.],
                                         [0., 0.]]]],
                                      [[[[0.5, 0.5],
                                         [0.5, 0.5]]]]]])

        plugin = ChooseWeightsLinear(weighting_coord_name, config_coord_name)
        result = plugin.process(cube, weights_cubes=weights_cubes)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.metadata,
                               weights_cubes[0].metadata)

    def test_height_and_realization(self):
        weighting_coord_name = "height"
        config_coord_name = "realization"

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
            break
        weights_cube_uk_det = cube_slice

        weights_cube_uk_ens = (
            set_up_weights_cube(data=data, timesteps=1, realizations=[1]))
        heights = [15., 25.]
        weights_cube_uk_ens = add_height(weights_cube_uk_ens, heights)
        weights_cube_uk_ens.data[1] = np.ones([1, 2, 2])
        for cube_slice in weights_cube_uk_ens.slices_over("realization"):
            break
        weights_cube_uk_ens = cube_slice

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

        plugin = ChooseWeightsLinear(weighting_coord_name, config_coord_name)
        result = plugin.process(cube, weights_cubes=weights_cubes)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.metadata,
                               weights_cubes[0].metadata)


if __name__ == '__main__':
    unittest.main()
