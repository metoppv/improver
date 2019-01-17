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
"""Unit tests for the ChooseWeightsLinear plugin."""

import unittest

import iris
from iris.coords import AuxCoord
from iris.tests import IrisTest
import numpy as np
from copy import deepcopy
from datetime import datetime as dt

from improver.blending.weights import ChooseWeightsLinear
from improver.utilities.temporal import forecast_period_coord
from improver.tests.set_up_test_cubes import (
    set_up_variable_cube, add_coordinate)


CONFIG_DICT_UKV = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                              "weights": [0, 1, 1, 0],
                              "units": "hours"}}


def set_up_basic_model_config_cube(frt=None, time_points=None):
    """Set up cube with dimensions of realization x time x lat x lon, plus
     model id and configuration scalar coords

    Kwargs:
        frt (datetime):
            Forecast reference time point
        time_points (list):
            List of times as datetime instances to create a dim coord
    """
    if frt is None:
        frt = dt(2017, 1, 10, 3, 0)
    if time_points is None:
        time_points = [dt(2017, 1, 10, 9, 0), dt(2017, 1, 10, 10, 0),
                       dt(2017, 1, 10, 11, 0)]

    model_id_coord = AuxCoord([1000], long_name="model_id")
    model_config_coord = AuxCoord(["uk_det"], long_name="model_configuration")

    data = np.full((1, 2, 2), 275.15, dtype=np.float32)
    cube = set_up_variable_cube(
        data, time=frt, frt=frt, include_scalar_coords=[model_id_coord,
                                                        model_config_coord])
    cube = add_coordinate(
        cube, time_points, "time", is_datetime=True, order=[1, 0, 2, 3])

    return cube


def set_up_basic_weights_cube(set_data=True, frt=None, time_points=None):
    """Set up weights cube with dimensions of realization x time x lat x lon,
    plus model id and configuration scalar coords

    Kwargs:
        set_data (bool):
            If True, update the np.zeros array in the weights cube to a
            specific data array which is used in several of the unit tests
            below.  Note this will cause an error if the cube dimensions do
            not match this array (shape (1, 4, 2, 2)).
        frt (datetime):
            Forecast reference time point
        time_points (list):
            List of times as datetime instances to create a dim coord
    """

    if frt is None:
        frt = dt(2017, 1, 10, 3, 0)
    if time_points is None:
        time_points = [dt(2017, 1, 10, 10, 0), dt(2017, 1, 10, 15, 0),
                       dt(2017, 1, 12, 3, 0), dt(2017, 1, 12, 9, 0)]

    model_id_coord = AuxCoord([1000], long_name="model_id")
    model_config_coord = AuxCoord(["uk_det"], long_name="model_configuration")

    weights_cube = set_up_variable_cube(
        np.zeros((1, 2, 2), dtype=np.float32), name="weights", units=1,
        time=frt, frt=frt, include_scalar_coords=[model_id_coord,
                                                  model_config_coord])

    weights_cube = add_coordinate(weights_cube, time_points, "time",
                                  is_datetime=True, order=[1, 0, 2, 3])
    if set_data:
        weights_cube.data = np.array(
            [[[[0., 0.], [0., 0.]],
              [[1., 1.], [1., 1.]],
              [[1., 1.], [1., 1.]],
              [[0., 0.], [0., 0.]]]], dtype=np.float32)

    return weights_cube


def update_time_and_forecast_period(cube, increment):
    """Updates time and forecast period points on an existing cube by a given
    increment (in units of time)"""
    cube.coord("time").points = cube.coord("time").points + increment
    forecast_period = forecast_period_coord(
        cube, force_lead_time_calculation=True)
    cube.replace_coord(forecast_period)
    return cube


class Test__init__(IrisTest):
    """Test the __init__ method"""

    def setUp(self):
        """Set up some initialisation arguments"""
        self.weighting_coord_name = "forecast_period"
        self.config_coord_name = "model_configuration"
        self.config_dict = CONFIG_DICT_UKV

    def test_basic(self):
        """Test default initialisation from cubes"""
        plugin = ChooseWeightsLinear(self.weighting_coord_name)
        self.assertEqual(
            plugin.weighting_coord_name, self.weighting_coord_name)
        self.assertEqual(plugin.config_coord_name, self.config_coord_name)
        self.assertIsNone(plugin.config_dict)
        self.assertEqual(plugin.weights_key_name, "weights")

    def test_config_coord_name(self):
        """Test different config coord name"""
        plugin = ChooseWeightsLinear(self.weighting_coord_name, "height")
        self.assertEqual(
            plugin.weighting_coord_name, self.weighting_coord_name)
        self.assertEqual(plugin.config_coord_name, "height")

    def test_with_config_dict(self):
        """Test initialisation from dict"""
        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, config_dict=self.config_dict)
        self.assertEqual(
            plugin.weighting_coord_name, self.weighting_coord_name)
        self.assertEqual(plugin.config_coord_name, self.config_coord_name)
        self.assertEqual(plugin.config_dict, self.config_dict)
        self.assertEqual(plugin.weights_key_name, "weights")


class Test__repr__(IrisTest):
    """Test the __repr__ method"""

    def test_cubes(self):
        """Test with default initialisation from cubes"""
        weighting_coord_name = "forecast_period"
        plugin = ChooseWeightsLinear(weighting_coord_name)
        expected_result = (
            "<ChooseWeightsLinear(): weighting_coord_name = forecast_period, "
            "config_coord_name = model_configuration, config_dict = None>")
        self.assertEqual(str(plugin), expected_result)

    def test_dict(self):
        """Test with configuration dictionary"""
        weighting_coord_name = "forecast_period"
        config_dict = CONFIG_DICT_UKV
        plugin = ChooseWeightsLinear(
            weighting_coord_name, config_dict=config_dict)
        expected_result = (
            "<ChooseWeightsLinear(): weighting_coord_name = forecast_period, "
            "config_coord_name = model_configuration, "
            "config_dict = {'uk_det': {'forecast_period': [7, 12, 48, 54], "
            "'weights': [0, 1, 1, 0], 'units': 'hours'}}>")
        self.assertEqual(str(plugin), expected_result)


class Test__check_config_dict(IrisTest):
    """Test the _check_config_dict method."""

    def setUp(self):
        """Set up some plugin inputs"""
        self.config_dict = deepcopy(CONFIG_DICT_UKV)
        self.weighting_coord_name = "forecast_period"

    def test_dictionary_key_mismatch(self):
        """Test whether there is a mismatch in the dictionary keys. As
        _check_config_dict is called within the initialisation,
        _check_config_dict is not called directly."""
        self.config_dict["uk_det"]["weights"] = [0, 1, 0]
        msg = ('These items in the configuration dictionary')
        with self.assertRaisesRegex(ValueError, msg):
            _ = ChooseWeightsLinear(
                self.weighting_coord_name, config_dict=self.config_dict)

    def test_error_weighting_coord_not_in_dict(self):
        """Test that an exception is raised when the required weighting_coord
        is not in the configuration dictionary"""
        with self.assertRaises(KeyError):
            ChooseWeightsLinear("height", config_dict=self.config_dict)


class Test__get_interpolation_inputs_from_cube(IrisTest):
    """Test the _get_interpolation_inputs_from_cube method."""

    def test_basic(self):
        """Test that the values for the source_points, target_points,
        source_weights, axis and fill_value are as expected."""
        cube = set_up_basic_model_config_cube()
        weights_cube = set_up_basic_weights_cube()

        expected_source_points = 3600*np.array([7, 12, 48, 54])
        expected_target_points = 3600*np.array([6., 7., 8.])
        expected_source_weights = weights_cube.data
        expected_axis = 1
        expected_fill_value = (np.array([[[0., 0.],
                                          [0., 0.]]]),
                               np.array([[[0., 0.],
                                          [0., 0.]]]))

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
        self.expected_source_points = 3600*np.array([7, 12, 48, 54])
        self.expected_target_points = 3600*np.array([6., 7., 8.])
        self.expected_source_weights = [0, 1, 1, 0]
        self.expected_fill_value = (0, 0)

        self.cube = set_up_basic_model_config_cube()
        self.weighting_coord_name = "forecast_period"

    def test_basic(self):
        """Test that the values for the source_points, target_points,
        source_weights, axis and fill_value are as expected."""
        config_dict = CONFIG_DICT_UKV

        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, config_dict=config_dict)

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
            self.weighting_coord_name, config_dict=config_dict)

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
        fill_value = (0, 0)
        weights = self.plugin._interpolate_to_find_weights(
            source_points, target_points, source_weights, fill_value, axis=0)
        self.assertArrayAlmostEqual(weights, expected_weights)

    def test_1d_array_use_fill_value(self):
        """Test that the interpolation produces the expected result for a
        1d input array where interpolation beyond of bounds of the input data
        uses the fill_value."""
        expected_weights = (
            np.array([3., 3., 0., 0.5, 1., 1., 1., 0.5, 0., 4., 4.]))
        source_points = np.array([0, 2, 4, 6])
        target_points = np.arange(-2, 9)
        source_weights = np.array([0, 1, 1, 0])
        fill_value = (3, 4)
        weights = self.plugin._interpolate_to_find_weights(
            source_points, target_points, source_weights, fill_value,
            axis=0)
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
        fill_value = (0, 0)
        weights = self.plugin._interpolate_to_find_weights(
            source_points, target_points, source_weights, fill_value, axis=1)
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
        fill_value = (0, 0)
        weights = self.plugin._interpolate_to_find_weights(
            source_points, target_points, source_weights, fill_value, axis=1)
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
        fill_value = (0, 0)
        weights = self.plugin._interpolate_to_find_weights(
            source_points, target_points, source_weights, fill_value, axis=2)
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
        self.config_dict = CONFIG_DICT_UKV
        self.weights = np.array([0., 0., 0.2])
        self.expected_weights = np.array([[[[0., 0.], [0., 0.]],
                                           [[0., 0.], [0., 0.]],
                                           [[0.2, 0.2], [0.2, 0.2]]]])

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
        plugin = ChooseWeightsLinear(self.weighting_coord_name)
        new_weights_cube = plugin._create_new_weights_cube(
            self.cube, weights, weights_cube)
        self.assertArrayAlmostEqual(new_weights_cube.data,
                                    self.expected_weights)
        self.assertAlmostEqual(new_weights_cube.metadata,
                               weights_cube.metadata)

    def test_with_dict(self):
        """Test a new weights cube is created as intended, with the desired
        cube name."""
        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, config_dict=self.config_dict)
        new_weights_cube = plugin._create_new_weights_cube(
            self.cube, self.weights)
        self.assertArrayAlmostEqual(new_weights_cube.data,
                                    self.expected_weights[..., 0, 0])
        self.assertEqual(new_weights_cube.name(), "weights")


class Test__calculate_weights(IrisTest):
    """Test the _calculate_weights method"""

    def setUp(self):
        """Set up some cubes and plugins to work with"""
        self.temp_cube = set_up_basic_model_config_cube()
        self.weights_cube = set_up_basic_weights_cube()

        config_dict = CONFIG_DICT_UKV
        weighting_coord_name = "forecast_period"

        self.plugin_cubes = ChooseWeightsLinear(weighting_coord_name)
        self.plugin_dict = ChooseWeightsLinear(
            weighting_coord_name, config_dict=config_dict)

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
        self.assertArrayAlmostEqual(
            new_weights_cube.data,
            self.expected_weights_below_range[..., 0, 0])
        self.assertEqual(new_weights_cube.name(), "weights")

    def test_within_range_cubes(self):
        """Test that interpolation works as intended when the forecast period
        required for the interpolation output is within the range specified
        within the inputs."""
        cube = update_time_and_forecast_period(self.temp_cube, 3600*5)
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
        cube = update_time_and_forecast_period(self.temp_cube, 3600*5)
        new_weights_cube = (
            self.plugin_dict._calculate_weights(cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            new_weights_cube.data,
            self.expected_weights_within_range[..., 0, 0])
        self.assertEqual(new_weights_cube.name(), "weights")

    def test_above_range_cubes(self):
        """Test that interpolation works as intended when the forecast period
        required for the interpolation output is above the range specified
        within the inputs."""
        cube = update_time_and_forecast_period(self.temp_cube, 3600*47)
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
        cube = update_time_and_forecast_period(self.temp_cube, 3600*47)
        new_weights_cube = (
            self.plugin_dict._calculate_weights(cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            new_weights_cube.data,
            self.expected_weights_above_range[..., 0, 0])
        self.assertEqual(new_weights_cube.name(), "weights")

    def test_spatial_varying_weights(self):
        """Test that interpolation works as intended when the weights vary
        spatially within the input cube."""

        # set up common attributes
        model_id_coord = AuxCoord([1000], long_name="model_id")
        model_config_coord = AuxCoord(
            ["uk_det"], long_name="model_configuration")
        frt_common = dt(2017, 1, 10, 3, 0)

        # set up data and weights cubes with suitable forecast periods
        data = np.full((1, 2, 2), 275.15, dtype=np.float32)
        time_points = [dt(2017, 1, 10, 12, 0), dt(2017, 1, 10, 18, 0),
                       dt(2017, 1, 11, 0, 0)]
        cube = set_up_variable_cube(
            data, time=time_points[0], frt=frt_common,
            include_scalar_coords=[model_id_coord, model_config_coord])
        cube = add_coordinate(
            cube, time_points, "time", is_datetime=True, order=[1, 0, 2, 3])

        time_points = [dt(2017, 1, 10, 9, 0), dt(2017, 1, 10, 15, 0),
                       dt(2017, 1, 10, 21, 0), dt(2017, 1, 11, 3, 0)]
        weights_cube = set_up_variable_cube(
            np.zeros((1, 2, 2), dtype=np.float32), name="weights", units=1,
            time=time_points[0], frt=frt_common,
            include_scalar_coords=[model_id_coord, model_config_coord])
        weights_cube = add_coordinate(weights_cube, time_points, "time",
                                      is_datetime=True, order=[1, 0, 2, 3])

        weights_cube.data = np.array([[[[1., 0.],
                                        [0., 0.]],
                                       [[1., 0.],
                                        [1., 1.]],
                                       [[0., 1.],
                                        [0., 1.]],
                                       [[0., 0.],
                                        [0., 1.]]]])

        # define expected output and run test
        expected_weights = np.array([[[[1., 0.],
                                       [0.5, 0.5]],
                                      [[0.5, 0.5],
                                       [0.5, 1.]],
                                      [[0., 0.5],
                                       [0., 1.]]]])

        new_weights_cube = (
            self.plugin_cubes._calculate_weights(
                cube, weights_cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data, expected_weights)
        self.assertAlmostEqual(new_weights_cube.metadata,
                               weights_cube.metadata)


class Test__slice_input_cubes(IrisTest):
    """Test the _slice_input_cubes method"""

    def setUp(self):
        """Set up plugin with suitable parameters (used for dict only)"""
        self.plugin = ChooseWeightsLinear(
            "forecast_period", config_dict=CONFIG_DICT_UKV)

        # create a cube with unnecessary realization coordinate (dimensions:
        # model_id: 2; realization: 1; latitude: 2; longitude: 2)
        cube = set_up_variable_cube(278.*np.ones((1, 2, 2), dtype=np.float32))
        self.cube = add_coordinate(
            cube, [1000, 2000], "model_id", dtype=np.int32)
        self.cube.add_aux_coord(
            AuxCoord(["uk_det", "uk_ens"], long_name="model_configuration"),
            data_dims=0)

        # create a reference cube as above WITHOUT realization
        new_data = self.cube.data[:, 0, :, :]
        dim_coords = [(self.cube.coord("model_id"), 0),
                      (self.cube.coord("latitude"), 1),
                      (self.cube.coord("longitude"), 2)]
        aux_coords = [(self.cube.coord("model_configuration"), 0),
                      (self.cube.coord("time"), None),
                      (self.cube.coord("forecast_period"), None),
                      (self.cube.coord("forecast_reference_time"), None)]
        self.reference_cube = iris.cube.Cube(
            new_data, "air_temperature", units="K",
            dim_coords_and_dims=dim_coords,
            aux_coords_and_dims=aux_coords)
        self.reference_cube.add_aux_coord(AuxCoord(0, "realization"))

        # split into a cubelist for each model
        self.reference_cubelist = iris.cube.CubeList([self.reference_cube[0],
                                                      self.reference_cube[1]])

    def test_slices(self):
        """Test function slices out extra dimensions"""
        result = self.plugin._slice_input_cubes(self.cube)
        self.assertIsInstance(result, iris.cube.CubeList)
        for cube, refcube in zip(result, self.reference_cubelist):
            self.assertArrayAlmostEqual(cube.data, refcube.data)
            self.assertEqual(cube.metadata, refcube.metadata)

    def test_cubelist(self):
        """Test function creates cubelist with same dimensions where needed"""
        result = self.plugin._slice_input_cubes(self.reference_cube)
        self.assertIsInstance(result, iris.cube.CubeList)
        for cube, refcube in zip(result, self.reference_cubelist):
            self.assertArrayAlmostEqual(cube.data, refcube.data)
            self.assertEqual(cube.metadata, refcube.metadata)

    def test_single_cube(self):
        """Test function populates a cubelist if given a cube with a scalar
        blending coordinate"""
        single_cube = self.reference_cube[0]
        result = self.plugin._slice_input_cubes(single_cube)
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(len(result), 1)
        self.assertArrayAlmostEqual(result[0].data, single_cube.data)
        self.assertEqual(result[0].metadata, single_cube.metadata)


class Test_process(IrisTest):
    """Test the process method"""

    def setUp(self):
        """Set up some cubes and plugin inputs"""
        self.weighting_coord_name = "forecast_period"
        self.plugin = ChooseWeightsLinear(self.weighting_coord_name)
        self.config_dict_fp = {"uk_det": {"forecast_period": [7, 12],
                                          "weights": [1, 0],
                                          "units": "hours"},
                               "uk_ens": {"forecast_period": [7, 12, 48, 54],
                                          "weights": [0, 1, 1, 0],
                                          "units": "hours"}}

        # set up UK deterministic weights cube for fps [ 7. 12.] hours
        time_points = [dt(2017, 1, 10, 10), dt(2017, 1, 10, 15)]
        self.weights_cube_uk_det = set_up_basic_weights_cube(
            frt=dt(2017, 1, 10, 3), time_points=time_points, set_data=False)
        self.weights_cube_uk_det.data[:, 0] = np.ones([1, 2, 2])

        # set up UK ensemble weights cube for fps [ 7. 12. 48. 54.] hours
        time_points = [dt(2017, 1, 10, 10), dt(2017, 1, 10, 15),
                       dt(2017, 1, 12, 3), dt(2017, 1, 12, 9)]
        self.weights_cube_uk_ens = set_up_basic_weights_cube(
            frt=dt(2017, 1, 10, 3), time_points=time_points, set_data=False)
        self.weights_cube_uk_ens.data[:, 1:3] = np.ones([1, 2, 2, 2])
        self.weights_cube_uk_ens.coord("model_id").points = [2000]
        self.weights_cube_uk_ens.coord("model_configuration").points = (
            ["uk_ens"])

    def test_error_incorrect_args(self):
        """Test error is raised if both config_dict and weights_cubes are set
        """
        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, config_dict=self.config_dict_fp)
        cubes = set_up_basic_model_config_cube()
        weights_cubes = set_up_basic_weights_cube()
        msg = 'Cannot calculate weights from both dict and cube'
        with self.assertRaisesRegex(ValueError, msg):
            _ = plugin.process(cubes, weights_cubes)

    def test_forecast_period_and_model_configuration_cubes(self):
        """Test when forecast_period is the weighting_coord_name. This
        demonstrates blending models whose relative weights differ with
        forecast period."""

        # set up data cubes with forecast periods [ 8. 20. 51.] hours
        time_points = [
            dt(2017, 1, 10, 11), dt(2017, 1, 10, 23), dt(2017, 1, 12, 6)]
        cube1 = set_up_basic_model_config_cube(
            frt=dt(2017, 1, 10, 3), time_points=time_points)
        cube2 = cube1.copy()
        cube2.coord("model_id").points = [2000]
        cube2.coord("model_configuration").points = ["uk_ens"]
        cubes = iris.cube.CubeList([cube1, cube2])

        # set up UK ensemble weights cube for fps [ 7. 12. 48. 52.] hours
        time_points = [dt(2017, 1, 10, 10), dt(2017, 1, 10, 15),
                       dt(2017, 1, 12, 3), dt(2017, 1, 12, 7)]
        weights_cube_uk_ens = set_up_basic_weights_cube(
            frt=dt(2017, 1, 10, 3), time_points=time_points, set_data=False)
        weights_cube_uk_ens.data[:, 1:3] = np.ones([1, 2, 2, 2])
        weights_cube_uk_ens.coord("model_id").points = [2000]
        weights_cube_uk_ens.coord("model_configuration").points = ["uk_ens"]

        weights_cubes = (
            iris.cube.CubeList([
                self.weights_cube_uk_det, weights_cube_uk_ens]))

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

        result = self.plugin.process(cubes, weights_cubes)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.metadata,
                               weights_cubes[0].metadata)

    def test_forecast_period_and_model_configuration_dict(self):
        """Test blending models over forecast_period with a configuration
        dictionary."""
        # set up data cubes with forecast periods [ 6. 7. 8.] hours
        time_points = [
            dt(2017, 1, 10, 9), dt(2017, 1, 10, 10), dt(2017, 1, 10, 11)]
        cube1 = set_up_basic_model_config_cube(
            frt=dt(2017, 1, 10, 3), time_points=time_points)
        cube2 = cube1.copy()
        cube2.coord("model_id").points = [2000]
        cube2.coord("model_configuration").points = ["uk_ens"]
        cubes = iris.cube.CubeList([cube1, cube2])

        expected_weights = np.array([[1., 1., 0.8], [0., 0., 0.2]])

        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, config_dict=self.config_dict_fp)
        result = plugin.process(cubes)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.name(), "weights")

    def test_forecast_period_and_model_configuration_three_models_cubes(self):
        """Test blending three models with relative weights varying along the
        forecast_period coordinate."""
        # set up data cubes with forecast periods [ 8. 20. 51.] hours
        time_points = [
            dt(2017, 1, 10, 11), dt(2017, 1, 10, 23), dt(2017, 1, 12, 6)]
        cube1 = set_up_basic_model_config_cube(
            frt=dt(2017, 1, 10, 3), time_points=time_points)
        cubes = iris.cube.CubeList([cube1])
        for i, model in enumerate(["uk_ens", "gl_ens"]):
            cube = cube1.copy()
            cube.coord("model_id").points = [1000*(i+2)]
            cube.coord("model_configuration").points = [model]
            cubes.append(cube)

        # set up global ensemble weights cube for fps [ 48. 54.] hours
        time_points = [dt(2017, 1, 12, 3), dt(2017, 1, 12, 9)]
        weights_cube_gl_ens = set_up_basic_weights_cube(
            frt=dt(2017, 1, 10, 3), time_points=time_points, set_data=False)
        weights_cube_gl_ens.data[:, 1] = np.ones([1, 2, 2])
        weights_cube_gl_ens.coord("model_id").points = [3000]
        weights_cube_gl_ens.coord("model_configuration").points = ["gl_ens"]

        weights_cubes = (
            iris.cube.CubeList([
                self.weights_cube_uk_det, self.weights_cube_uk_ens,
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

        result = self.plugin.process(cubes, weights_cubes)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.metadata,
                               weights_cubes[0].metadata)

    def test_forecast_period_and_model_configuration_three_models_dict(self):
        """Test blending three models over forecast period with a
        configuration dictionary returns a sorted weights cube."""
        time_points = [
            dt(2017, 1, 10, 9), dt(2017, 1, 10, 10), dt(2017, 1, 10, 11)]
        cube1 = set_up_basic_model_config_cube(
            frt=dt(2017, 1, 10, 3), time_points=time_points)
        cubes = iris.cube.CubeList([cube1])
        for i, model in enumerate(["uk_ens", "gl_ens"]):
            cube = cube1.copy()
            cube.coord("model_id").points = [1000*(i+2)]
            cube.coord("model_configuration").points = [model]
            cubes.append(cube)

        expected_weights = np.array([[1., 1., 0.72],
                                     [0., 0., 0.18],
                                     [0., 0., 0.1]])

        self.config_dict_fp["gl_ens"] = {"forecast_period": [7, 16, 48, 54],
                                         "weights": [0, 1, 1, 1],
                                         "units": "hours"}

        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, config_dict=self.config_dict_fp)
        result = plugin.process(cubes)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.name(), "weights")
        self.assertArrayAlmostEqual(result.coord('model_id').points,
                                    [1000, 2000, 3000])
        self.assertArrayEqual(
            result.coord('model_configuration').points,
            ["uk_det", "uk_ens", "gl_ens"])

    def test_height_and_realization_cubes(self):
        """Test when height is the weighting_coord_name and realization is the
        config_coord_name. This demonstrates blending in one member and
        blending out another member with height."""
        cube = set_up_variable_cube(274.*np.ones((2, 2, 2), dtype=np.float32))
        cube = add_coordinate(cube, [10., 20.], "height", coord_units="m")
        cubes = iris.cube.CubeList([])
        for cube_slice in cube.slices_over("realization"):
            cubes.append(cube_slice)

        data = np.zeros((1, 2, 2), dtype=np.float32)
        weights_cubes = iris.cube.CubeList()
        for i, model in enumerate(["uk_det", "uk_ens"]):
            model_id_coord = AuxCoord([1000*(i+1)], long_name="model_id")
            model_config_coord = AuxCoord(
                [model], long_name="model_configuration")
            weights_cube = set_up_variable_cube(
                data, name="weights", units="1", realizations=[i],
                include_scalar_coords=[model_id_coord, model_config_coord])
            weights_cube = add_coordinate(
                weights_cube, [15., 25.], "height", coord_units="m")
            weights_cubes.append(weights_cube)

        weights_cubes[0].data[0] = np.ones([2, 2])
        weights_cubes[1].data[1] = np.ones([2, 2])

        expected_weights = np.array([[[[1., 1.],
                                       [1., 1.]],
                                      [[0.5, 0.5],
                                       [0.5, 0.5]]],
                                     [[[0., 0.],
                                       [0., 0.]],
                                      [[0.5, 0.5],
                                       [0.5, 0.5]]]])

        plugin = ChooseWeightsLinear("height", config_coord_name="realization")
        result = plugin.process(cubes, weights_cubes)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.metadata,
                               weights_cubes[0].metadata)

    def test_height_and_realization_dict(self):
        """Test blending members with a configuration dictionary."""
        cube = set_up_variable_cube(274.*np.ones((2, 2, 2), dtype=np.float32))
        cube = add_coordinate(cube, [10., 20.], "height", coord_units="m")
        cubes = iris.cube.CubeList([])
        for cube_slice in cube.slices_over("realization"):
            cubes.append(cube_slice)

        expected_weights = np.array([[1., 0.5],
                                     [0., 0.5]])

        config_dict = {0: {"height": [15, 25],
                           "weights": [1, 0],
                           "units": "m"},
                       1: {"height": [15, 25],
                           "weights": [0, 1],
                           "units": "m"}}
        plugin = ChooseWeightsLinear(
            "height", config_coord_name="realization", config_dict=config_dict)
        result = plugin.process(cubes)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.name(), "weights")

    def test_exception_if_weights_incorrect(self):
        """Test that an exception is raised when the length of the
        weights_cubes cubelist differs from the number of points along the
        config_coord_name dimension within the input cube."""
        cube = iris.cube.CubeList([set_up_basic_model_config_cube()])
        weights_cubes = iris.cube.CubeList([])
        plugin = ChooseWeightsLinear(self.weighting_coord_name)
        msg = ('The number of cubes to be weighted')
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(cube, weights_cubes)


if __name__ == '__main__':
    unittest.main()
