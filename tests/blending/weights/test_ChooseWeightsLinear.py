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
from copy import deepcopy
from datetime import datetime as dt

import iris
import numpy as np
from iris.coords import AuxCoord
from iris.tests import IrisTest

from improver.blending.weights import ChooseWeightsLinear
from improver.metadata.forecast_times import forecast_period_coord

from ...set_up_test_cubes import (
    add_coordinate, set_up_probability_cube, set_up_variable_cube)

CONFIG_DICT_UKV = {"uk_det": {"forecast_period": [7, 12, 48, 54],
                              "weights": [0, 1, 1, 0],
                              "units": "hours"}}


def set_up_basic_model_config_cube(frt=None, time_points=None):
    """Set up cube with dimensions of time x air_temperature x lat x lon, plus
     model id and configuration scalar coordinates

    Args:
        frt (datetime.datetime):
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

    data = np.ones((2, 2, 2), dtype=np.float32)
    thresholds = np.array([275., 276.], dtype=np.float32)
    cube = set_up_probability_cube(
        data, thresholds, time=frt, frt=frt,
        include_scalar_coords=[model_id_coord, model_config_coord])

    cube = add_coordinate(
        cube, time_points, "time", is_datetime=True)

    return cube


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
        """Test default initialisations"""
        plugin = ChooseWeightsLinear(self.weighting_coord_name,
                                     self.config_dict)
        self.assertEqual(
            plugin.weighting_coord_name, self.weighting_coord_name)
        self.assertEqual(plugin.config_coord_name, self.config_coord_name)
        self.assertEqual(plugin.config_dict, self.config_dict)
        self.assertEqual(plugin.weights_key_name, "weights")

    def test_config_coord_name(self):
        """Test different config coord name"""
        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, self.config_dict, "height")
        self.assertEqual(
            plugin.weighting_coord_name, self.weighting_coord_name)
        self.assertEqual(plugin.config_coord_name, "height")

    def test_with_config_dict(self):
        """Test initialisation from dict"""
        plugin = ChooseWeightsLinear(
            self.weighting_coord_name, self.config_dict)
        self.assertEqual(
            plugin.weighting_coord_name, self.weighting_coord_name)
        self.assertEqual(plugin.config_coord_name, self.config_coord_name)
        self.assertEqual(plugin.config_dict, self.config_dict)
        self.assertEqual(plugin.weights_key_name, "weights")


class Test__repr__(IrisTest):
    """Test the __repr__ method"""

    def test_dict(self):
        """Test with configuration dictionary"""
        weighting_coord_name = "forecast_period"
        config_dict = CONFIG_DICT_UKV
        plugin = ChooseWeightsLinear(
            weighting_coord_name, config_dict)
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
                self.weighting_coord_name, self.config_dict)

    def test_error_weighting_coord_not_in_dict(self):
        """Test that an exception is raised when the required weighting_coord
        is not in the configuration dictionary"""
        with self.assertRaises(KeyError):
            ChooseWeightsLinear("height", self.config_dict)


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
            self.weighting_coord_name, config_dict)

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
            self.weighting_coord_name, config_dict)

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
        self.plugin = ChooseWeightsLinear("forecast_period", CONFIG_DICT_UKV)

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


class Test__create_new_weights_cube(IrisTest):
    """Test the _create_new_weights_cube function. """

    def setUp(self):
        """Set up some plugin inputs"""
        self.plugin = ChooseWeightsLinear(
            "forecast_period", config_dict=CONFIG_DICT_UKV)
        self.cube, = self.plugin._slice_input_cubes(
            set_up_basic_model_config_cube())
        self.weights = np.array([0., 0., 0.2])

    def test_with_dict(self):
        """Test a new weights cube is created as intended, with the desired
        cube name."""
        new_weights_cube = self.plugin._create_new_weights_cube(
            self.cube, self.weights)
        self.assertArrayAlmostEqual(new_weights_cube.data, self.weights)
        self.assertEqual(new_weights_cube.name(), "weights")
        # test only relevant coordinates have been retained on the weights cube
        expected_coords = {
            'time', 'forecast_reference_time', 'forecast_period',
            'model_id', 'model_configuration'}
        result_coords = {coord.name() for coord in new_weights_cube.coords()}
        self.assertSetEqual(result_coords, expected_coords)

    def test_with_dict_masked_input_cube(self):
        """Test a new weights cube is created as intended when we have a masked
        input cube."""
        self.cube.data = np.ma.masked_array(
            self.cube.data, np.ones(self.cube.data.shape)*True)
        new_weights_cube = self.plugin._create_new_weights_cube(
            self.cube, self.weights)
        self.assertEqual(np.ma.is_masked(new_weights_cube.data), False)
        self.assertArrayAlmostEqual(new_weights_cube.data, self.weights)


class Test__calculate_weights(IrisTest):
    """Test the _calculate_weights method"""

    def setUp(self):
        """Set up some cubes and plugins to work with"""
        config_dict = CONFIG_DICT_UKV
        weighting_coord_name = "forecast_period"

        self.plugin_dict = ChooseWeightsLinear(
            weighting_coord_name, config_dict)
        self.temp_cube, = self.plugin_dict._slice_input_cubes(
            set_up_basic_model_config_cube())

        self.expected_weights_below_range = np.array([0., 0., 0.2])
        self.expected_weights_within_range = np.array([0.8, 1., 1.])
        self.expected_weights_above_range = np.array([0.166667, 0., 0.])

    def test_below_range_dict(self):
        """Test that interpolation works as intended when the forecast period
        required for the interpolation output is below the range specified
        within the inputs."""
        new_weights_cube = (
            self.plugin_dict._calculate_weights(self.temp_cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            new_weights_cube.data, self.expected_weights_below_range)
        self.assertEqual(new_weights_cube.name(), "weights")

    def test_within_range_dict(self):
        """Test that interpolation works as intended when the forecast period
        required for the interpolation output is within the range specified
        within the inputs."""
        cube = update_time_and_forecast_period(self.temp_cube, 3600*5)
        new_weights_cube = (
            self.plugin_dict._calculate_weights(cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            new_weights_cube.data, self.expected_weights_within_range)
        self.assertEqual(new_weights_cube.name(), "weights")

    def test_above_range_dict(self):
        """Test that interpolation works as intended when the forecast period
        required for the interpolation output is above the range specified
        within the inputs."""
        cube = update_time_and_forecast_period(self.temp_cube, 3600*47)
        new_weights_cube = (
            self.plugin_dict._calculate_weights(cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            new_weights_cube.data, self.expected_weights_above_range)
        self.assertEqual(new_weights_cube.name(), "weights")


class Test__slice_input_cubes(IrisTest):
    """Test the _slice_input_cubes method"""

    def setUp(self):
        """Set up plugin with suitable parameters (used for dict only)"""
        self.plugin = ChooseWeightsLinear(
            "forecast_period", CONFIG_DICT_UKV)

        # create a cube with irrelevant threshold coordinate (dimensions:
        # model_id: 2; threshold: 2; latitude: 2; longitude: 2)
        cube = set_up_probability_cube(
            np.ones((2, 2, 2), dtype=np.float32),
            thresholds=np.array([278., 279.], dtype=np.float32))
        self.cube = add_coordinate(
            cube, [1000, 2000], "model_id", dtype=np.int32)
        self.cube.add_aux_coord(
            AuxCoord(["uk_det", "uk_ens"], long_name="model_configuration"),
            data_dims=0)

        # create a reference cube as above WITHOUT threshold
        self.reference_cube = iris.util.squeeze(self.cube[:, 0, :, :].copy())

        # split into a cubelist by model
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
        self.config_dict_fp = {"uk_det": {"forecast_period": [7, 12],
                                          "weights": [1, 0],
                                          "units": "hours"},
                               "uk_ens": {"forecast_period": [7, 12, 48, 54],
                                          "weights": [0, 1, 1, 0],
                                          "units": "hours"}}
        self.expected_coords_model_blend_weights = {
            'time', 'forecast_reference_time', 'forecast_period',
            'model_id', 'model_configuration'}

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
            self.weighting_coord_name, self.config_dict_fp)
        result = plugin.process(cubes)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.name(), "weights")
        result_coords = {coord.name() for coord in result.coords()}
        self.assertSetEqual(
            result_coords, self.expected_coords_model_blend_weights)

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
            self.weighting_coord_name, self.config_dict_fp)
        result = plugin.process(cubes)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.name(), "weights")
        self.assertArrayAlmostEqual(result.coord('model_id').points,
                                    [1000, 2000, 3000])
        self.assertArrayEqual(
            result.coord('model_configuration').points,
            ["uk_det", "uk_ens", "gl_ens"])
        result_coords = {coord.name() for coord in result.coords()}
        self.assertSetEqual(
            result_coords, self.expected_coords_model_blend_weights)

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
            "height", config_dict, config_coord_name="realization")
        result = plugin.process(cubes)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.name(), "weights")
        expected_coords = {
            'time', 'forecast_reference_time', 'forecast_period',
            'height', 'realization'}
        result_coords = {coord.name() for coord in result.coords()}
        self.assertSetEqual(result_coords, expected_coords)


if __name__ == '__main__':
    unittest.main()
