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
from iris.tests import IrisTest
import numpy as np

from improver.blending.weights import ChooseWeightsLinearFromCube
from improver.tests.blending.weights.helper_functions import (
    set_up_temperature_cube, set_up_basic_model_config_cube,
    set_up_weights_cube, set_up_basic_weights_cube,
    add_model_id_and_model_configuration, add_height)
from improver.tests.ensemble_calibration.ensemble_calibration.helper_functions\
    import add_forecast_reference_time_and_forecast_period


class Test__init__(IrisTest):
    """Test the __init__ method."""

    def test_basic(self):
        """Test the class is initialised correctly."""
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinearFromCube(
            weighting_coord_name, config_coord_name)
        self.assertEqual(weighting_coord_name, plugin.weighting_coord_name)
        self.assertEqual(config_coord_name, plugin.config_coord_name)


class Test__repr__(IrisTest):
    """Test the __repr__ method."""

    def test_basic(self):
        """Test the repr function formats the arguments correctly"""
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinearFromCube(
            weighting_coord_name, config_coord_name)
        result = str(plugin)
        expected = ("<ChooseWeightsLinearFromCube "
                    "weighting_coord_name = forecast_period, "
                    "config_coord_name = model_configuration>")
        self.assertEqual(result, expected)


class Test__check_weights_cubes(IrisTest):
    """Test the _check_weights_cubes method."""

    def test_exception_not_raised(self):
        """An exception is not raised when the length of the weights_cubes
        cubelist is the same as the number of points along the
        config_coord_name dimension within the input cube."""
        cube = set_up_basic_model_config_cube()
        weights_cubes = iris.cube.CubeList([set_up_basic_weights_cube()])
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinearFromCube(
            weighting_coord_name, config_coord_name)
        self.assertIsNone(plugin._check_weights_cubes(cube, weights_cubes))

    def test_exception_raised(self):
        """An exception is raised when the length of the weights_cubes
        cubelist is different to the number of points along the
        config_coord_name dimension within the input cube."""
        cube = set_up_basic_model_config_cube()
        weights_cubes = iris.cube.CubeList([])
        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinearFromCube(
            weighting_coord_name, config_coord_name)
        msg = ('The coordinate used to configure the weights')
        with self.assertRaisesRegex(ValueError, msg):
            plugin._check_weights_cubes(cube, weights_cubes)


class Test__arrange_interpolation_inputs(IrisTest):
    """Test the _arrange_interpolation_inputs function. """

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

        plugin = ChooseWeightsLinearFromCube(
            weighting_coord_name, config_coord_name)
        source_points, target_points, source_weights, axis, fill_value = (
            plugin._arrange_interpolation_inputs(cube, weights_cube))
        self.assertArrayAlmostEqual(source_points, expected_source_points)
        self.assertArrayAlmostEqual(target_points, expected_target_points)
        self.assertArrayAlmostEqual(source_weights, expected_source_weights)
        self.assertEqual(axis, expected_axis)
        self.assertArrayAlmostEqual(fill_value[0], expected_fill_value[0])
        self.assertArrayAlmostEqual(fill_value[1], expected_fill_value[1])


class Test__create_coord_and_dims_list(IrisTest):
    """Test the _create_coord_and_dims_list method."""

    def test_dim_coords(self):
        """Test that the expected list of coordinates is returned when the
        dimension coordinates are checked."""
        cube = set_up_basic_model_config_cube()
        weights_cube = set_up_basic_weights_cube()

        expected_coord_list = [(weights_cube.coord("realization"), 0),
                               (cube.coord("time"), 1),
                               (weights_cube.coord("latitude"), 2),
                               (weights_cube.coord("longitude"), 3)]

        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinearFromCube(
            weighting_coord_name, config_coord_name)
        new_coord_list = plugin._create_coord_and_dims_list(
            weights_cube, cube, weights_cube.dim_coords, weighting_coord_name)
        self.assertEqual(new_coord_list, expected_coord_list)

    def test_aux_coords(self):
        """Test that the expected list of coordinates is returned when the
        dimension coordinates are checked."""
        cube = set_up_basic_model_config_cube()
        weights_cube = set_up_basic_weights_cube()

        expected_coord_list = [
            (weights_cube.coord("forecast_reference_time"), None),
            (weights_cube.coord("model_configuration"), None),
            (weights_cube.coord("model_id"), None),
            (cube.coord("forecast_period"), 1)]

        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinearFromCube(
            weighting_coord_name, config_coord_name)
        new_coord_list = plugin._create_coord_and_dims_list(
            weights_cube, cube, weights_cube.aux_coords, weighting_coord_name)
        self.assertEqual(new_coord_list, expected_coord_list)


class Test__create_new_weights_cube(IrisTest):
    """Test the _create_new_weights_cube function. """

    def test_with_weights_cube(self):
        """Test that the the expected cube containg the new weights is
        returned."""
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
        plugin = ChooseWeightsLinearFromCube(
            weighting_coord_name, config_coord_name)
        new_weights_cube = plugin._create_new_weights_cube(
            cube, weights, weights_cube)
        self.assertArrayAlmostEqual(expected_weights, new_weights_cube.data)
        self.assertEqual(weights_cube.metadata, new_weights_cube.metadata)


class Test__interpolate_to_create_weights(IrisTest):
    """Test the _interpolate_to_create_weights function. """

    def test_below_range(self):
        """Test that interpolation works as intended when the forecast period
        required for the interpolation output is below the range specified
        within the inputs."""
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
        plugin = ChooseWeightsLinearFromCube(
            weighting_coord_name, config_coord_name)
        new_weights_cube = (
            plugin._interpolate_to_create_weights(cube, weights_cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data, expected_weights)
        self.assertAlmostEqual(new_weights_cube.metadata,
                               weights_cube.metadata)

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
        weights_cube = set_up_basic_weights_cube()

        expected_weights = np.array([[[[0.8, 0.8],
                                       [0.8, 0.8]],
                                      [[1., 1.],
                                       [1., 1.]],
                                      [[1., 1.],
                                       [1., 1.]]]])

        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinearFromCube(
            weighting_coord_name, config_coord_name)
        new_weights_cube = (
            plugin._interpolate_to_create_weights(cube, weights_cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data, expected_weights)
        self.assertAlmostEqual(new_weights_cube.metadata,
                               weights_cube.metadata)

    def test_above_range(self):
        """Test that interpolation works as intended when the forecast period
        required for the interpolation output is above the range specified
        within the inputs."""
        cube = add_model_id_and_model_configuration(
            set_up_temperature_cube(timesteps=3), model_ids=[1000],
            model_configurations=["uk_det"])
        cube = add_forecast_reference_time_and_forecast_period(
            cube, time_point=[412280.0, 412281.0, 412282.0],
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
        plugin = ChooseWeightsLinearFromCube(
            weighting_coord_name, config_coord_name)
        new_weights_cube = (
            plugin._interpolate_to_create_weights(cube, weights_cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data, expected_weights)
        self.assertAlmostEqual(new_weights_cube.metadata,
                               weights_cube.metadata)

    def test_spatial_varying_weights(self):
        """Test that interpolation works as intended when the weights vary
        spatially."""
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

        weighting_coord_name = "forecast_period"
        config_coord_name = "model_configuration"
        plugin = ChooseWeightsLinearFromCube(
            weighting_coord_name, config_coord_name)
        new_weights_cube = (
            plugin._interpolate_to_create_weights(cube, weights_cube))
        self.assertIsInstance(new_weights_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_weights_cube.data, expected_weights)
        self.assertAlmostEqual(new_weights_cube.metadata,
                               weights_cube.metadata)


class Test_process(IrisTest):
    """Test the process method."""

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
            cube, time_point=[412235.0, 412247.0, 412278],
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

        plugin = ChooseWeightsLinearFromCube(
            weighting_coord_name, config_coord_name)
        result = plugin.process(cube, weights_cubes)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.metadata,
                               weights_cubes[0].metadata)

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
            cube, time_point=[412235.0, 412247.0, 412278],
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

        plugin = ChooseWeightsLinearFromCube(
            weighting_coord_name, config_coord_name)
        result = plugin.process(cube, weights_cubes)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.metadata,
                               weights_cubes[0].metadata)

    def test_height_and_model_configuration(self):
        """Test when height is the weighting_coord_name and
        model_configuration is the config_coord_name. This demonstrates
        blending in one model and blending out another model with height."""
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

        plugin = ChooseWeightsLinearFromCube(
            weighting_coord_name, config_coord_name)
        result = plugin.process(cube, weights_cubes)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.metadata,
                               weights_cubes[0].metadata)

    def test_height_and_realization(self):
        """Test when height is the weighting_coord_name and realization is the
        config_coord_name. This demonstrates blending in one model and
        blending out another model with height."""
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

        plugin = ChooseWeightsLinearFromCube(
            weighting_coord_name, config_coord_name)
        result = plugin.process(cube, weights_cubes)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_weights)
        self.assertAlmostEqual(result.metadata,
                               weights_cubes[0].metadata)


if __name__ == '__main__':
    unittest.main()
