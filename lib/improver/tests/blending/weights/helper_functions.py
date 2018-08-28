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
"""Functions to set up cubes for use in weighting-related unit tests."""

import numpy as np

import iris

from improver.tests.ensemble_calibration.ensemble_calibration.helper_functions\
    import add_forecast_reference_time_and_forecast_period, set_up_cube


def set_up_zero_cube():
    """A helper function to set up input cubes for unit tests.
       The cube has latitude, longitude and time dimensions

    Returns:
        cube (iris.cube.Cube):
            dummy cube for testing

    """
    data = np.zeros((1, 2, 2, 2))
    cube = set_up_cube(
        data, standard_name="lwe_thickness_of_precipitation_amount",
        units="m", realizations=[0], timesteps=2, y_dimension_length=2,
        x_dimension_length=2)
    for cube_slice in cube.slices_over("realization"):
        cube_slice.remove_coord("realization")
    cube = add_forecast_reference_time_and_forecast_period(
        cube_slice, time_point=[412227, 412228.0], fp_point=[0, 1.])
    return cube


def set_up_cube_with_scalar_coord():
    """A helper function with a dummy scalar coordinate for testing.

    Returns:
        cube (iris.cube.Cube):
            Cube with dummy scalar coordinate.
    """
    cube = set_up_zero_cube()
    dummy_scalar_coord = (
        iris.coords.AuxCoord(1, long_name='scalar_coord', units='no_unit'))
    cube.add_aux_coord(dummy_scalar_coord)
    return cube


def set_up_precipitation_cube():
    """Set up a precipitation cube.

    Returns:
        cube (iris.cube.Cube):
            Precipitation cube with some pre-specified values in the cube data
            for testing.
    """
    cube = set_up_zero_cube()
    cube.data = np.zeros((2, 2, 2))
    cube.data[0][:][:] = 1.0
    cube.data[1][:][:] = 2.0
    return cube


def cubes_for_triangular_weighted_blend_tests():
    """Set up cubes for triangular weighted blend unit tests.

    Returns:
        cube (iris.cube.Cube):
            Cube for testing.
        central_cube (iris.cube.Cube):
            Cube containing a single forecast period.
        forecast_period (int):
            A forecast period.
    """
    cube = set_up_precipitation_cube()
    forecast_period = 0
    constr = iris.Constraint(forecast_period=forecast_period)
    central_cube = cube.extract(constr)
    return cube, central_cube, forecast_period


def set_up_temperature_cube(data=None, timesteps=3, realizations=None):
    """Create a cube with metadata and values suitable for air temperature.

     Keyword Args:
         data (np.ndarray):
             Data to use for creating the temperature cube.
         timesteps (int):
             Number of timesteps to create in cube.
         realizations (np.array):
             Values to be used as realizations.

    """
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


def set_up_basic_model_config_cube():
    """Set up cube with model configuration dimension.

    Returns:
        cube (iris.cube.Cube):
            Cube with model configuration dimension.
    """
    cube = add_model_id_and_model_configuration(
        set_up_temperature_cube(timesteps=3), model_ids=[1000],
        model_configurations=["uk_det"])
    cube = add_forecast_reference_time_and_forecast_period(
        cube, time_point=[412233.0, 412234.0, 412235.0],
        fp_point=[6., 7., 8.])
    return cube


def set_up_weights_cube(data=None, timesteps=3, realizations=None):
    """Create a cube with metadata and values suitable for weights.

     Keyword Args:
         data (np.ndarray):
             Data to use for creating the weights cube.
         timesteps (int):
             Number of timesteps to create in cube.
         realizations (np.array):
             Values to be used as realizations.

    """
    if realizations is None:
        realizations = [0]
    if data is None:
        data = np.zeros([1, timesteps, 2, 2])
    cube = set_up_cube(data, long_name="weights",
                       realizations=realizations, timesteps=timesteps,
                       y_dimension_length=2, x_dimension_length=2)
    return cube


def set_up_basic_weights_cube(
        model_ids=None, model_configurations=None, promote_to_new_axis=False):
    """Create a weights cube with additional coordinates (forecast period,
    forecast_reference_time, model_id, model_configuration).

    Keyword Args:
        model_ids (list):
            List of model ids.
        model_configurations (list):
            List of model configurations.
        promote_to_new_axis (bool):
            Specify whether the model_id coordinate should be promoted to
            be a dimension coordinate.

    Returns:
        weights_cube (iris.cube.Cube):
            Cube containing weights.
    """
    if model_ids is None:
        model_ids = [1000]
    if model_configurations is None:
        model_configurations = ["uk_det"]

    weights_cube = set_up_weights_cube(timesteps=4)
    weights_cube = add_forecast_reference_time_and_forecast_period(
        weights_cube, time_point=[412234.0, 412239.0, 412275.0, 412281.0],
        fp_point=[7., 12., 48., 54.])
    weights_cube.data[:, 1:3] = np.ones([1, 2, 2, 2])
    weights_cube = add_model_id_and_model_configuration(
        weights_cube, model_ids=model_ids,
        model_configurations=model_configurations,
        promote_to_new_axis=promote_to_new_axis)
    return weights_cube


def add_model_id_and_model_configuration(
        cube, model_ids=None, model_configurations=None,
        promote_to_new_axis=False):
    """Add model id and model configuration coordinates to an input cube.

    Args:
        cube (iris.cube.Cube):
            Cube to which the model id and model configuration coordinate
            will be added.

    Keyword Args:
        model_ids (list):
            List of the model ids to add to the input cube.
        model_configurations (list):
            List of the model configurations to add to the input cube.
        promote_to_new_axis (bool):
            Specify whether the model_id coordinate should be promoted to
            be a dimension coordinate.

    Returns:
        cubelist.concatenate_cube():
            Cube with a model id and model configuration coordinate added.
    """
    if model_ids is None:
        model_ids = [1000, 2000]
    if model_configurations is None:
        model_configurations = ["uk_det", "uk_ens"]

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
    return cubelist.concatenate_cube()


def add_height(cube, heights):
    """Add a heights coordinate to the input cube.

    Args:
        cube (iris.cube.Cube):
            Cube to which a height coordinate will be added.
        heights (list or np.ndarray):
            List of heights.

    Returns:
        cubelist.merge_cube():
            Cube with a height coordinate added.
    """
    cubelist = iris.cube.CubeList([])
    for height in heights:
        cube_copy = cube.copy()
        height_coord = iris.coords.AuxCoord(
            height, long_name='height', units="m")
        cube_copy.add_aux_coord(height_coord)
        cubelist.append(cube_copy)
    return cubelist.merge_cube()


def add_realizations(cube, num):
    """Create num realizations of input cube.
        Args:
            cube (iris.cube.Cube):
                   input cube.
            num (integer):
                   Number of realizations.
        Returns:
            cubeout (iris.cube.Cube):
                      copy of cube with num realizations added.
    """
    cubelist = iris.cube.CubeList()
    for i in range(0, num):
        newcube = cube.copy()
        new_ensemble_coord = (
            iris.coords.AuxCoord(i, standard_name='realization'))
        newcube.add_aux_coord(new_ensemble_coord)
        cubelist.append(newcube)
    cubeout = cubelist.merge_cube()
    return cubeout
