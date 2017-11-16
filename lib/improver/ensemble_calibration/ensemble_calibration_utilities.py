# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
This module defines all the utilities used by the "plugins"
specific for ensemble calibration.

"""
import numpy as np

import iris


def convert_cube_data_to_2d(
        forecast, coord="realization", transpose=True):
    """
    Function to convert data from a N-dimensional cube into a 2d
    numpy array. The result can be transposed, if required.

    Args:
        forecast (iris.cube.Cube):
            N-dimensional cube to be reshaped.
        coord (string):
            The data will be flattened along this coordinate.
        transpose (boolean):
            If True, the resulting flattened data is transposed.
            This will transpose a 2d array of the format [:, coord]
            to [coord, :].
            If False, the resulting flattened data is not transposed.
            This will result in a 2d array of format [:, coord].

    Returns:
        forecast_data (numpy.array):
            Reshaped 2d array.

    """
    forecast_data = []
    for coord_slice in forecast.slices_over(coord):
        forecast_data.append(coord_slice.data.flatten())
    if transpose:
        forecast_data = np.asarray(forecast_data).T
    return np.array(forecast_data)


def ensure_dimension_is_the_zeroth_dimension(cube, coord):
    """
    Function to ensure that the requested coordinate within the cube is
    the first dimension within the cube.

    If the requested dimension coordinate exists, the cube is transposed.
    If the requested coordinate exists, but it is not a dimension coordinate
    i.e. a scalar coordinate, then a new axis is created with the scalar
    coordinate becoming a dimension coordinate.
    If the coordinate is not present on the cube, then an error is raised.

    Args:
        cube (iris.cube.Cube):
            Cube where the requirement for the required dimension to be the
            first dimension will be enforced.
        coord (string):
            Name of the coordinate that is to be made the first dimension
            coordinate in the cube.

    """
    if cube.coords(coord, dim_coords=True):
        if cube.coord_dims(coord)[0] != 0:
            coords = []
            for acoord in cube.coords(dim_coords=True):
                if acoord.name() not in [coord]:
                    coords.append(cube.coord_dims(acoord)[0])
            first_coord = cube.coord_dims(coord)[0]
            cube.transpose([first_coord]+coords)
    elif cube.coords(coord, dim_coords=False):
        cube = iris.util.new_axis(cube, coord)
    else:
        msg = ("The coordinate {} is not a dimension coordinate "
               "in the cube: {}".format(coord, cube))
        raise ValueError(msg)
    return cube


def rename_coordinate(cubes, original_coord, renamed_coord):
    """
    Renames a coordinate to an alternative name for an
    input Iris Cube or Iris CubeList.

    Args:
        cubes (iris.cube.CubeList or iris.cube.Cube):
            Cubes with coordinates to be renamed.
        original_coord (string):
            Original name for the coordinate.
        renamed_coord (string):
            Name for the coordinate to be renamed to.

    """
    if isinstance(cubes, iris.cube.Cube):
        _renamer(cubes, original_coord, renamed_coord)
    elif isinstance(cubes, iris.cube.CubeList):
        for cube in cubes:
            _renamer(cube, original_coord, renamed_coord)
    else:
        msg = ("A Cube or CubeList is not provided for renaming "
               "{} to {}. Variable provided "
               "is of type: {}".format(
                   original_coord, renamed_coord, type(cubes)))
        raise TypeError(msg)


def _renamer(cube, original_coord, renamed_coord):
    """
    Renames a coordinate to an alternative name.
    If the coordinate is not found within the cube, then the
    original cube is returned.

    Args:
        cube (iris.cube.Cube):
            Cube with coordinates to be renamed.
        original_coord (string):
            Original name for the coordinate.
        renamed_coord (string):
            Name for the coordinate to be renamed to.

    """
    if cube.coords(original_coord):
        cube.coord(original_coord).rename(renamed_coord)


def check_predictor_of_mean_flag(predictor_of_mean_flag):
    """
    Check the predictor_of_mean_flag at the start of the
    estimate_coefficients_for_ngr method, to avoid having to check
    and raise an error later.

    Args:
        predictor_of_mean_flag (string):
            String to specify the input to calculate the calibrated mean.
            Currently the ensemble mean ("mean") and the ensemble members
            ("members") are supported as the predictors.

    """
    if predictor_of_mean_flag.lower() not in ["mean", "members"]:
        msg = ("The requested value for the predictor_of_mean_flag {}"
               "is not an accepted value."
               "Accepted values are 'mean' or 'members'").format(
                   predictor_of_mean_flag.lower())
        raise ValueError(msg)
