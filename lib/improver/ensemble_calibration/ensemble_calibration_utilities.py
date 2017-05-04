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

    Parameters
    ----------
    forecast : Iris cube
        N-dimensional cube to be reshaped.
    coord : String
        The data will be flattened along this coordinate.
    transpose : Logical
        If True, the resulting flattened data is transposed.
        If False, the resulting flattened data is not transposed.

    Returns
    -------
    forecast_data : Numpy array
        Reshaped 2d array.

    """
    forecast_data = []
    for coord_slice in forecast.slices_over(coord):
        forecast_data.append(coord_slice.data.flatten())
    if transpose:
        forecast_data = np.asarray(forecast_data).T
    return np.array(forecast_data)


def concatenate_cubes(
        cubes, coords_to_slice_over=["realization", "time"],
        master_coord="time",
        coordinates_for_association=["forecast_reference_time",
                                     "forecast_period"]):
    """
    Function to concatenate cubes, accounting for differences in the
    history attribute, and allow promotion of forecast_reference_time
    and forecast_period coordinates from scalar coordinates to auxiliary
    coordinates to allow concatenation.

    Parameters
    ----------
    cubes : Iris cubelist or Iris cube
        Cubes to be concatenated.
    coords_to_slice_over : List
        Coordinates to be sliced over.
    master_coord : String
        Coordinate that the other coordinates will be associated with.
    coordinates_for_association : List
        List of coordinates to be associated with the master_coord.

    Returns
    -------
    Iris cube
        Concatenated cube.

    """
    if isinstance(cubes, iris.cube.Cube):
        cubes = iris.cube.CubeList([cubes])

    for coord_to_slice_over in coords_to_slice_over:
        cubes = _slice_over_coordinate(cubes, coord_to_slice_over)

    cubes = _strip_var_names(cubes)

    associated_with_time_cubelist = iris.cube.CubeList([])
    for cube in cubes:
        associated_with_time_cubelist.append(
            _associate_any_coordinate_with_master_coordinate(
                cube, master_coord=master_coord,
                coordinates=coordinates_for_association))
    return associated_with_time_cubelist.concatenate_cube()


def _associate_any_coordinate_with_master_coordinate(
        cube, master_coord="time", coordinates=None):
    """
    Function to convert the given coordinates from scalar coordinates to
    auxiliary coordinates, where these auxiliary coordinates will be
    associated with the master coordinate.

    For example, forecast_reference_time and forecast_period can be converted
    from scalar coordinates to auxiliary coordinates, and associated with time.

    Parameters
    ----------
    cube : Iris cube
        Cube requiring addition of the specified coordinates as auxiliary
        coordinates.
    master_coord : String
        Coordinate that the other coordinates will be associated with.
    coordinates : None or List
        List of coordinates to be associated with the master_coord.

    Returns
    -------
    forecast_data : Iris cube
        Cube where the the requested coordinates have been added to the cube
        as auxiliary coordinates and associated with the desired master
        coordinate.

    """
    if coordinates is None:
        coordinates = []
    for coord in coordinates:
        if cube.coords(coord):
            if cube.coords(master_coord):
                temp_coord = cube.coord(coord)
                cube.remove_coord(coord)
                temp_aux_coord = iris.coords.AuxCoord(
                    temp_coord.points,
                    standard_name=temp_coord.standard_name,
                    long_name=temp_coord.long_name,
                    var_name=temp_coord.var_name, units=temp_coord.units,
                    bounds=temp_coord.bounds,
                    attributes=temp_coord.attributes,
                    coord_system=temp_coord.coord_system)
                coord_names = [
                    coord.standard_name for coord in cube.dim_coords]
                cube.add_aux_coord(
                    temp_aux_coord,
                    data_dims=coord_names.index(master_coord))
            else:
                msg = (
                    "The master coordinate for associating other " +
                    "coordinates with is not present: " +
                    "master_coord: {}, other coordinates: {}".format(
                        master_coord, coordinates))
                raise ValueError(msg)
    return cube


def _slice_over_coordinate(cubes, coord_to_slice_over, remove_history=True):
    """
    Function slice over the requested coordinate,
    promote the sliced coordinate into a dimension coordinate and
    remove the history attribute to help concatenation.

    Parameters
    ----------
    cubes : Iris cubelist or Iris cube
        Cubes to be concatenated.
    coords_to_slice_over : List
        Coordinates to be sliced over.
    remove_history : Logical
        Option to remove the history attribute to help make concatenation
        more likely. remove_history is set to True as default.

    Returns
    -------
    Iris CubeList
        CubeList containing sliced cubes.

    """
    sliced_by_coord_cubelist = iris.cube.CubeList([])
    if isinstance(cubes, iris.cube.Cube):
        cubes = iris.cube.CubeList([cubes])
    for cube in cubes:
        if cube.coords(coord_to_slice_over):
            for coord_slice in cube.slices_over(coord_to_slice_over):
                coord_slice = iris.util.new_axis(
                    coord_slice, coord_to_slice_over)
                if (remove_history and
                        "history" in coord_slice.attributes.keys()):
                    coord_slice.attributes.pop("history")
                sliced_by_coord_cubelist.append(coord_slice)
        else:
            sliced_by_coord_cubelist.append(cube)
    return sliced_by_coord_cubelist


def _strip_var_names(cubes):
    """
    Strips var_name from the cube and from all coordinates
    to help concatenation.

    Parameters
    ----------
    cubes : Iris cubelist or Iris cube
        Cubes to be concatenated.

    Returns
    -------
    Iris CubeList
        CubeList containing original cubes without a var_name on the cube,
        or on the coordinates.

    """
    if isinstance(cubes, iris.cube.Cube):
        cubes = iris.cube.CubeList([cubes])
    for cube in cubes:
        cube.var_name = None
        for coord in cube.coords():
            coord.var_name = None
    return cubes


def rename_coordinate(cubes, original_coord, renamed_coord):
    """
    Renames a coordinate to an alternative name for an
    input Iris Cube or Iris CubeList.

    Parameters
    ----------
    cubes : Iris cubelist or Iris cube
        Cubes with coordinates to be renamed.
    original_coord : String
        Original name for the coordinate.
    renamed_coord : String
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

    Parameters
    ----------
    cube : Iris cube
        Cube with coordinates to be renamed.
    original_coord : String
        Original name for the coordinate.
    renamed_coord : String
        Name for the coordinate to be renamed to.

    """
    if cube.coords(original_coord):
        cube.coord(original_coord).rename(renamed_coord)


def check_predictor_of_mean_flag(predictor_of_mean_flag):
    """
    Check the predictor_of_mean_flag at the start of the
    estimate_coefficients_for_ngr method, to avoid having to check
    and raise an error later.

    Parameters
    ----------
    predictor_of_mean_flag : String
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
