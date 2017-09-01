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
""" Provides support utilities for cube manipulation."""

import warnings
import numpy as np

import iris
from iris.coords import AuxCoord, DimCoord


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
                temp_aux_coord = (
                    build_coordinate(temp_coord.points,
                                     bounds=temp_coord.bounds,
                                     coord_type=AuxCoord,
                                     template_coord=temp_coord))
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


def _slice_over_coordinate(cubes, coord_to_slice_over):
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


def concatenate_cubes(
        cubes, coords_to_slice_over=None, master_coord="time",
        coordinates_for_association=None):
    """
    Function to concatenate cubes, accounting for differences in the
    history attribute, and allow promotion of forecast_reference_time
    and forecast_period coordinates from scalar coordinates to auxiliary
    coordinates to allow concatenation.

    Args:
    cubes : Iris cubelist or Iris cube
        Cubes to be concatenated.
    coords_to_slice_over : List
        Coordinates to be sliced over.
    master_coord : String
        Coordinate that the other coordinates will be associated with.
    coordinates_for_association : List
        List of coordinates to be associated with the master_coord.

    Returns:
    result: Iris cube
        Concatenated / merge cube.

    """
    if coords_to_slice_over is None:
        coords_to_slice_over = ["realization", "time"]
    if coordinates_for_association is None:
        coordinates_for_association = ["forecast_reference_time",
                                       "forecast_period"]
    if isinstance(cubes, iris.cube.Cube):
        cubes = iris.cube.CubeList([cubes])

    for coord_to_slice_over in coords_to_slice_over:
        cubes = _slice_over_coordinate(cubes, coord_to_slice_over)

    cubes = equalise_cubes(cubes, merging=False)

    associated_with_time_cubelist = iris.cube.CubeList([])
    for cube in cubes:
        associated_with_time_cubelist.append(
            _associate_any_coordinate_with_master_coordinate(
                cube, master_coord=master_coord,
                coordinates=coordinates_for_association))

    result = associated_with_time_cubelist.concatenate_cube()
    return result


def merge_cubes(cubes):
    """
    Function to merge cubes, accounting for differences in the
    attributes, and coords.

    Args:
    cubes : Iris cubelist or Iris cube
        Cubes to be merged.

    Returns:
    result : Iris cube
        Merged cube.

    """
    if isinstance(cubes, iris.cube.Cube):
        cubes = iris.cube.CubeList([cubes])

    cubelist = equalise_cubes(cubes)

    result = cubelist.merge_cube()
    return result


def equalise_cubes(cubes, merging=True):
    """
    Function to equalise attributes that do not match.

    Args:
        cubes : Iris cubelist
            List of cubes to check and equalise.
        merging : boolean
            Flag for whether the equalising is for merging
            as slightly different processing is required.
    Returns:
        cubelist : Iris cubelist
            List of cubes with revised cubes.
    """
    # Additional processing to deal with differences
    # between cubes e.g. different cell_methods
    # will probably need to be added but this function is
    # mainly here to provide the structure for where
    # these changes can be added.
    equalise_cube_attributes(cubes)
    _strip_var_names(cubes)
    if merging:
        cubelist = equalise_cube_coords(cubes)
    else:
        cubelist = cubes
    return cubelist


def equalise_cube_attributes(cubes):
    """
    Function to equalise attributes that do not match.

    Args:
        cubes : Iris cubelist
            List of cubes to check the attributes and revise.
    Warn:
        Warning: If it does not know what to do with an unmatching
                 attribute. Default is to delete it.
    """
    unmatching_attributes = compare_attributes(cubes)
    if len(unmatching_attributes) > 0:
        for i, cube in enumerate(cubes):
            # Remove history.
            if "history" in unmatching_attributes[i]:
                cube.attributes.pop("history")
                unmatching_attributes[i].pop("history")
            # Normalise grid_id to ukx_standard_1
            if "grid_id" in unmatching_attributes[i]:
                if cube.attributes['grid_id'] in ['enukx_standard_v1',
                                                  'ukvx_standard_v1']:
                    cube.attributes['grid_id'] = 'ukx_standard_v1'
                    unmatching_attributes[i].pop("grid_id")
            # Add model_id if titles do not match.
            if "title" in unmatching_attributes[i]:
                    model_title = cube.attributes.pop('title')
                    new_model_id_coord = build_coordinate([100*i],
                                                          long_name='model_id',
                                                          data_type=np.int)
                    new_model_coord = build_coordinate([model_title],
                                                       long_name='model',
                                                       coord_type=AuxCoord,
                                                       data_type=np.str)
                    cube.add_aux_coord(new_model_id_coord)
                    cube.add_aux_coord(new_model_coord)
                    unmatching_attributes[i].pop("title")
            # Remove any other mismatching attributes but raise warning.
            if len(unmatching_attributes[i]) != 0:
                for key in unmatching_attributes[i]:
                    msg = ('Do not know what to do with ' + key +
                           ' will delete it')
                    warnings.warn(msg)
                    cube.attributes.pop(key)


def equalise_cube_coords(cubes):
    """
    Function to equalise coordinates that do not match.

    Args:
        cubes : Iris cubelist
            List of cubes to check the coords and revise.
    Returns:
        cubelist : Iris cubelist
            List of cubes with revised coords.
    Raises:
        If Percentile coordinates do not match.
        If Threshold coordinates do not match.
        If model_id has more than one point.
    """
    unmatching_coords = compare_coords(cubes)
    cubelist = iris.cube.CubeList([])
    if len(unmatching_coords) > 0:
        for i, cube in enumerate(cubes):
            slice_over_keys = []
            for key in unmatching_coords[i]:
                # mismatching percentiles - raise error.
                if key.find('percentile_over') > 0:
                    msg = ("Percentile coordinates "
                           "must match to merge")
                    raise ValueError(msg)
                # mismatching thresholds - raise error.
                if key == 'threshold':
                    msg = ("Threshold coordinates "
                           "must match to merge")
                    raise ValueError(msg)
                # mismatching model id
                if key == 'model_id':
                    realization_found = False
                    # Check to see if there is a mismatch realization coord.
                    for j, check in enumerate(unmatching_coords):
                        if 'realization' in check:
                            realization_found = True
                            realization_coord = cubes[j].coord('realization')
                            break
                    # If there is add model_realization coord
                    # and realization coord if necessary.
                    if realization_found:
                        if len(cube.coord('model_id').points) != 1:
                            msg = ("Threshold coordinates "
                                   "must match to merge")
                            raise ValueError(msg)
                        else:
                            model_id_val = cube.coord('model_id').points[0]
                        if cube.coords('realization'):
                            unmatch = unmatching_coords[i]['realization']
                            if unmatch['data_dims'] >= 0:
                                data_dims = (
                                    unmatch['data_dims'])
                            else:
                                data_dims = None
                            new_model_real_coord = (
                                build_coordinate(
                                    cube.coord('realization').points +
                                    model_id_val,
                                    long_name='model_realization'))
                            cube.add_aux_coord(new_model_real_coord,
                                               data_dims=data_dims)
                        else:
                            new_model_real_coord = (
                                build_coordinate(
                                    [model_id_val],
                                    long_name='model_realization'))
                            cube.add_aux_coord(new_model_real_coord)
                            new_realization_coord = (
                                build_coordinate(
                                    [0],
                                    template_coord=realization_coord))
                            cube.add_aux_coord(new_realization_coord)
                # if mismatching is a dimension coord add to list to
                # slice over.
                if unmatching_coords[i][key]['data_dims'] >= 0:
                    slice_over_keys.append(key)

            if len(slice_over_keys) > 0:
                for slice_cube in cube.slices_over(slice_over_keys):
                    cubelist.append(slice_cube)
            else:
                cubelist.append(cube)
    else:
        cubelist = cubes
    return cubelist


def compare_attributes(cubes):
    """
    Function to compare attributes of cubes

    Args:
        cubes : Iris cubelist
            List of cubes to compare (must be more than 1)

    Returns:
        unmatching_attribues : List
            List of dictionaries of unmatching attributes

    Warns:
        Warning: If only a single cube is supplied
    """
    common_keys = list(cubes[0].attributes.keys())
    unmatching_attributes = []
    if isinstance(cubes, iris.cube.Cube) or len(cubes) == 1:
        msg = ('Only a single cube so no differences will be found ')
        warnings.warn(msg)
    else:
        for cube in cubes[1:]:
            cube_keys = list(cube.attributes.keys())
            common_keys = [
                key for key in common_keys
                if (key in cube_keys and
                    np.all(cube.attributes[key] == cubes[0].attributes[key]))]

        for i, cube in enumerate(cubes):
            unmatching_attributes.append(dict())
            for key in list(cube.attributes.keys()):
                if key not in common_keys:
                    unmatching_attributes[i].update({key:
                                                     cube.attributes[key]})
    return unmatching_attributes


def compare_coords(cubes):
    """
    Function to compare attributes of cubes

    Args:
        cubes : Iris cubelist
            List of cubes to compare (must be more than 1)

    Returns:
        unmatching_coords : List
            List of dictionaries of unmatching coordinates

    Warns:
        Warning: If only a single cube is supplied
    """
    common_coords = list(cubes[0].coords())
    unmatching_coords = []
    if isinstance(cubes, iris.cube.Cube) or len(cubes) == 1:
        msg = ('Only a single cube so no differences will be found ')
        warnings.warn(msg)
    else:
        for cube in cubes[1:]:
            cube_coords = list(cube.coords())
            common_coords = [
                coord for coord in common_coords
                if (coord in cube_coords and
                    np.all(cube.coords(coord) == cubes[0].coords(coord)))]

        for i, cube in enumerate(cubes):
            unmatching_coords.append(dict())
            for coord in list(cube.coords()):
                if coord not in common_coords:
                    dim_coords = cube.dim_coords
                    if coord in dim_coords:
                        dim_val = dim_coords.index(coord)
                    else:
                        dim_val = -1
                    unmatching_coords[i].update({coord.name():
                                                 {'data_dims': dim_val,
                                                  'coord': coord}})

    return unmatching_coords


def build_coordinate(data, long_name=None,
                     standard_name=None,
                     var_name=None,
                     coord_type=DimCoord,
                     data_type=None,
                     units='1',
                     bounds=None,
                     coord_system=None,
                     template_coord=None,
                     custom_function=None):
    """
    Construct an iris.coord.Dim/Auxcoord using the provided options.

    Args:
    -----
    data : number/list/np.array
        List or array of values to populate the coordinate points.
    long_name : str (optional)
        Name of the coordinate to be built.
    standard_name : str (optional)
        CF Name of the coordinate to be built.
    var_name : str (optional)
        Variable name
    coord_type : iris.coord.AuxCoord or iris.coord.DimCoord (optional)
        Selection between Dim and Aux coord.
    data_type : <type> (optional)
        The data type of the coordinate points, e.g. int
    units : str (optional)
        String defining the coordinate units.
    bounds : np.array (optional)
        A (len(data), 2) array that defines coordinate bounds.
    coord_system: iris.coord_systems.<coord_system> (optional)
        A coordinate system in which the dimension coordinates are defined.
    template_coord : iris.coord
        A coordinate to copy.
    custom_function : function (optional)
        A function to apply to the data values before constructing the
        coordinate, e.g. np.nan_to_num.

    Returns:
    --------
    crd_out: iris coordinate
        Dim or Auxcoord as chosen.

    """
    long_name_out = long_name
    std_name_out = standard_name
    var_name_out = var_name
    coord_type_out = coord_type
    data_type_out = data_type
    units_out = units
    bounds_out = bounds
    coord_system_out = coord_system

    if template_coord is not None:
        if long_name is None:
            long_name_out = template_coord.long_name
        if standard_name is None:
            std_name_out = template_coord.standard_name
        if var_name is None:
            var_name_out = template_coord.var_name
        if isinstance(coord_type, DimCoord):
            coord_type_out = type(template_coord)
            print coord_type_out
        if data_type is None:
            data_type_out = type(template_coord.points[0])
        if units == '1':
            units_out = template_coord.units
        if coord_system is None:
            coord_system_out = template_coord.coord_system

    if data_type_out is None:
        data_type_out = float

    data = np.array(data, data_type_out)
    if custom_function is not None:
        data = custom_function(data)

    crd_out = coord_type(data, long_name=long_name_out,
                         standard_name=std_name_out,
                         var_name=var_name_out,
                         units=units_out,
                         coord_system=coord_system_out,
                         bounds=bounds_out)

    return crd_out
