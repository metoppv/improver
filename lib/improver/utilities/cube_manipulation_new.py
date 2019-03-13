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
""" Provides support utilities for cube manipulation."""

import operator
import six
import warnings
import numpy as np

import iris
from iris.coords import AuxCoord, DimCoord
from iris.exceptions import CoordinateNotFoundError
from improver.utilities.cube_checker import check_cube_coordinates


def _associate_any_coordinate_with_master_coordinate(
        cube, master_coord="time", coordinates=None):
    """
    Function to convert the given coordinates from scalar coordinates to
    auxiliary coordinates, where these auxiliary coordinates will be
    associated with the master coordinate.

    For example, forecast_reference_time and forecast_period can be converted
    from scalar coordinates to auxiliary coordinates, and associated with time.

    Args:
        cube (Iris cube):
            Cube requiring addition of the specified coordinates as auxiliary
            coordinates.
        master_coord (String):
            Coordinate that the other coordinates will be associated with.
        coordinates (None or List):
            List of coordinates to be associated with the master_coord.

    Returns:
        forecast_data (Iris cube):
            Cube where the the requested coordinates have been added to the
            cube as auxiliary coordinates and associated with the desired
            master coordinate.

    Raises:
        ValueError: If the master coordinate is not present on the cube.

    """
    if coordinates is None:
        coordinates = []

    if not cube.coords(master_coord):
        msg = (
            "The master coordinate for associating other "
            "coordinates with is not present: "
            "master_coord: {}, other coordinates: {}".format(
                master_coord, coordinates))
        raise ValueError(msg)

    # If the master_coord is not a dimension coordinate, then the other
    # coordinates cannot be associated with it.
    if len(cube.coords(master_coord, dim_coords=True)) > 0:
        for coord in coordinates:
            if cube.coords(coord):
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
    return cube


def _slice_over_coordinate(cubes, coord_to_slice_over):
    """
    Function slice over the requested coordinate,
    promote the sliced coordinate into a dimension coordinate
    to help concatenation.

    Args:
        cubes (Iris cubelist or Iris cube):
            Cubes to be concatenated.
        coords_to_slice_over (List):
            Coordinates to be sliced over.

    Returns:
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


def strip_var_names(cubes):
    """
    Strips var_name from the cube and from all coordinates
    to help concatenation.

    Args:
        cubes (Iris cubelist or Iris cube):
            Cubes to be concatenated.

    Returns:
        cubes (Iris CubeList):
            CubeList containing original cubes without a var_name on the cube,
            or on the coordinates.
            Note: This internal function modifies the incoming cubes
    """
    if isinstance(cubes, iris.cube.Cube):
        cubes = iris.cube.CubeList([cubes])
    for cube in cubes:
        cube.var_name = None
        for coord in cube.coords():
            coord.var_name = None
    return cubes


def concatenate_cubes(
        cubes_in, coords_to_slice_over=None, master_coord="time",
        coordinates_for_association=None):
    """
    Function to concatenate cubes, accounting for differences in the
    history attribute, and allow promotion of forecast_reference_time
    and forecast_period coordinates from scalar coordinates to auxiliary
    coordinates to allow concatenation.

    Args:
        cubes_in (Iris cubelist or Iris cube):
            Cubes to be concatenated.
        coords_to_slice_over (List):
            Coordinates to be sliced over.
        master_coord (String):
            Coordinate that the other coordinates will be associated with.
        coordinates_for_association (List):
            List of coordinates to be associated with the master_coord.

    Returns:
        result (Iris cube):
            Concatenated / merge cube.

    """
    if coords_to_slice_over is None:
        coords_to_slice_over = ["realization", "time"]
    if coordinates_for_association is None:
        coordinates_for_association = ["forecast_reference_time",
                                       "forecast_period"]
    if isinstance(cubes_in, iris.cube.Cube):
        cubes = iris.cube.CubeList([cubes_in.copy()])
    else:
        cubes = iris.cube.CubeList([])
        for cube in cubes_in:
            cubes.append(cube.copy())

    for coord_to_slice_over in coords_to_slice_over:
        cubes = _slice_over_coordinate(cubes, coord_to_slice_over)

    cubes = _equalise_cubes(cubes, merging=False)

    associated_master_cubelist = iris.cube.CubeList([])
    for cube in cubes:
        associated_master_cubelist.append(
            _associate_any_coordinate_with_master_coordinate(
                cube, master_coord=master_coord,
                coordinates=coordinates_for_association))

    result = associated_master_cubelist.concatenate_cube()
    return result


def merge_cubes(cubes, model_id_attr=None, blend_coord=None):
    """
    Function to merge cubes, accounting for differences in the
    attributes, and coords.

    Args:
        cubes (Iris cubelist or Iris cube):
            Cubes to be merged.

    Kwargs:
        model_id_attr (str):
            Name of cube attribute used to identify the model for grid
            blending or None.
        blend_coord (str):
            Name of coordinate over which merged cube is to be blended, or
            None. If "time", triggers bounds range checks to avoid passing
            eg mismatched accumulation periods into blending.

    Returns:
        result (Iris cube):
            Merged cube.

    """
    if isinstance(cubes, iris.cube.Cube):
        cubes = iris.cube.CubeList([cubes])

    cubelist = _equalise_cubes(
        cubes, model_id_attr=model_id_attr, merging=True)

    for i, cube in enumerate(cubelist):
        cubelist[i] = iris.util.squeeze(cube)

    result = cubelist.merge_cube()

    if blend_coord is not None and blend_coord == "time":
        # If bounds ranges did not match, "result" will not have a name
        # on the leading dimension, but will have time and fp as aux
        # coords.  Checking bounds ranges here raises a useful error,
        # rather than encountering a missing blend coord downstream.
        check_bounds_range_coords = ["time", "forecast_period"]
        _check_bounds_ranges(result, check_bounds_range_coords)

    return result


def _equalise_cubes(cubes_in, model_id_attr=None, merging=True):
    """
    Function to equalise cubes where they do not match.

    Args:
        cubes_in (Iris cubelist):
            List of cubes to check and equalise.

    Kwargs:
        model_id_attr (str):
            Name of cube attribute used to identify the model for grid
            blending or None.
        merging (bool):
            Flag for whether equalisation is for merging or concatenation.

    Returns:
        cubelist (Iris cubelist):
            List of cubes with revised cubes.
            If merging the number of cubes in cubelist
            may be greater than the original number
            of cubes as the original cubes will be sliced
            so that that they can be merged together.
            Merging can only create new coords not add
            to existing mismatching coords.
    """
    cubes = iris.cube.CubeList([])
    for cube in cubes_in:
        cubes.append(cube.copy())
    _equalise_cube_attributes(cubes, model_id_attr=model_id_attr)
    strip_var_names(cubes)
    if merging:
        cubelist = _equalise_cube_coords(cubes)
        cubelist = _equalise_cell_methods(cubelist)
        demote_float64_precision(cubelist)
    else:
        cubelist = cubes
    return cubelist


def _equalise_cube_attributes(cubes, model_id_attr=None):
    """
    Function to equalise attributes that do not match.

    Args:
        cubes (Iris cubelist):
            List of cubes to check the attributes and revise.
        model_id_attr (str):
            Name of cube attribute used to identify the model
            for grid blending or None.

    Returns:
        cubelist (Iris cubelist):
        Note: This internal function modifies the incoming cubes
    Warns:
        Warning: If it does not know what to do with an unmatching
                 attribute. Default is to delete it.
    """
    # Unmatched warnings matching one of the silent_attributes are deleted
    # without raising a warning message
    silent_attributes = ['history', 'title', 'mosg__grid_version']
    unmatching_attributes = compare_attributes(cubes)

    if len(unmatching_attributes) > 0:
        for i, cube in enumerate(cubes):
            # Remove ignored attributes.
            for attr in silent_attributes:
                if attr in unmatching_attributes[i]:
                    cube.attributes.pop(attr)
                    unmatching_attributes[i].pop(attr)

            # If a model_id_attr has been specified, assume we are trying to
            # grid blend and throw an error if the model_id_attr does not
            # match that on the cubes to be blended.
            if model_id_attr is not None and \
                    model_id_attr not in cube.attributes:
                msg = ('Cannot create model ID coordinate for grid blending '
                       'as the model ID attribute specified is not found '
                       'within the cube attributes')
                raise ValueError(msg)

            if model_id_attr in unmatching_attributes[i]:
                model_title = cube.attributes.pop(model_id_attr)
                cube.attributes[model_id_attr] = "blend"

                new_model_id_coord = build_coordinate([1000 * i],
                                                      long_name='model_id',
                                                      data_type=np.int)
                new_model_coord = (
                    build_coordinate([model_title],
                                     long_name='model_configuration',
                                     coord_type=AuxCoord,
                                     data_type=np.str))

                cube.add_aux_coord(new_model_id_coord)
                cube.add_aux_coord(new_model_coord)
                unmatching_attributes[i].pop(model_id_attr)

            # Remove any other mismatching attributes but raise warning.
            if len(unmatching_attributes[i]) != 0:
                for key in unmatching_attributes[i]:
                    msg = ('Do not know what to do with ' + key +
                           ' will delete it'
                           ' - value is {}'.format(cube.attributes[key]))
                    warnings.warn(msg)
                    cube.attributes.pop(key)
    return cubes


def _equalise_cube_coords(cubes):
    """
    Function to equalise coordinates that do not match.

    Args:
        cubes (Iris cubelist):
            List of cubes to check the coords and revise.

    Returns:
        cubelist (Iris cubelist):
            List of cubes with revised coords.
            The number of cubes in cubelist
            may be greater than the original number
            of cubes as the original cubes will be sliced
            so that that they can be merged together.
            Merging can only create new coords not add
            to existing mismatching coords.
            Note: This internal function modifies the incoming cubes
    Raises:
        ValueError: If coordinates in error_keys do not match.
        ValueError: If model_id has more than one point.
    """
    unmatching_coords = compare_coords(cubes)
    # If len = 0 then cubes is a cube,
    # otherwise there will be a dict (possible empty) for
    # each cube in cubes.

    if len(unmatching_coords) == 0:
        cubelist = cubes
    else:
        # Check for mismatches in dim coord bounds
        _check_coord_bounds(cubes)

        # Throw an error for specific coordinate mismatches
        error_keys = ['threshold']
        for error_key in error_keys:
            for key in ([keyval for cube_dict in unmatching_coords
                         for keyval in cube_dict]):
                if error_key in key:
                    msg = ("{} ".format(error_key) +
                           "coordinates must match to merge")
                    raise ValueError(msg)

        cubelist = iris.cube.CubeList([])
        for i, cube in enumerate(cubes):
            slice_over_keys = []
            for key in unmatching_coords[i]:
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
                            msg = ("Model_id has more than one point")
                            raise ValueError(msg)
                        else:
                            model_id_val = cube.coord('model_id').points[0]
                        if cube.coords('realization'):
                            unmatch = unmatching_coords[i]['realization']
                            data_dims = unmatch['data_dims']
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
                if unmatching_coords[i][key]['data_dims'] is not None:
                    slice_over_keys.append(key)
                if unmatching_coords[i][key]['aux_dims'] is not None:
                    slice_over_keys.append(key)

            if len(slice_over_keys) > 0:
                for slice_cube in cube.slices_over(slice_over_keys):
                    cubelist.append(slice_cube)
            else:
                cubelist.append(cube)

    return cubelist


def _check_coord_bounds(cubes):
    """
    Function to check for dimension coordinate bounds that do not match.
    If a coordinate is not present in all cubes, it is ignored.

    Args:
        cubes (iris.cube.CubeList):
            List of cubes to check the cell methods and revise.
            These are modified in place.

    Raises:
        ValueError:
            If some but not all cubes have bounds on a shared dimension
            coordinate.
        ValueError:
            If existing bounds values on shared dimension coordinates do not
            match.
    """
    # Check each cube against all remaining cubes
    msg = 'Cubes with mismatching {} bounds are not compatible'
    for i, this_cube in enumerate(cubes):
        for later_cube in cubes[i+1:]:
            for coord in this_cube.coords(dim_coords=True):
                try:
                    match_coord = later_cube.coord(coord)
                except CoordinateNotFoundError:
                    continue
                if coord.bounds is None and match_coord.bounds is None:
                    continue
                elif coord.bounds is None and match_coord.bounds is not None:
                    raise ValueError(msg.format(coord.name()))
                elif coord.bounds is not None and match_coord.bounds is None:
                    raise ValueError(msg.format(coord.name()))
                else:
                    if np.allclose(np.array(coord.bounds),
                                   np.array(match_coord.bounds)):
                        continue
                    else:
                        raise ValueError(msg.format(coord.name()))


def _check_bounds_ranges(cube, coord_list):
    """
    Check the bounds ranges on a given list of coordinates match at each
    point along that dimension.  For example: to check time and forecast period
    ranges for accumulations to avoid blending 1 hr with 3 hr accumulations.

    Args:
        cube (iris.cube.Cube):
            Input cube with dimensional coordinates in coord_list
        coord_list (list):
            List of string coordinate names to check bounds

    Raises:
        ValueError:
            If bounds ranges at different points are incompatible
    """
    for name in coord_list:
        coord = cube.coord(name)
        if coord.bounds is None:
            continue
        if len(coord.points) == 1:
            continue

        bounds_ranges = np.abs(np.diff(coord.bounds))
        reference_range = bounds_ranges[0]
        if not np.all(np.isclose(bounds_ranges, reference_range)):
            msg = ('Cube with mismatching {} bounds ranges '
                   'cannot be blended'.format(name))
            raise ValueError(msg)


def _equalise_cell_methods(cubes):
    """
    Function to equalise cell methods that do not match.

    Args:
        cubes (iris.cube.CubeList):
            List of cubes to check the cell methods and revise.
    Returns:
        cubelist (iris.cube.CubeList):
            List of cubes with revised cell methods.
            Currently the cell methods are simply deleted if
            they do not match.
    Warns:
        Warning: If only a single cube.

    """
    if len(cubes) == 1:
        msg = ('Only a single cube so no differences will be found '
               'in cell methods')
        warnings.warn(msg)
        cubelist = cubes
    else:
        cell_methods = cubes[0].cell_methods
        for cube in cubes[1:]:
            cell_methods = list(set(cell_methods) & set(cube.cell_methods))
        cubelist = cubes
        for cube in cubelist:
            cube.cell_methods = tuple(cell_methods)
    return cubelist


def get_filtered_attributes(cube, attribute_filter=None):
    """
    Build dictionary of attributes that match the attribute_filter. If the
    attribute_filter is None, return all attributes.

    Args:
        cube (iris.cube.Cube):
            A cube from which attributes partially matching the
            attribute_filter will be returned.
    Keyword Args:
        attribute_filter (string or None):
            A string to match, or partially match, against attributes to build
            a filtered attribute dictionary. If None, all attributes are
            returned.
    Returns:
        attributes (dict):
            A dictionary of attributes partially matching the attribute_filter
            that were found on the input cube.
    """
    attributes = cube.attributes
    if attribute_filter is not None:
        attributes = {k: v for (k, v) in attributes.items()
                      if attribute_filter in k}
    return attributes


def compare_attributes(cubes, attribute_filter=None):
    """
    Function to compare attributes of cubes

    Args:
        cubes (Iris cubelist):
            List of cubes to compare (must be more than 1)
    Keyword Args:
        attribute_filter (string or None):
            A string to filter which attributes are actually compared. If None
            all attributes are compared.
    Returns:
        unmatching_attributes (List):
            List of dictionaries of unmatching attributes
    Warns:
        Warning: If only a single cube is supplied
    """
    unmatching_attributes = []
    if len(cubes) == 1:
        msg = ('Only a single cube so no differences will be found ')
        warnings.warn(msg)
    else:
        reference_attributes = get_filtered_attributes(
            cubes[0], attribute_filter=attribute_filter)

        common_keys = reference_attributes.keys()
        for cube in cubes[1:]:
            cube_attributes = get_filtered_attributes(
                cube, attribute_filter=attribute_filter)
            common_keys = {
                key for key in cube_attributes.keys()
                if key in common_keys and
                np.all(cube_attributes[key] == reference_attributes[key])}

        for cube in cubes:
            cube_attributes = get_filtered_attributes(
                cube, attribute_filter=attribute_filter)
            unique_attributes = {
                key: value for (key, value) in cube_attributes.items()
                if key not in common_keys}
            unmatching_attributes.append(unique_attributes)

    return unmatching_attributes


def compare_coords(cubes):
    """
    Function to compare the coordinates of the cubes

    Args:
        cubes (Iris cubelist):
            List of cubes to compare (must be more than 1)

    Returns:
        unmatching_coords (List):
            List of dictionaries of unmatching coordinates
            Number of dictionaries equals number of cubes
            unless cubes is a single cube in which case
            unmatching_coords returns an empty list.

    Warns:
        Warning: If only a single cube is supplied
    """
    unmatching_coords = []
    if len(cubes) == 1:
        msg = ('Only a single cube so no differences will be found ')
        warnings.warn(msg)
    else:
        common_coords = cubes[0].coords()
        for cube in cubes[1:]:
            cube_coords = cube.coords()
            common_coords = [
                coord for coord in common_coords
                if (coord in cube_coords and
                    np.all(cube.coords(coord) == cubes[0].coords(coord)))]

        for i, cube in enumerate(cubes):
            unmatching_coords.append(dict())
            for coord in cube.coords():
                if coord not in common_coords:
                    dim_coords = cube.dim_coords
                    if coord in dim_coords:
                        dim_val = dim_coords.index(coord)
                    else:
                        dim_val = None
                    aux_val = None
                    if dim_val is None and len(cube.coord_dims(coord)) > 0:
                        aux_val = cube.coord_dims(coord)[0]
                    unmatching_coords[i].update({coord.name():
                                                 {'data_dims': dim_val,
                                                  'aux_dims': aux_val,
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
        data (number/list/np.array):
            List or array of values to populate the coordinate points.
        long_name (str (optional)):
            Name of the coordinate to be built.
        standard_name (str (optional)):
            CF Name of the coordinate to be built.
        var_name (str (optional)):
            Variable name
        coord_type (iris.coord.AuxCoord or iris.coord.DimCoord (optional)):
            Selection between Dim and Aux coord.
        data_type (<type> (optional)):
            The data type of the coordinate points, e.g. int
        units (str (optional)):
            String defining the coordinate units.
        bounds (np.array (optional)):
            A (len(data), 2) array that defines coordinate bounds.
        coord_system(iris.coord_systems.<coord_system> (optional)):
            A coordinate system in which the dimension coordinates are defined.
        template_coord (iris.coord):
            A coordinate to copy.
        custom_function (function (optional)):
            A function to apply to the data values before constructing the
            coordinate, e.g. np.nan_to_num.

    Returns:
        crd_out(iris coordinate):
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

    crd_out = coord_type_out(data, long_name=long_name_out,
                             standard_name=std_name_out,
                             var_name=var_name_out,
                             units=units_out,
                             coord_system=coord_system_out,
                             bounds=bounds_out)

    if std_name_out is None and var_name_out is None:
        crd_out.rename(long_name_out)

    return crd_out


def sort_coord_in_cube(cube, coord, order="ascending"):
    """Sort a cube based on the ordering within the chosen coordinate.
    Sorting can either be in ascending or descending order.
    This code is based upon https://gist.github.com/pelson/9763057.

    Args:
        cube (iris.cube.Cube):
            The input cube to be sorted.
        coord (string):
            Name of the coordinate to be sorted.
        order (string):
            Choice of how to order the sorted coordinate.
            Options are either "ascending" or "descending".

    Returns:
        iris.cube.Cube:
            Cube where the chosen coordinate has been sorted into either
            ascending or descending order.

    Warns:
        Warning if the coordinate being processed is a circular coordinate.

    """
    coord_to_sort = cube.coord(coord)
    if isinstance(coord_to_sort, DimCoord):
        if coord_to_sort.circular:
            msg = ("The {} coordinate is circular. If the values in the "
                   "coordinate span a boundary then the sorting may return "
                   "an undesirable result.".format(coord_to_sort.name()))
            warnings.warn(msg)
    dim, = cube.coord_dims(coord_to_sort)
    index = [slice(None)] * cube.ndim
    index[dim] = np.argsort(coord_to_sort.points)
    if order == "descending":
        index[dim] = index[dim][::-1]
    if coord in ["height"] and order == "ascending":
        cube.coord(coord).attributes["positive"] = "up"
    elif coord in ["height"] and order == "descending":
        cube.coord(coord).attributes["positive"] = "down"
    return cube[tuple(index)]


def enforce_coordinate_ordering(
        cube, coord_names, anchor="start", promote_scalar=False,
        raise_exception=False):
    """
    Function to ensure that the requested coordinate within the cube are in
    the desired position.

    The reordering can either be anchored to the start or end of the available
    dimension coordinates using the "anchor" keyword argument. If desired,
    all the dimension coordinates can be reordered by specifying the coordinate
    names in the desired order.

    Note that the input cube is used as the output cube apart from if
    promote_scalar = True when a new cube instance is generated with an extra
    leading dimension.

    Args:
        cube (iris.cube.Cube):
            Cube where the ordering will be enforced to match the order within
            the coord_names. This input cube will be modified as part of this
            function.
        coord_names (list or str):
            List of the names of the coordinates to order. If a string is
            passed in, only the single specified coordinate is reordered.
        anchor (str):
            String to define where within the range of possible dimensions
            the specified coordinates should be located. If all the names
            of all the dimension coordinates are specified within the
            coord_names argument then this argument effectively does nothing.
            The options are either: "start" or "end". "start" indicates that
            the coordinates are inserted as the first coordinates within the
            cube. "end" indicates that the coordinates are inserted as the
            last coordinates within the cube. For example, if the specified
            coordinate names are ["time", "realization"] then "realization"
            will be the last coordinate within the cube, whilst "time" will be
            the last but one coordinate within the cube.
        promote_scalar (bool):
            If True, any coordinates that are matched and are not dimension
            coordinates are promoted to dimension coordinates.
            If False, any coordinates that are matched and are not dimension
            coordinates will not be considered in the reordering.
        raise_exception (bool):
            Option as to whether an exception should be raised, if the
            requested coordinate is not present.

    Returns:
        cube (iris.cube.Cube):
            Cube where the requirement for the dimensions to be in a particular
            order will have been enforced.

    Raises:
        ValueError: The anchor argument must have a value of either start or
            end.
        CoordinateNotFoundError: The requested coordinate is not available on
            the cube.
        ValueError: Multiple coordinates match the partial name provided.

    """
    if isinstance(coord_names, six.string_types):
        coord_names = str(coord_names)
    if isinstance(coord_names, str):
        coord_names = [coord_names]

    if anchor not in ["start", "end"]:
        msg = ("The value for the anchor must be either 'start' or 'end'."
               "The value specified for the anchor was {}".format(anchor))
        raise ValueError(msg)

    # Determine coordinate indices for use in creating a dictionary.
    # These indices are either relative to the start or end of the available
    # dimension coordinates.
    coord_indices = np.array(list(range(len(coord_names))))
    if anchor == "end":
        coord_indices = sorted(len(cube.dim_coords) - coord_indices)
    coord_dict = dict(list(zip(coord_names, coord_indices)))

    for coord_name in list(coord_dict.keys()):
        # Deal with the coord_name being a partial match to the actual
        # coordinate name.
        if cube.coords(coord_name):
            full_coord_name = coord_name
        else:
            coord = [coord for coord in cube.coords()
                     if coord_name in coord.name()]
            # If the coordinate is desired, raise an exception
            # if the coordinate is missing.
            if len(coord) == 0:
                if raise_exception:
                    msg = ("The requested coordinate {} is not a coordinate "
                           "in the cube: {}".format(coord, cube))
                    raise CoordinateNotFoundError(msg)
                else:
                    continue
            elif len(coord) == 1:
                # Replace the dictionary key with the actual coordinate name.
                full_coord_name = coord[0].name()
                coord_dict[full_coord_name] = coord_dict.pop(coord_name)
            else:
                msg = ("More than 1 coordinate: {} matched the specified "
                       "coordinate name: {}. Unable to distinguish which "
                       "coordinate should be reordered.".format(
                        coord, coord_name))
                raise ValueError(msg)

        # If the requested coordinate is not a dimension coordinate, make it
        # a dimension coordinate.
        if cube.coords(full_coord_name, dim_coords=False):
            if promote_scalar:
                cube = iris.util.new_axis(cube, full_coord_name)
            else:
                if raise_exception:
                    msg = ("The {} coordinate cannot be reordered as it "
                           "is a scalar coordinate.".format(full_coord_name))
                    raise ValueError(msg)
                else:
                    coord_dict.pop(full_coord_name, None)

    # Get the dimensions for the coordinates that have not been requested.
    remaining_coords = []
    for acoord in cube.coords(dim_coords=True):
        if acoord.name() not in coord_dict.keys():
            remaining_coords.append(cube.coord_dims(acoord)[0])
    remaining_coords = list(set(remaining_coords))

    # Get the dimensions for the coordinates that have been requested by
    # getting the keys from the dictionary that have been sorted by the values.
    coord_dims = []
    for coord_name, _ in sorted(
            list(coord_dict.items()), key=operator.itemgetter(1)):
        if cube.coords(coord_name, dim_coords=True):
            coord_dims.append(cube.coord_dims(coord_name)[0])

    # Transpose by inserting the requested coordinates at either the start
    # or the end.
    if anchor == "start":
        cube.transpose(coord_dims + remaining_coords)
    elif anchor == "end":
        cube.transpose(remaining_coords + coord_dims)
    return cube


def enforce_float32_precision(input_cubes):
    """Take input cube of any precision and convert to float32.

    Args:
        input_cubes (list):
            List containing one or more iris cubes to test if not float32
            precision and downscale to float32 if necessary. If a list item
            is not an Iris cube - then this item is skipped.
            Note: The code will modify the cubes in-place.

    """
    # If single cube - place within list.
    if isinstance(input_cubes, iris.cube.Cube):
        input_cubes = [input_cubes]

    for cube in input_cubes:
        if isinstance(cube, iris.cube.Cube):  # Skip if not cube.
            if cube.dtype != np.float32:
                cube.data = cube.data.astype(np.float32)


def demote_float64_precision(input_cubes):
    """Take input cube of any precision and convert any float64 data to
    float32.

    Args:
        input_cubes (cubelist or cube):
            List containing one or more iris cubes to test and adjust if
            necessary.
            Note: The code will modify the cubes in-place.

    """
    # If single cube - place within cubelist.
    if isinstance(input_cubes, iris.cube.Cube):
        input_cubes = iris.cube.CubeList([input_cubes])

    # Cycle through the cubes
    for cube in input_cubes:
        assert isinstance(cube, iris.cube.Cube), 'Object is not a cube'

        # Modify data if it is float64
        if cube.dtype == np.float64:
            cube.data = cube.data.astype(np.float32)


def clip_cube_data(cube, minimum_value, maximum_value):
    """Apply np.clip to data in a cube to ensure that the limits do not go
    beyond the provided minimum and maximum values.

    Args:
        cube (iris.cube.Cube):
            The cube that has been processed and contains data that is to be
            clipped.
        minimum_value (int or float):
            The minimum value, with data in the cube that falls below this
            threshold set to it.
        maximum_value (int or float):
            The maximum value, with data in the cube that falls above this
            threshold set to it.
    Returns:
        result (iris.cube.Cube):
            The processed cube with the data clipped to the limits of the
            original preprocessed cube.
    """
    original_attributes = cube.attributes
    original_methods = cube.cell_methods

    result = iris.cube.CubeList()
    for cube_slice in cube.slices([cube.coord(axis='y'),
                                   cube.coord(axis='x')]):
        cube_slice.data = np.clip(cube_slice.data,
                                  minimum_value, maximum_value)
        result.append(cube_slice)

    result = result.merge_cube()
    result.cell_methods = original_methods
    result.attributes = original_attributes
    result = check_cube_coordinates(cube, result)
    return result
