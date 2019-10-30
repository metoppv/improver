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
import warnings

import iris
import numpy as np
import six
from iris.coords import AuxCoord, DimCoord
from iris.exceptions import CoordinateNotFoundError

from improver.metadata.probabilistic import find_threshold_coordinate
from improver.utilities.cube_checker import check_cube_coordinates


def equalise_cube_attributes(cubes, silent=None):
    """
    Function to remove attributes that do not match between all cubes in the
    list.  Cubes are modified in place.

    Args:
        cubes (iris.cube.CubeList):
            List of cubes to check the attributes and revise.
        silent (list or None):
            List of attributes to remove silently if unmatched.

    Warns:
        UserWarning:
            If an unmatched attribute is not in the "silent" list,
            a warning will be raised.

    NOTE 16/05/19: iris.experimental now has an equalise_attributes function,
    which removes any unmatched attributes without raising a warning.

    TODO replace this function with the iris version once it is promoted into
    the standard iris package.  At that time, the silent_attributes member of
    the MergeCubes and ConcatenateCubes classes becomes obsolete and should be
    removed.
    """
    if silent is None:
        silent = []
    unmatched = compare_attributes(cubes)
    warning_msg = 'Deleting unmatched attribute {}, value {}'
    if len(unmatched) > 0:
        for i, cube in enumerate(cubes):
            for attr in unmatched[i]:
                if attr not in silent:
                    warnings.warn(
                        warning_msg.format(attr, cube.attributes[attr]))
                cube.attributes.pop(attr)


def strip_var_names(cubes):
    """
    Strips var_name from the cube and from all coordinates
    to help concatenation.

    Args:
        cubes (iris.cube.CubeList or iris.cube.Cube):
            Cubes to be concatenated.

    Returns:
        iris.cube.CubeList:
            CubeList containing original cubes without a var_name on the cube,
            or on the coordinates.
            Note: This internal function modifies the incoming cubes
    """
    if isinstance(cubes, iris.cube.Cube):
        cubes = iris.cube.CubeList([cubes])
    for cube in cubes:
        cube.var_name = None
        for coord in cube.coords():
            # retain var name required for threshold coordinate
            if coord.var_name != "threshold":
                coord.var_name = None
    return cubes


class ConcatenateCubes():
    """
    Class adding functionality to iris.concatenate_cubes().

    Accounts for differences in attributes and allows promotion of scalar
    coordinates to be associated with the dimension over which concatenation
    is to be performed (eg can promote forecast_period to auxiliary for single
    time point cube inputs).
    """

    def __init__(self, master_coord, coords_to_associate=None,
                 coords_to_slice_over=None):
        """
        Initialise parameters

        Args:
            master_coord (str):
                Coordinate to concatenate over.
            coords_to_associate (list):
                List of coordinates to be associated with the master_coord.  If
                master_coord is "time" this should be "forecast_reference_time"
                OR "forecast_period", NOT both.
            coords_to_slice_over (list):
                Dimension coordinates to slice over before concatenation.
                May cause the dimension order to change from input to output
                cubes.
        """
        self.master_coord = master_coord
        self.coords_to_associate = coords_to_associate
        self.coords_to_slice_over = coords_to_slice_over

        if self.coords_to_associate is None and self.master_coord == "time":
            self.coords_to_associate = ["forecast_period"]

        # Check for dangerous coordinate associations
        if self.coords_to_associate is not None:
            associated_coords = self.coords_to_associate.copy()
            associated_coords.append(self.master_coord)
            if ("time" in associated_coords and
                    "forecast_period" in associated_coords and
                    "forecast_reference_time" in associated_coords):
                msg = ("Time, forecast period and forecast reference time "
                       "cannot all be associated with a single dimension")
                raise ValueError(msg)

        # List of attributes to remove silently if unmatched
        self.silent_attributes = ["history", "title", "mosg__grid_version"]

    def _associate_any_coordinate_with_master_coordinate(self, cube):
        """
        Function to convert the given coordinates from scalar coordinates to
        auxiliary coordinates, where these auxiliary coordinates will be
        associated with the master coordinate.

        For example, forecast_period can be converted from scalar coordinate
        to auxiliary coordinate to be associated with a time dimension.

        Args:
            cube (iris.cube.Cube):
                Cube requiring promotion of the specified coordinates to
                auxiliary coordinates, to be associated with the master
                coordinate dimension.

        Returns:
            iris.cube.Cube:
                Cube where the the requested coordinates have been promoted to
                auxiliary coordinates.

        Raises:
            ValueError: If the master coordinate is not present on the cube.
        """
        coordinates = self.coords_to_associate
        if coordinates is None:
            coordinates = []

        # If the master_coord is not a dimension coordinate, then the other
        # coordinates cannot be associated with it.
        if cube.coords(self.master_coord, dim_coords=True):
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
                        data_dims=coord_names.index(self.master_coord))

        return cube

    @staticmethod
    def _slice_over_coordinate(cubes, coord_to_slice_over):
        """
        Function slices over the requested coordinate in each cube within a
        cubelist. The sliced coordinate is promoted into a one-point dimension
        to help concatenation. If the coord_to_slice_over is not found on a
        cube, the cube is added to the list in its original form.

        Args:
            cubes (iris.cube.Cube or iris.cube.CubeList):
                Cubes to be concatenated.
            coord_to_slice_over (str or iris.coords.Coord):
                Coordinate instance or name of coordinate to slice over.

        Returns:
            iris.cube.CubeList:
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

    def process(self, cubes_in):
        """
        Processes a list of cubes to ensure compatibility before calling the
        iris.cube.CubeList.concatenate_cube() method. Removes mismatched
        attributes, strips var_names from the cube and coordinates, and slices
        over any requested dimensions to avoid coordinate mismatch errors (eg
        for concatenating cubes with differently numbered realizations).

        If the input is a single Cube or CubeList of length 1, the input cube
        is returned unmodified.

        Args:
            cubes_in (iris.cube.CubeList or iris.cube.Cube):
                Cube or list of cubes to be concatenated

        Returns:
            iris.cube.Cube:
                Cube concatenated along master coord

        Raises:
            ValueError:
                If master coordinate is not present on all "cubes_in"
        """
        # if input is a single cube or list of length 1, return unchanged
        if isinstance(cubes_in, iris.cube.Cube):
            return cubes_in

        if len(cubes_in) == 1:
            return cubes_in[0]

        # create copies of input cubes so as not to modify in place
        cubes = iris.cube.CubeList([])
        for cube in cubes_in:
            cubes.append(cube.copy())

        # check master coordinate is on cubes - if not, throw error
        if not all(cube.coords(self.master_coord) for cube in cubes):
            raise ValueError(
                "Master coordinate {} is not present on input cube(s)".format(
                    self.master_coord))

        # slice over requested coordinates
        if self.coords_to_slice_over is not None:
            for coord_to_slice_over in self.coords_to_slice_over:
                cubes = self._slice_over_coordinate(cubes, coord_to_slice_over)

        # remove unmatched attributes
        equalise_cube_attributes(cubes, silent=self.silent_attributes)

        # remove cube variable names
        strip_var_names(cubes)

        # promote scalar coordinates to auxiliary as necessary
        associated_master_cubelist = iris.cube.CubeList([])
        for cube in cubes:
            associated_master_cubelist.append(
                self._associate_any_coordinate_with_master_coordinate(cube))

        # concatenate cube
        result = associated_master_cubelist.concatenate_cube()

        return result


def concatenate_cubes(
        cubes_in, coords_to_slice_over=None, master_coord="time",
        coordinates_for_association=None):
    """
    Wrapper for the ConcatenateCubes.process method

    Function to concatenate cubes, accounting for differences in the
    history attribute, and allow promotion of forecast_reference_time
    and forecast_period coordinates from scalar coordinates to auxiliary
    coordinates to allow concatenation.

    Args:
        cubes_in (iris.cube.CubeList or iris.cube.Cube):
            Cubes to be concatenated.
        coords_to_slice_over (list):
            Coordinates to be sliced over.
        master_coord (str):
            Coordinate that the other coordinates will be associated with.
        coordinates_for_association (list):
            List of coordinates to be associated with the master_coord.

    Returns:
        iris.cube.Cube:
            Concatenated cube.
    """
    plugin = ConcatenateCubes(
        master_coord, coords_to_associate=coordinates_for_association,
        coords_to_slice_over=coords_to_slice_over)
    result = plugin.process(cubes_in)
    return result


class MergeCubes():
    """
    Class adding functionality to iris.merge_cubes()

    Accounts for differences in attributes, cell methods and bounds ranges to
    avoid merge failures and anonymous dimensions.
    """
    def __init__(self):
        """Initialise constants"""
        # List of attributes to remove silently if unmatched
        self.silent_attributes = ["history", "title", "mosg__grid_version"]

    @staticmethod
    def _equalise_cell_methods(cubelist):
        """
        Function to equalise cell methods that do not match.  Modifies cubes
        in place.

        Args:
            cubelist (iris.cube.CubeList):
                List of cubes to check the cell methods and revise.
        """
        cell_methods = cubelist[0].cell_methods
        for cube in cubelist[1:]:
            cell_methods = list(set(cell_methods) & set(cube.cell_methods))
        for cube in cubelist:
            cube.cell_methods = tuple(cell_methods)

    @staticmethod
    def _check_time_bounds_ranges(cube):
        """
        Check the bounds on any dimensional time coordinates after merging.
        For example, to check time and forecast period ranges for accumulations
        to avoid blending 1 hr with 3 hr accumulations.  If points on the
        coordinate are not compatible, raise an error.

        Args:
            cube (iris.cube.Cube):
                Merged cube
        """
        for name in ["time", "forecast_period"]:
            try:
                coord = cube.coord(name)
            except CoordinateNotFoundError:
                continue

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

    def process(self, cubes_in, check_time_bounds_ranges=False):
        """
        Function to merge cubes, accounting for differences in attributes,
        coordinates and cell methods.  Note that cubes with different sets
        of coordinates (as opposed to cubes with the same coordinates with
        different values) cannot be merged.

        If the input is a single Cube, this is returned unmodified.  A
        CubeList of length 1 is checked for mismatched time bounds before
        returning the single Cube (since a CubeList of this form may be the
        result of premature iris merging on load).

        Args:
            cubes_in (iris.cube.CubeList or iris.cube.Cube):
                Cubes to be merged.
            check_time_bounds_ranges (bool):
                Flag to check whether scalar time bounds ranges match.
                This is for when we are expecting to create a new "time" axis
                through merging for eg precipitation accumulations, where we
                want to make sure that the bounds match so that we are not eg
                combining 1 hour with 3 hour accumulations.

        Returns:
            iris.cube.Cube:
                Merged cube.
        """
        # if input is already a single cube, return unchanged
        if isinstance(cubes_in, iris.cube.Cube):
            return cubes_in

        if len(cubes_in) == 1:
            # iris merges cubelist into shortest list possible on load
            # - may already have collapsed across invalid time bounds
            if check_time_bounds_ranges:
                self._check_time_bounds_ranges(cubes_in[0])
            return cubes_in[0]

        # create copies of input cubes so as not to modify in place
        cubelist = iris.cube.CubeList([])
        for cube in cubes_in:
            cubelist.append(cube.copy())

        # equalise cube attributes, cell methods and coordinate names
        equalise_cube_attributes(cubelist, silent=self.silent_attributes)
        strip_var_names(cubelist)
        self._equalise_cell_methods(cubelist)

        # merge resulting cubelist
        result = cubelist.merge_cube()

        # check time bounds if required
        if check_time_bounds_ranges:
            self._check_time_bounds_ranges(result)

        return result


def merge_cubes(cubes):
    """
    Wrapper for MergeCubes().process()

    Args:
        cubes (iris.cube.CubeList or iris.cube.Cube):
            Cubes to be merged.

    Returns:
        iris.cube.Cube:
            Merged cube.
    """
    result = MergeCubes().process(cubes)
    return result


def get_filtered_attributes(cube, attribute_filter=None):
    """
    Build dictionary of attributes that match the attribute_filter. If the
    attribute_filter is None, return all attributes.

    Args:
        cube (iris.cube.Cube):
            A cube from which attributes partially matching the
            attribute_filter will be returned.
        attribute_filter (str or None):
            A string to match, or partially match, against attributes to build
            a filtered attribute dictionary. If None, all attributes are
            returned.
    Returns:
        dict:
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
        cubes (iris.cube.CubeList):
            List of cubes to compare (must be more than 1)
        attribute_filter (str or None):
            A string to filter which attributes are actually compared. If None
            all attributes are compared.
    Returns:
        list of dict:
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
        cubes (iris.cube.CubeList):
            List of cubes to compare (must be more than 1)

    Returns:
        list of dict:
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
    Construct an iris.coords.Dim/Auxcoord using the provided options.

    Args:
        data (int or list of numpy.ndarray):
            List or array of values to populate the coordinate points.
        long_name (str):
            Name of the coordinate to be built.
        standard_name (str):
            CF Name of the coordinate to be built.
        var_name (str):
            Variable name
        coord_type (iris.coords.Coord):
            Selection between Dim and Aux coord.
        data_type (<type>):
            The data type of the coordinate points, e.g. int
        units (str):
            String defining the coordinate units.
        bounds (numpy.ndarray):
            A (len(data), 2) array that defines coordinate bounds.
        coord_system(iris.coord_systems.<coord_system>):
            A coordinate system in which the dimension coordinates are defined.
        template_coord (iris.coords.Coord):
            A coordinate to copy.
        custom_function (function):
            A function to apply to the data values before constructing the
            coordinate, e.g. np.nan_to_num.

    Returns:
        iris.coords.Coord:
            The constructed coord.

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
        coord (str):
            Name of the coordinate to be sorted.
        order (str):
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
        iris.cube.Cube:
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
            # Handle "threshold" as an argument
            if coord_name == "threshold":
                try:
                    coord = [find_threshold_coordinate(cube)]
                except CoordinateNotFoundError:
                    coord = []
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
        dim_order = coord_dims + remaining_coords
    elif anchor == "end":
        dim_order = remaining_coords + coord_dims
    # Don't reorder if dim_order is already correct:
    if dim_order != sorted(dim_order):
        cube.transpose(dim_order)
    return cube


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
        iris.cube.Cube:
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


def expand_bounds(result_cube, cubelist, expanded_coords):
    """Alter a coord such that bounds are expanded to cover
    the entire range of the input cubes.

    For example, in the case of time cubes if the input cubes have
    bounds of [0000Z, 0100Z] & [0100Z, 0200Z] then the output cube will
    have bounds of [0000Z,0200Z]

    Args:
        result_cube (iris.cube.Cube):
            A cube with metadata for the results.
        cubelist (iris.cube.CubeList):
            The list of cubes with coordinates to be combined
        expanded_coords (dict or None):
            Coordinates over which bounds should be expanded as a key, with the
            value indicating whether the upper or mid point of the coordinate
            should be used as the point value, e.g. {'time': 'upper'}.

                | 'mid' - halfway between the bounds
                | 'upper' - equal to the upper bound

    Returns:
        iris.cube.Cube:
            Cube with coords expanded.

            n.b. If argument point == 'mid' then python will convert
            result.coord('coord').points[0] to a float UNLESS the coord
            units contain 'seconds'.  This is to ensure that midpoints are
            not rounded down, for example when times are in hours.
    """
    for coord, point in expanded_coords.items():

        if len(result_cube.coord(coord).points) != 1:
            emsg = ('the expand bounds function should only be used on a'
                    'coordinate with a single point. The coordinate \"{}\" '
                    'has {} points.')
            raise ValueError(emsg.format(
                coord,
                len(result_cube.coord(coord).points)))

        bounds = ([cube.coord(coord).bounds for cube in cubelist])
        if any(b is None for b in bounds):
            points = ([cube.coord(coord).points for cube in cubelist])
            new_low_bound = np.min(points)
            new_top_bound = np.max(points)
        else:
            new_low_bound = np.min(bounds)
            new_top_bound = np.max(bounds)
        result_coord = result_cube.coord(coord)
        result_coord.bounds = np.array(
            [[new_low_bound, new_top_bound]])
        if result_coord.bounds.dtype == np.float64:
            result_coord.bounds = result_coord.bounds.astype(np.float32)

        if point == 'mid':
            if 'seconds' in str(result_coord.units):
                # integer division of seconds required to retain precision
                dtype_orig = result_coord.dtype
                result_coord.points = [
                    (new_top_bound - new_low_bound) // 2 + new_low_bound]
                # re-cast to original precision to avoid escalating int32s
                result_coord.points = result_coord.points.astype(dtype_orig)
            else:
                # float division of hours required for accuracy
                result_coord.points = [
                    (new_top_bound - new_low_bound) / 2. + new_low_bound]
        elif point == 'upper':
            result_coord.points = [new_top_bound]

        if result_coord.points.dtype == np.float64:
            result_coord.points = result_coord.points.astype(np.float32)

    return result_cube
