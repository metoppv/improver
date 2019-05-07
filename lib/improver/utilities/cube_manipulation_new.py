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

from improver.blending.weighted_blend import rationalise_blend_time_coords
from improver.utilities.cube_checker import (
    check_cube_coordinates, check_cube_not_float64, find_threshold_coordinate)


def equalise_cube_attributes(cubes, silent=None):
    """
    Function to remove attributes that do not match between all cubes in the
    list.  Cubes are modified in place.

    Args:
        cubes (iris.cube.CubeList):
            List of cubes to check the attributes and revise.

    Kwargs:
        silent (list or None):
            List of attributes to remove silently if unmatched.

    Warns:
        UserWarning:
            If an unmatched attribute is not in the "silent" list,
            a warning will be raised.
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

        Kwargs:
            coords_to_associate (list):
                List of coordinates to be associated with the master_coord.  If
                master_coord is "time" this should be "forecast_reference_time"
                OR "forecast_period", NOT both.
            coords_to_slice_over (list):
                Dimension coordinates to slice over before concatenation.
        """
        self.master_coord = master_coord
        self.coords_to_associate = coords_to_associate
        self.coords_to_slice_over = coords_to_slice_over

        if self.coords_to_slice_over is None:
            self.coords_to_slice_over = ["realization", "time"]
        if self.coords_to_associate is None and self.master_coord == "time":
            self.coords_to_associate = ["forecast_period"]

        # Check for dangerous coordinate associations
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
            cube (iris.cube.Cube):
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
            sliced_by_coord_cubelist (iris.cube.CubeList):
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

        Args:
            cubes_in (iris.cube.CubeList):
                Cube or list of cubes to be concatenated

        Returns:
            result (iris.cube.Cube):
                Cube concatenated along master coord

        Raises:
            ValueError:
                If master coordinate is not present on all "cubes_in"
        """
        # create copies of input cubes so as not to modify in place
        if isinstance(cubes_in, iris.cube.Cube):
            cubes = iris.cube.CubeList([cubes_in.copy()])
        else:
            cubes = iris.cube.CubeList([])
            for cube in cubes_in:
                cubes.append(cube.copy())

        # check master coordinate is on cubes - if not, throw error
        if not all(cube.coords(self.master_coord) for cube in cubes):
            raise ValueError(
                "Master coordinate {} is not present on input cube(s)".format(
                    self.master_coord))

        # slice over requested coordinates
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

    Accounts for differences in attributes and coordinates to avoid merge
    failures and anonymous dimensions.
    """
    def __init__(self):
        """Initialise constants"""
        # List of attributes to remove silently if unmatched
        self.silent_attributes = ["history", "title", "mosg__grid_version"]
        # List of coordinates that must strictly match on the input cubes.  If
        # unmatched, an exception will be raised.
        self.coord_mismatch_error_keys = ["threshold"]

    def _equalise_cubes(self, cubelist):
        """
        Function to equalise cubes where the attributes, coordinates or cell
        methods do not match.  Note this function cannot equalise cubes where
        different coordinates are present on each cube.

        Args:
            cubelist (iris.cube.CubeList):
                List of cubes to check and equalise.

        Returns:
            cubelist (iris.cube.CubeList):
                List of cubes with revised cubes. The number of cubes in this
                list may be greater than the original number of cubes if they
                have been sliced over mismatching dimension coordinates.
        """
        equalise_cube_attributes(cubelist, silent=self.silent_attributes)
        strip_var_names(cubelist)

        cubelist = self._equalise_cube_coords(cubelist)
        cubelist = self._equalise_cell_methods(cubelist)
        for cube in cubelist:
            check_cube_not_float64(cube, fix=True)  # TODO why here?
        return cubelist

    def _equalise_cube_coords(self, cubes):
        """
        Function to equalise coordinates that do not match, by slicing over
        dimension coordinate values and creating a cube list which can later
        be merged.  Raises errors if coordinates have bounds values that are
        unmatched, or if the coordinates specified by
        self.coord_mismatch_error_keys have differences.

        Args:
            cubes (iris.cube.CubeList):
                List of cubes to check the coords and revise.

        Returns:
            cubelist (iris.cube.CubeList):
                List of cubes with revised coords. The number of cubes in this
                list may be greater than the original number of cubes if they
                have been sliced over mismatching dimension coordinates.

        Raises:
            ValueError:
                If bounds on dimension coordinates do not match (through
                self._check_dim_coord_bounds(cubes)).
            ValueError:
                If coordinates in self.coord_mismatch_error_keys do not match.
        """
        # check for unmatching coords (returns a list of dictionaries, each
        # with an entry for each mismatched coord.  If all coords match,
        # returns a list of empty dictionaries.)
        unmatching_coords = compare_coords(cubes)

        # continue only if the list contains non-empty dictionaries
        if bool(unmatching_coords[0]):

            # check for mismatches in dim coord bounds
            self._check_dim_coord_bounds(cubes)

            # throw an error for specific coordinate mismatches
            for error_key in self.coord_mismatch_error_keys:
                for key in ([keyval for cube_dict in unmatching_coords
                             for keyval in cube_dict]):
                    if error_key in key:
                        msg = ("{} coordinates must match "
                               "to merge".format(error_key))
                        raise ValueError(msg)

            # slice over any remaining mismatched non-scalar coordinates
            cubelist = iris.cube.CubeList([])
            for i, cube in enumerate(cubes):
                slice_over_keys = []
                for key in unmatching_coords[i]:
                    # If mismatching is a dimension coord add to list to
                    # slice over (supports time lagging with different
                    # realizations).  Note this will not prevent failure to
                    # merge if the coordinate (eg realization) is not present
                    # on all input cubes.
                    if unmatching_coords[i][key]['data_dims'] is not None:
                        slice_over_keys.append(key)
                    if unmatching_coords[i][key]['aux_dims'] is not None:
                        slice_over_keys.append(key)

                if len(slice_over_keys) > 0:
                    for slice_cube in cube.slices_over(slice_over_keys):
                        cubelist.append(slice_cube)
                else:
                    cubelist.append(cube)
        else:
            cubelist = cubes

        return cubelist

    @staticmethod
    def _check_dim_coord_bounds(cubes):
        """
        Function to check for dimension coordinate bounds that do not match.
        This prevents the creation of anonymous auxiliary coordinates by
        iris.merge_cube(). If a coordinate is not present in all cubes, it is
        ignored here.

        Args:
            cubes (iris.cube.CubeList):
                List of cubes to check the cell methods and revise.
                These are modified in place.

        Raises:
            ValueError:
                If some but not all cubes have bounds on a shared dimension
                coordinate.
            ValueError:
                If existing bounds values on shared dimension coordinates do
                not match.
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
                    elif (coord.bounds is not None and
                          match_coord.bounds is not None):
                        if np.allclose(np.array(coord.bounds),
                                       np.array(match_coord.bounds)):
                            continue
                    raise ValueError(msg.format(coord.name()))

    @staticmethod
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
        """
        cell_methods = cubes[0].cell_methods
        for cube in cubes[1:]:
            cell_methods = list(set(cell_methods) & set(cube.cell_methods))
        cubelist = cubes
        for cube in cubelist:
            cube.cell_methods = tuple(cell_methods)
        return cubelist

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

        Args:
            cubes (iris.cube.CubeList or iris.cube.Cube):
                Cubes to be merged.

        Kwargs:
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
            return cubes_in[0]

        # create copies of input cubes so as not to modify in place
        cubelist = iris.cube.CubeList([])
        for cube in cubes_in:
            cubelist.append(cube.copy())

        # if coord_mismatch_error_keys includes "threshold", replace entry with
        # standard name of threshold-type coordinate on input cubes
        if "threshold" in self.coord_mismatch_error_keys:
            try:
                coord_name = find_threshold_coordinate(cubelist[0]).name()
            except CoordinateNotFoundError:
                pass
            else:
                self.coord_mismatch_error_keys.remove("threshold")
                self.coord_mismatch_error_keys.append(coord_name)

        # equalise cube attributes and coordinates
        cubelist = self._equalise_cubes(cubelist)

        # demote single-point dimensions to scalar coordinates
        for i, cube in enumerate(cubelist):
            cubelist[i] = iris.util.squeeze(cube)

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
        cubes (Iris cubelist or Iris cube):
            Cubes to be merged.

    Returns:
        result (Iris cube):
            Merged cube.
    """
    result = MergeCubes().process(cubes)
    return result


# TODO move this class into improver.blending.weighted_blend
class MergeCubesForWeightedBlending():
    """Prepares cubes for cycle and grid blending"""

    def __init__(self, blend_coord, weighting_coord=None, model_id_attr=None):
        """
        Initialise the class

        Args:
            blend_coord (str):
                Name of coordinate over which blending will be performed.  For
                multi-model blending this is flexible to any string containing
                "model".  For all other coordinates this is prescriptive:
                cube.coord(blend_coord) must return an iris.coords.Coord
                instance for all cubes passed into the "process" method.

        Kwargs:
            weighting_coord (str or None):
                The coordinate across which weights will be scaled in a
                multi-model blend.  Required for
                rationalise_blend_time_coordinates.
            model_id_attr (str or None):
                Name of attribute used to identify model for grid blending.
                None for cycle blending.

        Raises:
            ValueError:
                If trying to blend over model when model_id_attr is not set
        """
        if "model" in blend_coord and model_id_attr is None:
            raise ValueError(
                "model_id_attr required to blend over {}".format(blend_coord))

        # ensure model coordinates are not created for non-model blending
        if "model" not in blend_coord and model_id_attr is not None:
            warnings.warn(
                "model_id_attr not required for blending over {} - "
                "will be ignored".format(blend_coord))
            model_id_attr = None

        self.blend_coord = blend_coord
        self.weighting_coord = weighting_coord
        self.model_id_attr = model_id_attr

    def _create_model_coordinates(self, cubelist):
        """
        Adds numerical model ID and string model configuration scalar
        coordinates to input cubes if self.model_id_attr is specified.
        Sets the original attribute value to "blend", in anticipation.
        Modifies cubes in place.

        Args:
            cubelist (iris.cube.CubeList):
                List of cubes to be merged for blending

        Raises:
            ValueError:
                If self.model_id_attr is not present on all cubes
            ValueError:
                If input cubelist contains cubes from the same model
        """
        model_titles = []
        for i, cube in enumerate(cubelist):
            if self.model_id_attr not in cube.attributes:
                msg = ('Cannot create model ID coordinate for grid blending '
                       'as "model_id_attr={}" was not found within the cube '
                       'attributes'.format(self.model_id_attr))
                raise ValueError(msg)

            model_title = cube.attributes.pop(self.model_id_attr)
            if model_title in model_titles:
                raise ValueError('Cannot create model dimension coordinate '
                                 'with duplicate points')
            model_titles.append(model_title)
            cube.attributes[self.model_id_attr] = "blend"

            new_model_id_coord = build_coordinate([1000 * i],
                                                  long_name='model_id',
                                                  data_type=np.int32)
            new_model_coord = (
                build_coordinate([model_title],
                                 long_name='model_configuration',
                                 coord_type=AuxCoord,
                                 data_type=np.str))

            cube.add_aux_coord(new_model_id_coord)
            cube.add_aux_coord(new_model_coord)

    def process(self, cubes_in, cycletime=None):
        """
        Prepares merged input cube for cycle and grid blending

        Args:
            cubes (iris.cube.CubeList or iris.cube.Cube):
                Cubes to be merged.

        Kwargs:
            cycletime (str or None):
                The cycletime in a YYYYMMDDTHHMMZ format e.g. 20171122T0100Z.
                Can be used in rationalise_blend_time_coordinates.

        Returns:
            iris.cube.Cube:
                Merged cube.

        Raises:
            ValueError:
                If self.blend_coord is not present on all cubes (unless
                blending over models)
        """
        # if input is already a single cube, return unchanged
        if isinstance(cubes_in, iris.cube.Cube):
            return cubes_in

        if len(cubes_in) == 1:
            return cubes_in[0]

        # create copies of input cubes so as not to modify in place
        cubelist = iris.cube.CubeList([])
        for cube in cubes_in:
            if ("model" not in self.blend_coord and
                    not cube.coords(self.blend_coord)):
                raise ValueError(
                    "{} coordinate is not present on all input "
                    "cubes".format(self.blend_coord))
            cubelist.append(cube.copy())

        # TODO move rationalise_blend_time_coords into this class
        rationalise_blend_time_coords(
            cubelist, self.blend_coord, cycletime=cycletime,
            weighting_coord=self.weighting_coord)

        # create model ID and model configuration coordinates if blending
        # different models
        if self.model_id_attr is not None:
            self._create_model_coordinates(cubelist)

        # merge resulting cubelist
        result = MergeCubes().process(cubelist, check_time_bounds_ranges=True)

        return result


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
        cube.transpose(coord_dims + remaining_coords)
    elif anchor == "end":
        cube.transpose(remaining_coords + coord_dims)
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
