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
"""Module containing utilities for modifying cube metadata."""

import warnings
import numpy as np

import iris

from improver.utilities.cube_manipulation import compare_coords


def add_coord(cube, coord_name, changes, warnings_on=False):
    """Add coord to the cube.

    Args:
        cube (iris.cube.Cube):
            Cube containing combined data.
        coord_name (string):
            Name of the coordinate being added.
        changes (dict):
            Details on coordinate to be added to the cube.

    Keyword Args:
        warnings_on (bool):
            If True output warnings for mismatching metadata.

    Returns:
        result (iris.cube.Cube):
            Cube with added coordinate.

    Raises:
        ValueError: Trying to add new coord but no points defined.
        ValueError: Can not add a coordinate of length > 1
        UserWarning: adding new coordinate.

    """
    if 'points' not in changes:
        msg = ("Trying to add new coord but no points defined"
               " in metadata, coord  = {}".format(coord_name))
        raise ValueError(msg)
    if len(changes['points']) != 1:
        msg = ("Can not add a coordinate of length > 1,"
               " coord  = {}".format(coord_name))
        raise ValueError(msg)

    metatype = 'DimCoord'
    if 'metatype' in changes:
        if changes['metatype'] == 'AuxCoord':
            new_coord_method = iris.coords.AuxCoord
            metatype = 'AuxCoord'
        else:
            new_coord_method = iris.coords.DimCoord
    else:
        new_coord_method = iris.coords.DimCoord
    result = cube
    points = changes['points']
    bounds = None
    if 'bounds' in changes:
        bounds = changes['bounds']
    units = None
    if 'units' in changes:
        units = changes['units']
    new_coord = new_coord_method(long_name=coord_name,
                                 points=points,
                                 bounds=bounds,
                                 units=units)
    result.add_aux_coord(new_coord)
    if metatype == 'DimCoord':
        result = iris.util.new_axis(result, coord_name)
    if warnings_on:
        msg = ("Adding new coordinate "
               "{} with {}".format(coord_name,
                                   changes))
        warnings.warn(msg)
    return result


def update_coord(cube, coord_name, changes, warnings_on=False):
    """Amend the metadata in the combined cube.

    Args:
        cube (iris.cube.Cube):
            Cube containing combined data.
        coord_name (string):
            Name of the coordinate being updated.
        changes (string or dict):
            Details on coordinate to be updated.
            If changes = 'delete' the coordinate is deleted.

    Keyword Args:
        warnings_on (bool):
            If True output warnings for mismatching metadata.

    Returns:
        result (iris.cube.Cube):
            Cube with updated coordinate.

    Raises:
        ValueError : Can only remove a coordinate of length 1
        ValueError : Mismatch in points in existing coord
            and updated metadata.
        ValueError : Mismatch in bounds in existing coord
            and updated metadata.
        ValueError : The shape of the bounds array should
            be points.shape + (n_bounds,)
        UserWarning: Deleted coordinate.
        UserWarning: Updated coordinate

    """
    new_coord = cube.coord(coord_name)
    result = cube
    if changes == 'delete':
        if len(new_coord.points) != 1:
            msg = ("Can only remove a coordinate of length 1"
                   " coord  = {}".format(coord_name))
            raise ValueError(msg)
        result.remove_coord(coord_name)
        result = iris.util.squeeze(result)
        if warnings_on:
            msg = ("Deleted coordinate "
                   "{}".format(coord_name))
            warnings.warn(msg)
    else:
        if 'points' in changes:
            new_points = np.array(changes['points'])
            if (len(new_points) ==
                    len(new_coord.points)):
                new_coord.points = new_points
            else:
                msg = ("Mismatch in points in existing"
                       " coord and updated metadata for "
                       " coord {}".format(coord_name))
                raise ValueError(msg)
        if 'bounds' in changes:
            new_bounds = np.array(changes['bounds'])
            if new_coord.bounds is not None:
                if (len(new_bounds) == len(new_coord.bounds) and
                        len(new_coord.points)*2 ==
                        len(new_bounds.flatten())):
                    new_coord.bounds = new_bounds
                else:
                    msg = ("Mismatch in bounds in existing"
                           " coord and updated metadata for "
                           " coord {}".format(coord_name))
                    raise ValueError(msg)
            else:
                if (len(new_coord.points)*2 ==
                        len(new_bounds.flatten())):
                    new_coord.bounds = new_bounds
                else:
                    msg = ("The shape of the bounds array should"
                           " be points.shape + (n_bounds,)"
                           "for coord= {}".format(coord_name))
                    raise ValueError(msg)
        if 'units' in changes:
            new_coord.units = changes['units']
        if warnings_on:
            msg = ("Updated coordinate "
                   "{}".format(coord_name) +
                   "with {}".format(changes))
            warnings.warn(msg)
    return result


def update_attribute(cube, attribute_name, changes, warnings_on=False):
    """Update the attribute in the cube.

    Args:
        cube (iris.cube.Cube):
            Cube containing combined data.
        attribute_name (string):
            Name of the attribute being updated.
        changes (object):
            attribute value or
            If changes = 'delete' the coordinate is deleted.

    Keyword Args:
        warnings_on (bool):
            If True output warnings for mismatching metadata.

    Returns:
        result (iris.cube.Cube):
            Cube with updated coordinate.

    Raises:
        UserWarning: Deleted attributes.
        UserWarning: Updated coordinate.

    """
    result = cube
    if changes == 'delete':
        result.attributes.pop(attribute_name)
        if warnings_on:
            msg = ("Deleted attribute "
                   "{}".format(attribute_name))
            warnings.warn(msg)
    else:
        result.attributes[attribute_name] = changes
        if warnings_on:
            msg = ("Adding or updating attribute "
                   "{} with {}".format(attribute_name,
                                       changes))
            warnings.warn(msg)
    return result


def amend_metadata(cube,
                   new_diagnostic_name,
                   data_type,
                   revised_coords,
                   revised_attributes,
                   warnings_on=False):
    """Amend the metadata in the combined cube.

    Args:
        cube (iris.cube.Cube):
            Cube containing combined data.
        new_diagnostic_name (str):
            New name for the combined diagnostic.
        data_type (numpy.dtype):
            data type of cube data.
        revised_coords (dict or None):
            Revised coordinates for combined cube.
        revised_attributes (dict or None):
            Revised attributes for combined cube.

    Keyword Args:
        warnings_on (bool):
            If True output warnings for mismatching metadata.

    Returns:
        result (iris.cube.Cube):
            Cube with corrected Metadata.

    """
    result = cube
    result.data = result.data.astype(data_type)
    result.rename(new_diagnostic_name)

    if revised_coords is not None:
        for key in revised_coords:
            # If and exising coordinate.
            if key in [coord.name() for coord in cube.coords()]:
                changes = revised_coords[key]
                result = update_coord(result, key, changes,
                                      warnings_on=warnings_on)
            else:
                changes = revised_coords[key]
                result = add_coord(result, key, changes,
                                   warnings_on=warnings_on)

    if revised_attributes is not None:
        for key in revised_attributes:
            changes = revised_attributes[key]
            result = update_attribute(result, key, changes,
                                      warnings_on=warnings_on)

    return result


def resolve_metadata_diff(cube1, cube2, warnings_on=False):
    """Resolve any differences in metadata between cubes.

    Args:
        cube1 (iris.cube.Cube):
            Cube containing data to be combined.
        cube2 (iris.cube.Cube):
            Cube containing data to be combined.

    Keyword Args:
        warnings_on (bool):
            If True output warnings for mismatching metadata.

    Returns:
        (tuple): tuple containing
            **result1** (iris.cube.Cube):
                Cube with corrected Metadata.
            **result2** (iris.cube.Cube):
                Cube with corrected Metadata.

    """
    result1 = cube1
    result2 = cube2
    cubes = iris.cube.CubeList([result1, result2])

    # Processing will be based on cube1 so any unmatching
    # attributes will be ignored

    # Find mismatching coords
    unmatching_coords = compare_coords(cubes)
    # If extra dim coord length 1 on cube1 then add to cube2
    for coord in unmatching_coords[0]:
        if coord not in unmatching_coords[1]:
            if len(result1.coord(coord).points) == 1:
                if result1.coord_dims(coord) is not None:
                    coord_dict = dict()
                    coord_dict['points'] = result1.coord(coord).points
                    coord_dict['bounds'] = result1.coord(coord).bounds
                    coord_dict['units'] = result1.coord(coord).units
                    coord_dict['metatype'] = 'DimCoord'
                    result2 = add_coord(result2, coord, coord_dict,
                                        warnings_on=warnings_on)
                    result2 = iris.util.as_compatible_shape(result2,
                                                            result1)
    # If extra dim coord length 1 on cube2 then delete from cube2
    for coord in unmatching_coords[1]:
        if coord not in unmatching_coords[0]:
            if len(result2.coord(coord).points) == 1:
                result2 = update_coord(result2, coord, 'delete',
                                       warnings_on=warnings_on)

    # If shapes still do not match Raise an error
    if result1.data.shape != result2.data.shape:
        msg = "Can not combine cubes, mismatching shapes"
        raise ValueError(msg)
    return result1, result2
