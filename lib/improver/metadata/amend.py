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
"""Module containing utilities for modifying cube metadata"""

import warnings
from datetime import datetime

import iris
import numpy as np
from dateutil import tz

from improver.metadata.constants.mo_attributes import (
    MOSG_GRID_DEFINITION, GRID_ID_LOOKUP)
from improver.utilities.cube_manipulation import compare_coords


def update_stage_v110_metadata(cube):
    """Translates meta-data relating to the grid_id attribute from StaGE
    version 1.1.0 to later StaGE versions.
    Cubes that have no "grid_id" attribute are not recognised as v1.1.0 and
    are ignored.

    Args:
        cube (iris.cube.Cube):
            Cube to modify meta-data in (modified in place)

    Returns:
        bool:
            True if meta-data have been changed by this function.
    """
    try:
        grid_id = cube.attributes.pop('grid_id')
    except KeyError:
        # Not a version 1.1.0 grid, so exit.
        return False
    cube.attributes.update(MOSG_GRID_DEFINITION[GRID_ID_LOOKUP[grid_id]])
    cube.attributes['mosg__grid_version'] = '1.1.0'
    return True


def amend_attributes(cube, attributes_dict):
    """
    Add, update or remove attributes from a cube.  Modifies cube in place.

    Args:
        cube (iris.cube.Cube):
            Input cube
        attributes_dict (dict):
            Dictionary containing items of the form {attribute_name: value}.
            The "value" item is either the string "remove" or the new value
            of the attribute required.
    """
    for attribute_name, value in attributes_dict.items():
        if value == "remove":
            cube.attributes.pop(attribute_name, None)
        else:
            cube.attributes[attribute_name] = value


def add_coord(cube, coord_name, changes, warnings_on=False):
    """Add coord to the cube.

    Args:
        cube (iris.cube.Cube):
            Cube containing combined data.
        coord_name (str):
            Name of the coordinate being added.
        changes (dict):
            Details of coordinate to be added to the cube, with string keys.
            Valid keys are 'metatype' (which should have value 'DimCoord' or
            'AuxCoord'), 'points', 'bounds', 'units', 'attributes' and
            'var_name'. Any other key strings in the dictionary are ignored.
            More detail is available in
            :func:`improver.metadata.amend.amend_metadata`
        warnings_on (bool):
            If True output warnings for mismatching metadata.

    Returns:
        iris.cube.Cube:
            Cube with added coordinate.

    Raises:
        ValueError: Trying to add new coord but no points defined.
        ValueError: Can not add a coordinate of length > 1
        UserWarning: adding new coordinate.

    """
    result = cube.copy()
    # Get the points for the coordinate to be added.
    # The points must be defined.
    if 'points' in changes:
        if len(changes['points']) != 1:
            msg = ("Can not add a coordinate of length > 1,"
                   " coord  = {}".format(coord_name))
            raise ValueError(msg)
        points = changes['points']
    else:
        msg = ("Trying to add new coord but no points defined"
               " in metadata, coord  = {}".format(coord_name))
        raise ValueError(msg)

    # Get the bounds, units, var_name and attributes from the
    # changes dictionary.
    bounds = None
    if 'bounds' in changes:
        bounds = changes['bounds']
    units = None
    if 'units' in changes:
        units = changes['units']
    var_name = None
    if 'var_name' in changes:
        var_name = changes['var_name']
    attributes = None
    if 'attributes' in changes:
        attributes = changes['attributes']

    # Get the type of the coordinate, if specified.
    metatype = 'DimCoord'
    if 'metatype' in changes:
        if changes['metatype'] == 'AuxCoord':
            new_coord_method = iris.coords.AuxCoord
            metatype = 'AuxCoord'
        else:
            new_coord_method = iris.coords.DimCoord
    else:
        new_coord_method = iris.coords.DimCoord

    new_coord = new_coord_method(
        points=points, bounds=bounds, units=units, attributes=attributes)
    new_coord.rename(coord_name)
    new_coord.var_name = var_name

    result.add_aux_coord(new_coord)
    if metatype == 'DimCoord':
        result = iris.util.new_axis(result, coord_name)
    if warnings_on:
        msg = ("Adding new coordinate "
               "{} with {}".format(coord_name,
                                   changes))
        warnings.warn(msg)
    return result


def _update_coord(cube, coord_name, changes, warnings_on=False):
    """Amend the metadata in the combined cube.

    Args:
        cube (iris.cube.Cube):
            Cube containing combined data.
        coord_name (str):
            Name of the coordinate being updated.
        changes (str or dict):
            Details on coordinate to be updated.
            If changes = 'delete' the coordinate is deleted.
            More detail is available in
            :func:`improver.metadata.amend.amend_metadata`
        warnings_on (bool):
            If True output warnings for mismatching metadata.

    Returns:
        iris.cube.Cube:
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
    result = cube.copy()
    new_coord = result.coord(coord_name)
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
        if 'units' in changes and ('points' in changes or 'bounds' in changes):
            msg = ("When updating a coordinate, the 'units' and "
                   "'points'/'bounds' can only be updated independently. "
                   "The changes requested were {}".format(changes))
            raise ValueError(msg)
        if 'points' in changes:
            new_points = np.array(changes['points'], dtype=new_coord.dtype)
            if new_points.dtype == np.float64:
                new_points = new_points.astype(np.float32)
            if (len(new_points) ==
                    len(new_coord.points)):
                new_coord.points = new_points
            else:
                msg = ("Mismatch in points in existing"
                       " coord and updated metadata for "
                       " coord {}".format(coord_name))
                raise ValueError(msg)
        if 'bounds' in changes:
            new_bounds = np.array(changes['bounds'], dtype=new_coord.dtype)
            if new_bounds.dtype == np.float64:
                new_bounds = new_bounds.astype(np.float32)
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
            new_coord.convert_units(changes["units"])
        if 'attributes' in changes:
            new_coord.attributes.update(changes["attributes"])
        if warnings_on:
            msg = ("Updated coordinate "
                   "{}".format(coord_name) +
                   "with {}".format(changes))
            warnings.warn(msg)
    return result


def _update_attribute(cube, attribute_name, changes, warnings_on=False):
    """Update the attribute in the cube.

    Args:
        cube (iris.cube.Cube):
            Cube containing combined data.
        attribute_name (str):
            Name of the attribute being updated.
        changes (object):
            attribute value or
            If changes = 'delete' the coordinate is deleted.
            More detail is available in
            :func:`improver.metadata.amend.amend_metadata`
        warnings_on (bool):
            If True output warnings for mismatching metadata.

    Returns:
        iris.cube.Cube:
            Cube with updated coordinate.

    Raises:
        UserWarning: Deleted attributes.
        UserWarning: Updated coordinate.

    """
    result = cube.copy()
    if changes == 'delete':
        result.attributes.pop(attribute_name, None)
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
                   name=None,
                   data_type=None,
                   coordinates=None,
                   attributes=None,
                   units=None,
                   warnings_on=False):
    """Amend the metadata in the incoming cube. Please note that if keyword
    arguments to this function are supplied by unpacking a dictionary, then
    the keys of the dictionary need to correspond to the keyword arguments.

    Args:
        cube (iris.cube.Cube):
            Input cube.
        name (str):
            New name for the diagnostic.
        data_type (numpy.dtype):
            Data type that the cube data will be converted to.
        coordinates (dict or None):
            Revised coordinates for incoming cube.
        attributes (dict or None):
            Revised attributes for incoming cube.
        units (str, cf_units.Unit or None):
            Units for use in converting the units of the input cube.
        warnings_on (bool):
            If True output warnings for mismatching metadata.

    Returns:
        iris.cube.Cube:
            Cube with corrected metadata.

    Example inputs:
    ::

        coordinates: The name of the coordinate is required, in addition
            to details regarding the coordinate required by the coordinate.
            The type of the coordinate is specified using a "metatype" key.
            Available keys are:
                * metatype: Type of coordinate e.g. DimCoord or AuxCoord.
                * points: Point values for coordinate.
                * bounds: Bounds associated with each coordinate point.
                * units: Units of coordinate
            For example:
            "threshold": {
                "metatype": "DimCoord",
                "points": [1.0],
                "bounds": [[0.1, 1.0]],
                "units": "mm hr-1"
            }

        attributes: Attributes are specified using the name of the attribute
            to be modified as the key. For all keys, apart from "history",
            the value of the items in the dictionary can either be the value
            that will be added e.g. "source": "Met Office Radarnet" will add
            a "source" attribute with the value of "Met Office Radarnet", or
            "source": "delete" will delete the source attribute.
            For non-history attributes, the available options are e.g.:
                * "source": "Met Office Radarnet"
                * "source": "delete"
            For example:
            {
                "experiment_number": "delete",
                "field_code": "delete",
                "source": "Met Office Radarnet",
            }
            As the history attribute requires a timestamp to be created that
            represents now, this needs to be automatically created at runtime.
            If a history attribute is added, a name is also added.
            For the history attribute, the available options are e.g.
                * "history": ["add", "Nowcast"]
                * "history": "delete"

    """
    result = cube.copy()
    if data_type:
        result.data = result.data.astype(data_type)
    if name:
        result.rename(name)

    if coordinates is not None:
        for key in coordinates:
            # If the coordinate already exists in the cube, then update it.
            # Otherwise, add the coordinate.
            if key in [coord.name() for coord in cube.coords()]:
                changes = coordinates[key]
                result = _update_coord(result, key, changes,
                                       warnings_on=warnings_on)
            else:
                changes = coordinates[key]
                result = add_coord(result, key, changes,
                                   warnings_on=warnings_on)

    if attributes is not None:
        for key in attributes:
            changes = attributes[key]
            result = _update_attribute(result, key, changes,
                                       warnings_on=warnings_on)

    if units is not None:
        result.convert_units(units)

    return result


def set_history_attribute(cube, value, append=False):
    """Add a history attribute to a cube. This uses the current datetime to
    generate the timestamp for the history attribute. The new history attribute
    will overwrite any existing history attribute unless the "append" option is
    set to True. The history attribute is of the form "Timestamp: Description".

    Args:
        cube (iris.cube.Cube):
            The cube to which the history attribute will be added.
        value (str):
            String defining details to be included in the history attribute.
        append (bool):
            If True, add to the existing history rather than replacing the
            existing attribute.  Default is False.
    """
    tzinfo = tz.tzoffset('Z', 0)
    timestamp = datetime.strftime(datetime.now(tzinfo), "%Y-%m-%dT%H:%M:%S%Z")
    new_history = "{}: {}".format(timestamp, value)
    if append and "history" in cube.attributes.keys():
        cube.attributes["history"] += '; {}'.format(new_history)
    else:
        cube.attributes["history"] = new_history
