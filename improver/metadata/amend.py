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
from dateutil import tz

from improver.metadata.constants.mo_attributes import (
    GRID_ID_LOOKUP, MOSG_GRID_DEFINITION)


def update_stage_v110_metadata(cube):
    """Translates attributes relating to the grid_id attribute from StaGE
    version 1.1.0 to later StaGE versions.
    Cubes that have no "grid_id" attribute are not recognised as v1.1.0 and
    are ignored.

    Args:
        cube (iris.cube.Cube):
            Cube to modify attributes in (modified in place)

    Returns:
        None
    """
    try:
        grid_id = cube.attributes.pop('grid_id')
    except KeyError:
        # Not a version 1.1.0 grid, do nothing
        return
    cube.attributes.update(MOSG_GRID_DEFINITION[GRID_ID_LOOKUP[grid_id]])
    cube.attributes['mosg__grid_version'] = '1.1.0'


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
