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
            The "value" item is either the string "delete", or the new value
            of the attribute required.
    """
    for attribute_name, value in attributes_dict.items():
        if value == "delete":
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


def _update_cell_methods(cube, cell_method_definition):
    """Update cell methods. An "action" keyword is expected within the
    cell method definition to specify whether the cell method is to be added
    or deleted.

    The cube will be modified in-place.

    Args:
        cube (iris.cube.Cube):
            Cube containing cell methods that will be updated.
        cell_method_definition (dict):
            A dictionary which must contain an "action" keyword with a value of
            either "add" or "delete", which determines whether to add or delete
            the cell method. The rest of the keys are passed to the
            iris.coords.CellMethod function. Of these keys, "method", is
            compulsory, and "comments", "coords" and "intervals" are optional.
            If any additional keys are provided in the dictionary they are
            ignored.

    Raises:
        ValueError: If no action is specified for the cell method, then raise
                    an error.
        ValueError: If no method is specified for the cell method, then raise
                    an error.

    """
    if "action" not in cell_method_definition:
        msg = ("No action has been specified within the cell method "
               "definition. Please specify an action either 'add' or 'delete'."
               "The cell method definition provided "
               "was {}".format(cell_method_definition))
        raise ValueError(msg)

    if not cell_method_definition["method"]:
        msg = ("No method has been specified within the cell method "
               "definition. Please specify a method to describe "
               "the name of the operation, see iris.coords.CellMethod."
               "The cell method definition provided "
               "was {}".format(cell_method_definition))
        raise ValueError(msg)

    for key in ["coords", "intervals", "comments"]:
        if key not in cell_method_definition:
            cell_method_definition[key] = ()

    if not cell_method_definition["coords"]:
        coords = ()
    else:
        coords = tuple([cell_method_definition["coords"]])

    cell_method = iris.coords.CellMethod(
        method=cell_method_definition["method"],
        coords=coords,
        intervals=cell_method_definition["intervals"],
        comments=cell_method_definition["comments"])

    cm_list = []
    for cm in cube.cell_methods:
        if cm == cell_method and cell_method_definition["action"] == "delete":
            continue
        cm_list.append(cm)

    if cell_method_definition["action"] == "add":
        if cell_method not in cube.cell_methods:
            cm_list.append(cell_method)

    cube.cell_methods = cm_list


def amend_metadata(cube,
                   name=None,
                   data_type=None,
                   coordinates=None,
                   attributes=None,
                   cell_methods=None,
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
        cell_methods (dict or None):
            Cell methods for modification within the incoming cube.
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

        cell_methods: Cell methods are specified using a all arguments taken
            by iris.coords.CellMethod. Additionally, an action key is required
            to specify whether the specified cell method will be added or
            deleted.
            For example:
                {
                    "action": "delete",
                    "method": "point",
                    "coords": "time"
                }

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

    if cell_methods is not None:
        for key in cell_methods:
            _update_cell_methods(result, cell_methods[key])

    if units is not None:
        result.convert_units(units)

    return result


def resolve_metadata_diff(cube1, cube2, warnings_on=False):
    """Resolve any differences in metadata between cubes. This involves
    identifying coordinates that are mismatching between the cubes and
    attempting to add this coordinate where it is missing. This makes use of
    the points, bounds, units and attributes, as well as the coordinate type
    i.e. DimCoord or AuxCoord.

    Args:
        cube1 (iris.cube.Cube):
            Cube containing data to be combined.
        cube2 (iris.cube.Cube):
            Cube containing data to be combined.
        warnings_on (bool):
            If True output warnings for mismatching metadata.

    Returns:
        (tuple): tuple containing:
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
                if len(result1.coord_dims(coord)) > 0:
                    coord_dict = dict()
                    coord_dict['points'] = result1.coord(coord).points
                    coord_dict['bounds'] = result1.coord(coord).bounds
                    coord_dict['units'] = result1.coord(coord).units
                    coord_dict['attributes'] = result1.coord(coord).attributes
                    coord_dict['metatype'] = 'DimCoord'
                    if result1.coord(coord).var_name is not None:
                        coord_dict['var_name'] = result1.coord(coord).var_name
                    result2 = add_coord(result2, coord, coord_dict,
                                        warnings_on=warnings_on)
                    result2 = iris.util.as_compatible_shape(result2,
                                                            result1)
    # If extra dim coord length 1 on cube2 then delete from cube2
    for coord in unmatching_coords[1]:
        if coord not in unmatching_coords[0]:
            if len(result2.coord(coord).points) == 1:
                result2 = _update_coord(result2, coord, 'delete',
                                        warnings_on=warnings_on)

    # If shapes still do not match Raise an error
    if result1.data.shape != result2.data.shape:
        msg = "Can not combine cubes, mismatching shapes"
        raise ValueError(msg)
    return result1, result2


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
