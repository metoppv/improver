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
"""Module containing utilities for modifying cube metadata."""

import hashlib
import pickle
import re
import warnings
from datetime import datetime

import iris
import numpy as np
from dateutil import tz

from improver.utilities.cube_manipulation import compare_coords

GRID_TYPE = 'standard'
STAGE_VERSION = '1.3.0'

# Define current StaGE grid metadata
MOSG_GRID_DEFINITION = {
    'uk_ens': {'mosg__grid_type': GRID_TYPE,
               'mosg__model_configuration': 'uk_ens',
               'mosg__grid_domain': 'uk_extended',
               'mosg__grid_version': STAGE_VERSION},
    'gl_ens': {'mosg__grid_type': GRID_TYPE,
               'mosg__model_configuration': 'gl_ens',
               'mosg__grid_domain': 'global',
               'mosg__grid_version': STAGE_VERSION},
    'uk_det': {'mosg__grid_type': GRID_TYPE,
               'mosg__model_configuration': 'uk_det',
               'mosg__grid_domain': 'uk_extended',
               'mosg__grid_version': STAGE_VERSION},
    'gl_det': {'mosg__grid_type': GRID_TYPE,
               'mosg__model_configuration': 'gl_det',
               'mosg__grid_domain': 'global',
               'mosg__grid_version': STAGE_VERSION}
}


# Define correct v1.2.0 meta-data for v1.1.0 data.
GRID_ID_LOOKUP = {'enukx_standard_v1': 'uk_ens',
                  'engl_standard_v1': 'gl_ens',
                  'ukvx_standard_v1': 'uk_det',
                  'glm_standard_v1': 'gl_det'}


def update_stage_v110_metadata(cube):
    """Translates meta-data relating to the grid_id attribute from StaGE
    version 1.1.0 to later StaGE versions.
    Cubes that have no "grid_id" attribute are not recognised as v1.1.0 and
    are ignored.

    Args:
        cube (iris.cube.Cube):
            Cube to modify meta-data in (modified in place)

    Returns:
        boolean (bool):
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
            :func:`improver.utilities.cube_metadata.amend_metadata`
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
    result = cube
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

    try:
        new_coord = new_coord_method(
            standard_name=coord_name, var_name=var_name, points=points,
            bounds=bounds, units=units, attributes=attributes)
    except ValueError as cause:
        if 'is not a valid standard_name' in str(cause):
            new_coord = new_coord_method(
                long_name=coord_name, var_name=var_name, points=points,
                bounds=bounds, units=units, attributes=attributes)
        else:
            raise ValueError(cause)

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
        coord_name (str):
            Name of the coordinate being updated.
        changes (str or dict):
            Details on coordinate to be updated.
            If changes = 'delete' the coordinate is deleted.
            More detail is available in
            :func:`improver.utilities.cube_metadata.amend_metadata`
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
        if 'units' in changes and ('points' in changes or 'bounds' in changes):
            msg = ("When updating a coordinate, the 'units' and "
                   "'points'/'bounds' can only be updated independently. "
                   "The changes requested were {}".format(changes))
            raise ValueError(msg)
        if 'points' in changes:
            new_points = np.array(changes['points'])
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
            new_bounds = np.array(changes['bounds'])
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


def update_attribute(cube, attribute_name, changes, warnings_on=False):
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
            :func:`improver.utilities.cube_metadata.amend_metadata`
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
        result.attributes.pop(attribute_name, None)
        if warnings_on:
            msg = ("Deleted attribute "
                   "{}".format(attribute_name))
            warnings.warn(msg)
    elif "add" in changes:
        if attribute_name in ["history"]:
            new_history = changes
            new_history.remove("add")
            add_history_attribute(result, new_history[0])
        else:
            msg = ("Only the history attribute can be added. "
                   "The attribute specified was {}".format(attribute_name))
            raise ValueError(msg)
    else:
        result.attributes[attribute_name] = changes
        if warnings_on:
            msg = ("Adding or updating attribute "
                   "{} with {}".format(attribute_name,
                                       changes))
            warnings.warn(msg)
    return result


def update_cell_methods(cube, cell_method_definition):
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
        result (iris.cube.Cube):
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
    result = cube
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
                result = update_coord(result, key, changes,
                                      warnings_on=warnings_on)
            else:
                changes = coordinates[key]
                result = add_coord(result, key, changes,
                                   warnings_on=warnings_on)

    if attributes is not None:
        for key in attributes:
            changes = attributes[key]
            result = update_attribute(result, key, changes,
                                      warnings_on=warnings_on)

    if cell_methods is not None:
        for key in cell_methods:
            update_cell_methods(result, cell_methods[key])

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
                result2 = update_coord(result2, coord, 'delete',
                                       warnings_on=warnings_on)

    # If shapes still do not match Raise an error
    if result1.data.shape != result2.data.shape:
        msg = "Can not combine cubes, mismatching shapes"
        raise ValueError(msg)
    return result1, result2


def delete_attributes(cube, patterns):
    """
    Delete attributes that are complete or partial matches to elements in the
    list patterns.

    Args:
        cube (iris.cube.Cube):
            The cube from which attributes are to be deleted.
        patterns (list or tuple):
            A list of strings that match or partially match the keys of
            attributes to be deleted from the cube.
    """

    if not isinstance(patterns, (tuple, list)):
        patterns = [patterns]

    grid_attributes = []
    for pattern in patterns:
        grid_attributes.extend([k for k in cube.attributes if pattern in k])

    grid_attributes = list(set(grid_attributes))

    for key in grid_attributes:
        cube.attributes.pop(key)


def add_history_attribute(cube, value, append=False):
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


def probability_cube_name_regex(cube_name):
    """
    Regular expression matching IMPROVER probability cube name.  Returns
    None if the cube_name does not match the regular expression (ie does
    not start with 'probability_of').

    Args:
        cube_name (str):
            Probability cube name
    """
    regex = re.compile(
        '(probability_of_)'  # always starts this way
        '(?P<diag>.*?)'      # named group for the diagnostic name
        '(_in_vicinity|)'    # optional group, may be empty
        '(?P<thresh>_above_threshold|_below_threshold|_between_thresholds|$)')
    return regex.match(cube_name)


def in_vicinity_name_format(cube_name):
    """Generate the correct name format for an 'in_vicinity' probability
    cube, taking into account the 'above/below_threshold' or
    'between_thresholds' suffix required by convention.

    Args:
        cube_name (str):
            The non-vicinity probability cube name to be formatted.

    Returns:
        new_cube_name (str):
            Correctly formatted name following the accepted convention e.g.
            'probability_of_X_in_vicinity_above_threshold'.
    """
    regex = probability_cube_name_regex(cube_name)
    new_cube_name = 'probability_of_{diag}_in_vicinity{thresh}'.format(
        **regex.groupdict())
    return new_cube_name


def extract_diagnostic_name(cube_name):
    """
    Extract the standard or long name X of the diagnostic from a probability
    cube name of the form 'probability_of_X_above/below_threshold',
    'probability_of_X_between_thresholds', or
    'probability_of_X_in_vicinity_above/below_threshold'.

    Args:
        cube_name (str):
            The probability cube name

    Returns:
        diagnostic_name (str):
            The name of the diagnostic underlying this probability

    Raises:
        ValueError: If the input name does not match the expected regular
            expression (ie if cube_name_regex(cube_name) returns None).
    """
    try:
        diagnostic_name = probability_cube_name_regex(cube_name).group('diag')
    except AttributeError:
        raise ValueError(
            'Input {} is not a valid probability cube name'.format(cube_name))
    return diagnostic_name


def generate_hash(data_in):
    """
    Generate a hash from the data_in that can be used to uniquely identify
    equivalent data_in.

    Args:
        data_in (any):
            The data from which a hash is to be generated. This can be of any
            type that can be pickled.
    Returns:
        hash (str):
            A hexidecimal hash representing the data.
    """
    hashable_type = pickle.dumps(data_in)
    hash_result = hashlib.md5(hashable_type).hexdigest()
    return hash_result


def create_coordinate_hash(cube):
    """
    Generate a hash based on the input cube's x and y coordinates. This
    acts as a unique identifier for the grid which can be used to allow two
    grids to be compared.

    Args:
        cube (iris.cube.Cube):
            The cube from which x and y coordinates will be used to
            generate a hash.
    Returns:
        coordinate_hash (str):
            A hash created using the x and y coordinates of the input cube.
    """
    hashable_data = [cube.coord(axis='x'), cube.coord(axis='y')]
    return generate_hash(hashable_data)
