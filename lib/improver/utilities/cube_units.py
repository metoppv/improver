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
""" Provides support utilities for manipulating cube units."""

import iris
import numpy as np
from cf_units import Unit
from iris.exceptions import CoordinateNotFoundError

from improver.units import DEFAULT_UNITS


def enforce_units_and_dtypes(cubes, coords=None, enforce=True):
    """
    Function to check the units and datatypes of cube diagnostics and
    coordinates against the manifest in improver.units, with option to
    enforce or fail for non-conforming data.

    Args:
        cubes (iris.cube.Cube or iris.cube.CubeList):
            Cube or list of cubes to be checked
        coords (list or None):
            List of coordinate names to check.  If None, checks all
            coordinates present on the input cubes.
        enforce (bool):
            If True, this function returns a list of conformant cubes.
            If False, a ValueError is thrown if the cubes do not conform.
    Raises:
        ValueError: if "enforce=False" and the input cubes do not conform
            to the datatypes and units standard.
    Returns:
        new_cubes (iris.cube.CubeList):
            New cubelist with conformant datatypes and units
    """
    # convert input to CubeList
    if isinstance(cubes, iris.cube.Cube):
        cubes = [cubes]

    # create a list of copied cubes to modify
    new_cubes = [cube.copy() for cube in cubes]

    # construct a list of objects (cubes and coordinates) to be checked
    object_list = []
    for cube in new_cubes:
        object_list.append(cube)
        if coords is not None:
            for coord in coords:
                try:
                    object_list.append(cube.coord(coord))
                except CoordinateNotFoundError:
                    pass
        else:
            object_list.extend(cube.coords())

    error_string = ''
    for item in object_list:
        units, dtype = _get_required_units_and_dtype(item.name())

        if not enforce:
            # if not enforcing, throw an error if non-compliant
            conforms = _check_units_and_dtype(item, units, dtype)
            if not conforms:
                msg = ('{} with units {} and datatype {} does not conform'
                       ' to expected standard (units {}, datatype {})\n')
                msg = msg.format(item.name(), item.units, item.dtype,
                                 units, dtype)
                error_string += msg
            continue

        # attempt to convert units and record any errors
        try:
            item.convert_units(units)
        except ValueError:
            msg = '{} units cannot be converted to "{}"\n'
            error_string += msg

        # attempt to convert datatype and record any errors
        try:
            if isinstance(item, iris.cube.Cube):
                _convert_diagnostic_dtype(item, dtype)
            else:
                _convert_coordinate_dtype(item, dtype)
        except ValueError as cause:
            error_string += cause + '\n'

    # if any errors were raised, re-raise with all messages here
    if error_string:
        msg = 'The following errors were raised during processing:\n'
        raise ValueError(msg+error_string)

    return iris.cube.CubeList(new_cubes)


def _find_dict_key(input_key):
    """
    If input_key is not in the DEFAULT_UNITS dict, test for substrings of
    input_key that are available.  This allows, for example, use of
    "temperature" and "probability" in the DEFAULT_UNITS dict to avoid multiple
    duplicate entries.

    Args:
        input_key (str):
            Key that didn't return an entry in DEFAULT_UNITS

    Returns:
        str: New key to identify required entry

    Raises:
        KeyError: If the function finds either zero or multiple matches

    """
    if "probability" in input_key:
        # this avoids duplicate results from key matching below
        return "probability"

    matching_keys = []
    for key in DEFAULT_UNITS.keys():
        if key in input_key:
            matching_keys.append(key)
    if len(matching_keys) != 1:
        msg = ("Name '{}' is not uniquely defined in units.py; "
               "matching keys: {}")
        raise KeyError(msg.format(input_key, matching_keys))

    return matching_keys[0]


def _get_required_units_and_dtype(key):
    """
    Read DEFAULT_UNITS dict and return the required units and datatypes
    for the given coordinate / diagnostic name.

    Args:
        key (str):
            String name of coordinate or diagnostic to be checked
    Returns:
        units, dtype (tuple):
            Tuple with string and type object identifying the required units
            and datatype
    Raises:
        KeyError:
            If the input_key (or suitable substring) is not present in
            DEFAULT_UNITS.
    """
    try:
        unit = DEFAULT_UNITS[key]["unit"]
    except KeyError:
        # hold the error and check for valid substrings
        key = _find_dict_key(key)
        unit = DEFAULT_UNITS[key]["unit"]

    try:
        dtype = DEFAULT_UNITS[key]["dtype"]
    except KeyError:
        dtype = np.float32

    return unit, dtype


def _check_units_and_dtype(obj, units, dtype):
    """
    Check whether the units and datatype of the input object conform
    to the standard given.

    Args:
        obj (iris.cube.Cube or iris.coords.Coord):
            Cube or coordinate to be checked
        units (str):
            Required units
        dtype (type):
            Required datatype

    Returns:
        bool:
            True if object conforms; False if not
    """
    if Unit(obj.units) != Unit(units):
        return False

    if obj.dtype != dtype:
        return False

    return True


def _convert_coordinate_dtype(coord, dtype):
    """
    Convert a coordinate to the required units and datatype.

    Args:
        coord (iris.coords.Coord):
            Coordinate instance to be modified in place
        dtype (type):
            Required datatype
    """
    if check_precision_loss(dtype, coord.points):
        coord.points = coord.points.astype(dtype)
    else:
        msg = ('Data type of coordinate "{}" could not be'
               ' enforced without losing significant precision.')
        raise ValueError(msg.format(coord.name()))


def _convert_diagnostic_dtype(cube, dtype):
    """
    Convert cube data to the required units and datatype.

    Args:
        cube (iris.cube.Cube):
            Cube to be modified in place
        dtype (type):
            Required datatype
    """
    # if units conversion succeeded, convert datatype
    if check_precision_loss(dtype, cube.data):
        cube.data = cube.data.astype(dtype)
    else:
        msg = ('Data type of diagnostic "{}" could not be'
               ' enforced without losing significant precision.')
        raise ValueError(msg.format(cube.name()))


def check_precision_loss(dtype, data, precision=5):
    """
    This function checks that when converting data types there is not a loss
    of significant information. Float to integer conversion, and integer to
    integer conversion are covered by this function. Float to float conversion
    may be lossy if changing from 64 bit to 32 bit floats, but changes at this
    precision are not captured here by design.

    If the conversion is lossless (to the defined precision) this function
    returns True. If there is loss, the function returns False.

    .. See the documentation for examples of where such loss is important.
    .. include:: extended_documentation/utilities/cube_units/
       check_precision_loss_examples.rst

    Args:
        dtype (dtype):
            The data type to which the data is being converted.
        data (numpy.ndarray):
            The data that is to be checked for precision loss under data type
            conversion.
        precision (int):
            The number of decimal places beyond which differences are ignored.
    Returns:
        bool:
            True if the conversion is lossless to the given precision.
            False if the conversion if lossy to the given precision.
    """
    if not np.issubdtype(dtype, np.integer):
        return True
    if np.issubdtype(data.dtype, np.integer):
        values = dtype(data)
        integers = data
    else:
        values = np.round(data, precision)
        _, integers = np.modf(values)

    return (values == integers).all()
