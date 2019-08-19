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

import numpy as np
from cf_units import Unit

import iris
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

    Kwargs:
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

    # return a new cube list from the inputs in which data units and datatypes
    # have been enforced
    new_cubes = _enforce_diagnostic_units_and_dtypes(cubes, inplace=False)
    # set up coordinates to check
    if coords is not None:
        all_coords = list(set(coords))
    else:
        all_coords = [
            coord.name() for cube in cubes for coord in cube.coords()]
        all_coords = list(set(all_coords))
    # modify the copied cubes in place
    _enforce_coordinate_units_and_dtypes(new_cubes, all_coords)

    if not enforce:
        # check each cube against its counterpart and fail if changed
        for cube, ref in zip(new_cubes, cubes):
            if cube.units != ref.units:
                msg = ('Units {} of {} cube do not conform to '
                       'expected standard {}')
                raise ValueError(msg.format(ref.units, ref.name(), cube.units))
            if cube.dtype != ref.dtype:
                msg = ('Datatype {} of {} cube does not conform to '
                       'expected standard {}')
                raise ValueError(msg.format(ref.dtype, ref.name(), cube.dtype))
            for coord in all_coords:
                try:
                    if cube.coord(coord).units != ref.coord(coord).units:
                        msg = ('Units {} of coordinate {} on {} cube do not '
                               'conform to expected standard {}')
                        raise ValueError(msg.format(
                            ref.coord(coord).units, ref.coord(coord).name(),
                            ref.name(), cube.coord(coord).units))
                    if cube.coord(coord).dtype != ref.coord(coord).dtype:
                        msg = ('Datatype {} of coordinate {} on {} cube does '
                               'not conform to expected standard {}')
                        raise ValueError(msg.format(
                            ref.coord(coord).dtype, ref.coord(coord).name(),
                            ref.name(), cube.coord(coord).dtype))
                except CoordinateNotFoundError:
                    pass

    return iris.cube.CubeList(new_cubes)


def _find_dict_key(input_key, error_msg):
    """
    If input_key is not in the DEFAULT_UNITS dict, test for substrings of
    input_key that are available.  This allows, for example, use of
    "temperature" and "probability" in the DEFAULT_UNITS dict to avoid multiple
    duplicate entries.

    Args:
        input_key (str):
            Key that didn't return an entry in DEFAULT_UNITS
        error_msg (str):
            Error to raise if no unique match is found

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
        msg = '{}, matching keys: {}'.format(error_msg, matching_keys)
        raise KeyError(msg)

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
            Tuple with string and type objectt identifying the required units
            and datatype

    Raises:
        KeyError:
            If the input_key (or suitable substring) is not present in
            DEFAULT_UNITS.
    """
    try:
        unit = DEFAULT_UNITS[key]["unit"]
    except KeyError:
        msg = "Name '{}' not defined in units.py"
        # hold the error and check for valid substrings
        key = _find_dict_key(key, msg.format(key))
        unit = DEFAULT_UNITS[key]["unit"]

    try:
        dtype = DEFAULT_UNITS[key]["dtype"]
    except KeyError:
        dtype = np.float32

    return unit, dtype


def _check_units_and_dtype(object, units, dtype):
    """
    Check whether the units and datatype of the input object conform
    to the standard given.

    Args:
        object (iris.cube.Cube or iris.coords.Coord):
            Cube or coordinate to be checked
        units (str):
            Required units
        dtype (type):
            Required datatype

    Returns:
        bool:
            True if object conforms; False if not
    """
    if Unit(object.units) != Unit(units):
        return False

    if object.dtype != dtype:
        return False

    return True


def _enforce_coordinate_units_and_dtypes(cubes, coordinates, inplace=True):
    """
    Function to enforce standard units and data types as defined in units.py.
    If undefined, the expected datatype is np.float32.

    By default the cube units are changed in place, but setting inplace to
    False will return a copy of the cubes, leaving the originals unchanged.

    Args:
        cubes (iris.cube.CubeList):
            List of cubes on which to enforce coordinate units and data types.
        coordinates (list):
            List of coordinates for which units and dtypes should be enforced.
            This is a list of coordinate names.
    Keyword Args:
        inplace (bool):
            If True (default) the cubes are modified in place, if False this
            function returns a modified copy of the cubes.
    Returns:
        cubes (iris.cube.CubeList or insitu):
            The input cubes with units and data types of the chosen coordinates
            set to match the definitions in units.py.
    Raises:
        KeyError: If coordinate to enforce is not defined in units.py
        ValueError: If requested unit conversion is not possible.
        ValueError: If a unit data type could not be converted without losing
                    significant information (e.g. rounding time to the nearest
                    hour when there are sub-hourly components).
    """
    if not inplace:
        cubes = [cube.copy() for cube in cubes]

    for cube in cubes:
        for coord_name in coordinates:
            unit, dtype = _get_required_units_and_dtype(coord_name)

            try:
                coordinate = cube.coord(coord_name)
                coordinate.convert_units(unit)
            except ValueError:
                msg = '{} units cannot be converted to "{}"'
                raise ValueError(msg.format(coord_name, unit))
            except CoordinateNotFoundError:
                pass
            else:
                if check_precision_loss(dtype, coordinate.points):
                    coordinate.points = coordinate.points.astype(dtype)
                else:
                    msg = ('Data type of coordinate "{}" could not be'
                           ' enforced without losing significant precision.')
                    raise ValueError(msg.format(coord_name))

    if not inplace:
        return cubes


def _enforce_diagnostic_units_and_dtypes(cubes, inplace=True):
    """
    Function to enforce diagnostic units and data types as defined in units.py.
    If undefined, the expected datatype is np.float32.

    By default the diagnostic units are changed in place, but setting inplace
    to False will return a copy of the cubes, leaving the originals unchanged.

    Args:
        cubes (iris.cube.CubeList):
            List of cubes on which to enforce diagnostic units and data types.
    Keyword Args:
        inplace (bool):
            If True (default) the cubes are modified in place, if False this
            function returns a modified copy of the cubes.
    Returns:
        cubes (iris.cube.CubeList or insitu):
            The input cubes with units and data types of the diagnostic
            set to match the definitions in units.py.
    Raises:
        KeyError: If coordinate to enforce is not defined in units.py
        ValueError: If requested unit conversion is not possible.
        ValueError: If a unit data type could not be converted without losing
                    significant information (e.g. removing significant
                    fractional components when converting to integers).
    """
    if not inplace:
        cubes = [cube.copy() for cube in cubes]

    for cube in cubes:
        unit, dtype = _get_required_units_and_dtype(cube.name())

        try:
            cube.convert_units(unit)
        except ValueError:
            msg = '{} units cannot be converted to "{}"'
            raise ValueError(msg.format(cube.name(), unit))
        else:
            if check_precision_loss(dtype, cube.data):
                cube.data = cube.data.astype(dtype)
            else:
                msg = ('Data type of diagnostic "{}" could not be'
                       ' enforced without losing significant precision.')
                raise ValueError(msg.format(cube.name()))

    if not inplace:
        return cubes


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
    Keyword Args:
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
