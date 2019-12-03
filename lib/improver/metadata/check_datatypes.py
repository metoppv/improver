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
"""Utilities for datatype checking"""

import numpy as np
from cf_units import Unit
from iris.exceptions import CoordinateNotFoundError

TIME_REFERENCE_DTYPE = np.int64


def check_cube_not_float64(cube, fix=False):
    """Check a cube does not contain any float64 data, excepting time
    coordinates. The cube can be modified in place, if the fix keyword is
    specified to be True.

    Args:
        cube (iris.cube.Cube):
            The input cube that will be checked for float64 inclusion.
        fix (bool):
            If fix is True, then the cube is amended to not include float64
            data, otherwise, an error will be raised if float64 data is found.

    Raises:
        TypeError : Raised if 64 bit values are found in the cube.
    """
    if cube.dtype == np.float64:
        if fix:
            cube.data = cube.data.astype(np.float32)
        else:
            raise TypeError("64 bit cube not allowed: {!r}".format(cube))

    for coord in cube.coords():
        if coord.units.is_time_reference():
            continue

        if coord.points.dtype == np.float64:
            if fix:
                coord.points = coord.points.astype(np.float32)
            else:
                raise TypeError(
                    "64 bit coord points not allowed: {} in {!r}".format(
                        coord, cube))
        if coord.bounds is not None and coord.bounds.dtype == np.float64:
            if fix:
                coord.bounds = coord.bounds.astype(np.float32)
            else:
                raise TypeError(
                    "64 bit coord bounds not allowed: {} in {!r}".format(
                        coord, cube))


def _construct_object_list(cube, coord_names):
    """
    Construct a list of objects

    Args:
        cube (iris.cube.Cube):
            Cube to append to object list
        coord_names (list of str or None):
            List of coordinate names to take from cube.  If None, adds all
            coordinates present on the input cube.

    Returns:
        list of obj:
            List containing the original cube and specified coordinates
    """
    object_list = []
    object_list.append(cube)
    if coord_names is not None:
        for coord in coord_names:
            try:
                object_list.append(cube.coord(coord))
            except CoordinateNotFoundError:
                pass
    else:
        object_list.extend(cube.coords())
    return object_list


def _get_required_datatype(item):
    """
    Returns the required datatype of the object (cube or coordinate)
    passed in, according to the IMPROVER standard.  Input object must
    have attributes "units" and "dtype".
    """
    if item.units.is_time_reference():
        return TIME_REFERENCE_DTYPE
    if issubclass(item.dtype.type, np.integer):
        return np.int32
    return np.float32


def check_datatypes(cube, coords=None):
    """
    Function to check the datatypes of cube diagnostics and coordinates
    against the expected standard.  The default datatype is float32;
    integer quantities are expected to be 32-bit with the exception of
    absolute time.

    Args:
        cube (iris.cube.Cube):
            Cube to be checked
        coords (list or None):
            List of coordinate names to check.  If None, checks all
            coordinates present on the input cube.

    Raises:
        ValueError: if the input cube does not conform to the datatypes
            standard
    """
    # construct a list of objects (cube and coordinates) to be checked
    object_list = _construct_object_list(cube, coords)

    msg = ('{} datatype {} does not conform to expected standard ({})\n')
    error_string = ''
    for item in object_list:
        # allow string-type objects
        if item.dtype.type == np.unicode_:
            continue

        # check numerical datatypes
        required_dtype = _get_required_datatype(item)
        if item.dtype.type != required_dtype:
            error_string += msg.format(item.name(), item.dtype, required_dtype)

        if (hasattr(item, "bounds") and item.bounds is not None
                and item.bounds.dtype.type != required_dtype):
            error_string += msg.format(
                item.name()+' bounds', item.bounds.dtype, required_dtype)

    # if any data was non-compliant, raise details here
    if error_string:
        raise ValueError(error_string)


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

    if obj.dtype.type != dtype:
        return False

    return True


def check_time_coordinate_metadata(cube):
    """
    Function to check time coordinates against the expected standard. The
    standard for time coordinates is due to technical requirements and if
    violated the data integrity cannot be guaranteed; so if time coordinates
    are non-conformant an error is raised.

    Args:
        cube (iris.cube.Cube):
            Cube to be checked

    Raises:
        ValueError: if any the input cube's time coordinates do not conform
            to the standard datatypes and units
    """
    error_string = ''
    for time_coord in ["time", "forecast_reference_time", "forecast_period"]:
        try:
            coord = cube.coord(time_coord)
        except CoordinateNotFoundError:
            continue

        if coord.units.is_time_reference():
            required_unit = "seconds since 1970-01-01 00:00:00"
            required_dtype = TIME_REFERENCE_DTYPE
        else:
            required_unit = "seconds"
            required_dtype = np.int32

        if not _check_units_and_dtype(coord, required_unit, required_dtype):
            msg = ('Coordinate {} does not match required '
                   'standard (units {}, datatype {})\n')
            error_string += msg.format(
                coord.name(), required_unit, required_dtype)

    # if non-compliance was encountered, raise all messages here
    if error_string:
        raise ValueError(error_string)
