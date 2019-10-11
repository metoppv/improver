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

from improver.metadata.constants.datatypes import (
    INTEGER_QUANTITIES, TIME_METADATA)


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
        TypeError : Raised if float64 values are found in the cube.

    """
    if cube.dtype == np.float64:
        if fix:
            cube.data = cube.data.astype(np.float32)
        else:
            raise TypeError("64 bit cube not allowed: {!r}".format(cube))
    for coord in cube.coords():
        if coord.name() in ["time", "forecast_reference_time"]:
            continue
        if coord.points.dtype == np.float64:
            if fix:
                coord.points = coord.points.astype(np.float32)
            else:
                raise TypeError(
                    "64 bit coord points not allowed: {} in {!r}".format(
                        coord, cube))
        if (hasattr(coord, "bounds") and coord.bounds is not None and
                coord.bounds.dtype == np.float64):
            if fix:
                coord.bounds = coord.bounds.astype(np.float32)
            else:
                raise TypeError(
                    "64 bit coord bounds not allowed: {} in {!r}".format(
                        coord, cube))


def check_time_coordinate_metadata(cube):
    """
    Function to check time coordinates against the expected standard. The
    standard for time coordinates is due to technical requirements and if
    violated the data integrity cannot be guaranteed; so if time coordinates
    are non-conformant an error is raised.

    Args:
        cubes (iris.cube.Cube):
            Cube to be checked

    Raises:
        ValueError: if any the input cube's time coordinates do not conform
            to the standard datatypes and units
    """
    error_string = ''
    for time_coord in TIME_METADATA.keys():
        try:
            coord = cube.coord(time_coord)
        except CoordinateNotFoundError:
            continue
        reqd_unit = TIME_METADATA[time_coord]["unit"]
        reqd_dtype = TIME_METADATA[time_coord]["dtype"]

        if not _check_units_and_dtype(coord, reqd_unit, reqd_dtype):
            msg = ('Coordinate {} does not match required '
                   'standard (units {}, datatype {})\n')
            error_string += msg.format(
                coord.name(), reqd_unit, reqd_dtype)

    # if non-compliance was encountered, raise all messages here
    if error_string:
        raise ValueError(error_string)


def _construct_object_list(cube, coords):
    """
    Construct a list of objects to check

    Args:
        cube (iris.cube.Cube):
            Cube to be checked
        coords (list or None):
            List of coordinate names to check.  If None, adds all
            coordinates present on the input cube

    Returns:
        object (list):
            List containing the original cube and coordinates to check
    """
    object_list = []
    object_list.append(cube)
    if coords is not None:
        for coord in coords:
            try:
                object_list.append(cube.coord(coord))
            except CoordinateNotFoundError:
                pass
    else:
        object_list.extend(cube.coords())
    return object_list


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
    Returns:
        new_cube (iris.cube.Cube):
            New cube with conformant datatypes
    """
    # create a list of copied cubes to modify
    new_cube = cube.copy()

    # construct a list of objects (cube and coordinates) to be checked
    object_list = _construct_object_list(new_cube, coords)

    error_string = ''
    for item in object_list:
        # allow string-type objects
        if item.dtype.type == np.unicode_:
            continue

        if item.name() in TIME_METADATA.keys():
            reqd_dtype = TIME_METADATA[item.name()]["dtype"]
        elif item.name() in INTEGER_QUANTITIES:
            reqd_dtype = np.int32
        else:
            reqd_dtype = np.float32

        if item.dtype != reqd_dtype:
            msg = ('{} datatype {} does not conform '
                   'to expected standard ({})\n')
            msg = msg.format(item.name(), item.dtype, reqd_dtype)
            error_string += msg
        continue

    # if any data was non-compliant, raise details here
    if error_string:
        raise ValueError(error_string)

    return new_cube


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
