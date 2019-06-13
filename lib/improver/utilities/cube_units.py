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

import iris
from iris.exceptions import CoordinateNotFoundError

from improver.units import DEFAULT_UNITS

def enforce_coordinate_units_and_dtypes(cubes, coordinates, inplace=True):
    """
    Function to enforce standard units and data types as defined in units.py.

    By default the cube units are changed in place, but setting inplace to
    False will return a copy of the cubes, leaving the originals unchanged.

    Args:
        cubes (iris.cube.CubeList):
            List of cubes on which to enforce coordinate units and data types.
        coordinates (list):
            List of coordinates for which units and dtypes shoule be enforced.
            This is a list of names or iris coordinates.
    Kwargs:
        inplace (bool):
            If True (default) the cubes are modified in place, if False this
            function returns a modified copy of the cubes.
    Returns:
        cubes (cubelist or insitu):
            The input cubes with units and data types of the chosen coordinates
            set to match the definitions in units.py.
    Raises:
        KeyError: If coordinate to enforce is not defined in units.py
        ValueError: If requested unit conversion is not possible.
        ValueError: If a temporal unit could not be converted without losing
                    significant information (e.g. rounding to nearest hour).
    """
    if not inplace:
        cubes = [cube.copy() for cube in cubes]

    for cube in cubes:
        for coord_name in coordinates:
            try:
                unit = DEFAULT_UNITS[coord_name]["unit"]
                dtype = DEFAULT_UNITS[coord_name]["dtype"]
                utype = DEFAULT_UNITS[coord_name]["utype"]
            except KeyError:
                msg = "Coordinate {} not defined in units.py"
                raise KeyError(msg.format(coord_name))

            try:
                coordinate = cube.coord(coord_name)
                coordinate.convert_units(unit)
            except ValueError:
                msg = "{} units cannot be converted to {}"
                raise ValueError(msg.format(coord, unit))
            except CoordinateNotFoundError:
                pass
            else:
                if check_temporal_precision_loss(utype, dtype, coordinate):
                    coordinate.points = coordinate.points.astype(dtype)
                else:
                    msg = ('Data type of temporal coordinate "{}" could not be'
                           ' enforced without losing significant precision.')
                    raise ValueError(msg.format(coord_name))

    if not inplace:
        return cubes

def enforce_diagnostic_units_and_dtypes(cubes, inplace=True):
    """
    Function to enforce diagnostic units and data types as defined in units.py.

    By default the cube units are changed in place, but setting inplace to
    False will return a copy of the cubes, leaving the originals unchanged.

    Args:
        cubes (iris.cube.CubeList):
            List of cubes on which to enforce diagnostic units and data types.
    Kwargs:
        inplace (bool):
            If True (default) the cubes are modified in place, if False this
            function returns a modified copy of the cubes.
    Returns:
        cubes (cubelist or insitu):
            The input cubes with units and data types of the diagnostic
            set to match the definitions in units.py.
    Raises:
        KeyError: If coordinate to enforce is not defined in units.py
        ValueError: If requested unit conversion is not possible.
    """
    if not inplace:
        cubes = [cube.copy() for cube in cubes]

    for cube in cubes:
        diagnostic = cube.name()
        try:
            unit = DEFAULT_UNITS[diagnostic]["unit"]
            dtype = DEFAULT_UNITS[diagnostic]["dtype"]
        except KeyError:
            msg = "Diagnostic {} not defined in units.py"
            raise KeyError(msg.format(diagnostic))

        try:
            cube.convert_units(unit)
        except ValueError:
            msg = "{} units cannot be converted to {}"
            raise ValueError(msg.format(diagnostic, unit))
        else:
            cube.data = cube.data.astype(dtype)

    if not inplace:
        return cubes

def check_temporal_precision_loss(utype, dtype, coordinate):
    """
    When changing the data type time coordinates it is important to ensure
    that the loss of decimals does not fundamentally alter the time that is
    recorded. This function is used to check whether the conversion of a time
    coordinate to an integer type would result in the loss of important decimal
    information.

    Consider a validity time of 2018-11-03 16:15:00 expressed in seconds since
    1970-01-01 00:00:00 on an input cube.

    If units.py defines the fundamental units of time as hours since
    1970-01-01 00:00:00 and the data type as int64, we can see that there is no
    way to retain the 15 minutes, instead we would alter the validity time of
    the cube to 16:00 by changing the data type.

    This function returns True if the coordinate being considered is not a
    temporal coordinate or if the data type to which it is being converted is
    not of integer type.

    It returns True if there the time points are all integers to within
    reasonable precision; decimals beyond 5 decimal places are assumed to be
    precision errors. These values can be encoded as integers without loss of
    information.

    It returns False if the conversion would result in important decimals being
    removed and the time being fundamentally modified.
    """
    if not (utype is 'temporal' and np.issubdtype(dtype, np.integer)):
        return True

    points = coordinate.points
    rounded_values = np.round(points, 5)
    fractions, integers = np.modf(points)
    return (rounded_values == integers).all()
