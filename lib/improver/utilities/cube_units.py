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
            try:
                unit = DEFAULT_UNITS[coord_name]["unit"]
                dtype = DEFAULT_UNITS[coord_name]["dtype"]
            except KeyError:
                msg = "Coordinate {} not defined in units.py"
                raise KeyError(msg.format(coord_name))

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


def enforce_diagnostic_units_and_dtypes(cubes, inplace=True,
                                        check_precision=False):
    """
    Function to enforce diagnostic units and data types as defined in units.py.

    By default the diagnostic units are changed in place, but setting inplace
    to False will return a copy of the cubes, leaving the originals unchanged.

    Args:
        cubes (iris.cube.CubeList):
            List of cubes on which to enforce diagnostic units and data types.
    Keyword Args:
        inplace (bool):
            If True (default) the cubes are modified in place, if False this
            function returns a modified copy of the cubes.
        check_precision (bool):
            If True the change of cube units and data types is checked to
            ensure precision is not lost, e.g. losing decimal places when
            converting the data type. This defaults to False.
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
            msg = '{} units cannot be converted to "{}"'
            raise ValueError(msg.format(diagnostic, unit))
        else:
            if not check_precision or check_precision_loss(dtype, cube.data):
                cube.data = cube.data.astype(dtype)
            else:
                msg = ('Data type of diagnostic "{}" could not be'
                       ' enforced without losing significant precision.')
                raise ValueError(msg.format(diagnostic))

    if not inplace:
        return cubes


def check_precision_loss(dtype, data, precision=5):
    """
    This function checks that when converting data types there is not a loss
    of significant information. Float to integer conversion, and integer to
    integer conversion are covered by this function. Float to float conversion
    may be lossy if changing from 64 bit to 32 bit floats, but changes at this
    precision are not captured here by design.

    If the conversion is lossless (to the defined precision) this fuction
    returns True. If there is loss, the function returns False.

    .. See the documentation for examples of where such loss is important.
    .. include:: extended_documentation/utilities/cube_units/
       check_precision_loss_examples.rst

    Args:
        dtype (dtype):
            The data type to which the data is being converted.
        data (np.ndarray):
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
