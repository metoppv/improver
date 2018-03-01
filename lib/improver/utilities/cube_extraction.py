# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2018 Met Office.
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
""" Utilities to parse a list of constraints and extract matching subcube """

from ast import literal_eval
import iris


def parse_constraint_list(constraints, units=None):
    """
    Takes a list of string constraints and converts to key-value pairs

    Args:
        constraints (list):
            Space separated list of constraints with no space between key and
            value in each pair: e.g: kw1=val1 kw2=val2 kw3=val3.  Values must
            be of scalar types interpretable by ast.literal_eval: strings,
            numbers, booleans, or "None".

    Kwargs:
        units (list of strings):
            Space separated list of units for each coordinate in the list of
            constraints.  One or more "units" may be None, and units can only
            be associated with coordinate constraints.

    Returns:
        constraints_dict (dictionary):
            A dictionary of constraint keys and interpreted values
        units_dict (dictionary or None):
            A dictionary of unit keys and values
    """

    if units is None:
        list_units = len(constraints)*[None]
        units_dict = None
    else:
        if len(units) != len(constraints):
            msg = 'units list must match constraints'
            raise ValueError(msg)
        list_units = units
        units_dict = {}

    constraints_dict = {}
    for constraint_pair, units in zip(constraints, list_units):
        [key, value] = constraint_pair.split('=')
        try:
            constraints_dict[key] = literal_eval(value)
        except ValueError:
            constraints_dict[key] = value

        if units is not None and units.capitalize() != 'None':
            units_dict[key] = units

    return constraints_dict, units_dict


def extract_subcube(cube, constraints, units=None):
    """
    Using a set of constraints, extract a subcube from the provided cube if it
    is available.

    Args:
        cube (iris.cube.Cube):
            The cube from which a subcube is to be extracted.
        constraints (dictionary):
            A dictionary of constraints that define the subcube to be
            extracted.

    Kwargs:
        units (dictionary):
            A dictionary of units for the constraints.  Supplied if any
            coordinate constraints are provided in different units from those
            of the input cube (eg precip in mm/h for cube threshold in m/s).

    Returns:
        cube (iris.cube.Cube):
            A single cube matching the input constraints, or None.

    """
    constraint = iris.Constraint(**constraints)

    if units is not None:
        original_units = {}
        for coord in units.keys():
            original_units[coord] = cube.coord(coord).units
            cube.coord(coord).convert_units(units[coord])
        output_cube = cube.extract(constraint)
        for coord in original_units.keys():
            output_cube.coord(coord).convert_units(original_units[coord])
    else:
        output_cube = cube.extract(constraint)

    return output_cube
