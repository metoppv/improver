# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2022 Met Office.
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
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from iris import Constraint
from iris.cube import Cube

from improver.metadata.constants import FLOAT_DTYPE
from improver.utilities.cube_constraints import create_sorted_lambda_constraint
from improver.utilities.cube_manipulation import get_dim_coord_names


def parse_range_string_to_dict(value: str) -> Dict[str, str]:
    """
    Splits up a string in the form "[min:max:step]" into a list of
    [min, max, step].

    Args:
        value:
            A string containing the range information.
            It is assumed that the input value is of the form: "[2:10]".

    Returns:
        A list containing the min and max (and step).
    """
    value = value.replace("[", "").replace("]", "").split(":")
    return dict(zip(["min", "max", "step"], value))


def create_constraint(value: Union[float, List[float]]) -> Union[Callable, List[int]]:
    """
    Constructs an appropriate constraint for matching numerical values if they
    are floating point. If not, the original values are returned as a list
    (even if they were single valued on entry).

    Args:
        value:
            Constraint values that are being used to match against values in a
            cube for the purposes of extracting elements of the cube.

    Returns:
        If the input value(s) are floating point this function returns a
        lambda function that will enable for approximate matching to
        ensure they can be matched to cube values. If the inputs are int
        or non-numeric, they will be returned unchanged, except for single
        values that will have become lists.
    """
    if not isinstance(value, list):
        value = [value]

    if np.issubdtype(np.array(value).dtype, np.number):
        return lambda cell: any(np.isclose(cell.point, value))
    return value


def parse_constraint_list(
    constraints: List[str], units: Optional[List[str]] = None
) -> Tuple[Constraint, Optional[Dict], Optional[float], Optional[Dict]]:
    """
    For simple constraints of a key=value format, these are passed in as a
    list of strings and converted to key-value pairs prior to creating the
    constraints.
    For more complex constraints, the list of strings given as input
    are evaluated by parsing for specific identifiers and then the constraints
    are created as required.
    The simple key-value pairs and other constraints are merged into a single
    constraint.

    Args:
        constraints:
            List of string constraints with keys and values split by "=":
            e.g: ["kw1=val1", "kw2 = val2", "kw3=val3"], where the vals
            could include ranges e.g. [0:20] or ranges with a step value e.g.
            [0:20:3].
        units:
            List of units (as strings) corresponding to each coordinate in the
            list of constraints.  One or more "units" may be None, and units
            may only be associated with coordinate constraints.

    Returns:
        - A combination of all the constraints that were supplied.
        - A dictionary of unit keys and values
        - A list containing the min and max values for a longitude constraint
        - A dictionary of coordinate and the step value, i.e. a step of 2 will
          skip every other point
    """

    if units is None:
        list_units = len(constraints) * [None]
        units_dict = None
    else:
        if len(units) != len(constraints):
            msg = "units list must match constraints"
            raise ValueError(msg)
        list_units = units
        units_dict = {}

    simple_constraints_dict = {}
    complex_constraints = []
    longitude_constraint = None
    thinning_values = {}
    for constraint_pair, unit_val in zip(constraints, list_units):
        key, value = constraint_pair.split("=", 1)
        key = key.strip(" ")
        value = value.strip(" ")

        if ":" in value:
            range_dict = parse_range_string_to_dict(value)

            # longitude is a circular coordinate, so needs to be treated in a
            # different way to a normal constraint
            if key == "longitude":
                longitude_constraint = [
                    FLOAT_DTYPE(range_dict[k]) for k in ["min", "max"]
                ]
            else:
                complex_constraints.append(
                    create_sorted_lambda_constraint(
                        key, [range_dict["min"], range_dict["max"]]
                    )
                )
            if range_dict.get("step", None):
                thinning_values[key] = int(range_dict["step"])
        else:
            try:
                typed_value = literal_eval(value)
            except ValueError:
                simple_constraints_dict[key] = value
            else:
                simple_constraints_dict[key] = create_constraint(typed_value)

        if unit_val is not None and unit_val.capitalize() != "None":
            units_dict[key] = unit_val.strip(" ")

    if simple_constraints_dict:
        simple_constraints = Constraint(**simple_constraints_dict)
    else:
        simple_constraints = None

    constraints = simple_constraints
    for constr in complex_constraints:
        constraints = constraints & constr

    return constraints, units_dict, longitude_constraint, thinning_values


def apply_extraction(
    cube: Cube,
    constraint: Constraint,
    units: Optional[Dict] = None,
    use_original_units: bool = True,
    longitude_constraint: Optional[List] = None,
) -> Cube:
    """
    Using a set of constraints, extract a subcube from the provided cube if it
    is available.

    Args:
        cube:
            The cube from which a subcube is to be extracted.
        constraint:
            The constraint or ConstraintCombination that will be used to
            extract a subcube from the input cube.
        units:
            A dictionary of units for the constraints. Supplied if any
            coordinate constraints are provided in different units from those
            of the input cube (eg precip in mm/h for cube threshold in m/s).
        use_original_units:
            Boolean to state whether the coordinates used in the extraction
            should be converted back to their original units. The default is
            True, indicating that the units should be converted back to the
            original units.
        longitude_constraint:
            List containing the min and max values for the longitude.
            This has to be treated separately to the normal constraints due
            to the circular nature of longitude.

    Returns:
        A single cube matching the input constraints, or None if no subcube
        is found within cube that matches the constraints.
    """
    if units is None:
        output_cube = cube.extract(constraint)
    else:
        original_units = {}
        for coord in units.keys():
            original_units[coord] = cube.coord(coord).units
            cube.coord(coord).convert_units(units[coord])
        output_cube = cube.extract(constraint)
        if use_original_units:
            for coord in original_units:
                cube.coord(coord).convert_units(original_units[coord])
                try:
                    output_cube.coord(coord).convert_units(original_units[coord])
                except AttributeError:
                    # an empty output cube (None) is handled by the CLI
                    pass

    if longitude_constraint:
        output_cube = output_cube.intersection(
            longitude=longitude_constraint, ignore_bounds=True
        )
        # TODO: Below can be removed when https://github.com/SciTools/iris/issues/4119
        # is fixed
        output_cube.coord("longitude").points = output_cube.coord(
            "longitude"
        ).points.astype(FLOAT_DTYPE)
        if output_cube.coord("longitude").bounds is not None:
            output_cube.coord("longitude").bounds = output_cube.coord(
                "longitude"
            ).bounds.astype(FLOAT_DTYPE)

    return output_cube


def extract_subcube(
    cube: Cube,
    constraints: List[str],
    units: Optional[List[str]] = None,
    use_original_units: bool = True,
) -> Optional[Cube]:
    """
    Using a set of constraints, extract a subcube from the provided cube if it
    is available.

    Args:
        cube:
            The cube from which a subcube is to be extracted.
        constraints:
            List of string constraints with keys and values split by "=":
            e.g: ["kw1=val1", "kw2 = val2", "kw3=val3"].
        units:
            List of units (as strings) corresponding to each coordinate in the
            list of constraints.  One or more "units" may be None, and units
            may only be associated with coordinate constraints.
        use_original_units:
            Boolean to state whether the coordinates used in the extraction
            should be converted back to their original units. The default is
            True, indicating that the units should be converted back to the
            original units.

    Returns:
        A single cube matching the input constraints, or None if no subcube
        is found within cube that matches the constraints.
    """
    constraints, units, longitude_constraint, thinning_dict = parse_constraint_list(
        constraints, units=units
    )
    output_cube = apply_extraction(
        cube, constraints, units, use_original_units, longitude_constraint
    )

    if thinning_dict:
        output_cube = thin_cube(output_cube, thinning_dict)

    return output_cube


def thin_cube(cube: Cube, thinning_dict: Dict[str, int]) -> Cube:
    """
    Thin the coordinate by taking every X points, defined in the thinning dict
    as {coordinate: X}

    Args:
        cube:
            The cube containing the coordinates to be thinned.
        thinning_dict:
            A dictionary of coordinate and the step value, i.e. a step of 2
            will skip every other point

    Returns:
        A cube with thinned coordinates.
    """
    coord_names = get_dim_coord_names(cube)
    slices = [slice(None, None, None)] * len(coord_names)
    for key, val in thinning_dict.items():
        slices[coord_names.index(key)] = slice(None, None, val)
    return cube[tuple(slices)]
