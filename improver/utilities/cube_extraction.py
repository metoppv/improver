# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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

import numpy as np
from iris import Constraint
from iris.cube import CubeList

from improver.metadata.constants import FLOAT_DTYPE
from improver.utilities.cube_constraints import create_sorted_lambda_constraint


def create_range_constraint(coord_name, value):
    """
    Create a constraint that is representative of a range.

    Args:
        coord_name (str):
            Name of the coordinate for which the constraint will be created.
        value (str):
            A string containing the range information.
            It is assumed that the input value is of the form: "[2:10]".

    Returns:
        iris.Constraint:
            The constraint that has been created to represent the range.

    """
    value = value.replace("[", "").replace("]", "").split(":")
    constr = create_sorted_lambda_constraint(coord_name, value)
    return constr


def is_complex_parsing_required(value):
    """
    Determine if the string being parsed requires complex parsing.
    Currently, this is solely determined by the presence of a colon (:).

    Args:
        value (str):
           A string that will be parsed.

    Returns:
        bool:
            Flag value to indicate whether the string requires complex parsing.
    """
    return ":" in value


def create_constraint(value):
    """
    Constructs an appropriate constraint for matching numerical values if they
    are floating point. If not, the original values are returned as a list
    (even if they were single valued on entry).

    Args:
        value (float/int or list of float/int):
            Constraint values that are being used to match against values in a
            cube for the purposes of extracting elements of the cube.
    Returns:
        lambda function or list:
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


def parse_constraint_list(constraints, units=None):
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
        constraints (list):
            List of string constraints with keys and values split by "=":
            e.g: ["kw1=val1", "kw2 = val2", "kw3=val3"].
        units (list):
            List of units (as strings) corresponding to each coordinate in the
            list of constraints.  One or more "units" may be None, and units
            may only be associated with coordinate constraints.

    Returns:
        (tuple): tuple containing:
            **constraints** (iris.Constraint or \
            iris._constraints.ConstraintCombination):
                A combination of all the constraints that were supplied.

            **units_dict** (dictionary or None):
                A dictionary of unit keys and values
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
    for constraint_pair, unit_val in zip(constraints, list_units):
        key, value = constraint_pair.split("=", 1)
        key = key.strip(" ")
        value = value.strip(" ")

        if is_complex_parsing_required(value):
            complex_constraints.append(create_range_constraint(key, value))
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

    return constraints, units_dict


def apply_extraction(cube, constraint, units=None, use_original_units=True):
    """
    Using a set of constraints, extract a subcube from the provided cube if it
    is available.

    Args:
        cube (iris.cube.Cube):
            The cube from which a subcube is to be extracted.
        constraint (iris.Constraint or iris.ConstraintCombination):
            The constraint or ConstraintCombination that will be used to
            extract a subcube from the input cube.
        units (dict):
            A dictionary of units for the constraints. Supplied if any
            coordinate constraints are provided in different units from those
            of the input cube (eg precip in mm/h for cube threshold in m/s).
        use_original_units (bool):
            Boolean to state whether the coordinates used in the extraction
            should be converted back to their original units. The default is
            True, indicating that the units should be converted back to the
            original units.

    Returns:
        iris.cube.Cube:
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

    return output_cube


def extract_subcube(cube, constraints, units=None, use_original_units=True):
    """
    Using a set of constraints, extract a subcube from the provided cube if it
    is available.

    Args:
        cube (iris.cube.Cube):
            The cube from which a subcube is to be extracted.
        constraints (list):
            List of string constraints with keys and values split by "=":
            e.g: ["kw1=val1", "kw2 = val2", "kw3=val3"].
        units (list):
            List of units (as strings) corresponding to each coordinate in the
            list of constraints.  One or more "units" may be None, and units
            may only be associated with coordinate constraints.
        use_original_units (bool):
            Boolean to state whether the coordinates used in the extraction
            should be converted back to their original units. The default is
            True, indicating that the units should be converted back to the
            original units.

    Returns:
        iris.cube.Cube or None:
            A single cube matching the input constraints, or None if no subcube
            is found within cube that matches the constraints.
    """
    constraints, units = parse_constraint_list(constraints, units=units)
    output_cube = apply_extraction(
        cube, constraints, units=units, use_original_units=use_original_units
    )
    return output_cube


def _create_cutout(cube, grid_spec):
    """Given a gridded data cube and boundary limits for cutout dimensions,
    create cutout.  Expects cube on either lat-lon or equal area grid.
    """
    x_coord = cube.coord(axis="x").name()
    y_coord = cube.coord(axis="y").name()

    xmin = grid_spec[x_coord]["min"]
    xmax = grid_spec[x_coord]["max"]
    ymin = grid_spec[y_coord]["min"]
    ymax = grid_spec[y_coord]["max"]

    # need to use cube intersection for circular coordinates (longitude)
    if x_coord == "longitude":
        lat_constraint = Constraint(latitude=lambda x: ymin <= x.point <= ymax)
        cutout = cube.extract(lat_constraint)
        if cutout is None:
            return cutout

        cutout = cutout.intersection(longitude=(xmin, xmax))

        # intersection creates a new coordinate with default datatype - we
        # therefore need to re-cast to meet the IMPROVER standard
        cutout.coord("longitude").points = cutout.coord("longitude").points.astype(
            FLOAT_DTYPE
        )
        if cutout.coord("longitude").bounds is not None:
            cutout.coord("longitude").bounds = cutout.coord("longitude").bounds.astype(
                FLOAT_DTYPE
            )

    else:
        x_constraint = Constraint(
            projection_x_coordinate=lambda x: xmin <= x.point <= xmax
        )
        y_constraint = Constraint(
            projection_y_coordinate=lambda x: ymin <= x.point <= ymax
        )
        cutout = cube.extract(x_constraint & y_constraint)

    return cutout


def subset_data(cube, grid_spec=None, site_list=None):
    """Extract a spatial cutout or subset of sites from data
    to generate suite reference outputs.

    Args:
        cube (iris.cube.Cube):
            Input dataset
        grid_spec (dict):
            Dictionary containing bounding grid points and an integer "thinning
            factor" for each of UK and global grid, to create cutouts.  Eg a
            "thinning factor" of 10 would mean every 10th point being taken for
            the cutout.
        site_list (list of str):
            List of WMO site IDs to extract

    Returns:
        iris.cube.Cube:
            Subset of input cube as specified by input constraints

    Raises:
        ValueError:
            If grid_spec does not contain cutout parameters for the coordinates on
            the input gridded data.
        ValueError:
            If the grid_spec or site_list aren't present in the cube, so that the
            subset cube returned would be None.
    """
    if cube.coords("spot_index"):
        if site_list is None:
            raise ValueError("site_list required to extract from spot data")

        constraint = Constraint(coord_values={"wmo_id": lambda x: x in site_list})
        result = cube.extract(constraint)
        if result is None:
            raise ValueError(
                f"Cube does not contain any of the required sites: {site_list}"
            )

    else:
        if grid_spec is None:
            raise ValueError("grid_spec required to extract from gridded data")

        x_coord = cube.coord(axis="x").name()
        y_coord = cube.coord(axis="y").name()

        for coord in [y_coord, x_coord]:
            if coord not in grid_spec:
                raise ValueError(
                    f"Cube coordinates {y_coord}, {x_coord} do not match "
                    "expected values"
                )

        cutout = _create_cutout(cube, grid_spec)

        if cutout is None:
            raise ValueError(
                "Cube domain does not overlap with cutout specified:\n"
                f"{x_coord}: {grid_spec[x_coord]}, {y_coord}: {grid_spec[y_coord]}"
            )

        thin_x = grid_spec[x_coord]["thin"]
        thin_y = grid_spec[y_coord]["thin"]
        result_list = CubeList()
        for subcube in cutout.slices([y_coord, x_coord]):
            result_list.append(subcube[::thin_y, ::thin_x])
        result = result_list.merge_cube()

    return result
