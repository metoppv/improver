#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to extract a subset of input file data, given constraints."""

from improver import cli
from improver.cli import parameters


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube,
    *,
    constraints: parameters.multi(min=1),
    units: cli.comma_separated_list = None,
    ignore_failure=False,
):
    """ Extract a subset of a single cube.

    Extracts subset of data from a single cube, subject to equality-based
    constraints.
    Using a set of constraints, extract a sub-cube from the provided cube if it
    is available.

    Args:
        cube (iris.cube.Cube):
            The Cube from which a sub-cube is extracted
        constraints (list):
            The constraint(s) to be applied. These must be of the form
            "key=value", eg "threshold=1". Multiple constraints can be provided
            by repeating this keyword before each. Scalars, boolean and string
            values are supported. Lists of values can be provided
            e.g. key=[value1, value2, value3]. Alternatively, ranges can also
            be specified e.g. key=[value1:value3].
            When a range is specified, this is inclusive of the endpoints of
            the range. A range can also be specified with a step value,
            e.g. [value1:value2:step].
        units (list):
            List of units as strings corresponding to each coordinate in the
            list of constraints. One or more "units" may be None and units may
            only be associated with coordinate constraints. The list should be
            entered as a comma separated list without spaces, e.g. mm/hr,K.
        ignore_failure (bool):
            Option to ignore constraint match failure and return the input
            cube.

    Returns:
        iris.cube.Cube:
            A single cube matching the input constraints or None. If no
            sub-cube is found within the cube that matches the constraints.
    """
    from improver.utilities.cube_extraction import extract_subcube

    result = extract_subcube(cube, constraints, units)

    if result is None and ignore_failure:
        return cube
    if result is None:
        msg = "Constraint(s) could not be matched in input cube"
        raise ValueError(msg)
    return result
