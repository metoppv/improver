#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to multiply period average values by the period."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube, *, new_name: str = None):
    """Multiply a frequency or rate cube by the time period given by the
    time bounds over which it is defined to return a count or accumulation.
    The frequency or rate must be defined with time bounds, e.g. an average
    frequency across the period. This function will handle a cube with a
    non-scalar time coordinate, multiplying each time in the coordinate by the
    related bounds.

    The returned cube has units equivalent to the input cube multiplied by
    seconds. Any time related cell methods are removed from the output cube
    and a new "sum" over time cell method is added.

    Args:
        cube (iris.cube.Cube):
            Cube to multiply
        new_name (str):
            New name for the output diagnostic.

    Returns:
        iris.cube.Cube:
            A cube containing the time multiplied data.
    """
    from improver.utilities.temporal import integrate_time

    return integrate_time(cube, new_name=new_name)
