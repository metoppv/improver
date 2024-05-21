#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to compute the maximum within a time window for a period diagnostic."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube, minimum_realizations=None):
    """Find the maximum probability or maximum diagnostic value within a time window
    for a period diagnostic. For example, find the maximum probability of exceeding
    a given accumulation threshold in a period e.g. 20 mm in 3 hours, over the course
    of a longer interval e.g. a 24 hour time window.

    Args:
        cubes (iris.cube.CubeList or list of iris.cube.Cube):
            Cubes over which to find the maximum.
        minimum_realizations (int):
            If specified, the input cubes will be filtered to ensure that only
            realizations that include all available lead times are combined. If the
            number of realizations that meet this criteria are fewer than this integer,
            an error will be raised.

    Returns:
        result (iris.cube.Cube):
            Returns a cube that is representative of a maximum within a time window
            for the period diagnostic supplied.
    """
    from iris.cube import CubeList

    from improver.cube_combiner import MaxInTimeWindow

    return MaxInTimeWindow(minimum_realizations=minimum_realizations)(CubeList(cubes))
