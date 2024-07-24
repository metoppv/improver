#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to calculate the height at which the maximum vertical velocity
occurs. """

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube,
    max_cube: cli.inputcube,
    *,
    high_or_low: str = "low",
    new_name: str = "height_of_maximum_vertical_velocity",
):
    """Calculates the height level at which the maximum vertical velocity occurs for each
    grid point. It requires an input cube of vertical velocity and a cube with the maximum
    vertical velocity values at each grid point. For this case we are looking for the
    lowest height at which the maximum occurs.

    Args:
        cube (iris.cube.Cube):
            A cube containing vertical velocity.
        max_cube (iris.cube.Cube):
            A cube containing the maximum values of vertical velocity over all heights.
        high_or_low (str):
            Whether we are looking for the highest or lowest maximum height (for cases where
            there are two heights with the maximum vertical velocity.)
        new_name (str):
            The new name to be assigned to the output cube. In this case it will become
            height_of_maximum_vertical_velocity. If unspecified the name of the original
            cube is used.

    Returns:
        A cube of heights at which the maximum value occurs.
    """

    from improver.utilities.cube_manipulation import height_of_maximum

    return height_of_maximum(cube, max_cube, high_or_low, new_name)
