#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to calculate the maximum over the height coordinate"""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube,
    *,
    lower_height_bound: float = None,
    upper_height_bound: float = None,
    new_name: str = None,
):
    """Calculate the maximum value over the height coordinate of a cube. If height bounds are
    specified then the maximum value between these height levels is calculated.

    Args:
        cube (iris.cube.Cube):
            A cube with a height coordinate.
        lower_height_bound (float):
            The lower bound for the height coordinate. This is either a float or None if no lower
            bound is desired. Any specified bounds should have the same units as the height
            coordinate of cube.
        upper_height_bound (float):
            The upper bound for the height coordinate. This is either a float or None if no upper
            bound is desired. Any specified bounds should have the same units as the height
            coordinate of cube.
        new_name (str):
            The new name to be assigned to the output cube. If unspecified the name of the original
            cube is used.
    Returns:
        A cube of the maximum value over the height coordinate or maximum value between the provided
        height bounds."""

    from improver.utilities.cube_manipulation import maximum_in_height

    return maximum_in_height(
        cube,
        lower_height_bound=lower_height_bound,
        upper_height_bound=upper_height_bound,
        new_name=new_name,
    )
