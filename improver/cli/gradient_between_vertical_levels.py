# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to calculate a gradient between two vertical levels."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcubelist):
    """
    Calculate the gradient between two vertical levels. The gradient is calculated as the
    difference between the input cubes divided by the difference in height.

    Input cubes can be provided at height or pressure levels. If the cubes are provided
    at a pressure level, the height above sea level is extracted from height_of_pressure_levels
    cube. If the cubes are provided at height levels this is assumed to be a height above ground
    level and the height above sea level is calculated by adding the height of the orography to
    the "height" coordinate of the cube. It is possible for one cube to be defined at height
    levels and the other at pressure levels.

    Args:
        cubes (iris.cube.CubeList):
            Contains two cubes of a diagnostic at two different vertical levels. The cubes must
            either have a height coordinate or a pressure coordinate. If either cube is defined at
            height levels, an orography cube must also be provided. If either cube is defined at
            pressure levels, a geopotential_height cube must also be provided.

    Returns:
        iris.cube.Cube:
            A single cube containing the gradient between the two height levels. This cube will be
            renamed to "gradient_of" the cube name and will have a units attribute of the input
            cube units per metre.

    """
    from improver.utilities.gradient_between_vertical_levels import (
        GradientBetweenVerticalLevels,
    )

    return GradientBetweenVerticalLevels()(*cubes)
