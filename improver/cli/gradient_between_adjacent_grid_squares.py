#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to calculate the gradient between adjacent grid squares."""
from improver import cli


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube, *, regrid: bool = False):
    """Calculate the gradient between adjacent grid squares within
    a cube. The gradient is calculated along the x and y axis
    individually.

    Args:
        cube (iris.cube.Cube):
            Cube from which the gradients will be calculated.
        regrid (bool):
                If True, the gradient cube is regridded to match the spatial
                dimensions of the input cube. If False, the two output gradient cubes will have
                different spatial coords such that the coord matching the gradient axis will
                represent the midpoint of the input cube and will have one fewer points.
                If the x-axis is marked as circular, the gradient between the last and first points
                is also included.
    Returns:
        Cube after the gradients have been calculated along the x-axis and Cube after the gradients
        have been calculated along the y-axis.
    """
    from improver.utilities.spatial import GradientBetweenAdjacentGridSquares

    return GradientBetweenAdjacentGridSquares(regrid=regrid)(cube)
