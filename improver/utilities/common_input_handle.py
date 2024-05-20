# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from typing import Union

from iris.cube import Cube, CubeList

from improver.utilities.flatten import flatten


def as_cubelist(*cubes: Union[Cube,CubeList]):
    """
    Standardise input handling of cube/cubelist arguments.

    The role of this function is to flatten the provided inputs and thereby
    return a single CubeList object.

    Args:
        cubes:
            Input data provided in the form of one or more cubes or cubelists (or mixture thereof).

    Returns:
        CubeList:
            A CubeList containing all the cubes provided as input.
    """
    if not cubes or all([not cube for cube in cubes]):
        raise ValueError("One or more cube should be provided.")
    return CubeList(flatten(cubes))


def as_cube(cube: Union[Cube,CubeList]):
    """
    Standardise input handling of cube arguments.

    The role of this function is to return a single cube object.

    Args:
        cube:
            Input data provided in the form of a cube ir cubelists.

    Returns:
        Cube:
            A single cube.
    """
    if not cube:
        raise ValueError("A cube should be provided.")
    if isinstance(cube, CubeList):
        if len(cube) > 1:
            raise ValueError("A single cube should be provided.")
        cube = cube[0]
    elif not isinstance(cube, Cube):
        raise TypeError("A cube should be provided.")
    return cube