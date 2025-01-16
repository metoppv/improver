# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from typing import Union

from iris.cube import Cube, CubeList

from improver.utilities.flatten import flatten


def as_cubelist(*cubes: Union[Cube, CubeList]):
    """
    Standardise input handling of cube/cubelist arguments.

    The role of this function is to flatten the provided inputs and thereby
    return a single CubeList object.

    Args:
        cubes:
            Input data provided in the form of one or more cubes or cubelists (or mixture thereof).
            Any iterable is supported.

    Returns:
        CubeList:
            A CubeList containing all the cubes provided as input.
    """
    cubes = CubeList(flatten(cubes))
    # Remove CubeList verification for iris >=3.3.0
    for cube in cubes:
        if not hasattr(cube, "add_aux_coord"):
            raise TypeError("A non iris Cube object has been provided.")
    if len(cubes) == 0:
        raise ValueError("One or more cubes should be provided.")
    return cubes


def as_cube(*cube: Union[Cube, CubeList]):
    """
    Standardise input handling of cube arguments.

    The role of this function is to return a single cube object.
    Where more than one cube is provided, a cubelist merge is attempted.

    Args:
        cube:
            Input data provided in the form of a cube or cubelists.
            Any iterable is supported.

    Returns:
        Cube:
            A single cube.
    """
    cubelist = as_cubelist(*cube)
    if len(cubelist) != 1:
        # cube merge changes the object ID so this is conditional
        try:
            cubelist = [cubelist.merge_cube()]
        except Exception as err:
            err_msg = "Unable to return a single cube."
            raise ValueError(err_msg) from err
    return cubelist[0]
