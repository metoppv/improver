# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module for applying mask to a cube."""

from typing import Union

import iris
import numpy as np

from improver.utilities.common_input_handle import as_cubelist
from improver.utilities.cube_checker import find_dimension_coordinate_mismatch
from improver.utilities.cube_manipulation import (
    enforce_coordinate_ordering,
    get_coord_names,
)


def apply_mask(
    *cubes: Union[iris.cube.CubeList, iris.cube.Cube],
    mask_name: str,
    invert_mask: bool = False,
) -> iris.cube.Cube:
    """
    Apply a provided mask to a cube. If invert_mask is True, the mask will be inverted.

    Args:
        cubes:
            A list of iris cubes that should contain exactly two cubes: a mask and a cube
            to apply the mask to. The cubes should have the same dimensions.
        mask_name:
            The name of the mask cube. It should match with exactly one of the cubes in
            the input cubelist.
        invert_mask:
            If True, the mask will be inverted before it is applied.
    Returns:
        A cube with a mask applied to the data.

    Raises:
    ValueError: If the number of cubes provided is not equal to 2.
    ValueError: If the input cube and mask cube have different dimensions.

    """
    cubes = as_cubelist(*cubes)
    cube_names = [cube.name() for cube in cubes]
    if len(cubes) != 2:
        raise ValueError(
            f"""Two cubes are required for masking, a mask and the cube to be masked.
                         Provided cubes are {cube_names}"""
        )

    mask = cubes.extract_cube(mask_name)
    cubes.remove(mask)
    cube = cubes[0]

    # Ensure mask is in a boolean form and invert if requested
    mask.data = mask.data.astype(np.bool)
    if invert_mask:
        mask.data = ~mask.data

    coord_list = get_coord_names(cube)
    enforce_coordinate_ordering(mask, coord_list)

    # This error is required to stop the mask from being broadcasted to a new shape by numpy. When
    # the mask and cube have different shapes numpy will try to broadcast the mask to be the same
    # shape as the cube data. This might succeed but masks unexpected data points.
    if find_dimension_coordinate_mismatch(cube, mask):
        raise ValueError("Input cube and mask cube must have the same dimensions")

    cube.data = np.ma.array(cube.data, mask=mask.data)
    return cube
