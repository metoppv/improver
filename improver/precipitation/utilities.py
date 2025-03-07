# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Utilities for use by precipitation_type plugins / functions."""

from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError

from improver.metadata.constants import FLOAT_DTYPE
from improver.metadata.probabilistic import find_threshold_coordinate


def make_shower_condition_cube(cube: Cube, in_place: bool = False) -> Cube:
    """
    Modify the input cube's metadata and coordinates to produce a shower
    condition proxy. The input cube is expected to possess a single valued
    threshold coordinate.

    Args:
        cube:
            A thresholded diagnostic to be used as a proxy for showery conditions.
            The threshold coordinate should contain only one value, which denotes
            the key threshold that above which conditions are showery, and below
            which precipitation is more likely dynamic.
        in_place:
            If set true the cube is modified in place. By default a modified
            copy is returned.

    Returns:
        A shower condition probability cube that is an appropriately renamed
        version of the input with an updated threshold coordinate representing
        the probability of shower conditions occurring.

    Raises:
        CoordinateNotFoundError: Input has no threshold coordinate.
        ValueError: Input cube's threshold coordinate is multi-valued.
    """

    if not in_place:
        cube = cube.copy()

    shower_condition_name = "shower_condition"
    cube.rename(f"probability_of_{shower_condition_name}_above_threshold")
    try:
        shower_threshold = find_threshold_coordinate(cube)
    except CoordinateNotFoundError as err:
        msg = "Input has no threshold coordinate and cannot be used"
        raise CoordinateNotFoundError(msg) from err

    try:
        (_,) = shower_threshold.points
    except ValueError as err:
        msg = (
            "Expected a single valued threshold coordinate, but threshold "
            f"contains multiple points : {shower_threshold.points}"
        )
        raise ValueError(msg) from err

    cube.coord(shower_threshold).rename(shower_condition_name)
    cube.coord(shower_condition_name).var_name = "threshold"
    cube.coord(shower_condition_name).points = FLOAT_DTYPE(1.0)
    cube.coord(shower_condition_name).units = 1

    return cube
