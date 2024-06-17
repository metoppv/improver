# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Provides support utility for rescaling data."""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from iris.cube import Cube
from numpy import ndarray


def rescale(
    data: ndarray,
    data_range: Optional[Union[Tuple[float, float], List[float]]] = None,
    scale_range: Union[Tuple[float, float], List[float]] = (0.0, 1.0),
    clip: bool = False,
) -> ndarray:
    """
    Rescale data array so that data_min => scale_min
    and data_max => scale max.
    All adjustments are linear

    Args:
        data:
            Source values
        data_range:
            List containing two floats
            Lowest and highest source value to rescale.
            Default value of None is converted to [min(data), max(data)]
        scale_range:
            List containing two floats
            Lowest and highest value after rescaling.
            Defaults to (0., 1.)
        clip:
            If True, points where data were outside the scaling range
            will be set to the scale min or max appropriately.
            Default is False which continues the scaling beyond min and
            max.

    Returns:
        Output array of scaled data. Has same shape as data.
    """
    data_left = np.min(data) if data_range is None else data_range[0]
    data_right = np.max(data) if data_range is None else data_range[1]
    scale_left = scale_range[0]
    scale_right = scale_range[1]
    # Range check
    if data_left == data_right:
        raise ValueError(
            "Cannot rescale a zero input range ({} -> {})".format(data_left, data_right)
        )

    if scale_left == scale_right:
        raise ValueError(
            "Cannot rescale a zero output range ({} -> {})".format(
                scale_left, scale_right
            )
        )

    result = (
        (data - data_left) * (scale_right - scale_left) / (data_right - data_left)
    ) + scale_left
    if clip:
        result = np.clip(
            result, min(scale_left, scale_right), max(scale_left, scale_right)
        )
    return result


def apply_double_scaling(
    data_cube: Cube,
    scaled_cube: Cube,
    data_vals: Tuple[float, float, float],
    scaling_vals: Tuple[float, float, float],
    combine_function: Callable[[ndarray, ndarray], ndarray] = np.minimum,
) -> ndarray:
    """
    From data_cube, an array of limiting values is created based on a linear
    rescaling from three data_vals to three scaling_vals.
    The three values refer to a lower-bound, a mid-point and an upper-bound.
    This rescaled data_cube is combined with scaled_cube to produce an array
    containing either the higher or lower value as needed.

    Args:
        data_cube:
            Data from which to create a rescaled data array
        scaled_cube:
            Data already in the rescaled frame of reference which will be
            combined with the rescaled data_cube using the combine_function.
        data_vals:
            Lower, mid and upper points to rescale data_cube from
        scaling_vals:
            Lower, mid and upper points to rescale data_cube to
        combine_function:
            Function that takes two arrays of the same shape and returns
            one array of the same shape.
            Expected to be numpy.minimum (default) or numpy.maximum.

    Returns:
        Output data from data_cube after rescaling and combining with
        scaled_cube.
        This array will have the same dimensions as scaled_cube.
    """
    # Where data are below the specified mid-point (data_vals[1]):
    #  Set rescaled_data to be a rescaled value between the first and mid-point
    # Elsewhere
    #  Set rescaled_data to be a rescaled value between the mid- and last point
    rescaled_data = np.where(
        data_cube.data < data_vals[1],
        rescale(
            data_cube.data,
            data_range=(data_vals[0], data_vals[1]),
            scale_range=(scaling_vals[0], scaling_vals[1]),
            clip=True,
        ),
        rescale(
            data_cube.data,
            data_range=(data_vals[1], data_vals[2]),
            scale_range=(scaling_vals[1], scaling_vals[2]),
            clip=True,
        ),
    )
    # Ensure scaled_cube is no larger or smaller than the rescaled_data:
    return combine_function(scaled_cube.data, rescaled_data)
