# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Provides support utility for rescaling data."""

import numpy as np


def rescale(data, data_range=None, scale_range=(0., 1.),
            clip=False):
    """
    Rescale data array so that data_min => scale_min
    and data_max => scale max.
    All adjustments are linear

    Args:
        data (numpy.ndarray):
            Source values
        data_range (list):
            List containing two floats
            Lowest and highest source value to rescale.
            Default value of None is converted to [min(data), max(data)]
        scale_range (list):
            List containing two floats
            Lowest and highest value after rescaling.
            Defaults to (0., 1.)
        clip (bool):
            If True, points where data were outside the scaling range
            will be set to the scale min or max appropriately.
            Default is False which continues the scaling beyond min and
            max.

    Returns:
        numpy.ndarray:
            Output array of scaled data. Has same shape as data.
    """
    data_min = np.min(data) if data_range is None else data_range[0]
    data_max = np.max(data) if data_range is None else data_range[1]
    scale_min = scale_range[0]
    scale_max = scale_range[1]
    # Range check
    if data_min == data_max:
        raise ValueError("Cannot rescale a zero input range " +
                         "({} -> {})".format(data_min, data_max))

    if scale_min == scale_max:
        raise ValueError("Cannot rescale a zero output range " +
                         "({} -> {})".format(scale_min, scale_max))

    result = ((data - data_min) * (scale_max - scale_min) /
              (data_max - data_min)) + scale_min
    if clip:
        result = np.clip(result, scale_min, scale_max)
    return result


def apply_double_scaling(data_cube, scaled_cube,
                         data_vals, scaling_vals,
                         combine_function=np.minimum):
    """
    From data_cube, an array of limiting values is created based on a linear
    rescaling from three data_vals to three scaling_vals.
    The three values refer to a lower-bound, a mid-point and an upper-bound.
    This rescaled data_cube is combined with scaled_cube to produce an array
    containing either the higher or lower value as needed.

    Args:
        data_cube (iris.cube.Cube):
            Data from which to create a rescaled data array
        scaled_cube (iris.cube.Cube):
            Data already in the rescaled frame of reference which will be
            combined with the rescaled data_cube using the combine_function.
        data_vals (tuple of three values):
            Lower, mid and upper points to rescale data_cube from
        scaling_vals (tuple of three values):
            Lower, mid and upper points to rescale data_cube to
        combine_function (Callable[[numpy.ndarray, numpy.ndarray],
        numpy.ndarray]):
            Function that takes two arrays of the same shape and returns
            one array of the same shape.
            Expected to be numpy.minimum (default) or numpy.maximum.

    Returns:
        numpy.ndarray:
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
        rescale(data_cube.data,
                data_range=(data_vals[0], data_vals[1]),
                scale_range=(scaling_vals[0], scaling_vals[1]),
                clip=True),
        rescale(data_cube.data,
                data_range=(data_vals[1], data_vals[2]),
                scale_range=(scaling_vals[1], scaling_vals[2]),
                clip=True))
    # Ensure scaled_cube is no larger or smaller than the rescaled_data:
    return combine_function(scaled_cube.data, rescaled_data)
