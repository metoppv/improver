# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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

    Keyword Args:
        data_range (list):
            List containing two floats
            Lowest and highest source value to rescale.
            Default value of None is converted to [min(data), max(data)]
        scale_range (list):
            List containing two floats
            Lowest and highest value after rescaling.
            Defaults to (0., 1.)
        clip (boolean):
            If True, points where data were outside the scaling range
            will be set to the scale min or max appropriately.
            Default is False which continues the scaling beyond min and
            max.

    Returns:
        result (numpy.ndarray):
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
