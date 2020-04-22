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
"""Provides tools for neighbourhood generation"""

import numpy as np


def rolling_window(A, shape):
    """Creates views of the array A, this avoids
    creating a massive matrix of points in a neighbourhood
    calculation.
    Args:
        A:
            The input array padded with nans for half the
            neighbourhood size (2D).
        shape:
            The neighbourhood shape e.g. is the neighbourhood
            size is 3, the shape would be (3, 3) to create a
            3x3 array.

    Returns:
        ndarray of "views" into the data, each view represents
        a neighbourhood of points.
    """
    nwd = len(shape)
    nad = len(A.shape)
    assert nad >= nwd
    adjshp = (
        *A.shape[:-nwd],
        *(ad - wd + 1 for ad, wd in zip(A.shape[-nwd:], shape)),
        *shape,
    )
    assert all(ad > 0 for ad in adjshp)
    strides = A.strides + A.strides[-nwd:]
    return np.lib.stride_tricks.as_strided(
        A, shape=adjshp, strides=strides, writeable=False
    )


def pad_and_roll(A, shape, pad_value=np.nan):
    """Pads the input arrays and passes them to _rolling_window
    to create windows of the data.
    Args:
        A:
            The dataset to pad and create rolling windows for.
        shape:
            Desired shape of the neighbourhood.
        pad_values:
            (Optional) the fill value for the padded array.
            Defaults to np.nan

    Returns:
        ndarray, containing views of the dataset A.
    """
    pad_extent = [(0, 0)] * (len(A.shape) - len(shape))
    pad_extent.extend((d // 2, d // 2) for d in shape)
    A = np.pad(A, pad_extent, mode="constant", constant_values=pad_value)
    return rolling_window(A, shape)
