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


def rolling_window(a, shape, writeable=False):
    """Creates a rolling window neighbourhoods of the given `shape` from the
    last `len(shape)` axes of the input array. avoids creating large output
    array by constructing a non-continuous view mapped onto the input array.

    args:
        a:
            The input array padded with nans for half the
            neighbourhood size (2D).
        shape:
            The neighbourhood shape e.g. is the neighbourhood
            size is 3, the shape would be (3, 3) to create a
            3x3 array.
        writeable:
            If True the returned view will be writeable. This will modify
            the input array, so use with caution.

    Returns:
        ndarray of "views" into the data, each view represents
        a neighbourhood of points.
    """
    nwd = len(shape)
    nad = len(a.shape)
    assert nad >= nwd
    adjshp = (
        *a.shape[:-nwd],
        *(ad - wd + 1 for ad, wd in zip(a.shape[-nwd:], shape)),
        *shape,
    )
    assert all(ad > 0 for ad in adjshp)
    strides = a.strides + a.strides[-nwd:]
    return np.lib.stride_tricks.as_strided(
        a, shape=adjshp, strides=strides, writeable=writeable
    )


def pad_and_roll(a, shape, **kwargs):
    """Pads the last `len(shape)` axes of the input array for `rolling_window`
    to create 'neighbourhood' views of the data of a given `shape` as the last
    axes in the returned array. Collapsing over the last `len(shape)` axes
    results in a shape of the original input array.

    args:
        a:
            The dataset to pad and create rolling windows for.
        shape:
            Desired shape of the neighbourhood.
        kwargs:
            additional keyword arguments passed to `numpy.pad` function.

    Returns:
        ndarray, containing views of the dataset a.
    """
    writeable = kwargs.pop("writeable", False)
    pad_extent = [(0, 0)] * (len(a.shape) - len(shape))
    pad_extent.extend((d // 2, d // 2) for d in shape)
    a = np.pad(a, pad_extent, **kwargs)
    return rolling_window(a, shape, writeable=writeable)
