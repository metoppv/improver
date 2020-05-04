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


def rolling_window(input_array, shape, writeable=False):
    """Creates a rolling window neighbourhoods of the given `shape` from the
    last `len(shape)` axes of the input array. avoids creating large output
    array by constructing a non-continuous view mapped onto the input array.

    args:
        input_array (numpy.ndarray):
            A 2-D array padded with nans for half the
            neighbourhood size.
        shape (tuple(int)):
            The neighbourhood shape e.g. if the neighbourhood
            size is 3, the shape would be (3, 3) to create a
            3x3 array around each point in the input_array.
        writeable (bool):
            If True the returned view will be writeable. This will modify
            the input array, so use with caution.

    Returns:
        numpy.ndarray:
            "views" into the data, each view represents
            a neighbourhood of points.
    """
    num_window_dims = len(shape)
    num_arr_dims = len(input_array.shape)
    assert num_arr_dims >= num_window_dims
    adjshp = (
        *input_array.shape[:-num_window_dims],
        *(
            arr_dims - win_dims + 1
            for arr_dims, win_dims in zip(input_array.shape[-num_window_dims:], shape)
        ),
        *shape,
    )
    assert all(arr_dims > 0 for arr_dims in adjshp)
    strides = input_array.strides + input_array.strides[-num_window_dims:]
    return np.lib.stride_tricks.as_strided(
        input_array, shape=adjshp, strides=strides, writeable=writeable
    )


def pad_and_roll(input_array, shape, **kwargs):
    """Pads the last `len(shape)` axes of the input array for `rolling_window`
    to create 'neighbourhood' views of the data of a given `shape` as the last
    axes in the returned array. Collapsing over the last `len(shape)` axes
    results in a shape of the original input array.

    args:
        input_array (numpy.ndarray):
            The dataset of points to pad and create rolling windows for.
        shape (tuple(int)):
            Desired shape of the neighbourhood. E.g. if a neighbourhood
            width of 1 around the point is desired, this shape should be (3, 3)::

                X X X
                X O X
                X X X

            Where O is our central point and X represent the neighbour points.
        kwargs:
            additional keyword arguments passed to `numpy.pad` function.

    Returns:
        numpy.ndarray:
            Contains the views of the input_array, the final dimension of
            the array will be the specified shape in the input arguments,
            the leading dimensions will depend on the shape of the input array.
    """
    writeable = kwargs.pop("writeable", False)
    pad_extent = [(0, 0)] * (len(input_array.shape) - len(shape))
    pad_extent.extend((d // 2, d // 2) for d in shape)
    input_array = np.pad(input_array, pad_extent, **kwargs)
    return rolling_window(input_array, shape, writeable=writeable)


def pad_boxsum(data, boxsize, **pad_options):
    """Pad an array to shape suitable for `boxsum`.

    Args:
        data (numpy.ndarray):
            The input data array.
        boxsize (int or pair of int):
            The size of the neighbourhood.
        pad_options (dict):
            Additional keyword arguments passed to `numpy.pad` function.
    Returns:
        numpy.ndarray:
            Array padded to shape suitable for `boxsum`.
    """
    boxsize = np.atleast_1d(boxsize)
    ih, jh = boxsize[0] // 2, boxsize[-1] // 2
    padding = [(0, 0)] * (data.ndim - 2) + [(ih + 1, ih), (jh + 1, jh)]
    padded = np.pad(data, padding, **pad_options)
    return padded


def boxsum(data, boxsize, cumsum=True, **pad_options):
    """Fast vectorised approach to calculating neighbourhood totals.

    Displacements are calculated as follows for the following input array,
    where the accumulation has occurred from top to bottom and left to right::

    | 1 | 2 | 2 | 2 |
    | 1 | 3 | 4 | 4 |
    | 2 | 4 | 5 | 6 |
    | 2 | 4 | 6 | 7 |


    For a 3x3 neighbourhood centred around the point with a value of 5::

    | 1 (C) | 2 | 2                 | 2 (D) |
    | 1     | 3 | 4                 | 4     |
    | 2     | 4 | 5 (Central point) | 6     |
    | 2 (A) | 4 | 6                 | 7 (B) |

    To calculate the value for the neighbourhood sum at the "Central point"
    with a value of 5, calculate::

      Neighbourhood sum = B - A - D + C

    At the central point, this will yield::

      Neighbourhood sum = 7 - 2 - 2 +1 => 4

    Args:
        data (numpy.ndarray):
            The input data array.
        boxsize (int or pair of int):
            The size of the neighbourhood.
        cumsum (bool or callable):
            If False, assume the input data is already cumulative. If True
            (default), calculate cumsum along the last two dimensions of
            the input array. If callable is passed, use it for this calculation
            instead of `data.cumsum(-2).cumsum(-1)`.
        pad_options (dict):
            Additional keyword arguments passed to `numpy.pad` function.
            If given, the returned result will have the same shape as the input
            array.

    Returns:
        numpy.ndarray:
            Array containing the calculated neighbourhood total.
    """
    boxsize = np.atleast_1d(boxsize)
    if pad_options:
        data = pad_boxsum(data, boxsize, **pad_options)
    if callable(cumsum):
        data = cumsum(data)
    elif cumsum:
        data = data.cumsum(-2).cumsum(-1)
    i, j = boxsize[0], boxsize[-1]
    m, n = data.shape[-2] - i, data.shape[-1] - j
    result = (
        data[..., i : i + m, j : j + n]
        - data[..., :m, j : j + n]
        + data[..., :m, :n]
        - data[..., i : i + m, :n]
    )
    return result
