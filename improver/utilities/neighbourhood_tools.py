# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2022 Met Office.
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

from typing import Any, Tuple, Union

import numpy as np
from numpy import ndarray


def rolling_window(
    input_array: ndarray, shape: Tuple[int, int], writeable: bool = False
) -> ndarray:
    """Creates a rolling window neighbourhood of the given `shape` from the
    last `len(shape)` axes of the input array. Avoids creating a large output
    array by constructing a non-continuous view mapped onto the input array.

    args:
        input_array:
            An array from which rolling window neighbourhoods will be created.
        shape:
            The neighbourhood shape e.g. if the neighbourhood
            size is 3, the shape would be (3, 3) to create a
            3x3 array around each point in the input_array.
        writeable:
            If True the returned view will be writeable. This will modify
            the input array, so use with caution.

    Returns:
        "views" into the data, each view represents
        a neighbourhood of points.

    Raises:
        ValueError: If `input_array` has fewer dimensions than `shape`.
        RuntimeError: If any dimension of `shape` is larger than
            the corresponding dimension of `input_array`.
    """
    num_window_dims = len(shape)
    num_arr_dims = len(input_array.shape)
    if num_arr_dims < num_window_dims:
        raise ValueError(
            "Number of dimensions of the input array must be greater than or "
            "equal to the length of the neighbourhood shape used for "
            "constructing rolling window neighbourhoods."
        )
    adjshp = (
        *input_array.shape[:-num_window_dims],
        *(
            arr_dims - win_dims + 1
            for arr_dims, win_dims in zip(input_array.shape[-num_window_dims:], shape)
        ),
        *shape,
    )
    if any(arr_dims <= 0 for arr_dims in adjshp):
        raise RuntimeError(
            "The calculated shape of the output array view contains a "
            "dimension that is negative or zero. Each dimension of the "
            "neighbourhood shape must be less than or equal to the "
            "corresponding dimension of the input array."
        )
    strides = input_array.strides + input_array.strides[-num_window_dims:]
    return np.lib.stride_tricks.as_strided(
        input_array, shape=adjshp, strides=strides, writeable=writeable
    )


def pad_and_roll(
    input_array: ndarray, shape: Tuple[int, int], **kwargs: Any
) -> ndarray:
    """Pads the last `len(shape)` axes of the input array for `rolling_window`
    to create 'neighbourhood' views of the data of a given `shape` as the last
    axes in the returned array. Collapsing over the last `len(shape)` axes
    results in a shape of the original input array.

    args:
        input_array:
            The dataset of points to pad and create rolling windows for.
        shape:
            Desired shape of the neighbourhood. E.g. if a neighbourhood
            width of 1 around the point is desired, this shape should be (3, 3)::

                X X X
                X O X
                X X X

            Where O is our central point and X represent the neighbour points.
        kwargs:
            additional keyword arguments passed to `numpy.pad` function.

    Returns:
        Contains the views of the input_array, the final dimension of
        the array will be the specified shape in the input arguments,
        the leading dimensions will depend on the shape of the input array.
    """
    writeable = kwargs.pop("writeable", False)
    pad_extent = [(0, 0)] * (len(input_array.shape) - len(shape))
    pad_extent.extend((d // 2, d // 2) for d in shape)
    input_array = np.pad(input_array, pad_extent, **kwargs)
    return rolling_window(input_array, shape, writeable=writeable)


def pad_boxsum(
    data: ndarray, boxsize: Union[int, Tuple[int, int]], **pad_options: Any
) -> ndarray:
    """Pad an array to shape suitable for `boxsum`.

    Note that padding is not symmetric: there is an extra row/column at
    the top/left (as required for calculating the boxsum).

    Args:
        data:
            The input data array.
        boxsize:
            The size of the neighbourhood.
        pad_options:
            Additional keyword arguments passed to `numpy.pad` function.
    Returns:
        Array padded to shape suitable for `boxsum`.
    """
    boxsize = np.atleast_1d(boxsize)
    ih, jh = boxsize[0] // 2, boxsize[-1] // 2
    padding = [(0, 0)] * (data.ndim - 2) + [(ih + 1, ih), (jh + 1, jh)]
    padded = np.pad(data, padding, **pad_options)
    return padded


def boxsum(
    data: ndarray,
    boxsize: Union[int, Tuple[int, int]],
    cumsum: bool = True,
    **pad_options: Any,
) -> ndarray:
    """Fast vectorised approach to calculating neighbourhood totals.

    This function makes use of the summed-area table method. An input
    array is accumulated top to bottom and left to right. This accumulated
    array can then be used to efficiently calculate the total within a
    neighbourhood about any point. An example input data array::

        | 1 | 1 | 1 | 1 | 1 |
        | 1 | 1 | 1 | 1 | 1 |
        | 1 | 1 | 1 | 1 | 1 |
        | 1 | 1 | 1 | 1 | 1 |

    is accumulated to become::

        | 1 | 2  | 3  | 4  | 5  |
        | 2 | 4  | 6  | 8  | 10 |
        | 3 | 6  | 9  | 12 | 15 |
        | 4 | 8  | 12 | 16 | 20 |
        | 5 | 10 | 15 | 20 | 25 |

    If we wish to calculate the total in a 3x3 neighbourhood about
    some point (*) of our array we use the following points::

        | 1 (C) | 2  | 3     | 4 (D)  | 5  |
        | 2     | 4  | 6     | 8      | 10 |
        | 3     | 6  | 9 (*) | 12     | 15 |
        | 4 (A) | 8  | 12    | 16 (B) | 20 |
        | 5     | 10 | 15    | 20     | 25 |

    And the calculation is::

        Neighbourhood sum = C - A - D + B
        = 1 - 4 - 4 + 16
        = 9

    This is the value we would expect for a 3x3 neighbourhood
    in an array filled with ones.

    Args:
        data:
            The input data array.
        boxsize:
            The size of the neighbourhood. Must be an odd number.
        cumsum:
            If False, assume the input data is already cumulative. If True
            (default), calculate cumsum along the last two dimensions of
            the input array.
        pad_options:
            Additional keyword arguments passed to `numpy.pad` function.
            If given, the returned result will have the same shape as the input
            array.

    Returns:
        Array containing the calculated neighbourhood total.

    Raises:
        ValueError: If `boxsize` has non-integer type.
        ValueError: If any member of `boxsize` is not an odd number.
    """
    boxsize = np.atleast_1d(boxsize)
    if not issubclass(boxsize.dtype.type, np.integer):
        raise ValueError("The size of the neighbourhood must be of an integer type.")
    if not np.all(boxsize % 2):
        raise ValueError("The size of the neighbourhood must be an odd number.")
    if pad_options:
        data = pad_boxsum(data, boxsize, **pad_options)
    if cumsum:
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
