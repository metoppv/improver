# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""Unit tests for spatial padding utilities"""

import numpy as np
import pytest

from improver.utilities.neighbourhood_tools import (
    boxsum,
    pad_and_roll,
    pad_boxsum,
    rolling_window,
)


@pytest.fixture
def array_size_5():
    return np.arange(25).astype(np.int32).reshape((5, 5))


@pytest.fixture
def array_size_3():
    return np.arange(9).astype(np.int32).reshape((3, 3))


def test_rolling_window_neighbourhood_size_2(array_size_5):
    """Test producing a 2 * 2 neighbourhood."""
    windows = rolling_window(array_size_5, (2, 2))
    expected = np.zeros((4, 4, 2, 2), dtype=np.int32)
    for i in range(4):
        for j in range(4):
            expected[i, j] = array_size_5[i : i + 2, j : j + 2]
    np.testing.assert_array_equal(windows, expected)


def test_rolling_window_exception_too_many_dims(array_size_5):
    """Test an exception is raised if shape has too many dimensions."""
    msg = (
        "Number of dimensions of the input array must be greater than or "
        "equal to the length of the neighbourhood shape used for "
        "constructing rolling window neighbourhoods."
    )
    with pytest.raises(ValueError) as exc_info:
        rolling_window(array_size_5, (2, 2, 2))
    assert msg in str(exc_info.value)


def test_rolling_window_exception_dims_too_large(array_size_5):
    """Test an exception is raised if dimensions of shape are larger than 
    corresponding dimensions of input array."""
    msg = (
        "The calculated shape of the output array view contains a "
        "dimension that is negative or zero. Each dimension of the "
        "neighbourhood shape must be less than or equal to the "
        "corresponding dimension of the input array."
    )
    with pytest.raises(RuntimeError) as exc_info:
        rolling_window(array_size_5, (2, 6))
    assert msg in str(exc_info.value)


def test_rolling_window_writable(array_size_5):
    """Test that result is writable if and only if `writable` is True."""
    windows = rolling_window(array_size_5, (2, 2))
    msg = "assignment destination is read-only"
    with pytest.raises(ValueError) as exc_info:
        windows[0, 0, 0, 0] = -1
    assert msg in str(exc_info.value)
    windows = rolling_window(array_size_5, (2, 2), writeable=True)
    windows[0, 0, 0, 0] = -1
    assert windows[0, 0, 0, 0] == -1


def test_padding_neighbourhood_size_2(array_size_5):
    """Test that result is same as result of rolling_window with a border of zeros."""
    padded = pad_and_roll(array_size_5, (2, 2), mode="constant")
    window = rolling_window(array_size_5, (2, 2))
    inner_part = padded[1:-1, 1:-1, ::]
    np.testing.assert_array_equal(inner_part, window)
    border_index = (
        [[0, i, 0, j] for i in range(5) for j in [0, 1]]
        + [[5, i, 1, j] for i in range(5) for j in [0, 1]]
        + [[i, 0, j, 0] for i in range(5) for j in [0, 1]]
        + [[i, 5, j, 1] for i in range(5) for j in [0, 1]]
    )
    outer_part = padded[list(zip(*border_index))]
    np.testing.assert_array_equal(outer_part, np.zeros(40, dtype=np.int32))


def test_padding_non_zero(array_size_5):
    """Test padding with a number other than the default of 0."""
    padded = pad_and_roll(array_size_5, (2, 2), mode="constant", constant_values=1)
    border_index = (
        [[0, i, 0, j] for i in range(5) for j in [0, 1]]
        + [[5, i, 1, j] for i in range(5) for j in [0, 1]]
        + [[i, 0, j, 0] for i in range(5) for j in [0, 1]]
        + [[i, 5, j, 1] for i in range(5) for j in [0, 1]]
    )
    outer_part = padded[list(zip(*border_index))]
    np.testing.assert_array_equal(outer_part, np.ones(40, dtype=np.int32))


def test_pad_boxsum(array_size_3):
    """Test that padded array consists of input array surrounded by border of zeros."""
    padded = pad_boxsum(array_size_3, 3, mode="constant")
    expected = np.zeros((6, 6), dtype=np.int32)
    expected[2:5, 2:5] = array_size_3
    np.testing.assert_array_equal(padded, expected)


def test_pad_boxsum_non_zero(array_size_3):
    """Test padding with a number other than the default of 0."""
    padded = pad_boxsum(array_size_3, 3, mode="constant", constant_values=2)
    expected = 2 * np.ones((6, 6), dtype=np.int32)
    expected[2:5, 2:5] = array_size_3
    np.testing.assert_array_equal(padded, expected)


def test_boxsum_with_automatic_cumsum(array_size_5):
    """Test that boxsum correctly calculates neighbourhood sums using raw array."""
    result = boxsum(array_size_5, 3)
    expected = np.array(
        [  # pylint: disable=E1136
            [np.sum(array_size_5[i - 1 : i + 2, j - 1 : j + 2]) for j in [2, 3]]
            for i in [2, 3]
        ]
    )
    np.testing.assert_array_equal(result, expected)


def test_boxsum_non_square(array_size_5):
    """Test that boxsum correctly calculates neighbourhood sums using 
    non-square box."""
    result = boxsum(array_size_5, (1, 3))
    expected = np.array(
        [  # pylint: disable=E1136
            [np.sum(array_size_5[i, j - 1 : j + 2]) for j in [2, 3]]
            for i in range(1, 5)
        ]
    )
    np.testing.assert_array_equal(result, expected)


def test_boxsum_with_precalculated_cumsum(array_size_5):
    """Test that boxsum correctly calculates neighbourhood sums using 
    pre-calculated cumsum."""
    cumsum_arr = np.array(
        [  # pylint: disable=E1136
            [np.sum(array_size_5[: i + 1, : j + 1]) for j in range(5)] for i in range(5)
        ]
    )
    result = boxsum(cumsum_arr, 3, cumsum=False)
    expected = np.array(
        [  # pylint: disable=E1136
            [np.sum(array_size_5[i - 1 : i + 2, j - 1 : j + 2]) for j in [2, 3]]
            for i in [2, 3]
        ]
    )
    np.testing.assert_array_equal(result, expected)


def test_boxsum_with_padding(array_size_5):
    """Test that boxsum correctly calculates neighbourhood sums when adding padding to array."""
    result = boxsum(array_size_5, 3, mode="constant", constant_values=0)
    expected = np.array(
        [
            [  # pylint: disable=E1136
                np.sum(array_size_5[max(0, i - 1) : i + 2, max(0, j - 1) : j + 2])
                for j in range(5)
            ]
            for i in range(5)
        ]
    )
    np.testing.assert_array_equal(result, expected)


def test_boxsum_exception_non_integer(array_size_5):
    """Test that an exception is raised if `boxsize` is not an integer."""
    msg = "The size of the neighbourhood must be of an integer type."
    with pytest.raises(ValueError) as exc_info:
        boxsum(array_size_5, 1.5)
    assert msg in str(exc_info.value)


def test_boxsum_exception_not_odd(array_size_5):
    """Test that an exception is raised if `boxsize` contains a number that is not odd."""
    msg = "The size of the neighbourhood must be an odd number."
    with pytest.raises(ValueError) as exc_info:
        boxsum(array_size_5, (1, 2))
    assert msg in str(exc_info.value)
