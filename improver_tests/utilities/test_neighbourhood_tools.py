# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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

import unittest

import numpy as np
from iris.tests import IrisTest

from improver.utilities.neighbourhood_tools import (
    boxsum,
    pad_and_roll,
    pad_boxsum,
    rolling_window,
)

class Test_creating_rolling_window_neighbourhoods(IrisTest):

    """Test creating rolling window neighbourhoods of an array."""

    def setUp(self):
        """Set up a 5 * 5 array."""
        self.array = np.arange(25).astype(np.int32).reshape((5, 5))

    def test_neighbourhood_size_2(self):
        """Test producing a 2 * 2 neighbourhood."""
        windows = rolling_window(self.array, (2, 2))
        expected = np.zeros((4, 4, 2, 2), dtype=np.int32)
        for i in range(4):
            for j in range(4):
                expected[i, j] = self.array[i : i + 2, j : j + 2]
        self.assertArrayEqual(windows, expected)

    def test_exception_too_many_dims(self):
        """Test an exception is raised if shape has too many dimensions."""
        msg = (
            "Number of dimensions of the input array must be greater than or equal to "
            "the length of the neighbourhood shape used for constructing rolling window neighbourhoods"
        )
        with self.assertRaisesRegex(ValueError, msg):
            rolling_window(self.array, (2, 2, 2))

    def test_exception_dims_too_large(self):
        """Test an exception is raised if dimensions of shape are larger than 
        corresponding dimensions of input array."""
        msg = (
            "The calculated shape of the output array view contains a dimension that is negative or zero. "
            "Each dimension of the neighbourhood shape must be less than or equal to the corresponding "
            "dimension of the input array."
        )
        with self.assertRaisesRegex(RuntimeError, msg):
            rolling_window(self.array, (2, 6))

    def test_writable(self):
        """Test that result is writable if and only if `writable` is True."""
        windows = rolling_window(self.array, (2, 2))
        msg = "assignment destination is read-only"
        with self.assertRaisesRegex(ValueError, msg):
            windows[0, 0, 0, 0] = -1
        windows = rolling_window(self.array, (2, 2), writeable=True)
        windows[0, 0, 0, 0] = -1
        self.assertEqual(windows[0, 0, 0, 0], -1)


class Test_padding_and_creating_rolling_window_neighbourhoods(IrisTest):

    """Test creating rolling window neighbourhoods with padding."""

    def setUp(self):
        """Set up a 5 * 5 array."""
        self.array = np.arange(25).astype(np.int32).reshape((5, 5))

    def test_neighbourhood_size_2(self):
        """Test that result is same as result of rolling_window with a border of zeros"""
        padded = pad_and_roll(self.array, (2, 2))
        window = rolling_window(self.array, (2, 2))
        inner_part = padded[1: -1, 1: -1, : :]
        self.assertArrayEqual(inner_part, window)
        border_index = [[0, i, 0, j] for i in range(5) for j in [0, 1]] \
                       + [[5, i, 1, j] for i in range(5) for j in [0, 1]] \
                       + [[i, 0, j, 0] for i in range(5) for j in [0, 1]] \
                       + [[i, 5, j, 1] for i in range(5) for j in [0, 1]]
        outer_part = padded[list(zip(*border_index))]
        self.assertArrayEqual(outer_part, np.zeros(40, dtype=np.int32))

    def test_non_zero_padding(self):
        """Test padding with a number other than the default of 0"""
        padded = pad_and_roll(self.array, (2, 2), mode='constant', constant_values=1)
        border_index = [[0, i, 0, j] for i in range(5) for j in [0, 1]] \
                       + [[5, i, 1, j] for i in range(5) for j in [0, 1]] \
                       + [[i, 0, j, 0] for i in range(5) for j in [0, 1]] \
                       + [[i, 5, j, 1] for i in range(5) for j in [0, 1]]
        outer_part = padded[list(zip(*border_index))]
        self.assertArrayEqual(outer_part, np.ones(40, dtype=np.int32))


class Test_padding_for_boxsum(IrisTest):

    """Test padding an array to shape suitable for `boxsum`."""

    def setUp(self):
        """Set up a 3 * 3 array."""
        self.array = np.arange(9).astype(np.int32).reshape((3, 3))

    def test_padding(self):
        """Test that padded array consists of input array surrounded by border of zeros."""
        padded = pad_boxsum(self.array, 3)
        expected = np.zeros((6, 6), dtype=np.int32)
        expected[2:5, 2:5] = self.array
        self.assertArrayEqual(padded, expected)

    def test_padding_non_zero(self):
        """Test padding with a number other than the default of 0"""
        padded = pad_boxsum(self.array, 3, mode='constant', constant_values=2)
        expected = 2 * np.ones((6, 6), dtype=np.int32)
        expected[2:5, 2:5] = self.array
        self.assertArrayEqual(padded, expected)

if __name__ == "__main__":
    unittest.main()
