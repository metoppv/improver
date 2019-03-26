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
"""Unit tests for the plugins and functions within indexing_operations.py"""

import unittest
import numpy as np

from iris.tests import IrisTest
from improver.utilities.indexing_operations import choose


class Test_choose(IrisTest):

    """Test the choose function behaves as expected, giving the same results
    as the numpy choose method, but without the 32 leading dimensions limit."""

    def setUp(self):
        """Set up the data arrays."""
        self.data = np.arange(132).reshape(33, 2, 2) + 1
        self.small_data = np.arange(12).reshape(3, 2, 2) + 1

    def test_single_index(self):
        """Test that an array of indices containing only one value returns the
        expected values."""
        index_array = [0]
        expected = np.array([[1, 2], [3, 4]])
        result = choose(index_array, self.data)
        self.assertArrayEqual(result, expected)
        self.assertEqual(result.shape, expected.shape)

    def test_1D_index_array_columns(self):
        """Test that a 1D array of indices returns the expected values."""
        index_array = [0, 1]
        expected = np.array([[1, 6], [3, 8]])
        result = choose(index_array, self.data)
        self.assertArrayEqual(result, expected)
        self.assertEqual(result.shape, expected.shape)

    def test_2D_index_array_rows_only(self):
        """Test that a 2D array of indices, arranged as a single column,
        returns the expected values."""
        index_array = [[0], [1]]
        expected = np.array([[1, 2], [7, 8]])
        result = choose(index_array, self.data)
        self.assertArrayEqual(result, expected)
        self.assertEqual(result.shape, expected.shape)

    def test_2D_index_array(self):
        """Test that a 2D array of indices returns the expected values."""
        index_array = [[0, 1], [1, 2]]
        expected = np.array([[1, 6], [7, 12]])
        result = choose(index_array, self.data)
        self.assertArrayEqual(result, expected)
        self.assertEqual(result.shape, expected.shape)

    def test_utilising_indices_beyond_32(self):
        """An explicit test that this function is handling indices beyond 32.
        The numpy choose function does not support a data array with a leading
        dimension of longer than 32. Note that due to indexing from 0, an index
        of 32 here is for array 33, beyond numpy's limit."""
        index_array = [32]
        expected = np.array([[129, 130], [131, 132]])
        result = choose(index_array, self.data)
        self.assertArrayEqual(result, expected)
        self.assertEqual(result.shape, expected.shape)

    def test_numpy_choose_comparison_single_index(self):
        """Test that an array of indices containing only one value returns the
        same values and array shape as numpy choose."""
        index_array = [0]
        expected = np.choose(index_array, self.small_data)
        result = choose(index_array, self.small_data)
        self.assertArrayEqual(result, expected)
        self.assertEqual(result.shape, expected.shape)

    def test_nupmy_choose_comparison_2D_index_array_rows_only(self):
        """Test that a 2D array of indices, arranged as a single column,
        returns the same values and array shape as numpy choose."""
        index_array = [[0], [1]]
        expected = np.choose(index_array, self.small_data)
        result = choose(index_array, self.small_data)
        self.assertArrayEqual(result, expected)
        self.assertEqual(result.shape, expected.shape)

    def test_numpy_choose_comparison_2D_index_array(self):
        """Test that a 2D array of indices returns the same values and array
        shape as numpy choose."""
        index_array = [[0, 1], [1, 2]]
        expected = np.choose(index_array, self.small_data)
        result = choose(index_array, self.small_data)
        self.assertArrayEqual(result, expected)
        self.assertEqual(result.shape, expected.shape)

    def test_numpy_choose_comparison_1D_index_array_columns(self):
        """Test that a 1D array of indices returns the same values and array
        shape as numpy choose."""
        index_array = [0, 1]
        expected = np.choose(index_array, self.small_data)
        result = choose(index_array, self.small_data)
        self.assertArrayEqual(result, expected)
        self.assertEqual(result.shape, expected.shape)


if __name__ == '__main__':
    unittest.main()
