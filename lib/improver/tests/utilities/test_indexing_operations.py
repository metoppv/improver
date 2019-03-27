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

    def test_3D_index_array(self):
        """Test that a 3D array of indices returns the expected values. Note
        that in this case there is an additional dimension in the returned
        array as the index_array has as many dimensions as the data array."""
        index_array = [[[0, 1], [1, 2]]]
        expected = np.array([[[1, 6], [7, 12]]])
        result = choose(index_array, self.data)
        self.assertArrayEqual(result, expected)
        self.assertEqual(result.shape, expected.shape)

    def test_4D_index_array(self):
        """Test that using an array of indices of higher dimesionality than the
        data_array results in a sensible error."""
        index_array = [[[[0, 1], [1, 2]]]]
        msg = ("Dimensionality of array_set has increased which will prevent "
               "indexing from working as expected.")
        with self.assertRaisesRegexp(IndexError, msg):
            choose(index_array, self.data)

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

    def test_numpy_choose_comparison_1D_index_array_columns(self):
        """Test that a 1D array of indices returns the same values and array
        shape as numpy choose."""
        index_array = [0, 1]
        expected = np.choose(index_array, self.small_data)
        result = choose(index_array, self.small_data)
        self.assertArrayEqual(result, expected)
        self.assertEqual(result.shape, expected.shape)

    def test_numpy_choose_comparison_2D_index_array_rows_only(self):
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

    def test_numpy_choose_comparison_3D_index_array(self):
        """Test that a 3D array of indices returns the same values and array
        shape as numpy choose. Note that in this case there is an additional
        dimension in the returned array as the index_array has as many
        dimensions as the data array."""
        index_array = [[[0, 1], [1, 2]]]
        expected = np.array([[[1, 6], [7, 12]]])
        result = choose(index_array, self.data)
        self.assertArrayEqual(result, expected)
        self.assertEqual(result.shape, expected.shape)

    def test_unbroadcastable_shape_error(self):
        """Test that a useful error is raised when the array of indices and the
        data arrays cannot be broadcast to a common shape."""
        index_array = [0, 1, 2]
        msg = ("shape mismatch: objects cannot be broadcast to a single shape")
        with self.assertRaisesRegexp(ValueError, msg):
            choose(index_array, self.small_data)

    def test_invalid_array_indices(self):
        """Test that a useful error is raised when the array that is indexed
        to provide data does not exist because the index is out of range. More
        simply, if there are only 3 sub-arays, indices of 3 or more should lead
        to a sensbile error. Note that the behaviour of this function is
        equivalent to numpy choose with mode=raise, there is no attempt to wrap
        or clip invalid index values."""
        index_array = [0, 3]
        msg = 'index_array contains an index that is larger than the number'
        with self.assertRaisesRegexp(IndexError, msg):
            choose(index_array, self.small_data)


if __name__ == '__main__':
    unittest.main()
