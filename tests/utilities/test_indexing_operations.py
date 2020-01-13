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

    def test_3D_index_array_test1(self):
        """Test that a 3D array of indices with a shape matching the data array
        returns the expected values. Here values are taken from a mix of
        sub-arrays. This example can be seen graphically in the documentation
        for the choose function."""
        index_array = np.array([[[0, 1], [1, 0]],
                                [[0, 2], [0, 1]],
                                [[1, 1], [2, 0]]])
        expected = np.array([[[1, 6], [7, 4]],
                             [[1, 10], [3, 8]],
                             [[5, 6], [11, 4]]])
        result = choose(index_array, self.small_data)
        self.assertArrayEqual(result, expected)
        self.assertEqual(result.shape, expected.shape)

    def test_3D_index_array_test2(self):
        """Test that a 3D array of indices with a shape matching the data array
        returns the expected values. Here the sub-arrays are rearranged as
        complete units."""
        index_array = np.array([[[1, 1], [1, 1]],
                                [[2, 2], [2, 2]],
                                [[0, 0], [0, 0]]])
        expected = np.array([self.small_data[1],
                             self.small_data[2],
                             self.small_data[0]])
        result = choose(index_array, self.small_data)
        self.assertArrayEqual(result, expected)
        self.assertEqual(result.shape, expected.shape)

    def test_3D_index_array_utilising_indices_beyond_32(self):
        """An explicit test that this function is handling indices beyond 32.
        The numpy choose function does not support a data array with a leading
        dimension of longer than 32. Note that due to indexing from 0, an index
        of 32 here is for array 33, beyond numpy's limit."""
        index_array = np.ones(self.data.shape).astype(int)
        expected = np.array([self.data[1]] * 33)
        result = choose(index_array, self.data)
        self.assertArrayEqual(result, expected)
        self.assertEqual(result.shape, expected.shape)

    def test_numpy_choose_comparison_3D_index_array_test1(self):
        """Test that a 3D array of indices with a shape matching the data array
        returns the same result as numpy choose. Here values are taken from a
        mix of sub-arrays."""
        index_array = np.array([[[0, 1], [1, 0]],
                                [[0, 2], [0, 1]],
                                [[1, 1], [2, 0]]])
        choose_result = choose(index_array, self.small_data)
        npchoose_result = np.choose(index_array, self.small_data)
        self.assertArrayEqual(choose_result, npchoose_result)
        self.assertEqual(choose_result.shape, npchoose_result.shape)

    def test_numpy_choose_comparison_3D_index_array_test2(self):
        """Test that a 3D array of indices with a shape matching the data array
        returns the same result as numpy choose. Here the sub-arrays are
        rearranged as complete units."""
        index_array = np.array([[[1, 1], [1, 1]],
                                [[2, 2], [2, 2]],
                                [[0, 0], [0, 0]]])
        choose_result = choose(index_array, self.small_data)
        npchoose_result = np.choose(index_array, self.small_data)
        self.assertArrayEqual(choose_result, npchoose_result)
        self.assertEqual(choose_result.shape, npchoose_result.shape)

    def test_invalid_array_indices(self):
        """Test that a useful error is raised when the array that is indexed
        to provide data does not exist because the index is out of range. More
        simply, if there are only 3 sub-arays, indices of 3 or more should lead
        to a sensbile error. Note that the behaviour of this function is
        equivalent to numpy choose with mode=raise, there is no attempt to wrap
        or clip invalid index values."""
        index_array = np.array([[[0, 1], [1, 0]],
                                [[0, 2], [0, 1]],
                                [[3, 3], [3, 3]]])
        msg = 'index_array contains an index that is larger than the number'
        with self.assertRaisesRegex(IndexError, msg):
            choose(index_array, self.small_data)

    def test_unmatched_array_shapes(self):
        """Test that a useful error is raised when the index_array and
        data_array have different shapes. This choose function provides only
        a subset of the full numpy choose features, and one of its limitations
        is to work only with arrays that match; there is no broadcasting."""
        index_array = np.array([[[0, 1], [1, 0]],
                                [[0, 2], [0, 1]]])
        msg = ("The choose function only works with an index_array that "
               "matches the shape of array_set.")
        with self.assertRaisesRegex(ValueError, msg):
            choose(index_array, self.small_data)


if __name__ == '__main__':
    unittest.main()
