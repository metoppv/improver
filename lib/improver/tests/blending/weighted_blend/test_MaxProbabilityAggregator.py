# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
"""Unit tests for the weighted_blend.PercentileBlendingAggregator class."""


import unittest

from iris.tests import IrisTest
import numpy as np

from improver.blending.weighted_blend import MaxProbabilityAggregator


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(MaxProbabilityAggregator())
        msg = '<MaxProbabilityAggregator>'
        self.assertEqual(result, msg)


class Test_aggregate(IrisTest):
    """Test the aggregate method"""
    def test_basic(self):
        """Test a simple case with only ones"""
        data = np.ones((1, 1, 2, 1))
        axis = 2
        weights = np.array([0, 1])
        plugin = MaxProbabilityAggregator
        result = plugin.aggregate(data, axis, weights)
        self.assertEqual(result.shape, (1, 1, 1))
        self.assertArrayEqual(result, np.ones((1, 1, 1)))

    def test_1D_data(self):
        """Test a simple case with one dimensional data"""
        data = np.array([1, 2, 3, 4, 5])
        axis = 0
        weights = np.array([0, 0.25, 0.5, 0.25, 0])
        plugin = MaxProbabilityAggregator
        result = plugin.aggregate(data, axis, weights)
        self.assertEqual(result.shape, ())
        self.assertArrayEqual(result, np.array([1.5]))

    def test_3D_data(self):
        """Test a simple case with dimensions, collapsing along axis 2 and
           so removing this axis from the result"""
        data = np.array([[[2, 2, 2, 2, 2],
                          [1, 2, 3, 4, 5]],
                         [[5, 5, 5, 5, 5],
                          [1, 4, 3, 8, 10]]])
        axis = 2
        weights = np.array([0, 0.25, 0.5, 0.25, 0])
        expected_data = np.array([[1, 1.5],
                                  [2.5, 2]])
        plugin = MaxProbabilityAggregator
        result = plugin.aggregate(data, axis, weights)
        self.assertEqual(result.shape, (2, 2))
        self.assertArrayEqual(result, expected_data)

    def test_negative_axis(self):
        """Test a case where a negative axis is provided, using the same 3D
           test data as test_3D_data"""
        data = np.array([[[2, 2, 2, 2, 2],
                          [1, 2, 3, 4, 5]],
                         [[5, 5, 5, 5, 5],
                          [1, 4, 3, 8, 10]]])
        axis = -1
        weights = np.array([0, 0.25, 0.5, 0.25, 0])
        expected_data = np.array([[1, 1.5],
                                  [2.5, 2]])
        plugin = MaxProbabilityAggregator
        result = plugin.aggregate(data, axis, weights)
        self.assertEqual(result.shape, (2, 2))
        self.assertArrayEqual(result, expected_data)


if __name__ == '__main__':
    unittest.main()
