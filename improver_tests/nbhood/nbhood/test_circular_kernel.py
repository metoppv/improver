# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the nbhood.nbhood.circular_kernel function."""

import unittest

import numpy as np
from iris.tests import IrisTest

from improver.nbhood.nbhood import circular_kernel


class Test_circular_kernel(IrisTest):

    """Test neighbourhood processing plugin."""

    def test_basic(self):
        """Test that the plugin returns a Numpy array."""
        ranges = 2
        weighted_mode = False
        result = circular_kernel(ranges, weighted_mode)
        self.assertIsInstance(result, np.ndarray)

    def test_single_point_weighted(self):
        """Test behaviour for a unitary range, with weighting."""
        ranges = 1
        weighted_mode = True
        expected = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        result = circular_kernel(ranges, weighted_mode)
        self.assertArrayAlmostEqual(result, expected)

    def test_single_point_unweighted(self):
        """Test behaviour for a unitary range without weighting.
        Note that this gives one more grid cell range than weighted! As the
        affected area is one grid cell more in each direction, an equivalent
        range of 2 was chosen for this test."""
        ranges = 1
        weighted_mode = False
        expected = [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]
        result = circular_kernel(ranges, weighted_mode)
        self.assertArrayEqual(result, expected)

    def test_range5_weighted(self):
        """Test behaviour for a range of 5, with weighting."""
        ranges = 5
        weighted_mode = True
        expected = [
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.20, 0.32, 0.36, 0.32, 0.20, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.28, 0.48, 0.60, 0.64, 0.60, 0.48, 0.28, 0.00, 0.00],
            [0.00, 0.20, 0.48, 0.68, 0.80, 0.84, 0.80, 0.68, 0.48, 0.20, 0.00],
            [0.00, 0.32, 0.60, 0.80, 0.92, 0.96, 0.92, 0.80, 0.60, 0.32, 0.00],
            [0.00, 0.36, 0.64, 0.84, 0.96, 1.00, 0.96, 0.84, 0.64, 0.36, 0.00],
            [0.00, 0.32, 0.60, 0.80, 0.92, 0.96, 0.92, 0.80, 0.60, 0.32, 0.00],
            [0.00, 0.20, 0.48, 0.68, 0.80, 0.84, 0.80, 0.68, 0.48, 0.20, 0.00],
            [0.00, 0.00, 0.28, 0.48, 0.60, 0.64, 0.60, 0.48, 0.28, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.20, 0.32, 0.36, 0.32, 0.20, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        ]
        result = circular_kernel(ranges, weighted_mode)
        self.assertArrayAlmostEqual(result, expected)

    def test_range5_unweighted(self):
        """Test behaviour for a range of 5 without weighting."""
        ranges = 5
        weighted_mode = False
        expected = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        result = circular_kernel(ranges, weighted_mode)
        self.assertArrayEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
