# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the function "cube_manipulation.clip_cube_data".
"""

import unittest

import numpy as np
from iris.cube import Cube
from iris.tests import IrisTest

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import clip_cube_data


class Test_clip_cube_data(IrisTest):
    """Test the clip_cube_data utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        data = np.array(
            [
                [
                    [226.15, 237.4, 248.65],
                    [259.9, 271.15, 282.4],
                    [293.65, 304.9, 316.15],
                ],
                [
                    [230.15, 241.4, 252.65],
                    [263.9, 275.15, 286.4],
                    [297.65, 308.9, 320.15],
                ],
            ],
            dtype=np.float32,
        )
        cube = set_up_variable_cube(data)
        self.minimum_value = cube.data.min()
        self.maximum_value = cube.data.max()
        self.processed_cube = cube.copy(data=cube.data * 2.0 - cube.data.mean())

    def test_basic(self):
        """Test that the utility returns a cube."""
        result = clip_cube_data(
            self.processed_cube, self.minimum_value, self.maximum_value
        )
        self.assertIsInstance(result, Cube)

    def test_clipping(self):
        """Test that the utility clips the processed cube to the same limits
        as the input cube when slicing over multiple x-y planes"""
        result = clip_cube_data(
            self.processed_cube, self.minimum_value, self.maximum_value
        )
        self.assertEqual(result.data.min(), self.minimum_value)
        self.assertEqual(result.data.max(), self.maximum_value)
        self.assertEqual(result.attributes, self.processed_cube.attributes)
        self.assertEqual(result.cell_methods, self.processed_cube.cell_methods)


if __name__ == "__main__":
    unittest.main()
