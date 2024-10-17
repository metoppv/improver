# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the function collapsed.
"""

import unittest

import iris
import numpy as np

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import collapsed


class Test_collapsed(unittest.TestCase):
    """Test the collapsed utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        data = 281 * np.ones((3, 3, 3), dtype=np.float32)
        self.cube = set_up_variable_cube(data, realizations=[0, 1, 2])
        self.expected_data = self.cube.collapsed(
            ["realization"], iris.analysis.MEAN
        ).data

    def test_single_method(self):
        """Test that a collapsed cube is returned with no cell method added"""
        result = collapsed(self.cube, "realization", iris.analysis.MEAN)
        self.assertTupleEqual(result.cell_methods, ())
        self.assertTrue((result.data == self.expected_data).all())

    def test_two_methods(self):
        """Test that a cube keeps its original cell method but another
        isn't added.
        """
        cube = self.cube
        method = iris.coords.CellMethod("test")
        cube.add_cell_method(method)
        result = collapsed(cube, "realization", iris.analysis.MEAN)
        self.assertTupleEqual(result.cell_methods, (method,))
        self.assertTrue((result.data == self.expected_data).all())

    def test_two_coords(self):
        """Test behaviour collapsing over 2 coordinates, including not escalating
        precision when collapsing a float coordinate (latitude)"""
        expected_data = self.cube.collapsed(
            ["realization", "latitude"], iris.analysis.MEAN
        ).data
        result = collapsed(self.cube, ["realization", "latitude"], iris.analysis.MEAN)
        self.assertTrue((result.data == expected_data).all())
        self.assertEqual(
            result.coord("latitude").dtype, self.cube.coord("latitude").dtype
        )


if __name__ == "__main__":
    unittest.main()
