# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import unittest

import numpy as np

from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)
from improver.utilities.cube_checker import find_dimension_coordinate_mismatch


class Test_find_dimension_coordinate_mismatch(unittest.TestCase):
    """Test if two cubes have the dimension coordinates."""

    def setUp(self):
        """Set up a cube."""
        data = np.ones((2, 16, 16), dtype=np.float32)
        self.cube = set_up_variable_cube(
            data, name="precipitation_amount", units="kg m^-2", spatial_grid="equalarea"
        )

    def test_no_mismatch(self):
        """Test if there is no mismatch between the dimension coordinates."""
        result = find_dimension_coordinate_mismatch(self.cube, self.cube)
        self.assertIsInstance(result, list)
        self.assertFalse(result)

    def test_mismatch_in_first_cube(self):
        """Test when finding a one-way mismatch, so that the second cube has
        a missing coordinate. This returns an empty list."""
        first_cube = self.cube.copy()
        second_cube = next(self.cube.slices_over("realization")).copy()
        second_cube.remove_coord("realization")
        result = find_dimension_coordinate_mismatch(
            first_cube, second_cube, two_way_mismatch=False
        )
        self.assertIsInstance(result, list)
        self.assertFalse(result)

    def test_mismatch_in_second_cube(self):
        """Test when finding a one-way mismatch, so that the first cube has
        a missing coordinate. This returns a list with the missing coordinate
        name."""
        first_cube = next(self.cube.slices_over("realization")).copy()
        first_cube.remove_coord("realization")
        second_cube = self.cube.copy()
        result = find_dimension_coordinate_mismatch(
            first_cube, second_cube, two_way_mismatch=False
        )
        self.assertIsInstance(result, list)
        self.assertListEqual(result, ["realization"])

    def test_two_way_mismatch(self):
        """Test when finding a two-way mismatch, when the first and second
        cube contain different coordinates."""
        first_cube = self.cube.copy()
        second_cube = next(self.cube.slices_over("realization")).copy()
        second_cube.remove_coord("realization")
        second_cube = add_coordinate(second_cube, [10, 20], "height", "m")
        result = find_dimension_coordinate_mismatch(first_cube, second_cube)
        self.assertIsInstance(result, list)
        self.assertListEqual(result, ["height", "realization"])
