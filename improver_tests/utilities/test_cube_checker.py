# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the cube_checker utility."""

import unittest

import numpy as np

from improver.synthetic_data.set_up_test_cubes import (
    set_up_variable_cube,
)
from improver.utilities.cube_checker import (
    check_for_x_and_y_axes,
)


class Test_check_for_x_and_y_axes(unittest.TestCase):
    """Test whether the cube has an x and y axis."""

    def setUp(self):
        """Set up a cube."""
        data = np.ones((1, 5, 5), dtype=np.float32)
        self.cube = set_up_variable_cube(
            data, name="precipitation_amount", units="kg m^-2", spatial_grid="equalarea"
        )

    def test_no_y_coordinate(self):
        """Test that the expected exception is raised, if there is no
        y coordinate."""
        sliced_cube = next(self.cube.slices(["projection_x_coordinate"]))
        sliced_cube.remove_coord("projection_y_coordinate")
        msg = "The cube does not contain the expected"
        with self.assertRaisesRegex(ValueError, msg):
            check_for_x_and_y_axes(sliced_cube)

    def test_no_x_coordinate(self):
        """Test that the expected exception is raised, if there is no
        x coordinate."""

        sliced_cube = next(self.cube.slices(["projection_y_coordinate"]))
        sliced_cube.remove_coord("projection_x_coordinate")
        msg = "The cube does not contain the expected"
        with self.assertRaisesRegex(ValueError, msg):
            check_for_x_and_y_axes(sliced_cube)

    def test_pass_dimension_requirement(self):
        """Pass in compatible cubes that should not raise an exception. No
        assert statement required as any other input will raise an
        exception."""
        check_for_x_and_y_axes(self.cube, require_dim_coords=True)

    def test_fail_dimension_requirement(self):
        """Test that the expected exception is raised, if there the x and y
        coordinates are not dimensional coordinates."""
        msg = "The cube does not contain the expected"
        cube = self.cube[0, :, 0]
        with self.assertRaisesRegex(ValueError, msg):
            check_for_x_and_y_axes(cube, require_dim_coords=True)


if __name__ == "__main__":
    unittest.main()
