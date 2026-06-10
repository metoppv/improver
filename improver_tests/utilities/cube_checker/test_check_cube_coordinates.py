# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import unittest

import iris
import numpy as np
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_checker import check_cube_coordinates


class Test_check_cube_coordinates(unittest.TestCase):
    """Test check_cube_coordinates successfully promotes scalar coordinates to
    dimension coordinates in a new cube if they were dimension coordinates in
    the progenitor cube."""

    def setUp(self):
        """Set up a cube."""
        data = np.ones((1, 16, 16), dtype=np.float32)
        self.cube = set_up_variable_cube(
            data, name="precipitation_amount", units="kg m^-2", spatial_grid="equalarea"
        )
        self.squeezed_cube = iris.util.squeeze(self.cube)

    def test_basic(self):
        """Test returns iris.cube.Cube."""
        result = check_cube_coordinates(self.cube, self.cube)
        self.assertIsInstance(result, Cube)

    def test_basic_transpose(self):
        """Test when we only want to transpose the new_cube."""
        new_cube = self.cube.copy()
        new_cube.transpose([2, 1, 0])
        result = check_cube_coordinates(self.cube, new_cube)
        self.assertEqual(result.dim_coords, self.cube.dim_coords)

    def test_coord_promotion(self):
        """Test that scalar coordinates in new_cube are promoted to dimension
        coordinates to match the parent cube."""
        result = check_cube_coordinates(self.cube, self.squeezed_cube)
        self.assertEqual(result.dim_coords, self.cube.dim_coords)
        self.assertEqual(
            result.coords(dim_coords=False), self.cube.coords(dim_coords=False)
        )

    def test_coord_promotion_and_reordering(self):
        """Test case in which a scalar coordinate are promoted but the order
        must be corrected to match the progenitor cube."""
        self.cube.transpose(new_order=[1, 0, 2])
        result = check_cube_coordinates(self.cube, self.squeezed_cube)
        self.assertEqual(result.dim_coords, self.cube.dim_coords)

    def test_permitted_exception_coordinates(self):
        """Test that if the new_cube is known to have additional coordinates
        compared with the original cube, these coordinates are listed as
        exception_coordinates and handled correctly."""
        exception_coordinates = ["realization"]
        result = check_cube_coordinates(
            self.squeezed_cube, self.cube, exception_coordinates=exception_coordinates
        )
        dim_coords = (
            tuple(self.cube.coord("realization")) + self.squeezed_cube.dim_coords
        )
        self.assertEqual(result.dim_coords, dim_coords)

    def test_no_permitted_exception_coordinates(self):
        """Test that if the new_cube has additional coordinates compared with
        the original cube, if no coordinates are listed as exception
        coordinates, then an exception will be raised."""
        msg = "The number of dimension coordinates within the new cube"
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            check_cube_coordinates(self.squeezed_cube, self.cube)

    def test_missing_exception_coordinates(self):
        """Test that if the new_cube has additional coordinates compared with
        the original cube, if these coordinates are not listed as exception
        coordinates, then an exception will be raised."""
        exception_coordinates = ["height"]
        msg = "All permitted exception_coordinates must be on the new_cube."
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            check_cube_coordinates(
                self.squeezed_cube,
                self.cube,
                exception_coordinates=exception_coordinates,
            )

    def test_coord_promotion_missing_scalar(self):
        """Test case in which a scalar coordinate has been lost from new_cube,
        meaning the cube undergoing checking ends up with different dimension
        coordinates to the progenitor cube. This raises an error."""
        self.squeezed_cube.remove_coord("realization")
        msg = "The number of dimension coordinates within the new cube"
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            check_cube_coordinates(self.cube, self.squeezed_cube)


if __name__ == "__main__":
    unittest.main()
