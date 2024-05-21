# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the function "cube_manipulation.sort_coord_in_cube".
"""

import unittest

import iris
import numpy as np
import pytest
from iris.coords import AuxCoord
from iris.tests import IrisTest

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import sort_coord_in_cube


class Test_sort_coord_in_cube(IrisTest):
    """Class to test the sort_coord_in_cube function."""

    def setUp(self):
        """Set up ascending and descending cubes"""
        self.ascending_height_points = np.array([5.0, 10.0, 20.0], dtype=np.float32)
        self.descending_height_points = np.flip(self.ascending_height_points)

        self.data = np.array(
            [np.ones((3, 3)), 2 * np.ones((3, 3)), 3 * np.ones((3, 3))],
            dtype=np.float32,
        )
        self.ascending_cube = set_up_variable_cube(self.data)
        self.ascending_cube.coord("realization").rename("height")
        self.ascending_cube.coord("height").points = self.ascending_height_points
        self.ascending_cube.coord("height").units = "m"

        self.descending_cube = self.ascending_cube.copy()
        self.descending_cube.coord("height").points = self.descending_height_points

    def test_ascending_then_ascending(self):
        """Test that the sorting successfully sorts the cube based
        on the points within the given coordinate. The points in the resulting
        cube should now be in ascending order."""
        expected_data = self.data
        coord_name = "height"
        result = sort_coord_in_cube(self.ascending_cube, coord_name)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(
            self.ascending_cube.coord_dims(coord_name), result.coord_dims(coord_name)
        )
        self.assertArrayAlmostEqual(
            self.ascending_height_points, result.coord(coord_name).points
        )
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_auxcoord(self):
        """Test that the above sorting is successful when an AuxCoord is
        used."""
        expected_data = self.data
        coord_name = "height_aux"
        height_coord = self.ascending_cube.coord("height")
        (height_coord_index,) = self.ascending_cube.coord_dims("height")
        new_coord = AuxCoord(height_coord.points, long_name=coord_name)
        self.ascending_cube.add_aux_coord(new_coord, height_coord_index)
        result = sort_coord_in_cube(self.ascending_cube, coord_name)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(
            self.ascending_cube.coord_dims(coord_name), result.coord_dims(coord_name)
        )
        self.assertArrayAlmostEqual(
            self.ascending_height_points, result.coord(coord_name).points
        )
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_ascending_then_descending(self):
        """Test that the sorting successfully sorts the cube based
        on the points within the given coordinate. The points in the resulting
        cube should now be in descending order."""
        expected_data = np.flip(self.data)
        coord_name = "height"
        result = sort_coord_in_cube(self.ascending_cube, coord_name, descending=True)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(
            self.descending_cube.coord_dims(coord_name), result.coord_dims(coord_name)
        )
        self.assertArrayAlmostEqual(
            self.descending_height_points, result.coord(coord_name).points
        )
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_descending_then_ascending(self):
        """Test that the sorting successfully sorts the cube based
        on the points within the given coordinate. The points in the resulting
        cube should now be in ascending order."""
        expected_data = np.flip(self.data)
        coord_name = "height"
        result = sort_coord_in_cube(self.descending_cube, coord_name)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(
            self.ascending_cube.coord_dims(coord_name), result.coord_dims(coord_name)
        )
        self.assertArrayAlmostEqual(
            self.ascending_height_points, result.coord(coord_name).points
        )
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_descending_then_descending(self):
        """Test that the sorting successfully sorts the cube based
        on the points within the given coordinate. The points in the resulting
        cube should now be in descending order."""
        expected_data = self.data
        coord_name = "height"
        result = sort_coord_in_cube(self.descending_cube, coord_name, descending=True)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(
            self.descending_cube.coord_dims(coord_name), result.coord_dims(coord_name)
        )
        self.assertArrayAlmostEqual(
            self.descending_height_points, result.coord(coord_name).points
        )
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_latitude(self):
        """Test that the sorting successfully sorts the cube based
        on the points within the given coordinate (latitude).
        The points in the resulting cube should now be in descending order."""
        expected_data = np.array(
            [
                [[1.00, 1.00, 1.00], [1.00, 1.00, 1.00], [6.00, 1.00, 1.00]],
                [[2.00, 2.00, 2.00], [2.00, 2.00, 2.00], [6.00, 2.00, 2.00]],
                [[3.00, 3.00, 3.00], [3.00, 3.00, 3.00], [6.00, 3.00, 3.00]],
            ]
        )
        self.ascending_cube.data[:, 0, 0] = 6.0
        expected_points = np.flip(self.ascending_cube.coord("latitude").points)
        coord_name = "latitude"
        result = sort_coord_in_cube(self.ascending_cube, coord_name, descending=True)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(
            self.ascending_cube.coord_dims(coord_name), result.coord_dims(coord_name)
        )
        self.assertArrayAlmostEqual(expected_points, result.coord(coord_name).points)
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_warn_raised_for_circular_coordinate(self):
        """Test that a warning is successfully raised when circular
        coordinates are sorted."""
        self.ascending_cube.data[:, 0, 0] = 6.0
        coord_name = "latitude"
        self.ascending_cube.coord(coord_name).circular = True
        warning_msg = "The latitude coordinate is circular."

        with pytest.warns(UserWarning, match=warning_msg):
            result = sort_coord_in_cube(
                self.ascending_cube, coord_name, descending=True
            )

        self.assertIsInstance(result, iris.cube.Cube)


if __name__ == "__main__":
    unittest.main()
