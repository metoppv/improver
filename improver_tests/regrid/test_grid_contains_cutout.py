# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for grid_contains_cutout"""

import unittest

import numpy as np

from improver.regrid.landsea import grid_contains_cutout
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.spatial import calculate_grid_spacing


class Test_grid_contains_cutout(unittest.TestCase):
    """Test the grid_contains_cutout method"""

    def test_basic(self):
        """Test success for matching cubes"""
        grid = set_up_variable_cube(np.ones((10, 10), dtype=np.float32))
        cutout = set_up_variable_cube(np.zeros((10, 10), dtype=np.float32))
        self.assertTrue(grid_contains_cutout(grid, cutout))

    def test_success_equal_area(self):
        """Test success for an equal area cube created by subsetting another
        cube"""
        grid = set_up_variable_cube(
            np.ones((10, 10), dtype=np.float32), spatial_grid="equalarea"
        )
        cutout = grid[2:5, 3:7]
        self.assertTrue(grid_contains_cutout(grid, cutout))

    def test_success_latlon(self):
        """Test success for a lat/lon cube created by subsetting another
        cube"""
        grid = set_up_variable_cube(
            np.ones((10, 10), dtype=np.float32), spatial_grid="latlon"
        )
        cutout = grid[2:5, 3:7]
        self.assertTrue(grid_contains_cutout(grid, cutout))

    def test_failure_different_grids(self):
        """Test failure comparing an equal area with a lat/lon grid"""
        grid = set_up_variable_cube(np.ones((10, 10), dtype=np.float32))
        cutout = set_up_variable_cube(
            np.ones((10, 10), dtype=np.float32), spatial_grid="equalarea"
        )
        self.assertFalse(grid_contains_cutout(grid, cutout))

    def test_failure_different_units(self):
        """Test failure comparing grids in different units"""
        grid = set_up_variable_cube(
            np.ones((10, 10), dtype=np.float32), spatial_grid="equalarea"
        )
        cutout = grid.copy()
        for axis in ["x", "y"]:
            cutout.coord(axis=axis).convert_units("feet")
        self.assertFalse(grid_contains_cutout(grid, cutout))

    def test_failure_outside_domain(self):
        """Test failure if the cutout begins outside the grid domain"""
        grid = set_up_variable_cube(
            np.ones((10, 10), dtype=np.float32), spatial_grid="equalarea"
        )
        cutout = grid.copy()
        grid_spacing = calculate_grid_spacing(cutout, cutout.coord(axis="x").units)
        cutout.coord(axis="x").points = (
            cutout.coord(axis="x").points - 10 * grid_spacing
        )
        self.assertFalse(grid_contains_cutout(grid, cutout))

    def test_failure_partial_overlap(self):
        """Test failure if the cutout is only partially included in the
        grid"""
        grid = set_up_variable_cube(
            np.ones((10, 10), dtype=np.float32), spatial_grid="equalarea"
        )
        cutout = grid.copy()
        grid_spacing = calculate_grid_spacing(cutout, cutout.coord(axis="x").units)
        cutout.coord(axis="x").points = cutout.coord(axis="x").points + 2 * grid_spacing
        self.assertFalse(grid_contains_cutout(grid, cutout))

    def test_failure_mismatched_grid_points(self):
        """Test failure when grids overlap but points do not match"""
        grid = set_up_variable_cube(np.ones((10, 10), dtype=np.float32))
        cutout = set_up_variable_cube(np.ones((6, 7), dtype=np.float32))
        self.assertFalse(grid_contains_cutout(grid, cutout))


if __name__ == "__main__":
    unittest.main()
