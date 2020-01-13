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
"""Unit tests for grid_contains_cutout"""

import unittest

import numpy as np

from improver.standardise import grid_contains_cutout
from improver.utilities.spatial import calculate_grid_spacing

from ..set_up_test_cubes import set_up_variable_cube


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
            np.ones((10, 10), dtype=np.float32), spatial_grid='equalarea')
        cutout = grid[2:5, 3:7]
        self.assertTrue(grid_contains_cutout(grid, cutout))

    def test_success_latlon(self):
        """Test success for a lat/lon cube created by subsetting another
        cube"""
        grid = set_up_variable_cube(
            np.ones((10, 10), dtype=np.float32), spatial_grid='latlon')
        cutout = grid[2:5, 3:7]
        self.assertTrue(grid_contains_cutout(grid, cutout))

    def test_failure_different_grids(self):
        """Test failure comparing an equal area with a lat/lon grid"""
        grid = set_up_variable_cube(np.ones((10, 10), dtype=np.float32))
        cutout = set_up_variable_cube(np.ones((10, 10), dtype=np.float32),
                                      spatial_grid='equalarea')
        self.assertFalse(grid_contains_cutout(grid, cutout))

    def test_failure_different_units(self):
        """Test failure comparing grids in different units"""
        grid = set_up_variable_cube(
            np.ones((10, 10), dtype=np.float32), spatial_grid='equalarea')
        cutout = grid.copy()
        for axis in ['x', 'y']:
            cutout.coord(axis=axis).convert_units('feet')
        self.assertFalse(grid_contains_cutout(grid, cutout))

    def test_failure_outside_domain(self):
        """Test failure if the cutout begins outside the grid domain"""
        grid = set_up_variable_cube(
            np.ones((10, 10), dtype=np.float32), spatial_grid='equalarea')
        cutout = grid.copy()
        grid_spacing = calculate_grid_spacing(
            cutout, cutout.coord(axis='x').units)
        cutout.coord(axis='x').points = (
            cutout.coord(axis='x').points - 10*grid_spacing)
        self.assertFalse(grid_contains_cutout(grid, cutout))

    def test_failure_partial_overlap(self):
        """Test failure if the cutout is only partially included in the
        grid"""
        grid = set_up_variable_cube(
            np.ones((10, 10), dtype=np.float32), spatial_grid='equalarea')
        cutout = grid.copy()
        grid_spacing = calculate_grid_spacing(
            cutout, cutout.coord(axis='x').units)
        cutout.coord(axis='x').points = (
            cutout.coord(axis='x').points + 2*grid_spacing)
        self.assertFalse(grid_contains_cutout(grid, cutout))

    def test_failure_mismatched_grid_points(self):
        """Test failure when grids overlap but points do not match"""
        grid = set_up_variable_cube(np.ones((10, 10), dtype=np.float32))
        cutout = set_up_variable_cube(np.ones((6, 7), dtype=np.float32))
        self.assertFalse(grid_contains_cutout(grid, cutout))


if __name__ == '__main__':
    unittest.main()
