# -*- coding: iso-8859-1 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
"""Unit tests for the convert_distance_into_number_of_grid_cells function from
 spatial.py."""

import unittest

from iris.tests import IrisTest

from improver.tests.nbhood.nbhood.test_NeighbourhoodProcessing import (
    set_up_cube, set_up_cube_lat_long)
from improver.utilities.spatial import (
    convert_distance_into_number_of_grid_cells)


class Test_convert_distance_into_number_of_grid_cells(IrisTest):

    """Test conversion of distance in metres into number of grid cells."""

    def setUp(self):
        """Set up the cube."""
        self.DISTANCE = 6100
        self.MAX_DISTANCE_IN_GRID_CELLS = 500
        self.cube = set_up_cube()

    def test_basic_distance_to_grid_cells(self):
        """Test the distance in metres to grid cell conversion."""
        result = convert_distance_into_number_of_grid_cells(
            self.cube, self.DISTANCE, self.MAX_DISTANCE_IN_GRID_CELLS)
        self.assertEqual(result, (3, 3))

    def test_basic_distance_to_grid_cells_different_max_distance(self):
        """
        Test the distance in metres to grid cell conversion for an alternative
        max distance in grid cells.
        """
        max_distance_in_grid_cells = 50
        result = convert_distance_into_number_of_grid_cells(
            self.cube, self.DISTANCE, max_distance_in_grid_cells)
        self.assertEqual(result, (3, 3))

    def test_basic_distance_to_grid_cells_km_grid(self):
        """Test the distance-to-grid-cell conversion, grid in km."""
        self.cube.coord("projection_x_coordinate").convert_units("kilometres")
        self.cube.coord("projection_y_coordinate").convert_units("kilometres")
        result = convert_distance_into_number_of_grid_cells(
            self.cube, self.DISTANCE, self.MAX_DISTANCE_IN_GRID_CELLS)
        self.assertEqual(result, (3, 3))

    def test_single_point_lat_long(self):
        """Test behaviour for a single grid cell on lat long grid."""
        cube = set_up_cube_lat_long()
        msg = "Invalid grid: projection_x/y coords required"
        with self.assertRaisesRegexp(ValueError, msg):
            convert_distance_into_number_of_grid_cells(
                cube, self.DISTANCE, self.MAX_DISTANCE_IN_GRID_CELLS)

    def test_single_point_range_negative(self):
        """Test behaviour with a non-zero point with negative range."""
        distance = -1.0 * self.DISTANCE
        msg = "distance of -6100.0m gives a negative cell extent"
        with self.assertRaisesRegexp(ValueError, msg):
            convert_distance_into_number_of_grid_cells(
                self.cube, distance, self.MAX_DISTANCE_IN_GRID_CELLS)

    def test_single_point_range_0(self):
        """Test behaviour with a non-zero point with zero range."""
        distance = 5
        msg = "Distance of 5m gives zero cell extent"
        with self.assertRaisesRegexp(ValueError, msg):
            convert_distance_into_number_of_grid_cells(
                self.cube, distance, self.MAX_DISTANCE_IN_GRID_CELLS)

    def test_single_point_range_lots(self):
        """Test behaviour with a non-zero point with unhandleable range."""
        distance = 40000.0
        max_distance_in_grid_cells = 10
        msg = "distance of 40000.0m exceeds maximum grid cell extent"
        with self.assertRaisesRegexp(ValueError, msg):
            convert_distance_into_number_of_grid_cells(
                self.cube, distance, max_distance_in_grid_cells)

    def test_single_point_range_greater_than_domain(self):
        """Test correct exception raised when the distance is larger than the
           corner-to-corner distance of the domain."""
        distance = 42500.0
        msg = "Distance of 42500.0m exceeds max domain distance of "
        with self.assertRaisesRegexp(ValueError, msg):
            convert_distance_into_number_of_grid_cells(
                self.cube, distance, self.MAX_DISTANCE_IN_GRID_CELLS)


if __name__ == '__main__':
    unittest.main()
