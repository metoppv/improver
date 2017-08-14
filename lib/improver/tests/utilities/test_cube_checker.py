# -*- coding: utf-8 -*-
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
"""Unit tests for the cube_checker utility."""


import unittest

import iris
from iris.tests import IrisTest

from improver.utilities.cube_checker import (
    check_cube_coordinates, check_for_x_and_y_axes,
    find_dimension_coordinate_mismatch)
from improver.tests.nbhood.nbhood.test_NeighbourhoodProcessing import (
    set_up_cube)


class Test_check_for_x_and_y_axes(IrisTest):

    """Test whether the cube has an x and y axis."""

    def setUp(self):
        """Set up a cube."""
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)

    def test_no_y_coordinate(self):
        """Test that the expected exception is raised, if there is no
        y coordinate."""
        for sliced_cube in self.cube.slices(
                ["projection_x_coordinate"]):
            break
        sliced_cube.remove_coord("projection_y_coordinate")
        msg = "The cube does not contain the expected"
        with self.assertRaisesRegexp(ValueError, msg):
            check_for_x_and_y_axes(sliced_cube)

    def test_no_x_coordinate(self):
        """Test that the expected exception is raised, if there is no
        x coordinate."""
        for sliced_cube in self.cube.slices(
                ["projection_y_coordinate"]):
            break
        sliced_cube.remove_coord("projection_x_coordinate")
        msg = "The cube does not contain the expected"
        with self.assertRaisesRegexp(ValueError, msg):
            check_for_x_and_y_axes(sliced_cube)


class Test_check_cube_coordinates(IrisTest):

    """Test check_cube_coordinates successfully promotes scalar coordinates to
    dimension coordinates in a new cube if they were dimension coordinates in
    the progenitor cube."""

    def test_basic(self):
        """Test returns iris.cube.Cube."""
        cube = set_up_cube()
        result = check_cube_coordinates(cube, cube)
        self.assertIsInstance(result, iris.cube.Cube)

    def test_coord_promotion(self):
        """Test that scalar coordinates in new_cube are promoted to dimension
        coordinates to match the parent cube."""
        cube = set_up_cube()
        new_cube = iris.util.squeeze(cube)
        result = check_cube_coordinates(cube, new_cube)
        self.assertEqual(result.dim_coords, cube.dim_coords)

    def test_coord_promotion_only_dim_coords_in_parent(self):
        """Test that only dimension coordinates in the parent cube are matched
        when promoting the scalar coordinates in new_cube. Here realization is
        made into a scalar coordinate on the parent, and so should remain a
        scalar in new_cube as well."""
        cube = set_up_cube()
        new_cube = iris.util.squeeze(cube)
        cube = cube[0]
        result = check_cube_coordinates(cube, new_cube)
        self.assertEqual(result.dim_coords, cube.dim_coords)

    def test_coord_promotion_and_reordering(self):
        """Test case in which a scalar coordinate are promoted but the order
        must be corrected to match the progenitor cube."""
        cube = set_up_cube()
        new_cube = iris.util.squeeze(cube)
        cube.transpose(new_order=[1, 0, 2, 3])
        result = check_cube_coordinates(cube, new_cube)
        self.assertEqual(result.dim_coords, cube.dim_coords)

    def test_coord_promotion_missing_scalar(self):
        """Test case in which a scalar coordinate has been lost from new_cube,
        meaning the cube undergoing checking ends up with different dimension
        coordinates to the progenitor cube. This raises an error."""
        cube = set_up_cube()
        new_cube = iris.util.squeeze(cube)
        new_cube.remove_coord('realization')
        msg = 'Returned cube dimension coordinates'
        with self.assertRaisesRegexp(iris.exceptions.CoordinateNotFoundError,
                                     msg):
            check_cube_coordinates(cube, new_cube)


class Test_find_dimension_coordinate_mismatch(IrisTest):

    """Test if two cubes have the dimension coordinates."""

    def test_no_mismatch(self):
        """Test if there is no mismatch between the dimension coordinates."""
        cube = set_up_cube()
        result = find_dimension_coordinate_mismatch(cube, cube)
        self.assertIsInstance(result, list)
        self.assertFalse(result)

    def test_mismatch_in_first_cube(self):
        """Test when finding a one-way mismatch, so that the second cube has
        a missing coordinate. This returns an empty list."""
        cube = set_up_cube()
        first_cube = cube.copy()
        second_cube = cube.copy()
        second_cube.remove_coord("time")
        result = find_dimension_coordinate_mismatch(
            first_cube, second_cube, two_way_mismatch=False)
        self.assertIsInstance(result, list)
        self.assertFalse(result)

    def test_mismatch_in_second_cube(self):
        """Test when finding a one-way mismatch, so that the first cube has
        a missing coordinate. This returns a list with the missing coordinate
        name.l"""
        cube = set_up_cube()
        first_cube = cube.copy()
        first_cube.remove_coord("time")
        second_cube = cube.copy()
        result = find_dimension_coordinate_mismatch(
            first_cube, second_cube, two_way_mismatch=False)
        self.assertIsInstance(result, list)
        self.assertListEqual(result, ["time"])

    def test_two_way_mismatch(self):
        """Test when finding a two-way mismatch, when the first and second
        cube contain different coordinates."""
        cube = set_up_cube()
        first_cube = cube.copy()
        first_cube.remove_coord("time")
        second_cube = cube.copy()
        second_cube.remove_coord("realization")
        result = find_dimension_coordinate_mismatch(first_cube, second_cube)
        self.assertIsInstance(result, list)
        self.assertListEqual(result, ["time", "realization"])


if __name__ == '__main__':
    unittest.main()
