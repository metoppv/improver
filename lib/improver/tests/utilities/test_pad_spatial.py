# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
import numpy as np
import iris
from iris.coords import DimCoord
from iris.tests import IrisTest

from improver.utilities.pad_spatial import (
    pad_coord, pad_cube_with_halo, remove_halo_from_cube,
    create_cube_with_new_data)
from improver.tests.nbhood.nbhood.test_NeighbourhoodProcessing import (
    set_up_cube)


class Test_pad_coord(IrisTest):

    """Test the padding of a coordinate."""

    def setUp(self):
        """Set up a cube."""
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        coord_points_x = np.arange(10., 60., 10.)
        x_bounds = np.array([coord_points_x - 5, coord_points_x + 5]).T
        coord_points_y = np.arange(5., 105., 20.)
        y_bounds = np.array([coord_points_y - 10, coord_points_y + 10]).T

        self.cube.coord("projection_x_coordinate").points = coord_points_x
        self.cube.coord("projection_x_coordinate").bounds = x_bounds
        self.cube.coord("projection_y_coordinate").points = coord_points_y
        self.cube.coord("projection_y_coordinate").bounds = y_bounds

        self.cube_y_reorder = self.cube.copy()
        coord_points_y_reorder = np.arange(85., -15., -20.)
        y_bounds_reorder = np.array([coord_points_y_reorder + 10,
                                     coord_points_y_reorder - 10]).T
        self.cube_y_reorder.coord("projection_y_coordinate").points = (
            coord_points_y_reorder)
        self.cube_y_reorder.coord("projection_y_coordinate").bounds = (
            y_bounds_reorder)

    def test_add(self):
        """Test the functionality to add padding to the chosen coordinate.
        Includes a test that the coordinate bounds array is modified to reflect
        the new values."""
        expected = np.array(
            [-10., 0., 10., 20., 30., 40., 50., 60., 70.])
        coord = self.cube.coord("projection_x_coordinate")
        expected_bounds = np.array([expected - 5, expected + 5]).T
        width = 1
        method = "add"
        new_coord = pad_coord(coord, width, method)
        self.assertIsInstance(new_coord, DimCoord)
        self.assertArrayAlmostEqual(new_coord.points, expected)
        self.assertArrayEqual(new_coord.bounds, expected_bounds)

    def test_add_y_reorder(self):
        """Test the functionality to add still works if y is negative."""
        expected = np.array(
            [125.0, 105.0, 85.0, 65.0, 45.0, 25.0, 5.0, -15.0, -35.0])
        y_coord = self.cube_y_reorder.coord("projection_y_coordinate")
        expected_bounds = np.array([expected + 10, expected - 10]).T
        width = 1
        method = "add"
        new_coord = pad_coord(y_coord, width, method)
        self.assertIsInstance(new_coord, DimCoord)
        self.assertArrayAlmostEqual(new_coord.points, expected)
        self.assertArrayEqual(new_coord.bounds, expected_bounds)

    def test_exception(self):
        """Test an exception is raised if the chosen coordinate is
        non-uniform."""
        coord_points = np.arange(10., 60., 10.)
        coord_points[0] = -200.
        self.cube.coord("projection_x_coordinate").points = coord_points
        coord = self.cube.coord("projection_x_coordinate")
        width = 1
        method = "add"
        msg = "Non-uniform increments between grid points"
        with self.assertRaisesRegex(ValueError, msg):
            pad_coord(coord, width, method)

    def test_remove(self):
        """Test the functionality to remove padding from the chosen
        coordinate. Includes a test that the coordinate bounds array is
        modified to reflect the new values."""
        expected = np.array([30.])
        expected_bounds = np.array([expected - 5, expected + 5]).T
        coord = self.cube.coord("projection_x_coordinate")
        width = 1
        method = "remove"
        new_coord = pad_coord(coord, width, method)
        self.assertIsInstance(new_coord, DimCoord)
        self.assertArrayAlmostEqual(new_coord.points, expected)
        self.assertArrayEqual(new_coord.bounds, expected_bounds)


class Test__create_cube_with_new_data(IrisTest):

    """Test creating a new cube using a template cube."""

    def setUp(self):
        """Set up a cube."""
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)

    def test_yx_order(self):
        """Test that a cube is created with the expected order for the y and x
        coordinates within the output cube, if the input cube has dimensions
        of projection_y_coordinate and projection_x_coordinate."""
        for sliced_cube in self.cube.slices(
                ["projection_y_coordinate", "projection_x_coordinate"]):
            break
        data = sliced_cube.data
        coord_x = sliced_cube.coord("projection_x_coordinate")
        coord_y = sliced_cube.coord("projection_y_coordinate")

        new_cube = create_cube_with_new_data(
            sliced_cube, data, coord_x, coord_y)
        self.assertIsInstance(new_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_cube.data, data)
        self.assertEqual(new_cube.coord_dims("projection_y_coordinate")[0], 0)
        self.assertEqual(new_cube.coord_dims("projection_x_coordinate")[0], 1)

    def test_xy_order(self):
        """Test that a cube is created with the expected order for the y and x
        coordinates within the output cube, if the input cube has dimensions
        of projection_x_coordinate and projection_y_coordinate."""
        for sliced_cube in self.cube.slices(
                ["projection_x_coordinate", "projection_y_coordinate"]):
            break
        data = sliced_cube.data
        coord_x = sliced_cube.coord("projection_x_coordinate")
        coord_y = sliced_cube.coord("projection_y_coordinate")

        new_cube = create_cube_with_new_data(
            sliced_cube, data, coord_x, coord_y)
        self.assertIsInstance(new_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_cube.data, data)
        self.assertEqual(new_cube.coord_dims("projection_y_coordinate")[0], 1)
        self.assertEqual(new_cube.coord_dims("projection_x_coordinate")[0], 0)

    def test_realization(self):
        """Test that a cube is created with the expected order for coordinates
        within the output cube, if the input cube has dimensions of
        realization, projection_y_coordinate and projection_x_coordinate."""
        for sliced_cube in self.cube.slices(["realization",
                                             "projection_y_coordinate",
                                             "projection_x_coordinate"]):
            break
        data = sliced_cube.data
        coord_x = sliced_cube.coord("projection_x_coordinate")
        coord_y = sliced_cube.coord("projection_y_coordinate")

        new_cube = create_cube_with_new_data(
            sliced_cube, data, coord_x, coord_y)
        self.assertIsInstance(new_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_cube.data, data)
        self.assertEqual(new_cube.coord_dims("realization")[0], 0)
        self.assertEqual(new_cube.coord_dims("projection_y_coordinate")[0], 1)
        self.assertEqual(new_cube.coord_dims("projection_x_coordinate")[0], 2)

    def test_forecast_period(self):
        """Test that a cube is created with the expected order for coordinates
        within the output cube, if the input cube has dimensions of
        projection_y_coordinate and projection_x_coordinate, and where the
        input cube also has a forecast_period coordinate."""
        for sliced_cube in self.cube.slices(["projection_y_coordinate",
                                             "projection_x_coordinate"]):
            break
        fp_coord = DimCoord(np.array([10]), standard_name="forecast_period")
        sliced_cube.add_aux_coord(fp_coord)
        data = sliced_cube.data
        coord_x = sliced_cube.coord("projection_x_coordinate")
        coord_y = sliced_cube.coord("projection_y_coordinate")

        new_cube = create_cube_with_new_data(
            sliced_cube, data, coord_x, coord_y)
        self.assertIsInstance(new_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_cube.data, data)
        self.assertEqual(new_cube.coord("forecast_period").points, 10)
        self.assertEqual(new_cube.coord_dims("projection_y_coordinate")[0], 0)
        self.assertEqual(new_cube.coord_dims("projection_x_coordinate")[0], 1)

    def test_no_y_dimension_coordinate(self):
        """Test that a cube is created with the expected y and x coordinates
        within the output cube, if the input cube only has a
        projection_y_coordinate dimension coordinate."""
        for sliced_cube in self.cube.slices(
                ["projection_x_coordinate"]):
            break
        data = sliced_cube.data
        coord_x = sliced_cube.coord("projection_x_coordinate")
        coord_y = sliced_cube.coord("projection_y_coordinate")
        new_cube = create_cube_with_new_data(
            sliced_cube, data, coord_x, coord_y)
        self.assertIsInstance(new_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_cube.data, data)
        self.assertEqual(new_cube.coord_dims("projection_x_coordinate")[0], 0)
        self.assertTrue(
            new_cube.coords("projection_y_coordinate", dim_coords=False))


class Test_pad_cube_with_halo(IrisTest):

    """Test for padding a cube with a halo."""

    def setUp(self):
        """Set up a cube."""
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)

    def test_basic(self):
        """Test that padding the data in a cube with a halo has worked as
        intended."""
        for sliced_cube in self.cube.slices(["projection_y_coordinate",
                                             "projection_x_coordinate"]):
            break
        expected = np.array(
            [[0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 1., 1., 1., 1., 1., 0., 0.],
             [0., 0., 1., 1., 1., 1., 1., 0., 0.],
             [0., 0., 1., 1., 0., 1., 1., 0., 0.],
             [0., 0., 1., 1., 1., 1., 1., 0., 0.],
             [0., 0., 1., 1., 1., 1., 1., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        width_x = width_y = 1
        padded_cube = pad_cube_with_halo(
            sliced_cube, width_x, width_y, masked_halo=True)
        self.assertIsInstance(padded_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(padded_cube.data, expected)

    def test_masked_halo_false(self):
        """Test the halo data is as expeceted when masked_halo is false"""

        for sliced_cube in self.cube.slices(["projection_y_coordinate",
                                             "projection_x_coordinate"]):
            break
        expected = np.array(
            [[1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 0., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1.]])
        width_x = width_y = 1
        padded_cube = pad_cube_with_halo(
            sliced_cube, width_x, width_y)
        self.assertIsInstance(padded_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(padded_cube.data, expected)

    def test_different_widths(self):
        """Test that padding the data in a cube with different widths has
        worked as intended."""
        for sliced_cube in self.cube.slices(["projection_y_coordinate",
                                             "projection_x_coordinate"]):
            break
        expected = np.array(
            [[0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 1., 1., 1., 1., 1., 0., 0.],
             [0., 0., 1., 1., 1., 1., 1., 0., 0.],
             [0., 0., 1., 1., 0., 1., 1., 0., 0.],
             [0., 0., 1., 1., 1., 1., 1., 0., 0.],
             [0., 0., 1., 1., 1., 1., 1., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        width_x = 1
        width_y = 2
        padded_cube = pad_cube_with_halo(
            sliced_cube, width_x, width_y, masked_halo=True)
        self.assertIsInstance(padded_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(padded_cube.data, expected)

    def test_zero_width(self):
        """Test that padding the data in a cube with a width of zero has
        worked as intended."""
        for sliced_cube in self.cube.slices(["projection_y_coordinate",
                                             "projection_x_coordinate"]):
            break
        expected = np.array(
            [[0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 0., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.]])
        width_x = 0
        width_y = 2
        padded_cube = pad_cube_with_halo(
            sliced_cube, width_x, width_y, masked_halo=True)
        self.assertIsInstance(padded_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(padded_cube.data, expected)


class Test_remove_halo_from_cube(IrisTest):

    """Test a halo is removed from the cube data."""

    def setUp(self):
        """Set up a cube."""
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)

    def test_basic(self):
        """Test that removing a halo of points from the data on a cube
        has worked as intended."""
        for sliced_cube in self.cube.slices(["projection_y_coordinate",
                                             "projection_x_coordinate"]):
            break
        expected = np.array([[0.]])
        width_x = width_y = 1
        padded_cube = remove_halo_from_cube(sliced_cube, width_x, width_y)
        self.assertIsInstance(padded_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(padded_cube.data, expected)

    def test_different_widths(self):
        """Test that removing a halo of points from the data on a cube
        has worked as intended for different x and y widths."""
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 5, 5),), num_time_points=1,
            num_grid_points=10)
        for sliced_cube in self.cube.slices(["projection_y_coordinate",
                                             "projection_x_coordinate"]):
            break
        expected = np.array(
            [[1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 0., 1., 1.]])
        width_x = 1
        width_y = 2
        padded_cube = remove_halo_from_cube(sliced_cube, width_x, width_y)
        self.assertIsInstance(padded_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(padded_cube.data, expected)

    def test_zero_width(self):
        """Test that removing a halo of points from the data on a cube
        has worked as intended, if a width of zero is specified."""
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 5, 5),), num_time_points=1,
            num_grid_points=10)
        for sliced_cube in self.cube.slices(["projection_y_coordinate",
                                             "projection_x_coordinate"]):
            break
        expected = np.array(
            [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 0., 1., 1., 1., 1.]])
        width_x = 0
        width_y = 2
        padded_cube = remove_halo_from_cube(sliced_cube, width_x, width_y)
        self.assertIsInstance(padded_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(padded_cube.data, expected)


if __name__ == '__main__':
    unittest.main()
