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
"""Unit tests for spatial padding utilities"""

import unittest

import iris
import numpy as np
from iris.coords import DimCoord
from iris.tests import IrisTest

from improver.utilities.pad_spatial import (
    _create_cube_with_padded_data, create_cube_with_halo, pad_coord,
    pad_cube_with_halo, remove_cube_halo, remove_halo_from_cube)

from ..set_up_test_cubes import set_up_variable_cube


class Test_pad_coord(IrisTest):

    """Test the padding of a coordinate."""

    def setUp(self):
        """Set up a cube."""
        self.cube = set_up_variable_cube(np.ones((5, 5), dtype=np.float32),
                                         spatial_grid="equalarea")
        coord_points_x = np.linspace(10., 50., 5)
        x_bounds = np.array([coord_points_x - 5, coord_points_x + 5]).T
        coord_points_y = np.linspace(5., 85., 5)
        y_bounds = np.array([coord_points_y - 10, coord_points_y + 10]).T

        self.cube.coord("projection_x_coordinate").points = coord_points_x
        self.cube.coord("projection_x_coordinate").bounds = x_bounds
        self.cube.coord("projection_y_coordinate").points = coord_points_y
        self.cube.coord("projection_y_coordinate").bounds = y_bounds

        self.cube_y_reorder = self.cube.copy()
        coord_points_y_reorder = np.flip(coord_points_y)
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
        expected = np.linspace(0., 60., 7)
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
        expected = np.linspace(105., -15., 7)
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
        expected = np.array([20., 30., 40.])
        expected_bounds = np.array([expected - 5, expected + 5]).T
        coord = self.cube.coord("projection_x_coordinate")
        width = 1
        method = "remove"
        new_coord = pad_coord(coord, width, method)
        self.assertIsInstance(new_coord, DimCoord)
        self.assertArrayAlmostEqual(new_coord.points, expected)
        self.assertArrayEqual(new_coord.bounds, expected_bounds)


class Test_create_cube_with_halo(IrisTest):
    """Tests for the create_cube_with_halo function"""

    def setUp(self):
        """Set up a realistic input cube with lots of metadata.  Input cube
        grid is 1000x1000 km with points spaced 100 km apart."""
        attrs = {'history': '2018-12-10Z: StaGE Decoupler',
                 'title': 'Temperature on UK 2 km Standard Grid',
                 'source': 'Met Office Unified Model'}

        self.cube = set_up_variable_cube(
            np.ones((11, 11), dtype=np.float32), spatial_grid='equalarea',
            standard_grid_metadata='uk_det', attributes=attrs)
        self.grid_spacing = np.diff(self.cube.coord(axis='x').points)[0]

    def test_basic(self):
        """Test function returns a cube with expected metadata"""
        halo_size_km = 162.
        result = create_cube_with_halo(self.cube, 1000.*halo_size_km)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), 'grid_with_halo')
        self.assertFalse(result.attributes)

    def test_values(self):
        """Test coordinate values with standard halo radius (rounds down to 1
        grid cell)"""
        halo_size_km = 162.

        x_min = self.cube.coord(axis='x').points[0] - self.grid_spacing
        x_max = self.cube.coord(axis='x').points[-1] + self.grid_spacing
        expected_x_points = np.arange(x_min, x_max+1, self.grid_spacing)
        y_min = self.cube.coord(axis='y').points[0] - self.grid_spacing
        y_max = self.cube.coord(axis='y').points[-1] + self.grid_spacing
        expected_y_points = np.arange(y_min, y_max+1, self.grid_spacing)

        result = create_cube_with_halo(self.cube, 1000.*halo_size_km)
        self.assertSequenceEqual(result.data.shape, (13, 13))
        self.assertArrayAlmostEqual(
            result.coord(axis='x').points, expected_x_points)
        self.assertArrayAlmostEqual(
            result.coord(axis='y').points, expected_y_points)

        # check explicitly that the original grid remains an exact subset of
        # the output cube (ie that padding hasn't shifted the existing grid)
        self.assertArrayAlmostEqual(result.coord(axis='x').points[1:-1],
                                    self.cube.coord(axis='x').points)
        self.assertArrayAlmostEqual(result.coord(axis='y').points[1:-1],
                                    self.cube.coord(axis='y').points)


class Test__create_cube_with_padded_data(IrisTest):

    """Test creating a new cube using a template cube."""

    def setUp(self):
        """Set up a cube."""
        self.cube = set_up_variable_cube(np.ones((1, 5, 5), dtype=np.float32),
                                         spatial_grid="equalarea")
        self.cube.data[0, 2, 2] = 0

    def test_yx_order(self):
        """Test that a cube is created with the expected order for the y and x
        coordinates within the output cube, if the input cube has dimensions
        of projection_y_coordinate and projection_x_coordinate."""
        sliced_cube = next(self.cube.slices(
            ["projection_y_coordinate", "projection_x_coordinate"]))
        data = sliced_cube.data
        coord_x = sliced_cube.coord("projection_x_coordinate")
        coord_y = sliced_cube.coord("projection_y_coordinate")

        new_cube = _create_cube_with_padded_data(
            sliced_cube, data, coord_x, coord_y)
        self.assertIsInstance(new_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_cube.data, data)
        self.assertEqual(new_cube.coord_dims("projection_y_coordinate")[0], 0)
        self.assertEqual(new_cube.coord_dims("projection_x_coordinate")[0], 1)

    def test_xy_order(self):
        """Test that a cube is created with the expected order for the y and x
        coordinates within the output cube, if the input cube has dimensions
        of projection_x_coordinate and projection_y_coordinate."""
        sliced_cube = next(self.cube.slices(
            ["projection_x_coordinate", "projection_y_coordinate"]))
        data = sliced_cube.data
        coord_x = sliced_cube.coord("projection_x_coordinate")
        coord_y = sliced_cube.coord("projection_y_coordinate")

        new_cube = _create_cube_with_padded_data(
            sliced_cube, data, coord_x, coord_y)
        self.assertIsInstance(new_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_cube.data, data)
        self.assertEqual(new_cube.coord_dims("projection_y_coordinate")[0], 1)
        self.assertEqual(new_cube.coord_dims("projection_x_coordinate")[0], 0)

    def test_realization(self):
        """Test that a cube is created with the expected order for coordinates
        within the output cube, if the input cube has dimensions of
        realization, projection_y_coordinate and projection_x_coordinate."""
        sliced_cube = next(self.cube.slices(["realization",
                                             "projection_y_coordinate",
                                             "projection_x_coordinate"]))
        data = sliced_cube.data
        coord_x = sliced_cube.coord("projection_x_coordinate")
        coord_y = sliced_cube.coord("projection_y_coordinate")

        new_cube = _create_cube_with_padded_data(
            sliced_cube, data, coord_x, coord_y)
        self.assertIsInstance(new_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(new_cube.data, data)
        self.assertEqual(new_cube.coord_dims("realization")[0], 0)
        self.assertEqual(new_cube.coord_dims("projection_y_coordinate")[0], 1)
        self.assertEqual(new_cube.coord_dims("projection_x_coordinate")[0], 2)

    def test_no_y_dimension_coordinate(self):
        """Test that a cube is created with the expected y and x coordinates
        within the output cube, if the input cube only has a
        projection_y_coordinate dimension coordinate."""
        sliced_cube = next(self.cube.slices(["projection_x_coordinate"]))
        data = sliced_cube.data
        coord_x = sliced_cube.coord("projection_x_coordinate")
        coord_y = sliced_cube.coord("projection_y_coordinate")
        new_cube = _create_cube_with_padded_data(
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
        self.cube = set_up_variable_cube(np.ones((5, 5), dtype=np.float32),
                                         spatial_grid="equalarea")
        self.cube.data[2, 2] = 0

    def test_basic(self):
        """Test that padding the data in a cube with a halo has worked as
        intended."""
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
        width_x = width_y = 2
        padded_cube = pad_cube_with_halo(
            self.cube, width_x, width_y, halo_mean_data=False)
        self.assertIsInstance(padded_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(padded_cube.data, expected)

    def test_different_widths(self):
        """Test that padding the data in a cube with different widths has
        worked as intended."""
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
        width_x = 2
        width_y = 4
        padded_cube = pad_cube_with_halo(
            self.cube, width_x, width_y, halo_mean_data=False)
        self.assertIsInstance(padded_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(padded_cube.data, expected)

    def test_zero_width(self):
        """Test that padding the data in a cube with a width of zero has
        worked as intended."""
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
        width_y = 4
        padded_cube = pad_cube_with_halo(
            self.cube, width_x, width_y, halo_mean_data=False)
        self.assertIsInstance(padded_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(padded_cube.data, expected)

    def test_halo_using_mean_smoothing(self):
        """Test values in halo are correctly smoothed when halo_mean_data=True.
        This impacts recursive filter outputs."""
        data = np.array([[0., 0., 0.1, 0., 0.],
                         [0., 0., 0.25, 0., 0.],
                         [0.1, 0.25, 0.5, 0.25, 0.1],
                         [0., 0., 0.25, 0., 0.],
                         [0., 0., 0.1, 0., 0.]], dtype=np.float32)
        self.cube.data = data
        expected_data = np.array([
            [0., 0., 0., 0., 0.1, 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.1, 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.1, 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.25, 0., 0., 0., 0.],
            [0.1, 0.1, 0.1, 0.25, 0.5, 0.25, 0.1, 0.1, 0.1],
            [0., 0., 0., 0., 0.25, 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.1, 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.1, 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.1, 0., 0., 0., 0.]], dtype=np.float32)

        padded_cube = pad_cube_with_halo(self.cube, 2, 2)
        self.assertArrayAlmostEqual(padded_cube.data, expected_data)


class Test_remove_cube_halo(IrisTest):
    """Tests for the  remove_cube_halo function"""

    def setUp(self):
        """Set up a realistic input cube with lots of metadata.  Input cube
        grid is 1000x1000 km with points spaced 100 km apart."""
        self.attrs = {'history': '2018-12-10Z: StaGE Decoupler',
                      'title': 'Temperature on UK 2 km Standard Grid',
                      'source': 'Met Office Unified Model'}
        self.cube = set_up_variable_cube(
            np.ones((3, 11, 11), dtype=np.float32), spatial_grid='equalarea',
            standard_grid_metadata='uk_det', attributes=self.attrs)

        self.cube_1d = set_up_variable_cube(
            np.ones((1, 11, 11), dtype=np.float32), spatial_grid='equalarea',
            standard_grid_metadata='uk_det', attributes=self.attrs)
        self.grid_spacing = np.diff(self.cube.coord(axis='x').points)[0]

    def test_basic(self):
        """Test function returns a cube with expected attributes and shape"""
        halo_size_km = 162.
        result = remove_cube_halo(self.cube, 1000.*halo_size_km)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.attributes['history'],
                         self.attrs['history'])
        self.assertEqual(result.attributes['title'], self.attrs['title'])
        self.assertEqual(result.attributes['source'], self.attrs['source'])
        self.assertSequenceEqual(result.data.shape, (3, 9, 9))

    def test_values(self):
        """Test function returns a cube with expected shape and data"""
        halo_size_km = 162.
        self.cube_1d.data[0, 2, :] = np.arange(0, 11)
        self.cube_1d.data[0, :, 2] = np.arange(0, 11)
        result = remove_cube_halo(self.cube_1d, 1000.*halo_size_km)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertSequenceEqual(result.data.shape, (1, 9, 9))
        self.assertArrayEqual(result.data[0, 1, :], np.arange(1, 10))
        self.assertArrayEqual(result.data[0, :, 1], np.arange(1, 10))


class Test_remove_halo_from_cube(IrisTest):

    """Test a halo is removed from the cube data."""

    def setUp(self):
        """Set up a cube."""
        self.cube = set_up_variable_cube(np.ones((5, 5), dtype=np.float32),
                                         spatial_grid="equalarea")
        self.cube.data[2, 2] = 0
        self.large_cube = set_up_variable_cube(
            np.ones((10, 10), dtype=np.float32), spatial_grid="equalarea")
        # set equally-spaced coordinate points
        self.large_cube.coord(axis='y').points = np.linspace(
            0, 900000, 10, dtype=np.float32)
        self.large_cube.coord(axis='x').points = np.linspace(
            -300000, 600000, 10, dtype=np.float32)
        self.large_cube.data[5, 5] = 0

    def test_basic(self):
        """Test that removing a halo of points from the data on a cube
        has worked as intended."""
        expected = np.array([[0.]])
        width_x = width_y = 2
        padded_cube = remove_halo_from_cube(self.cube, width_x, width_y)
        self.assertIsInstance(padded_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(padded_cube.data, expected)

    def test_different_widths(self):
        """Test that removing a halo of points from the data on a cube
        has worked as intended for different x and y widths."""
        expected = np.array(
            [[1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 0., 1., 1.]])
        width_x = 2
        width_y = 4
        padded_cube = remove_halo_from_cube(self.large_cube, width_x, width_y)
        self.assertIsInstance(padded_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(padded_cube.data, expected)

    def test_zero_width(self):
        """Test that removing a halo of points from the data on a cube
        has worked as intended, if a width of zero is specified."""
        expected = np.array(
            [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 0., 1., 1., 1., 1.]])
        width_x = 0
        width_y = 4
        padded_cube = remove_halo_from_cube(self.large_cube, width_x, width_y)
        self.assertIsInstance(padded_cube, iris.cube.Cube)
        self.assertArrayAlmostEqual(padded_cube.data, expected)


if __name__ == '__main__':
    unittest.main()
