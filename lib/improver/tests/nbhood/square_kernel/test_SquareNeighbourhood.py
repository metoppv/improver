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
"""Unit tests for the nbhood.square_kernel.SquareNeighbourhood plugin."""


import unittest

import copy

import iris
from iris.coords import CellMethod, DimCoord
from iris.cube import Cube, CubeList
from iris.tests import IrisTest

import numpy as np

from improver.nbhood.square_kernel import SquareNeighbourhood
from improver.tests.nbhood.nbhood.test_NeighbourhoodProcessing import (
    set_up_cube)


class Test__init__(IrisTest):

    """Test the init method."""

    def test_sum_or_fraction(self):
        """Test that a ValueError is raised if an invalid option is passed
        in for sum_or_fraction."""
        sum_or_fraction = "nonsense"
        msg = "option is invalid"
        with self.assertRaisesRegexp(ValueError, msg):
            SquareNeighbourhood(sum_or_fraction=sum_or_fraction)


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(SquareNeighbourhood())
        msg = ('<SquareNeighbourhood: weighted_mode: {}, '
               'sum_or_fraction: {}, re_mask: {}>'.format(
                   True, "fraction", True))
        self.assertEqual(result, msg)


class Test_cumulate_array(IrisTest):

    """Test for cumulating an array in the y and x dimension."""

    def test_basic(self):
        """
        Test that the y-dimension and x-dimension accumulation produces the
        intended result. A 2d cube is passed in.
        """
        data = np.array([[1., 2., 3., 4., 5.],
                         [2., 4., 6., 8., 10.],
                         [3., 6., 8., 11., 14.],
                         [4., 8., 11., 15., 19.],
                         [5., 10., 14., 19., 24.]])
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        cube = iris.util.squeeze(cube)
        mask = cube.copy()
        cube_and_mask = iris.cube.CubeList([cube, mask])
        nan_mask = np.zeros(cube.data.shape, dtype=bool)
        (result_cube_and_mask,
         result_nan_mask) = SquareNeighbourhood().cumulate_array(cube_and_mask)
        self.assertIsInstance(result_cube_and_mask[0], Cube)
        self.assertIsInstance(result_cube_and_mask[1], Cube)
        self.assertArrayAlmostEqual(result_cube_and_mask[0].data, data)
        self.assertArrayAlmostEqual(result_cube_and_mask[1].data, data)
        self.assertArrayAlmostEqual(result_nan_mask, nan_mask)

    def test_nan_array(self):
        """Test correct nanmask is returned when array containing nan data
           is input."""
        data = np.array([[0., 1., 2., 3., 4.],
                         [1., 3., 5., 7., 9.],
                         [2., 5., 7., 10., 13.],
                         [3., 7., 10., 14., 18.],
                         [4., 9., 13., 18., 23.]])
        nanmask = np.zeros([5, 5]).astype(bool)
        nanmask[0, 0] = True
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        cube = iris.util.squeeze(cube)
        mask = cube.copy()
        cube.data[0, 0] = np.nan
        cube_and_mask = iris.cube.CubeList([cube, mask])
        (result,
         result_nan_mask) = SquareNeighbourhood().cumulate_array(cube_and_mask)
        self.assertArrayAlmostEqual(result[0].data, data)
        self.assertArrayAlmostEqual(result[1].data, data)
        self.assertArrayAlmostEqual(result_nan_mask, nanmask)


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
        new_coord = SquareNeighbourhood.pad_coord(coord, width, method)
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
        new_coord = SquareNeighbourhood.pad_coord(y_coord, width, method)
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
        with self.assertRaisesRegexp(ValueError, msg):
            SquareNeighbourhood.pad_coord(coord, width, method)

    def test_remove(self):
        """Test the functionality to remove padding from the chosen
        coordinate. Includes a test that the coordinate bounds array is
        modified to reflect the new values."""
        expected = np.array([30.])
        expected_bounds = np.array([expected - 5, expected + 5]).T
        coord = self.cube.coord("projection_x_coordinate")
        width = 1
        method = "remove"
        new_coord = SquareNeighbourhood.pad_coord(coord, width, method)
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

        new_cube = SquareNeighbourhood()._create_cube_with_new_data(
            sliced_cube, data, coord_x, coord_y)
        self.assertIsInstance(new_cube, Cube)
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

        new_cube = SquareNeighbourhood()._create_cube_with_new_data(
            sliced_cube, data, coord_x, coord_y)
        self.assertIsInstance(new_cube, Cube)
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

        new_cube = SquareNeighbourhood()._create_cube_with_new_data(
            sliced_cube, data, coord_x, coord_y)
        self.assertIsInstance(new_cube, Cube)
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

        new_cube = SquareNeighbourhood()._create_cube_with_new_data(
            sliced_cube, data, coord_x, coord_y)
        self.assertIsInstance(new_cube, Cube)
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
        new_cube = SquareNeighbourhood()._create_cube_with_new_data(
            sliced_cube, data, coord_x, coord_y)
        self.assertIsInstance(new_cube, Cube)
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
        padded_cube = SquareNeighbourhood().pad_cube_with_halo(
            sliced_cube, width_x, width_y, masked_halo=True)
        self.assertIsInstance(padded_cube, Cube)
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
        padded_cube = SquareNeighbourhood().pad_cube_with_halo(
            sliced_cube, width_x, width_y)
        self.assertIsInstance(padded_cube, Cube)
        self.assertArrayAlmostEqual(padded_cube.data, expected)

    def test_mask_halo(self):
        """Test that padding the data in a cube with a halo when mask=True."""
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
        padded_cube = SquareNeighbourhood().pad_cube_with_halo(
            sliced_cube, width_x, width_y, masked_halo=True)
        self.assertIsInstance(padded_cube, Cube)
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
        padded_cube = SquareNeighbourhood().pad_cube_with_halo(
            sliced_cube, width_x, width_y, masked_halo=True)
        self.assertIsInstance(padded_cube, Cube)
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
        padded_cube = SquareNeighbourhood().pad_cube_with_halo(
            sliced_cube, width_x, width_y, masked_halo=True)
        self.assertIsInstance(padded_cube, Cube)
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
        padded_cube = SquareNeighbourhood().remove_halo_from_cube(
            sliced_cube, width_x, width_y)
        self.assertIsInstance(padded_cube, Cube)
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
        padded_cube = SquareNeighbourhood().remove_halo_from_cube(
            sliced_cube, width_x, width_y)
        self.assertIsInstance(padded_cube, Cube)
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
        padded_cube = SquareNeighbourhood().remove_halo_from_cube(
            sliced_cube, width_x, width_y)
        self.assertIsInstance(padded_cube, Cube)
        self.assertArrayAlmostEqual(padded_cube.data, expected)


class Test_mean_over_neighbourhood(IrisTest):

    """Test for calculating mean value in neighbourhood."""

    def setUp(self):
        """Set up cube and expected results for tests."""

        # This array is the output from cumulate_array when a 3x3 array of 1's
        # with a 0 at the centre point (1,1) is passed in.
        # A Halo has been added to the data.
        self.data = np.array(
            [[0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 1., 2., 3., 3., 3.],
             [0., 0., 2., 3., 5., 5., 5.],
             [0., 0., 3., 5., 8., 8., 8.],
             [0., 0., 3., 5., 8., 8., 8.],
             [0., 0., 3., 5., 8., 8., 8.]])
        self.cube = Cube(self.data, long_name='test')
        self.x_coord = DimCoord([0, 1, 2, 3, 4, 5, 6],
                                standard_name='longitude')
        self.y_coord = DimCoord([0, 1, 2, 3, 4, 5, 6],
                                standard_name='latitude')
        self.cube.add_dim_coord(self.x_coord, 1)
        self.cube.add_dim_coord(self.y_coord, 0)
        self.mask = self.cube.copy()
        self.mask.rename('mask_data')
        # This array is the output from cumulate_array when a 3x3 array of 1's
        # A Halo of missing points has been added to the data.
        self.mask.data = np.array(
            [[0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 1., 2., 3., 3., 3.],
             [0., 0., 2., 4., 6., 6., 6.],
             [0., 0., 3., 6., 9., 9., 9.],
             [0., 0., 3., 6., 9., 9., 9.],
             [0., 0., 3., 6., 9., 9., 9.]])
        self.nan_mask = np.zeros(self.cube.data.shape, dtype=int)
        self.width = 1
        self.expected = np.array(
            [[0.88888889, 0.83333333, 0.83333333, 0.88888889,
              0.83333333, 1.0, 0.88888889],
             [0.88888889, 0.85714286, 0.75000000, 0.83333333,
              0.75000000, 1.0, 0.83333333],
             [0.83333333, 0.87500000, 0.75000000, 0.83333333,
              0.75000000, 1.0, 0.83333333],
             [0.83333333, 0.66666667, 0.83333333, 0.88888889,
              0.83333333, 1.0, 0.88888889],
             [0.88888889, 0.85714286, 0.75000000, 0.83333333,
              0.75000000, 1.0, 0.83333333],
             [0.83333333, 0.80000000, 1.00000000, 1.00000000,
              1.00000000, 1.0, 1.00000000],
             [1.00000000, 1.00000000, 0.83333333, 0.88888889,
              0.83333333, 1.0, 0.88888889]])

    def test_basic_fraction(self):
        """Test cube with correct data is produced when mean over
           neighbourhood is calculated where the sum_or_fraction option is
           set to "fraction"."""
        cube_and_mask = iris.cube.CubeList([self.cube, self.mask])
        result = SquareNeighbourhood().mean_over_neighbourhood(
            cube_and_mask, self.width, self.width, self.nan_mask)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, self.expected)

    def test_basic_sum(self):
        """Test cube with correct data is produced when mean over
           neighbourhood is calculated where the sum_or_fraction option is
           set to "sum"."""
        expected = np.array(
            [[8., 5., -5., -8., -5., -3., 8.],
             [8., 6., -3., -5., -3., -2., 5.],
             [5., 7., 3., 5., 3., 2., -5.],
             [-5., -2., 5., 8., 5., 3., -8.],
             [-8., -6., 3., 5., 3., 2., -5.],
             [-5., -4., 2., 3., 2., 1., -3.],
             [-3., -6., -5., -8., -5., -3., 8.]])
        cube_and_mask = iris.cube.CubeList([self.cube, self.mask])
        result = SquareNeighbourhood(
            sum_or_fraction="sum").mean_over_neighbourhood(
                cube_and_mask, self.width, self.width, self.nan_mask)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_nan_mask(self):
        """Test the correct result is returned when a nan must be substituted
           into the final array."""
        cube_and_mask = iris.cube.CubeList([self.cube, self.mask])
        nan_mask = self.nan_mask
        nan_mask[2, 2] = True
        expected_data = self.expected
        expected_data[2, 2] = np.nan
        result = SquareNeighbourhood().mean_over_neighbourhood(
            cube_and_mask, self.width, self.width, nan_mask)
        self.assertArrayAlmostEqual(result.data, expected_data)


class Test__set_up_cubes_to_be_neighbourhooded(IrisTest):

    """Test the set up of cubes prior to neighbourhooding."""

    def setUp(self):
        """Set up a cube."""
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        self.cube = iris.util.squeeze(self.cube)

    def test_without_masked_data(self):
        """Test setting up cubes to be neighbourhooded when the input cube
        does not contain masked arrays."""
        expected_mask = np.ones((5, 5))
        cubes = (
            SquareNeighbourhood._set_up_cubes_to_be_neighbourhooded(self.cube))
        self.assertIsInstance(cubes, CubeList)
        self.assertEqual(len(cubes), 2)
        self.assertEqual(cubes[0], self.cube)
        self.assertArrayEqual(cubes[1].data, expected_mask)

    def test_with_masked_data(self):
        """Test setting up cubes to be neighbourhooded when the input cube
        contains masked arrays."""
        cube = self.cube
        data = cube.data
        cube.data[1, 3] = 0.5
        cube.data[3, 3] = 0.5
        cube.data = np.ma.masked_equal(data, 0.5)
        mask = np.logical_not(cube.data.mask.astype(int))
        data = cube.data.data * mask
        cubes = (
            SquareNeighbourhood._set_up_cubes_to_be_neighbourhooded(
                cube.copy()))
        self.assertIsInstance(cubes, CubeList)
        self.assertEqual(len(cubes), 2)
        self.assertArrayAlmostEqual(cubes[0].data, data)
        self.assertArrayAlmostEqual(cubes[1].data, mask)

    def test_with_separate_mask_cube(self):
        """Test setting up cubes to be neighbourhooded for an input cube and
        an additional mask cube."""
        self.cube.data[1, 3] = 0.5
        self.cube.data[3, 3] = 0.5
        mask_cube = self.cube.copy()
        mask_cube.data[mask_cube.data == 0.5] = 0.0
        mask_cube.data = mask_cube.data.astype(int)
        expected_data = self.cube.data * mask_cube.data
        cubes = (
            SquareNeighbourhood._set_up_cubes_to_be_neighbourhooded(
                self.cube.copy(), mask_cube=mask_cube))
        self.assertIsInstance(cubes, CubeList)
        self.assertEqual(len(cubes), 2)
        self.assertArrayAlmostEqual(cubes[0].data, expected_data)


class Test__pad_and_calculate_neighbourhood(IrisTest):

    """Test the padding and calculation of neighbourhood processing."""

    def setUp(self):
        """Set up a cube."""
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 1, 1),), num_time_points=1,
            num_grid_points=3)
        for sliced_cube in self.cube.slices(["projection_y_coordinate",
                                             "projection_x_coordinate"]):
            break
        sliced_cube.remove_coord("realization")
        sliced_cube.remove_coord("time")
        self.cube = sliced_cube
        self.mask = self.cube.copy()
        self.mask.data[::] = 1.0
        self.mask.rename('masked_data')

    def test_basic(self):
        """Test setting up cubes to be padded and then passed into
        neighbourhood processing."""
        expected_data = np.array(
            [[0.85714286, 0.75, 0.83333333, 0.85714286, 0.75, 1., 0.85714286],
             [0.85714286, 0.8, 0.75, 0.75, 0.5, np.nan, 0.75],
             [0.75, 0.83333333, 0.75, 0.8, 0.66666667, 1., 0.8],
             [0.8, 0.5, 0.83333333, 0.85714286, 0.75, 1., 0.85714286],
             [0.85714286, 0.8, 0.75, 0.75, 0.5, np.nan, 0.75],
             [0.75, 0.66666667, 1., 1., 1., np.nan, 1.],
             [1., 1., 0.83333333, 0.85714286, 0.75, 1., 0.85714286]])

        grid_cells_x = grid_cells_y = 1
        cube = self.cube
        mask_cube = cube.copy()
        mask_cube.data[::] = 1.0
        mask_cube.data[1, 2] = 0.0
        mask_cube.data[2, 2] = 0.0
        # _set_up_cubes_to_be_neighbourhooded would set masked points to 0.0
        cube.data[1, 2] = 0.0
        cube.data[2, 2] = 0.0
        mask_cube.rename('mask_data')
        cubes = CubeList([cube, mask_cube])
        nbcube = (
            SquareNeighbourhood()._pad_and_calculate_neighbourhood(
                cubes, grid_cells_x, grid_cells_y))
        self.assertIsInstance(nbcube, Cube)
        self.assertArrayAlmostEqual(nbcube.data, expected_data)


class Test__remove_padding_and_mask(IrisTest):

    """Test the removal of padding and dealing with masked data."""

    def setUp(self):
        """Set up a cube."""
        self.padded_cube = set_up_cube(
            zero_point_indices=((0, 0, 3, 3),), num_time_points=1,
            num_grid_points=7)
        self.padded_cube = iris.util.squeeze(self.padded_cube)
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 1, 1),), num_time_points=1,
            num_grid_points=3)
        self.cube = iris.util.squeeze(self.cube)
        self.mask_cube = self.cube.copy()
        masked_array = np.ones(self.mask_cube.data.shape)
        masked_array[1, 1] = 0
        masked_array[0, 1] = 0
        self.mask_cube.rename('mask_data')
        self.mask_cube.data = masked_array.astype(bool)

    def test_without_masked_data(self):
        """Test that removing a halo of points from the data on a cube
        has worked as intended when the input data is not masked."""
        expected = np.array(
            [[1., 1., 1.],
             [1., 0., 1.],
             [1., 1., 1.]])
        grid_cells_x = grid_cells_y = 1
        nbcube = (
            SquareNeighbourhood()._remove_padding_and_mask(
                self.padded_cube, self.cube, None,
                grid_cells_x, grid_cells_y))
        self.assertIsInstance(nbcube, Cube)
        self.assertArrayAlmostEqual(nbcube.data, expected)

    def test_with_masked_data(self):
        """Test that removing a halo of points from the data on a cube
        has worked as intended when the input data has an associated mask."""
        expected = np.array(
            [[1., 1., 1.],
             [1., 0., 1.],
             [1., 1., 1.]])
        expected_mask = np.array(
            [[False, True, False],
             [False, True, False],
             [False, False, False]])
        grid_cells_x = grid_cells_y = 1
        nbcube = (
            SquareNeighbourhood()._remove_padding_and_mask(
                self.padded_cube, self.cube, self.mask_cube,
                grid_cells_x, grid_cells_y))
        print nbcube.data
        self.assertIsInstance(nbcube, Cube)
        self.assertArrayAlmostEqual(nbcube.data.data, expected)
        self.assertArrayAlmostEqual(nbcube.data.mask, expected_mask)

    def test_with_masked_data_and_no_remasking(self):
        """Test that removing halo works correctly with remask=False"""
        expected = np.array(
            [[1., 1., 1.],
             [1., 0., 1.],
             [1., 1., 1.]])
        grid_cells_x = grid_cells_y = 1
        nbcube = (
            SquareNeighbourhood(re_mask=False)._remove_padding_and_mask(
                self.padded_cube, self.cube, self.mask_cube,
                grid_cells_x, grid_cells_y))
        self.assertIsInstance(nbcube, Cube)
        self.assertArrayAlmostEqual(nbcube.data, expected)


class Test_run(IrisTest):

    """Test the run method on the SquareNeighbourhood class."""

    RADIUS = 2500

    def test_basic_re_mask_true(self):
        """Test that a cube with correct data is produced by the run method
        when re-masking is applied."""
        data = np.array(
            [[[[1., 1., 1., 1., 1.],
               [1., 0.88888889, 0.88888889, 0.88888889, 1.],
               [1., 0.88888889, 0.88888889, 0.88888889, 1.],
               [1., 0.88888889, 0.88888889, 0.88888889, 1.],
               [1., 1., 1., 1., 1.]]]])
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        result = SquareNeighbourhood().run(cube, self.RADIUS)
        self.assertIsInstance(cube, Cube)
        self.assertArrayAlmostEqual(result.data, data)

    def test_negative_strides_re_mask_true(self):
        """Test that a cube still works if there are negative-strides."""
        data = np.array(
            [[[[1., 1., 1., 1., 1.],
               [1., 0.88888889, 0.88888889, 0.88888889, 1.],
               [1., 0.88888889, 0.88888889, 0.88888889, 1.],
               [1., 0.88888889, 0.88888889, 0.88888889, 1.],
               [1., 1., 1., 1., 1.]]]])
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        coord_points_x = np.arange(-42000, -52000., -2000)
        coord_points_y = np.arange(8000., -2000, -2000)

        cube.coord("projection_x_coordinate").points = coord_points_x
        cube.coord("projection_y_coordinate").points = coord_points_y

        result = SquareNeighbourhood().run(cube, self.RADIUS)
        self.assertIsInstance(cube, Cube)
        self.assertArrayAlmostEqual(result.data, data)

    def test_basic_re_mask_false(self):
        """Test that a cube with correct data is produced by the run method."""
        data = np.array(
            [[[[1., 1., 1., 1., 1.],
               [1., 0.88888889, 0.88888889, 0.88888889, 1.],
               [1., 0.88888889, 0.88888889, 0.88888889, 1.],
               [1., 0.88888889, 0.88888889, 0.88888889, 1.],
               [1., 1., 1., 1., 1.]]]])
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        result = SquareNeighbourhood(re_mask=False).run(cube, self.RADIUS)
        self.assertIsInstance(cube, Cube)
        self.assertArrayAlmostEqual(result.data, data)

    def test_masked_array_re_mask_true(self):
        """Test that the run method produces a cube with correct data when a
        cube containing masked data is passed in and re-masking is applied."""
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        cube.data = np.array([[[[1, 1, 0, 1, 1],
                                [1, 1, 1, 0, 0],
                                [1, 0, 1, 0, 0],
                                [0, 0, 1, 1, 0],
                                [0, 1, 1, 0, 1]]]])
        mask = np.array([[[[0, 0, 1, 1, 0],
                           [0, 1, 1, 1, 0],
                           [0, 0, 1, 1, 1],
                           [0, 0, 1, 1, 0],
                           [0, 0, 1, 1, 0]]]])
        expected_array = np.array(
            [[[[1.0000, 0.666667, 0.600000, 0.500000, 0.50],
               [1.0000, 0.750000, 0.571429, 0.428571, 0.25],
               [1.0000, 1.000000, 0.714286, 0.571429, 0.25],
               [np.nan, 1.000000, 0.666667, 0.571429, 0.25],
               [np.nan, 1.000000, 0.750000, 0.750000, 0.50]]]])
        cube.data = np.ma.masked_where(mask == 0, cube.data)
        result = SquareNeighbourhood().run(cube, self.RADIUS)
        self.assertArrayAlmostEqual(result.data, expected_array)

    def test_masked_array_re_mask_false(self):
        """Test that the run method produces a cube with correct data when a
           cube containing masked data is passed in."""
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        cube.data = np.array([[[[1, 1, 0, 1, 1],
                                [1, 1, 1, 0, 0],
                                [1, 0, 1, 0, 0],
                                [0, 0, 1, 1, 0],
                                [0, 1, 1, 0, 1]]]])
        mask = np.array([[[[0, 0, 1, 1, 0],
                           [0, 1, 1, 1, 0],
                           [0, 0, 1, 1, 1],
                           [0, 0, 1, 1, 0],
                           [0, 0, 1, 1, 0]]]])
        expected_array = np.array(
            [[[[1.0000, 0.666667, 0.600000, 0.500000, 0.50],
               [1.0000, 0.750000, 0.571429, 0.428571, 0.25],
               [1.0000, 1.000000, 0.714286, 0.571429, 0.25],
               [np.nan, 1.000000, 0.666667, 0.571429, 0.25],
               [np.nan, 1.000000, 0.750000, 0.750000, 0.50]]]])
        cube.data = np.ma.masked_where(mask == 0, cube.data)
        result = SquareNeighbourhood(re_mask=False).run(cube, self.RADIUS)
        self.assertArrayAlmostEqual(result.data, expected_array)

    def test_nan_array_re_mask_true(self):
        """Test that an array containing nans is handled correctly when
        re-masking is applied."""
        data = np.array(
            [[[[np.nan, 1., 1., 1., 1.],
               [1., 0.8750000, 0.88888889, 0.88888889, 1.],
               [1., 0.88888889, 0.88888889, 0.88888889, 1.],
               [1., 0.88888889, 0.88888889, 0.88888889, 1.],
               [1., 1., 1., 1., 1.]]]])
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        cube.data[0, 0, 0, 0] = np.nan
        result = SquareNeighbourhood().run(cube, self.RADIUS)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, data)

    def test_nan_array_re_mask_false(self):
        """Test that an array containing nans is handled correctly."""
        data = np.array(
            [[[[np.nan, 1., 1., 1., 1.],
               [1., 0.8750000, 0.88888889, 0.88888889, 1.],
               [1., 0.88888889, 0.88888889, 0.88888889, 1.],
               [1., 0.88888889, 0.88888889, 0.88888889, 1.],
               [1., 1., 1., 1., 1.]]]])
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        cube.data[0, 0, 0, 0] = np.nan
        result = SquareNeighbourhood(re_mask=False).run(cube, self.RADIUS)
        self.assertIsInstance(cube, Cube)
        self.assertArrayAlmostEqual(result.data, data)

    def test_masked_array_with_nans_re_mask_true(self):
        """Test that the run method produces a cube with correct data when a
        cube containing masked nans is passed in and when re-masking is
        applied."""
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        cube.data = np.array([[[[np.nan, 1, 0, 1, 1],
                                [1, 1, 1, 0, 0],
                                [1, 0, 1, 0, 0],
                                [0, 0, 1, 1, 0],
                                [0, 1, 1, 0, 1]]]])
        mask = np.array([[[[0, 0, 1, 1, 0],
                           [0, 1, 1, 1, 0],
                           [0, 0, 1, 1, 1],
                           [0, 0, 1, 1, 0],
                           [0, 0, 1, 1, 0]]]])
        expected_array = np.array(
            [[[[np.nan, 0.666667, 0.600000, 0.500000, 0.50],
               [1.0000, 0.750000, 0.571429, 0.428571, 0.25],
               [1.0000, 1.000000, 0.714286, 0.571429, 0.25],
               [np.nan, 1.000000, 0.666667, 0.571429, 0.25],
               [np.nan, 1.000000, 0.750000, 0.750000, 0.50]]]])
        cube.data = np.ma.masked_where(mask == 0, cube.data)
        result = SquareNeighbourhood().run(cube, self.RADIUS)
        self.assertArrayAlmostEqual(result.data, expected_array)

    def test_masked_array_with_nans_re_mask_false(self):
        """Test that the run method produces a cube with correct data when a
           cube containing masked nans is passed in."""
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        cube.data = np.array([[[[np.nan, 1, 0, 1, 1],
                                [1, 1, 1, 0, 0],
                                [1, 0, 1, 0, 0],
                                [0, 0, 1, 1, 0],
                                [0, 1, 1, 0, 1]]]])
        mask = np.array([[[[0, 0, 1, 1, 0],
                           [0, 1, 1, 1, 0],
                           [0, 0, 1, 1, 1],
                           [0, 0, 1, 1, 0],
                           [0, 0, 1, 1, 0]]]])
        expected_array = np.array(
            [[[[np.nan, 0.666667, 0.600000, 0.500000, 0.50],
               [1.0000, 0.750000, 0.571429, 0.428571, 0.25],
               [1.0000, 1.000000, 0.714286, 0.571429, 0.25],
               [np.nan, 1.000000, 0.666667, 0.571429, 0.25],
               [np.nan, 1.000000, 0.750000, 0.750000, 0.50]]]])
        cube.data = np.ma.masked_where(mask == 0, cube.data)
        result = SquareNeighbourhood(re_mask=False).run(cube, self.RADIUS)
        self.assertArrayAlmostEqual(result.data, expected_array)

    def test_multiple_times(self):
        """Test that a cube with correct data is produced by the run method
        when multiple times are supplied."""
        expected_1 = np.array(
            [[1., 1., 1., 1., 1.],
             [1., 0.88888889, 0.88888889, 0.88888889, 1.],
             [1., 0.88888889, 0.88888889, 0.88888889, 1.],
             [1., 0.88888889, 0.88888889, 0.88888889, 1.],
             [1., 1., 1., 1., 1.]])
        expected_2 = np.array(
            [[1., 0.83333333, 0.83333333, 0.83333333, 1.],
             [1., 0.88888889, 0.88888889, 0.88888889, 1.],
             [1., 0.88888889, 0.88888889, 0.88888889, 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.]])
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2), (0, 1, 1, 2)), num_time_points=2,
            num_grid_points=5)
        result = SquareNeighbourhood().run(cube, self.RADIUS)
        self.assertIsInstance(cube, Cube)
        self.assertArrayAlmostEqual(result.data[0, 0], expected_1)
        self.assertArrayAlmostEqual(result.data[0, 1], expected_2)

    def test_multiple_times_with_mask(self):
        """Test that the run method produces a cube with correct data when a
        cube containing masked data at multiple time steps is passed in.
        Re-masking is disabled."""
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=2,
            num_grid_points=5)
        data = np.array([[[[1, 1, 0, 1, 1],
                           [1, 1, 1, 0, 0],
                           [1, 0, 1, 0, 0],
                           [0, 0, 1, 1, 0],
                           [0, 1, 1, 0, 1]],
                          [[1, 1, 0, 1, 1],
                           [1, 1, 1, 0, 0],
                           [1, 0, 1, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 1, 1, 0, 1]]]])
        mask = np.array([[[[0, 0, 1, 1, 0],
                           [0, 1, 1, 1, 0],
                           [0, 0, 1, 1, 1],
                           [0, 0, 1, 1, 0],
                           [0, 0, 1, 1, 0]],
                          [[0, 0, 1, 1, 0],
                           [0, 1, 1, 1, 0],
                           [0, 1, 1, 1, 1],
                           [0, 0, 0, 1, 0],
                           [0, 0, 1, 1, 0]]]])
        masked_data = np.ma.masked_where(mask == 0, data)
        cube.data = masked_data
        expected_array = np.array(
            [[[[1.0000, 0.666667, 0.600000, 0.500000, 0.500000],
               [1.0000, 0.750000, 0.571429, 0.428571, 0.250000],
               [1.0000, 1.000000, 0.714286, 0.571429, 0.250000],
               [np.nan, 1.000000, 0.666667, 0.571429, 0.250000],
               [np.nan, 1.000000, 0.750000, 0.750000, 0.500000]],
              [[1.0000, 0.666667, 0.600000, 0.500000, 0.500000],
               [0.5000, 0.600000, 0.500000, 0.428571, 0.250000],
               [0.5000, 0.750000, 0.428571, 0.333333, 0.000000],
               [0.0000, 0.666667, 0.333333, 0.333333, 0.000000],
               [np.nan, 1.000000, 0.333333, 0.333333, 0.000000]]]])
        result = SquareNeighbourhood(re_mask=False).run(cube, self.RADIUS)
        self.assertArrayAlmostEqual(result.data, expected_array)

    def test_multiple_times_nan(self):
        """Test that a cube with correct data is produced by the run method
        for multiple times and for when nans are present."""
        expected_1 = np.array(
            [[np.nan, 0.8, 0.8333333, 0.8333333, 1.],
             [1., 0.75, 0.77777778, 0.77777778, 1.],
             [1., 0.77777778, 0.77777778, 0.77777778, 1.],
             [1., 0.88888889, 0.88888889, 0.88888889, 1.],
             [1., 1., 1., 1., 1.]])
        expected_2 = np.array(
            [[1., 1., 1., 1., 1.],
             [1., np.nan, 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.]])
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2), (0, 0, 1, 2)), num_time_points=2,
            num_grid_points=5)
        cube.data[0, 0, 0, 0] = np.nan
        cube.data[0, 1, 1, 1] = np.nan
        result = SquareNeighbourhood().run(cube, self.RADIUS)
        self.assertIsInstance(cube, Cube)
        self.assertArrayAlmostEqual(result.data[0, 0], expected_1)
        self.assertArrayAlmostEqual(result.data[0, 1], expected_2)

    def test_metadata(self):
        """Test that a cube with correct metadata is produced by the run
        method."""
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        cube.attributes = {"Conventions": "CF-1.5"}
        cube.add_cell_method(CellMethod("mean", coords="time"))
        result = SquareNeighbourhood().run(cube, self.RADIUS)
        self.assertIsInstance(cube, Cube)
        self.assertTupleEqual(result.cell_methods, cube.cell_methods)
        self.assertDictEqual(result.attributes, cube.attributes)


if __name__ == '__main__':
    unittest.main()
