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
            sliced_cube, width_x, width_y, masked_data=True)
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
            sliced_cube, width_x, width_y)
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
            sliced_cube, width_x, width_y)
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

        # This array is the output from cumulate_array when a 5x5 array of 1's
        # with a 0 at the centre point (2,2) is passed in.
        # Note that edge points are not handled correctly by default as no
        # padding is applied.
        self.data = np.array(
            [[1., 2., 3., 4., 5.],
             [2., 4., 6., 8., 10.],
             [3., 6., 8., 11., 14.],
             [4., 8., 11., 15., 19.],
             [5., 10., 14., 19., 24.]])
        self.cube = Cube(self.data, long_name='test')
        self.x_coord = DimCoord([0, 1, 2, 3, 4], standard_name='longitude')
        self.y_coord = DimCoord([0, 1, 2, 3, 4], standard_name='latitude')
        self.cube.add_dim_coord(self.x_coord, 0)
        self.cube.add_dim_coord(self.y_coord, 1)
        self.mask = self.cube.copy()
        self.mask.rename('mask_data')
        self.nan_masks = np.zeros(self.cube.data.shape, dtype=int)
        self.width = 1
        # Set up padded dataset to simulate padding.
        self.padded_data = np.array(
            [[1., 2., 3., 4., 5., 6., 7., 8., 9.],
             [2., 4., 6., 8., 10., 12., 14., 16., 18.],
             [3., 6., 9., 12., 15., 18., 21., 24., 27.],
             [4., 8., 12., 16., 20., 24., 28., 32., 36.],
             [5., 10., 15., 20., 24., 29., 34., 39., 44.],
             [6., 12., 18., 24., 29., 35., 41., 47., 53.],
             [7., 14., 21., 28., 34., 41., 48., 55., 62.],
             [8., 16., 24., 32., 39., 47., 55., 63., 71.],
             [9., 18., 27., 36., 44., 53., 62., 71., 80.]])
        self.padded_cube = Cube(self.padded_data, long_name='test')
        self.padded_x_coord = DimCoord(range(0, 9), standard_name='longitude')
        self.padded_y_coord = DimCoord(range(0, 9), standard_name='latitude')
        self.padded_cube.add_dim_coord(self.padded_x_coord, 0)
        self.padded_cube.add_dim_coord(self.padded_y_coord, 1)
        self.padded_result = np.array(
            [[1., 1., 1., 1., 1.],
             [1., 0.88888889, 0.88888889, 0.88888889, 1.],
             [1., 0.88888889, 0.88888889, 0.88888889, 1.],
             [1., 0.88888889, 0.88888889, 0.88888889, 1.],
             [1., 1., 1., 1., 1.]])

    def test_basic_fraction(self):
        """Test cube with correct data is produced when mean over
           neighbourhood is calculated where the sum_or_fraction option is
           set to "fraction"."""
        expected = np.array(
            [[0.33333333, 0.44444444, -0.55555556, -0.55555556, 0.33333333],
             [0.33333333, 0.33333333, -0.66666667, -0.66666667, 1.],
             [1.55555556, 2., 0.88888889, 0.88888889, -0.55555556],
             [-0.55555556, -0.66666667, 0.88888889, 0.88888889, -1.11111111],
             [-1.66666667, -2.11111111, -0.55555556, -0.55555556, 0.33333333]])
        cube_and_mask = iris.cube.CubeList([self.cube, self.mask])
        result = SquareNeighbourhood().mean_over_neighbourhood(
            cube_and_mask, self.width, self.width, self.nan_masks)
        self.assertIsInstance(result, Cube)
        print result.data
        #self.assertArrayAlmostEqual(result.data, self.result)


if __name__ == '__main__':
    unittest.main()
