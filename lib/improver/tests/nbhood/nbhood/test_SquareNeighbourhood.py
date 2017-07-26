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
"""Unit tests for the nbhood.CircularNeighbourhood plugin."""


import unittest

import copy

from iris.coords import CellMethod, DimCoord
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest

import numpy as np

from improver.nbhood.nbhood import SquareNeighbourhood
from improver.tests.nbhood.nbhood.test_NeighbourhoodProcessing import (
    set_up_cube)


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(SquareNeighbourhood())
        msg = '<SquareNeighbourhood: unweighted_mode: False>'
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
        nanmask = np.zeros([5, 5]).astype(bool)
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        nan_mask = np.zeros(cube.data.shape, dtype=int).flatten()
        result = SquareNeighbourhood().cumulate_array(cube)
        self.assertIsInstance(result[0], Cube)
        self.assertArrayAlmostEqual(result[0].data, data)
        self.assertArrayAlmostEqual(result[1][0].data, nan_mask)

    def test_for_multiple_times(self):
        """
        Test that the y-dimension and x-dimension accumulation produces the
        intended result when the input cube has multiple times. The input
        cube has an extra time dimension to ensure that a 3d cube is correctly
        handled.
        """
        data = np.array([[[1., 2., 3., 4., 5.],
                          [2., 4., 6., 8., 10.],
                          [3., 6., 8., 11., 14.],
                          [4., 8., 11., 15., 19.],
                          [5., 10., 14., 19., 24.]],
                         [[1., 2., 3., 4., 5.],
                          [2., 4., 6., 8., 10.],
                          [3., 6., 9., 12., 15.],
                          [4., 8., 12., 15., 19.],
                          [5., 10., 15., 19., 24.]],
                         [[0., 1., 2., 3., 4.],
                          [1., 3., 5., 7., 9.],
                          [2., 5., 8., 11., 14.],
                          [3., 7., 11., 15., 19.],
                          [4., 9., 14., 19., 24.]]])
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2), (0, 1, 3, 3), (0, 2, 0, 0)),
            num_time_points=3, num_grid_points=5)
        nan_mask = np.zeros(cube[0, 0, :, :].data.shape, dtype=int).flatten()
        result = SquareNeighbourhood().cumulate_array(cube)
        self.assertIsInstance(result[0], Cube)
        self.assertArrayAlmostEqual(result[0].data, data)
        self.assertArrayAlmostEqual(result[1][0].data, nan_mask)

    def test_for_multiple_times_NaNs(self):
        """
        Test that the y-dimension and x-dimension accumulation produces the
        intended result when the input cube has multiple times. The input
        cube has an extra time dimension to ensure that a 3d cube is correctly
        handled.
        """
        data = np.array(
            [[[0., 1., 2., 3., 4.],
              [1., 3., 5., 7., 9.],
              [2., 5., 7., 10., 13.],
              [3., 7., 10., 14., 18.],
              [4., 9., 13., 18., 23.]],
             [[1., 2., 3., 4., 5.],
              [2., 3., 5., 7., 9.],
              [3., 5., 8., 11., 14.],
              [4., 7., 11., 14., 18.],
              [5., 9., 14., 18., 23.]],
             [[0., 1., 2., 3., 4.],
              [1., 3., 5., 7., 9.],
              [2., 5., 8., 10., 13.],
              [3., 7., 11., 14., 18.],
              [4., 9., 14., 18., 23.]]])

        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2), (0, 1, 3, 3), (0, 2, 0, 0)),
            num_time_points=3, num_grid_points=5)
        cube.data[0, 0, 0, 0] = np.nan
        cube.data[0, 1, 1, 1] = np.nan
        cube.data[0, 2, 2, 3] = np.nan
        nan_mask = np.zeros(cube[0, 0, :, :].data.shape, dtype=int).flatten()
        nan_mask_1 = copy.copy(nan_mask)
        nan_mask_2 = copy.copy(nan_mask)
        nan_mask_3 = copy.copy(nan_mask)
        nan_mask_1[0] = 1
        nan_mask_2[6] = 1
        nan_mask_3[13] = 1
        result = SquareNeighbourhood().cumulate_array(cube)
        self.assertIsInstance(result[0], Cube)
        self.assertArrayAlmostEqual(result[0].data, data)
        self.assertArrayAlmostEqual(result[1][0].data, nan_mask_1)
        self.assertArrayAlmostEqual(result[1][1].data, nan_mask_2)
        self.assertArrayAlmostEqual(result[1][2].data, nan_mask_3)

    def test_for_multiple_realizations_and_times(self):
        """
        Test that the y-dimension and x-dimension accumulation produces the
        intended result when the input cube has multiple realizations and
        times. The input cube has extra time and realization dimensions to
        ensure that a 4d cube is correctly handled.
        """
        data = np.array([[[[1., 2., 3., 4., 5.],
                           [2., 4., 6., 8., 10.],
                           [3., 6., 8., 11., 14.],
                           [4., 8., 11., 15., 19.],
                           [5., 10., 14., 19., 24.]],
                          [[0., 1., 2., 3., 4.],
                           [1., 3., 5., 7., 9.],
                           [2., 5., 8., 11., 14.],
                           [3., 7., 11., 15., 19.],
                           [4., 9., 14., 19., 24.]]],
                         [[[1., 2., 3., 4., 5.],
                           [2., 4., 6., 8., 10.],
                           [3., 6., 9., 12., 15.],
                           [4., 8., 12., 15., 19.],
                           [5., 10., 15., 19., 24.]],
                          [[1., 2., 3., 4., 5.],
                           [2., 4., 6., 8., 10.],
                           [3., 5., 8., 11., 14.],
                           [4., 7., 11., 15., 19.],
                           [5., 9., 14., 19., 24.]]]])
        cube = set_up_cube(
            zero_point_indices=(
                (0, 0, 2, 2), (1, 0, 3, 3), (0, 1, 0, 0), (1, 1, 2, 1)),
            num_time_points=2, num_grid_points=5, num_realization_points=2)
        nan_mask = np.zeros(cube[0, 0, :, :].data.shape, dtype=int).flatten()
        result = SquareNeighbourhood().cumulate_array(cube)
        self.assertIsInstance(result[0], Cube)
        self.assertArrayAlmostEqual(result[0].data, data)
        self.assertArrayAlmostEqual(result[1][0].data, nan_mask)


class Test_pad_coord(IrisTest):

    """Test the padding of a coordinate."""

    def setUp(self):
        """Set up a cube."""
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)

    def test_add(self):
        """Test the functionality to add padding to the chosen coordinate."""
        expected = np.array(
            [-245010.968921, -242506.398537, -240001.828154, -237497.25777,
             -234992.687386, -232488.117002, -229983.546618, -227478.976234,
             -224974.40585])
        coord = self.cube.coord("projection_x_coordinate")
        width = 1
        method = "add"
        new_coord = SquareNeighbourhood.pad_coord(coord, width, method)
        self.assertIsInstance(new_coord, DimCoord)
        self.assertArrayAlmostEqual(new_coord.points, expected)

    def test_exception(self):
        """Test an exception is raised if the chosen coordinate is
        non-uniform."""
        coord_points = np.array([10, 20, 30, 45, 60])
        self.cube.coord("projection_x_coordinate").points = coord_points
        coord = self.cube.coord("projection_x_coordinate")
        width = 1
        method = "add"
        msg = "Non-uniform increments between grid points"
        with self.assertRaisesRegexp(ValueError, msg):
            SquareNeighbourhood.pad_coord(coord, width, method)

    def test_remove(self):
        """Test the functionality to remove padding from the chosen
        coordinate."""
        expected = np.array([-234992.6875])
        coord = self.cube.coord("projection_x_coordinate")
        width = 1
        method = "remove"
        new_coord = SquareNeighbourhood.pad_coord(coord, width, method)
        self.assertIsInstance(new_coord, DimCoord)
        self.assertArrayAlmostEqual(new_coord.points, expected)


class Test__check_for_x_and_y_axes(IrisTest):

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
        data = sliced_cube.data
        coord_x = sliced_cube.coord("projection_x_coordinate")
        coord_y = sliced_cube.coord("projection_y_coordinate")
        sliced_cube.remove_coord("projection_y_coordinate")
        msg = "The cube does not contain the expected"
        with self.assertRaisesRegexp(ValueError, msg):
            SquareNeighbourhood._check_for_x_and_y_axes(sliced_cube)


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
        for sliced_cube in self.cube.slices(
            ["realization", "projection_y_coordinate",
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
        for sliced_cube in self.cube.slices(
            ["projection_y_coordinate",
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
        for sliced_cube in self.cube.slices(
            ["projection_y_coordinate",
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

    def test_different_widths(self):
        """Test that padding the data in a cube with different widths has
        worked as intended."""
        for sliced_cube in self.cube.slices(
            ["projection_y_coordinate",
             "projection_x_coordinate"]):
            break
        expected = np.array(
            [[1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 0., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1.]])
        width_x = 1
        width_y = 2
        padded_cube = SquareNeighbourhood().pad_cube_with_halo(
                          sliced_cube, width_x, width_y)
        self.assertIsInstance(padded_cube, Cube)
        self.assertArrayAlmostEqual(padded_cube.data, expected)

    def test_zero_width(self):
        """Test that padding the data in a cube with a width of zero has
        worked as intended."""
        for sliced_cube in self.cube.slices(
            ["projection_y_coordinate",
             "projection_x_coordinate"]):
            break
        expected = np.array(
            [[1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 0., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.]])
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
        for sliced_cube in self.cube.slices(
            ["projection_y_coordinate",
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
        has worked as intended for difference x and y widths."""
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 5, 5),), num_time_points=1,
            num_grid_points=10)
        for sliced_cube in self.cube.slices(
            ["projection_y_coordinate",
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
        for sliced_cube in self.cube.slices(
            ["projection_y_coordinate",
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

    def test_nan_array(self):
        """Test correct nanmask is returned when array containing NaN data
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
        cube.data[0, 0, 0, 0] = np.nan
        result, nan_mask = SquareNeighbourhood().cumulate_array(cube)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, data)
        self.assertArrayAlmostEqual(nan_mask, nanmask)


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
        self.result = np.array(
            [[0.33333333, 0.44444444, -0.55555556, -0.55555556, 0.33333333],
             [0.33333333, 0.33333333, -0.66666667, -0.66666667, 1.],
             [1.55555556, 2., 0.88888889, 0.88888889, -0.55555556],
             [-0.55555556, -0.66666667, 0.88888889, 0.88888889, -1.11111111],
             [-1.66666667, -2.11111111, -0.55555556, -0.55555556, 0.33333333]])
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

    def test_basic(self):
        """Test cube with correct data is produced when mean over
           neighbourhood is calculated."""
        nan_masks = [np.zeros(self.cube.data.shape, dtype=int)]
        result = SquareNeighbourhood().mean_over_neighbourhood(
            self.cube, self.width, self.width, nan_masks)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, self.result)

    def test_basic_padded(self):
        """Test cube with correct data is produced when mean over
           neighbourhood is calculated."""
        nan_masks = [np.zeros(self.padded_cube.data.shape, dtype=int)]
        result = SquareNeighbourhood().mean_over_neighbourhood(
            self.padded_cube, self.width, self.width, nan_masks)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(
            result.data[2:-2, 2:-2], self.padded_result)

    def test_multiple_times(self):
        """Test mean over neighbourhood with more than two dimensions."""
        data = np.array([self.padded_data, self.padded_data])
        cube = Cube(data, long_name='two times test')
        cube.add_dim_coord(self.padded_x_coord, 1)
        cube.add_dim_coord(self.padded_y_coord, 2)
        t_coord = DimCoord([0, 1], standard_name='time')
        cube.add_dim_coord(t_coord, 0)
        nan_masks = [np.zeros(cube.data[0].shape, dtype=int)] * 2
        result = SquareNeighbourhood().mean_over_neighbourhood(
            cube, self.width, self.width, nan_masks)
        self.assertArrayAlmostEqual(
            result.data[0, 2:-2, 2:-2], self.padded_result)
        self.assertArrayAlmostEqual(
            result.data[1, 2:-2, 2:-2], self.padded_result)


class Test__set_up_cubes_to_be_neighbourhooded(IrisTest):

    """Test the set up of cubes prior to neighbourhooding."""

    def setUp(self):
        """Set up a cube."""
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)

    def test_without_masked_data(self):
        """Test setting up cubes to be neighbourhooded when the input cube
        does not contain masked arrays."""
        cubes = (
            SquareNeighbourhood._set_up_cubes_to_be_neighbourhooded(self.cube))
        self.assertIsInstance(cubes, CubeList)
        self.assertEqual(len(cubes), 1)
        self.assertEqual(cubes[0], self.cube)

    def test_with_masked_data(self):
        """Test setting up cubes to be neighbourhooded when the input cube
        contains masked arrays."""
        cube = self.cube
        data = cube.data
        cube.data[0, 0, 1, 3] = 0.5
        cube.data[0, 0, 3, 3] = 0.5
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


class Test__pad_and_calculate_neighbourhood(IrisTest):

    """Test the padding and calculation of neighbourhood processing."""

    def setUp(self):
        """Set up a cube."""
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 1, 1),), num_time_points=1,
            num_grid_points=3)
        for sliced_cube in self.cube.slices(
            ["projection_y_coordinate",
             "projection_x_coordinate"]):
            break
        sliced_cube.remove_coord("realization")
        sliced_cube.remove_coord("time")
        self.cube = sliced_cube

    def test_without_masked_data(self):
        """Test setting up cubes to be padded and then passed into
        neighbourhood processing when the input cubes do not contain masked
        arrays."""
        expected = np.array(
            [[1.66666667, 1.66666667, -1.22222222, -1.22222222, -1.22222222,
              -1.33333333, 1.66666667],
             [1.66666667, 1.66666667, -1.22222222, -1.22222222, -1.22222222,
              -1.33333333, 2.44444444],
             [3.22222222, 4., 0.88888889, 0.88888889, 0.88888889,
              1., -1.22222222],
             [-1.22222222, -1.22222222, 0.88888889, 0.88888889, 0.88888889,
              1., -1.22222222],
             [-1.22222222, -1.22222222, 0.88888889, 0.88888889, 0.88888889,
              1., -1.22222222],
             [-1.22222222, -1.22222222, 1., 1., 1.,
              1., -2.11111111],
             [-2.88888889, -3.66666667, -1.22222222, -1.22222222, -1.22222222,
              -1.33333333,  1.66666667]])
        grid_cells_x = grid_cells_y = 1
        cubes = CubeList([self.cube])
        nbcubes = (
            SquareNeighbourhood()._pad_and_calculate_neighbourhood(
                cubes, grid_cells_x, grid_cells_y))
        self.assertIsInstance(nbcubes, CubeList)
        self.assertEqual(len(nbcubes), 1)
        self.assertArrayAlmostEqual(nbcubes[0].data, expected)

    def test_with_masked_data(self):
        """Test setting up cubes to be padded and then passed into
        neighbourhood processing when the input cubes contain masked
        arrays."""
        expected_data = np.array(
            [[1.66666667, 1.66666667, -1.22222222, -1.22222222, -1.22222222,
              -1.33333333, 1.66666667],
             [1.66666667, 1.66666667, -1.22222222, -1.22222222, -1.22222222,
              -1.33333333, 2.44444444],
             [3.22222222, 4., 0.88888889, 0.88888889, 0.88888889,
              1., -1.22222222],
             [-1.22222222, -1.22222222, 0.88888889, 0.88888889, 0.88888889,
              1., -1.22222222],
             [-1.22222222, -1.22222222, 0.88888889, 0.88888889, 0.88888889,
              1., -1.22222222],
             [-1.22222222, -1.22222222, 1., 1., 1.,
              1., -2.11111111],
             [-2.88888889, -3.66666667, -1.22222222, -1.22222222, -1.22222222,
              -1.33333333,  1.66666667]])
        expected_mask = np.array(
            [[0.55555556, 0.77777778, -0.11111111, -0.44444444, -0.77777778,
              -1., 0.44444444],
             [0.77777778, 1.11111111, -0.11111111, -0.55555556, -1.,
              -1.33333333, 0.55555556],
             [1., 1.44444444, 0.11111111, 0.22222222, 0.33333333,
              0.33333333, -0.22222222],
             [-0.33333333, -0.44444444, 0.11111111, 0.33333333, 0.55555556,
              0.66666667, -0.33333333],
             [-0.55555556, -0.77777778, 0.11111111, 0.44444444, 0.77777778,
              1., -0.44444444],
             [-0.77777778, -1.11111111, 0., 0.33333333, 0.66666667,
              1., -0.33333333],
             [-0.66666667, -1., -0.11111111, -0.33333333, -0.55555556,
              -0.66666667, 0.33333333]])

        grid_cells_x = grid_cells_y = 1
        cube = self.cube
        mask_cube = cube.copy()
        mask_cube.data[1, 2] = 0.5
        mask_cube.data[2, 2] = 0.5
        mask_cube.data = np.ma.masked_equal(mask_cube.data, 0.5)
        mask_cube.rename('mask_data')
        mask_cube.data = np.logical_not(mask_cube.data.astype(int))
        cubes = CubeList([cube, mask_cube])
        nbcubes = (
            SquareNeighbourhood()._pad_and_calculate_neighbourhood(
                cubes, grid_cells_x, grid_cells_y))
        self.assertIsInstance(nbcubes, CubeList)
        self.assertEqual(len(nbcubes), 2)
        self.assertArrayAlmostEqual(nbcubes[0].data, expected_data)
        self.assertArrayAlmostEqual(nbcubes[1].data, expected_mask)


class Test__remove_padding_and_mask(IrisTest):

    """Test the removal of padding and dealing with masked data."""

    def setUp(self):
        """Set up a cube."""
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=7)

    def test_without_masked_data(self):
        """Test that removing a halo of points from the data on a cube
        has worked as intended when the input data is not masked."""
        expected = np.array(
            [[0., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.]])
        grid_cells_x = grid_cells_y = 1
        cubes = CubeList([self.cube])
        nbcube = (
            SquareNeighbourhood()._remove_padding_and_mask(
                cubes, self.cube.name(), grid_cells_x, grid_cells_y))
        self.assertIsInstance(nbcube, Cube)
        self.assertArrayAlmostEqual(nbcube.data, expected)

    def test_with_masked_data(self):
        """Test that removing a halo of points from the data on a cube
        has worked as intended when the input data is masked."""
        expected = np.array(
            [[0., 0., 1.],
             [1., 0., 1.],
             [1., 1., 1.]])
        grid_cells_x = grid_cells_y = 1
        cube = self.cube
        mask_cube = cube.copy()
        masked_array = np.ones(mask_cube.data.shape)
        masked_array[0, 0, 3, 3] = 0
        masked_array[0, 0, 2, 3] = 0
        mask_cube.rename('mask_data')
        mask_cube.data = masked_array.astype(bool)
        cubes = CubeList([cube, mask_cube])
        nbcube = (
            SquareNeighbourhood()._remove_padding_and_mask(
                cubes, cube.name(), grid_cells_x, grid_cells_y))
        self.assertIsInstance(nbcube, Cube)
        self.assertArrayAlmostEqual(nbcube.data, expected)

    def test_nanmask(self):
        """Test the correct result is returned when a nans must be substituded
           into the final array. Note: this type of data should also be masked,
           so the expected_data array looks strange because there is further
           processing to be done on it."""
        data = np.array([[0., 1., 2., 3., 4.],
                         [1., 3., 5., 7., 9.],
                         [2., 5., 7., 10., 13.],
                         [3., 7., 10., 14., 18.],
                         [4., 9., 13., 18., 23.]])
        cube = self.cube
        cube.data = data
        nanmask = np.zeros([5, 5]).astype(bool)
        nanmask[0, 0] = True
        expected_data = np.array(
            [[np.nan, 0.83333333, 1., 1., 1.],
             [0.83333333, 0.77777778, 0.88888889, 0.88888889, 1.],
             [1., 0.88888889, 0.88888889, 0.88888889, 1.],
             [1., 0.88888889, 0.88888889, 0.88888889, 1.],
             [1., 1., 1., 1., 1.]])
        result = SquareNeighbourhood().mean_over_neighbourhood(
            cube, self.width, self.width, nanmask)
        self.assertArrayAlmostEqual(result.data, expected_data)


class Test_run(IrisTest):

    """Test the run method on the SquareNeighbourhood class."""

    RADIUS = 2500

    def test_basic(self):
        """Test that a cube with correct data is produced by the run method."""
        data = np.array(
            [[1., 1., 1., 1., 1.],
             [1., 0.88888889, 0.88888889, 0.88888889, 1.],
             [1., 0.88888889, 0.88888889, 0.88888889, 1.],
             [1., 0.88888889, 0.88888889, 0.88888889, 1.],
             [1., 1., 1., 1., 1.]])
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        result = SquareNeighbourhood().run(cube, self.RADIUS)
        self.assertIsInstance(cube, Cube)
        self.assertArrayAlmostEqual(result.data, data)

    def test_masked_array(self):
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
        expected_array = np.array([[0., 0., 0.6, 0.5, 0.],
                                   [0., 0.75, 0.57142857, 0.42857143, 0.],
                                   [0., 0., 0.71428571, 0.57142857, 0.25],
                                   [0., 0., 0.66666667, 0.57142857, 0.],
                                   [0., 0., 0.75, 0.75, 0.]])
        cube.data = np.ma.masked_where(mask == 0, cube.data)
        result = SquareNeighbourhood().run(cube, self.RADIUS)
        self.assertArrayAlmostEqual(result.data, expected_array)

    def test_NaN_array(self):
        """Test that the an array containing NaNs is handled correctly."""
        data = np.array(
            [[np.nan, 0.777778, 1., 1., 1.],
             [0.777778, 0.77777778, 0.88888889, 0.88888889, 1.],
             [1., 0.88888889, 0.88888889, 0.88888889, 1.],
             [1., 0.88888889, 0.88888889, 0.88888889, 1.],
             [1., 1., 1., 1., 1.]])
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        cube.data[0, 0, 0, 0] = np.NaN
        result = SquareNeighbourhood().run(cube, self.RADIUS)
        self.assertIsInstance(cube, Cube)
        self.assertArrayAlmostEqual(result.data, data)

    def test_masked_array_with_nans(self):
        """Test that the run method produces a cube with correct data when a
           cube containing masked nans is passed in."""
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        cube.data = np.array([[[[np.NaN, 1, 0, 1, 1],
                                [1, 1, 1, 0, 0],
                                [1, 0, 1, 0, 0],
                                [0, 0, 1, 1, 0],
                                [0, 1, 1, 0, 1]]]])
        mask = np.array([[[[0, 0, 1, 1, 0],
                           [0, 1, 1, 1, 0],
                           [0, 0, 1, 1, 1],
                           [0, 0, 1, 1, 0],
                           [0, 0, 1, 1, 0]]]])
        expected_array = np.array([[0., 0., 0.6, 0.5, 0.],
                                   [0., 0.75, 0.57142857, 0.42857143, 0.],
                                   [0., 0., 0.71428571, 0.57142857, 0.25],
                                   [0., 0., 0.66666667, 0.57142857, 0.],
                                   [0., 0., 0.75, 0.75, 0.]])
        expected_array_data = np.array(
            [[np.nan, 0.66666667, 0.6, 0.5, 0.5],
             [1., 0.75, 0.57142857, 0.42857143, 0.25],
             [1., 1., 0.71428571, 0.57142857, 0.25],
             [np.nan, 1., 0.66666667, 0.57142857, 0.25],
             [np.nan, 1., 0.75, 0.75, 0.5]])
        cube.data = np.ma.masked_where(mask == 0, cube.data)
        result = SquareNeighbourhood().run(cube, self.RADIUS)
        self.assertArrayAlmostEqual(result.data, expected_array)
        self.assertArrayAlmostEqual(result.data.data, expected_array_data)

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
            [[1., 0.88888889, 0.88888889, 0.88888889, 1.],
             [1., 0.88888889, 0.88888889, 0.88888889, 1.],
             [1., 0.88888889, 0.88888889, 0.88888889, 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.]])
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2), (0, 1, 1, 2)), num_time_points=2,
            num_grid_points=5)
        result = SquareNeighbourhood().run(cube, self.RADIUS)
        self.assertIsInstance(cube, Cube)
        self.assertArrayAlmostEqual(result.data[0], expected_1)
        self.assertArrayAlmostEqual(result.data[1], expected_2)

    def test_multiple_times_with_mask(self):
        """Test that the run method produces a cube with correct data when a
           cube containing masked data at multiple time steps is passed in."""
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
        expected_array = np.array([[[0., 0., 0.6, 0.5, 0.],
                                    [0., 0.75, 0.57142857, 0.42857143, 0.],
                                    [0., 0., 0.71428571, 0.57142857, 0.25],
                                    [0., 0., 0.66666667, 0.57142857, 0.],
                                    [0., 0., 0.75, 0.75, 0.]],
                                   [[0., 0., 0.6, 0.5, 0.],
                                    [0., 0.6, 0.5, 0.42857143, 0.],
                                    [0., 0.75, 0.42857143,  0.33333333, 0.],
                                    [0., 0., 0., 0.33333333, 0.],
                                    [0., 0., 0.33333333, 0.33333333, 0.]]])
        result = SquareNeighbourhood().run(cube, self.RADIUS)
        self.assertArrayAlmostEqual(result.data, expected_array)

    def test_multiple_times_NaN(self):
        """Test that a cube with correct data is produced by the run method
        for multiple times and for when NaNs are present."""
        expected_1 = np.array(
            [[np.nan, 0.666667, 0.88888889, 0.88888889, 1.],
             [0.777778, 0.66666667, 0.77777778, 0.77777778,  1.],
             [1., 0.77777778, 0.77777778, 0.77777778, 1.],
             [1., 0.88888889, 0.88888889, 0.88888889, 1.],
             [1., 1., 1., 1., 1.]])
        expected_2 = np.array(
            [[0.88888889, 0.88888889, 0.88888889, 1., 1.],
             [0.88888889, np.nan, 0.88888889, 1., 1.],
             [0.88888889, 0.88888889, 0.88888889, 1., 1.],
             [1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1.]])
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2), (0, 0, 1, 2)), num_time_points=2,
            num_grid_points=5)
        cube.data[0, 0, 0, 0] = np.NaN
        cube.data[0, 1, 1, 1] = np.NaN
        result = SquareNeighbourhood().run(cube, self.RADIUS)
        self.assertIsInstance(cube, Cube)
        self.assertArrayAlmostEqual(result.data[0], expected_1)
        self.assertArrayAlmostEqual(result.data[1], expected_2)

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
