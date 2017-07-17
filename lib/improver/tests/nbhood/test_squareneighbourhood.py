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

from iris.cube import Cube
from iris.tests import IrisTest
from iris.coords import DimCoord

import numpy as np

from improver.nbhood import SquareNeighbourhood
from improver.tests.nbhood.test_neighbourhoodprocessing import set_up_cube


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
        nanmask = np.zeros([5,5]).astype(bool)
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        result, nan_mask = SquareNeighbourhood().cumulate_array(cube)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, data)
        self.assertArrayAlmostEqual(nan_mask, nanmask)

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
        result, nan_mask = SquareNeighbourhood().cumulate_array(cube)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, data)

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
        result, nan_mask = SquareNeighbourhood().cumulate_array(cube)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, data)

    def test_nan_array(self):
        """Test correct nanmask is returned when array containing NaN data
           is input."""
        data = np.array([[0., 1., 2., 3., 4.],
                         [1., 3., 5., 7., 9.],
                         [2., 5., 7., 10., 13.],
                         [3., 7., 10., 14., 18.],
                         [4., 9., 13., 18., 23.]])
        nanmask = np.zeros([5,5]).astype(bool)
        nanmask[0,0] = True
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        cube.data[0,0,0,0] = np.nan
        result, nan_mask = SquareNeighbourhood().cumulate_array(cube)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, data)
        self.assertArrayAlmostEqual(nan_mask, nanmask)

class Test_mean_over_neighbourhood(IrisTest):

    """Test for calculating mean value in neighbourhood"""

    def setUp(self):
        """Set up cube and expected results for tests."""

        # This array is the output from cumulate_array when a 5x5 array of 1's
        # with a 0 at the centre point (2,2) is passed in.
        data = np.array([[1., 2., 3., 4., 5.],
                         [2., 4., 6., 8., 10.],
                         [3., 6., 8., 11., 14.],
                         [4., 8., 11., 15., 19.],
                         [5., 10., 14., 19., 24.]])
        self.cube = Cube(data, long_name='test')
        self.x_coord = DimCoord([0, 1, 2, 3, 4], standard_name='longitude')
        self.y_coord = DimCoord([0, 1, 2, 3, 4], standard_name='latitude')
        self.cube.add_dim_coord(self.x_coord, 0)
        self.cube.add_dim_coord(self.y_coord, 1)
        self.result = np.array([[1., 1., 1., 1., 1.],
                                [1., 0.88888889, 0.88888889, 0.88888889, 1.],
                                [1., 0.88888889, 0.88888889, 0.88888889, 1.],
                                [1., 0.88888889, 0.88888889, 0.88888889, 1.],
                                [1., 1., 1., 1., 1.]])
        self.nanmask = np.array([[False, False, False, False, False],
                                 [False, False, False, False, False],
                                 [False, False, False, False, False],
                                 [False, False, False, False, False],
                                 [False, False, False, False, False]])
        self.width = 1

    def test_basic(self):
        """Test cube with correct data is produced when mean over
           neighbourhood is calculated."""
        result = SquareNeighbourhood().mean_over_neighbourhood(
            self.cube, self.width, self.width, self.nanmask)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, self.result)

    def test_multiple_times(self):
        """Test mean over neighbourhood with more than two dimensions."""
        data = np.array([[[1., 2., 3., 4., 5.],
                          [2., 4., 6., 8., 10.],
                          [3., 6., 8., 11., 14.],
                          [4., 8., 11., 15., 19.],
                          [5., 10., 14., 19., 24.]],
                         [[0., 2., 3., 4., 5.],
                          [2., 4., 6., 8., 10.],
                          [3., 6., 8., 11., 14.],
                          [4., 8., 11., 15., 19.],
                          [5., 10., 14., 19., 24.]]])
        cube = Cube(data, long_name='two times test')
        cube.add_dim_coord(self.x_coord, 1)
        cube.add_dim_coord(self.y_coord, 2)
        t_coord = DimCoord([0, 1], standard_name='time')
        cube.add_dim_coord(t_coord, 0)
        expected_result = np.array([self.result, self.result])
        expected_result[1, 2, 2] = 0.77777778
        result = SquareNeighbourhood().mean_over_neighbourhood(
            cube, self.width, self.width, self.nanmask)
        self.assertArrayAlmostEqual(result.data, expected_result)

    def test_neighbourhood_larger_than_half_domain(self):
        """Test the calculation of edge columns and rows is ok when the
           neighbourhood size is larger than half the domain."""
        expected_result = np.array([[0.9375, 0.95, 0.95, 0.95, 0.9375],
                                    [0.95, 0.96, 0.96, 0.96, 0.95],
                                    [0.95, 0.96, 0.96, 0.96, 0.95],
                                    [0.95, 0.96, 0.96, 0.96, 0.95],
                                    [0.9375, 0.95, 0.95, 0.95, 0.9375]])
        width = 3
        result = SquareNeighbourhood().mean_over_neighbourhood(
            self.cube, width, width, self.nanmask)
        self.assertArrayAlmostEqual(result.data, expected_result)

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
        nanmask = np.zeros([5,5]).astype(bool)
        nanmask[0,0] = True
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
        data = np.array([[1., 1., 1., 1., 1.],
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

    def test_NaN_array_fail(self):
        """Test that the correct exception is raised when an array containing
           NaNs is passed in."""
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        cube.data[0, 0] = np.NaN
        msg = 'Data array contains NaNs'
        with self.assertRaisesRegexp(ValueError, msg):
            SquareNeighbourhood().run(cube, self.RADIUS)


if __name__ == '__main__':
    unittest.main()
