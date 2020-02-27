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
"""Unit tests for the nbhood.square_kernel.SquareNeighbourhood plugin."""


import unittest

import iris
import numpy as np
from iris.coords import CellMethod, DimCoord
from iris.cube import Cube
from iris.tests import IrisTest

from improver.nbhood.square_kernel import SquareNeighbourhood
from improver.wind_calculations.wind_direction import WindDirection

from ..nbhood.test_BaseNeighbourhoodProcessing import set_up_cube


class Test__init__(IrisTest):

    """Test the init method."""

    def test_sum_or_fraction(self):
        """Test that a ValueError is raised if an invalid option is passed
        in for sum_or_fraction."""
        sum_or_fraction = "nonsense"
        msg = "option is invalid"
        with self.assertRaisesRegex(ValueError, msg):
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
            zero_point_indices=((0, 0, 2, 2),), num_grid_points=5)
        cube = iris.util.squeeze(cube)
        result_cube = SquareNeighbourhood().cumulate_array(cube)
        self.assertIsInstance(result_cube, Cube)
        self.assertArrayAlmostEqual(result_cube.data, data)


class Test_calculate_neighbourhood(IrisTest):

    """ Test calculating neighbourhood """

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
        cells_x = cells_y = 1
        self.n_rows = 7
        self.n_columns = 7
        self.ymax_xmax_disp = (cells_y*self.n_columns) + cells_x
        self.ymax_xmin_disp = (cells_y*self.n_columns) - cells_x - 1
        self.ymin_xmax_disp = (-1*(cells_y+1)*self.n_columns) + cells_x
        self.ymin_xmin_disp = (-1*(cells_y+1)*self.n_columns) - cells_x - 1

    def test_basic(self):
        """ Test that calculate neighbourhood returns correct values """
        expected = np.array(
            [[8., 5., -5., -8., -5., -3., 8.],
             [8., 6., -3., -5., -3., -2., 5.],
             [5., 7., 3., 5., 3., 2., -5.],
             [-5., -2., 5., 8., 5., 3., -8.],
             [-8., -6., 3., 5., 3., 2., -5.],
             [-5., -4., 2., 3., 2., 1., -3.],
             [-3., -6., -5., -8., -5., -3., 8.]])
        result = (
            SquareNeighbourhood().calculate_neighbourhood(self.cube,
                                                          self.ymax_xmax_disp,
                                                          self.ymin_xmax_disp,
                                                          self.ymin_xmin_disp,
                                                          self.ymax_xmin_disp,
                                                          self.n_rows,
                                                          self.n_columns))
        self.assertArrayEqual(result, expected)


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
        result = SquareNeighbourhood().mean_over_neighbourhood(
            self.cube, self.mask, self.width)
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
        result = SquareNeighbourhood(
            sum_or_fraction="sum").mean_over_neighbourhood(
                self.cube, self.mask, self.width)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, expected)


class Test_set_up_cubes_to_be_neighbourhooded(IrisTest):

    """Test the set up of cubes prior to neighbourhooding."""

    def setUp(self):
        """Set up a cube."""
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_grid_points=5)
        self.cube = iris.util.squeeze(self.cube)

    def test_without_masked_data(self):
        """Test setting up cubes to be neighbourhooded when the input cube
        does not contain masked arrays."""
        expected_mask = np.ones((5, 5))
        expected_nans = expected_mask.astype(bool)*False
        cube, mask, nan_array = (
            SquareNeighbourhood.set_up_cubes_to_be_neighbourhooded(self.cube))
        self.assertIsInstance(cube, Cube)
        self.assertIsInstance(mask, Cube)
        self.assertEqual(cube, self.cube)
        self.assertArrayEqual(nan_array, expected_nans)
        self.assertArrayEqual(mask.data, expected_mask)

    def test_with_masked_data(self):
        """Test setting up cubes to be neighbourhooded when the input cube
        contains masked arrays."""
        cube = self.cube
        data = cube.data
        cube.data[1, 3] = 0.5
        cube.data[3, 3] = 0.5
        cube.data = np.ma.masked_equal(data, 0.5)
        mask = np.logical_not(cube.data.mask.astype(int))
        expected_nans = np.ones((5, 5)).astype(bool)*False
        data = cube.data.data * mask
        result_cube, result_mask, result_nan_array = (
            SquareNeighbourhood.set_up_cubes_to_be_neighbourhooded(
                cube.copy()))
        self.assertArrayAlmostEqual(result_cube.data, data)
        self.assertArrayAlmostEqual(result_mask.data, mask)
        self.assertArrayEqual(result_nan_array, expected_nans)

    def test_with_separate_mask_cube(self):
        """Test for an input cube and an additional mask cube."""
        self.cube.data[1, 3] = 0.5
        self.cube.data[3, 3] = 0.5
        mask_cube = self.cube.copy()
        mask_cube.data = np.ones((5, 5))
        mask_cube.data[self.cube.data == 0.5] = 0
        mask_cube.data = mask_cube.data.astype(int)
        expected_data = self.cube.data * mask_cube.data
        expected_mask = np.ones((5, 5))
        expected_mask[1, 3] = 0.0
        expected_mask[3, 3] = 0.0
        expected_nans = np.ones((5, 5)).astype(bool)*False
        result_cube, result_mask, result_nan_array = (
            SquareNeighbourhood.set_up_cubes_to_be_neighbourhooded(
                self.cube.copy(), mask_cube=mask_cube))
        self.assertIsInstance(result_cube, Cube)
        self.assertIsInstance(result_mask, Cube)
        self.assertArrayAlmostEqual(result_cube.data, expected_data)
        self.assertArrayAlmostEqual(result_mask.data, expected_mask)
        self.assertArrayEqual(result_nan_array, expected_nans)

    def test_with_separate_mask_cube_and_nan(self):
        """Test for an input cube and an additional mask cube."""
        mask_cube = self.cube.copy()
        self.cube.data[1, 3] = 0.5
        self.cube.data[3, 3] = 0.5
        self.cube.data[1, 2] = np.nan
        self.cube.data[3, 1] = np.nan
        mask_cube.data = np.ones((5, 5))
        mask_cube.data[self.cube.data == 0.5] = 0
        mask_cube.data = mask_cube.data.astype(int)

        expected_mask = np.ones((5, 5))
        expected_mask[1, 3] = 0.0
        expected_mask[3, 3] = 0.0
        expected_mask[1, 2] = 0.0
        expected_mask[3, 1] = 0.0
        expected_data = self.cube.data * expected_mask
        expected_data[1, 2] = 0.0
        expected_data[3, 1] = 0.0
        expected_nans = np.ones((5, 5)).astype(bool)*False
        expected_nans[1, 2] = True
        expected_nans[3, 1] = True

        result_cube, result_mask, result_nan_array = (
            SquareNeighbourhood.set_up_cubes_to_be_neighbourhooded(
                self.cube.copy(), mask_cube=mask_cube))

        self.assertIsInstance(result_cube, Cube)
        self.assertIsInstance(result_mask, Cube)
        self.assertArrayAlmostEqual(result_cube.data, expected_data)
        self.assertArrayAlmostEqual(result_mask.data, expected_mask)
        self.assertArrayEqual(result_nan_array, expected_nans)


class Test__pad_and_calculate_neighbourhood(IrisTest):

    """Test the padding and calculation of neighbourhood processing."""

    def setUp(self):
        """Set up a cube."""
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 1, 1),), num_grid_points=3)
        sliced_cube = next(self.cube.slices(["projection_y_coordinate",
                                             "projection_x_coordinate"]))
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

        grid_cells = 1
        cube = self.cube
        mask_cube = cube.copy()
        mask_cube.data[::] = 1.0
        mask_cube.data[1, 2] = 0.0
        mask_cube.data[2, 2] = 0.0
        # set_up_cubes_to_be_neighbourhooded would set masked points to 0.0
        cube.data[1, 2] = 0.0
        cube.data[2, 2] = 0.0
        mask_cube.rename('mask_data')

        nbcube = (
            SquareNeighbourhood()._pad_and_calculate_neighbourhood(
                cube, mask_cube, grid_cells))
        self.assertIsInstance(nbcube, Cube)
        self.assertArrayAlmostEqual(nbcube.data, expected_data)

    def test_complex(self):
        """Test neighbourhooding with an array of complex wind directions"""
        # set up cube with complex wind directions 30-60 degrees
        a = 0.54066949+0.82535591j
        b = 0.56100423+0.80502117j
        c = 0.5+0.8660254j
        d = 0.59150635+0.77451905j

        expected_data_complex = np.array(
            [[a, b, b, a, b, c, a],
             [a, 0.55228934+0.81373606j, d, b, d, c, b],
             [b, 0.54575318+0.82027223j, d, b, d, c, b],
             [b, 0.62200847+0.74401694j, b, a, b, c, a],
             [a, 0.55228934+0.81373606j, d, b, d, c, b],
             [b, 0.57320508+0.79282032j, c, c, c, c, c],
             [c, c, b, a, b, c, a]])

        expected_data_deg = np.array(
            [[56.77222443, 55.12808228, 55.12808228, 56.77222443, 55.12808228,
              60.00000381, 56.77222443],
             [56.77222443, 55.83494186, 52.63074112, 55.12808228, 52.63074112,
              60.00000381, 55.12808228],
             [55.12808228, 56.36291885, 52.63074112, 55.12808228, 52.63074112,
              60.00000381, 55.12808228],
             [55.12808228, 50.10391235, 55.12808228, 56.77222443, 55.12808228,
              60.00000381, 56.77222443],
             [56.77222443, 55.83494186, 52.63074112, 55.12808228, 52.63074112,
              60.00000381, 55.12808228],
             [55.12808228, 54.13326263, 60.00000381, 60.00000381, 60.00000381,
              60.00000381, 60.00000381],
             [60.00000381, 60.00000381, 55.12808228, 56.77222443, 55.12808228,
              60.00000381, 56.77222443]],
            dtype=np.float32
        )

        self.cube.data = WindDirection.deg_to_complex(30.*self.cube.data + 30.)
        nbcube = (
            SquareNeighbourhood()._pad_and_calculate_neighbourhood(
                self.cube, self.mask, 1))
        self.assertTrue(np.any(np.iscomplex(nbcube.data)))
        self.assertArrayAlmostEqual(nbcube.data, expected_data_complex)
        self.assertArrayAlmostEqual(WindDirection.complex_to_deg(nbcube.data),
                                    expected_data_deg)

    def test_complex_masked(self):
        """Test complex neighbourhooding works with a mask"""

        mask_cube = self.cube.copy()
        mask_cube.data[::] = 1.0
        mask_cube.data[1, 2] = 0.0
        mask_cube.data[2, 2] = 0.0

        self.cube.data = WindDirection.deg_to_complex(30.*self.cube.data + 30.)

        # set_up_cubes_to_be_neighbourhooded would set masked points to 0.0
        self.cube.data[1, 2] = 0.0
        self.cube.data[2, 2] = 0.0
        mask_cube.rename('mask_data')

        nbcube = (
            SquareNeighbourhood()._pad_and_calculate_neighbourhood(
                self.cube, mask_cube, 1))
        self.assertIsInstance(nbcube, Cube)


class Test__remove_padding_and_mask(IrisTest):

    """Test the removal of padding and dealing with masked data."""

    def setUp(self):
        """Set up a cube."""
        self.padded_cube = set_up_cube(
            zero_point_indices=((0, 0, 3, 3),), num_grid_points=7)
        self.padded_cube = iris.util.squeeze(self.padded_cube)
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 1, 1),), num_grid_points=3)
        self.cube = iris.util.squeeze(self.cube)
        self.mask_cube = self.cube.copy()
        masked_array = np.ones(self.mask_cube.data.shape)
        masked_array[1, 2] = 0
        masked_array[0, 1] = 0
        self.mask_cube.data = masked_array
        self.mask_cube.rename('mask_data')
        self.no_mask = self.mask_cube.copy()
        self.no_mask.data = np.ones(self.mask_cube.data.shape)

    def test_without_masked_data(self):
        """Test that removing a halo of points from the data on a cube
        has worked as intended when the input data is not masked."""
        expected = np.array(
            [[1., 1., 1.],
             [1., 0., 1.],
             [1., 1., 1.]])
        grid_cells = 1
        nbcube = (
            SquareNeighbourhood()._remove_padding_and_mask(
                self.padded_cube, self.cube, self.no_mask,
                grid_cells))
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
             [False, False, True],
             [False, False, False]])
        grid_cells = 1
        nbcube = (
            SquareNeighbourhood()._remove_padding_and_mask(
                self.padded_cube, self.cube, self.mask_cube,
                grid_cells))
        self.assertIsInstance(nbcube, Cube)
        self.assertArrayAlmostEqual(nbcube.data.data, expected)
        self.assertArrayAlmostEqual(nbcube.data.mask, expected_mask)

    def test_with_masked_data_and_no_remasking(self):
        """Test that removing halo works correctly with remask=False"""
        expected = np.array(
            [[1., 1., 1.],
             [1., 0., 1.],
             [1., 1., 1.]])
        grid_cells = 1
        nbcube = (
            SquareNeighbourhood(re_mask=False)._remove_padding_and_mask(
                self.padded_cube, self.cube, self.mask_cube,
                grid_cells))
        self.assertIsInstance(nbcube, Cube)
        self.assertArrayAlmostEqual(nbcube.data, expected)

    def test_clipping(self):
        """Test that clipping is working"""
        expected = np.array(
            [[1., 1., 1.],
             [1., 0., 1.],
             [1., 1., 1.]])
        grid_cells = 1
        self.padded_cube.data[2, 2] = 1.1
        self.padded_cube.data[3, 3] = -0.1
        nbcube = (
            SquareNeighbourhood(re_mask=False)._remove_padding_and_mask(
                self.padded_cube, self.cube, self.mask_cube,
                grid_cells))
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
        expected_mask_array = np.array(
            [[[[True, True, False, False, True],
               [True, False, False, False, True],
               [True, True, False, False, False],
               [True, True, False, False, True],
               [True, True, False, False, True]]]])
        cube.data = np.ma.masked_where(mask == 0, cube.data)
        result = SquareNeighbourhood().run(cube, self.RADIUS)
        self.assertArrayAlmostEqual(result.data.data, expected_array)
        self.assertArrayAlmostEqual(result.data.mask, expected_mask_array)

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

    def test_complex(self):
        """Test that a cube containing complex numbers is sensibly processed"""
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        cube.data = cube.data.astype(complex)
        cube.data[0, 0, 1, 3] = 0.5+0.5j
        cube.data[0, 0, 4, 3] = 0.4+0.6j
        expected_data = np.array(
            [[[[1.0+0.0j, 1.0+0.0j, 0.91666667+0.083333333j,
                0.91666667+0.083333333j, 0.875+0.125j],
               [1.0+0.0j, 0.88888889+0.0j, 0.83333333+0.055555556j,
                0.83333333+0.055555556j, 0.91666667+0.083333333j],
               [1.0+0.0j, 0.88888889+0.0j, 0.83333333+0.055555556j,
                0.83333333+0.055555556j, 0.91666667+0.083333333j],
               [1.0+0.0j, 0.88888889+0.0j, 0.82222222+0.066666667j,
                0.82222222+0.066666667j, 0.9+0.1j],
               [1.0+0.0j, 1.0+0.0j, 0.9+0.1j, 0.9+0.1j, 0.85+0.15j]]]])
        result = SquareNeighbourhood().run(cube, self.RADIUS)
        self.assertArrayAlmostEqual(result.data, expected_data)

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
