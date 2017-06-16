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
import numpy as np

from improver.nbhood import SquareNeighbourhood
from improver.tests.test_nbhood_neighbourhoodprocessing import (
    set_up_cube, set_up_cube_lat_long)


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
        result = SquareNeighbourhood(cube).cumulate_array()
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, data)

    def test_masked_array(self):
        """
        Test that the y-dimension and x-dimension accumulation produces the
        intended result for a masked array. A 2d cube is passed in.
        """
        data = np.array([[0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0.5, 0.5, 0.5, 0.5, 1.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]])
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        cube.data[0, 0, 2, 0] = 0.5
        cube.data[0, 0, 2, 4] = 0.5
        cube.data = np.ma.masked_greater(cube.data, 0.5)
        result = SquareNeighbourhood(cube).cumulate_array()
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data.data, data)

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
        result = SquareNeighbourhood(cube).cumulate_array()
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
        result = SquareNeighbourhood(cube).cumulate_array()
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, data)


class Test_run(IrisTest):

    """Test the run method on the SquareNeighbourhood class."""

    def test_basic(self):
        """Test that a cube with correct data is produced by the run method"""
        data = np.array([[1., 2., 3., 4., 5.],
                         [2., 4., 6., 8., 10.],
                         [3., 6., 8., 11., 14.],
                         [4., 8., 11., 15., 19.],
                         [5., 10., 14., 19., 24.]])
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        result = SquareNeighbourhood(
            cube).run()
        self.assertIsInstance(cube, Cube)
        self.assertArrayAlmostEqual(result.data, data)


if __name__ == '__main__':
    unittest.main()
