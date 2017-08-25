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
"""Unit tests for the
nbhood.circular_kernel.GeneratePercentilesFromACircularNeighbourhood
plugin."""


import unittest

import iris
from iris.cube import Cube
from iris.tests import IrisTest
import numpy as np

from improver.constants import DEFAULT_PERCENTILES
from improver.nbhood.circular_kernel import (
    GeneratePercentilesFromACircularNeighbourhood)
from improver.tests.nbhood.nbhood.test_BaseNeighbourhoodProcessing import (
    set_up_cube, set_up_cube_lat_long)


# Standard percentile answers for radius 2 grid points
# (==> circle of 13 points)
# Array name indicates how many of the 13 grid points were ones;
# remaining points are zeroes.
PERCENTILES_1_IN_13 = np.ones(15)
PERCENTILES_1_IN_13[:2] = [0., 0.6]
PERCENTILES_2_IN_13 = np.ones(15)
PERCENTILES_2_IN_13[:3] = [0., 0., 0.2]

# For 1 zero in group of 5 (other 4 are ones):
PERCENTILES_1_IN_5 = np.ones(15)
PERCENTILES_1_IN_5[:4] = [0., 0.2, 0.4, 0.8]

# Standard percentile answers for radius 3 grid points
# (==> circle of 25 points)
PERCENTILES_1_IN_25 = np.ones(15)
PERCENTILES_1_IN_25[:1] = [0.]


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(GeneratePercentilesFromACircularNeighbourhood())
        msg = ('<GeneratePercentilesFromACircularNeighbourhood: '
               'percentiles: {}>'.format(DEFAULT_PERCENTILES))
        self.assertEqual(str(result), msg)


class Test_make_percentile_cube(IrisTest):

    """Test the make_percentile_cube method from
       GeneratePercentilesFromACircularNeighbourhood."""

    def test_basic(self):
        """
        Test that the plugin returns an iris.cube.Cube and that percentile
        coord is added.
        """
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        result = (
            GeneratePercentilesFromACircularNeighbourhood(
            ).make_percentile_cube(cube))
        self.assertIsInstance(result, Cube)
        self.assertIsInstance(result.coord(
            'percentiles_over_neighbourhood'), iris.coords.Coord)
        self.assertArrayEqual(result.coord(
            'percentiles_over_neighbourhood').points, DEFAULT_PERCENTILES)


class Test_run(IrisTest):

    """Test neighbourhood circular percentile plugin."""

    def test_basic(self):
        """Test that the plugin returns an iris.cube.Cube."""
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        radius = 4000.
        result = (
            GeneratePercentilesFromACircularNeighbourhood().run(
                cube, radius))
        self.assertIsInstance(result, Cube)

    def test_single_point(self):
        """Test behaviour for a single non-zero grid cell."""
        cube = set_up_cube()
        expected = np.ones([15] + list(np.shape(cube.data)))
        expected = np.transpose(expected, [1, 0, 2, 3, 4])
        expected[0, :, 0, 7, 5] = PERCENTILES_1_IN_13
        expected[0, :, 0, 6:9, 6] = (
            np.transpose(np.tile(PERCENTILES_1_IN_13, (3, 1))))
        expected[0, :, 0, 5:10, 7] = (
            np.transpose(np.tile(PERCENTILES_1_IN_13, (5, 1))))
        expected[0, :, 0, 6:9, 8] = (
            np.transpose(np.tile(PERCENTILES_1_IN_13, (3, 1))))
        expected[0, :, 0, 7, 9] = PERCENTILES_1_IN_13
        radius = 4000.
        result = (
            GeneratePercentilesFromACircularNeighbourhood().run(
                cube, radius))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_multi_point_multitimes(self):
        """Test behaviour for points over multiple times."""
        cube = set_up_cube(
            zero_point_indices=[(0, 0, 10, 10), (0, 1, 7, 7)],
            num_time_points=2
        )
        expected = np.ones([15] + list(np.shape(cube.data)))
        expected = np.transpose(expected, [1, 0, 2, 3, 4])
        expected[0, :, 0, 10, 8] = PERCENTILES_1_IN_13
        expected[0, :, 0, 9:12, 9] = (
            np.transpose(np.tile(PERCENTILES_1_IN_13, (3, 1))))
        expected[0, :, 0, 8:13, 10] = (
            np.transpose(np.tile(PERCENTILES_1_IN_13, (5, 1))))
        expected[0, :, 0, 9:12, 11] = (
            np.transpose(np.tile(PERCENTILES_1_IN_13, (3, 1))))
        expected[0, :, 0, 10, 12] = PERCENTILES_1_IN_13
        expected[0, :, 1, 7, 5] = PERCENTILES_1_IN_13
        expected[0, :, 1, 6:9, 6] = (
            np.transpose(np.tile(PERCENTILES_1_IN_13, (3, 1))))
        expected[0, :, 1, 5:10, 7] = (
            np.transpose(np.tile(PERCENTILES_1_IN_13, (5, 1))))
        expected[0, :, 1, 6:9, 8] = (
            np.transpose(np.tile(PERCENTILES_1_IN_13, (3, 1))))
        expected[0, :, 1, 7, 9] = PERCENTILES_1_IN_13
        radius = 4000.
        result = (
            GeneratePercentilesFromACircularNeighbourhood().run(
                cube, radius))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_lat_long(self):
        """Test behaviour for a single grid cell on lat long grid."""
        cube = set_up_cube_lat_long()
        msg = "Invalid grid: projection_x/y coords required"
        radius = 6000.
        with self.assertRaisesRegexp(ValueError, msg):
            GeneratePercentilesFromACircularNeighbourhood().run(cube, radius)

    def test_single_point_masked_to_null(self):
        """Test behaviour with a masked non-zero point.
        The behaviour here is not right, as the mask is ignored.
        This comes directly from the numpy.percentile base
        behaviour.
        """
        cube = set_up_cube()
        mask = np.zeros_like(cube.data)
        mask[0, 0, 7, 7] = 1
        cube.data = np.ma.masked_array(cube.data, mask=mask)
        expected = np.ones([15] + list(np.shape(cube.data)))
        expected = np.transpose(expected, [1, 0, 2, 3, 4])
        expected[0, :, 0, 7, 5] = PERCENTILES_1_IN_13
        expected[0, :, 0, 6:9, 6] = (
            np.transpose(np.tile(PERCENTILES_1_IN_13, (3, 1))))
        expected[0, :, 0, 5:10, 7] = (
            np.transpose(np.tile(PERCENTILES_1_IN_13, (5, 1))))
        expected[0, :, 0, 6:9, 8] = (
            np.transpose(np.tile(PERCENTILES_1_IN_13, (3, 1))))
        expected[0, :, 0, 7, 9] = PERCENTILES_1_IN_13
        radius = 4000.
        result = (
            GeneratePercentilesFromACircularNeighbourhood().run(
                cube, radius))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_masked_other_point(self):
        """Test behaviour with a non-zero point next to a masked point.
        The behaviour here is not right, as the mask is ignored.
        """
        cube = set_up_cube()
        mask = np.zeros_like(cube.data)
        mask[0, 0, 6, 7] = 1
        cube.data = np.ma.masked_array(cube.data, mask=mask)
        expected = np.ones([15] + list(np.shape(cube.data)))
        expected = np.transpose(expected, [1, 0, 2, 3, 4])
        expected[0, :, 0, 7, 5] = PERCENTILES_1_IN_13
        expected[0, :, 0, 6:9, 6] = (
            np.transpose(np.tile(PERCENTILES_1_IN_13, (3, 1))))
        expected[0, :, 0, 5:10, 7] = (
            np.transpose(np.tile(PERCENTILES_1_IN_13, (5, 1))))
        expected[0, :, 0, 6:9, 8] = (
            np.transpose(np.tile(PERCENTILES_1_IN_13, (3, 1))))
        expected[0, :, 0, 7, 9] = PERCENTILES_1_IN_13
        radius = 4000.
        result = (
            GeneratePercentilesFromACircularNeighbourhood().run(
                cube, radius))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_range_1(self):
        """Test behaviour with a non-zero point with unit range."""
        cube = set_up_cube()
        expected = np.ones([15] + list(np.shape(cube.data)))
        expected = np.transpose(expected, [1, 0, 2, 3, 4])
        expected[0, :, 0, 7, 6] = PERCENTILES_1_IN_5
        expected[0, :, 0, 6:9, 7] = (
            np.transpose(np.tile(PERCENTILES_1_IN_5, (3, 1))))
        expected[0, :, 0, 7, 8] = PERCENTILES_1_IN_5
        radius = 2000.
        result = (
            GeneratePercentilesFromACircularNeighbourhood().run(
                cube, radius))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_range_0(self):
        """Test behaviour with zero range."""
        cube = set_up_cube()
        radius = 0.
        msg = "Distance of {0}m gives zero cell extent".format(radius)
        with self.assertRaisesRegexp(ValueError, msg):
            GeneratePercentilesFromACircularNeighbourhood().run(cube, radius)

    def test_point_pair(self):
        """Test behaviour for two nearby non-zero grid cells."""
        cube = set_up_cube(
            zero_point_indices=[(0, 0, 7, 6), (0, 0, 7, 8)])
        expected = np.ones([15] + list(np.shape(cube.data)))
        expected = np.transpose(expected, [1, 0, 2, 3, 4])
        expected[0, :, 0, 7, 4] = PERCENTILES_1_IN_13
        expected[0, :, 0, 6:9, 5] = (
            np.transpose(np.tile(PERCENTILES_1_IN_13, (3, 1))))
        expected[0, :, 0, 5:10, 6] = (
            np.transpose(np.tile(PERCENTILES_1_IN_13, (5, 1))))
        expected[0, :, 0, 7, 6] = PERCENTILES_2_IN_13
        expected[0, :, 0, 6:9, 7] = (
            np.transpose(np.tile(PERCENTILES_2_IN_13, (3, 1))))
        expected[0, :, 0, 5:10, 8] = (
            np.transpose(np.tile(PERCENTILES_1_IN_13, (5, 1))))
        expected[0, :, 0, 7, 8] = PERCENTILES_2_IN_13
        expected[0, :, 0, 6:9, 9] = (
            np.transpose(np.tile(PERCENTILES_1_IN_13, (3, 1))))
        expected[0, :, 0, 7, 10] = PERCENTILES_1_IN_13
        radius = 4000.
        result = (
            GeneratePercentilesFromACircularNeighbourhood().run(
                cube, radius))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_almost_edge(self):
        """Test behaviour for a non-zero grid cell quite near the edge."""
        cube = set_up_cube(
            zero_point_indices=[
                (0, 0, 7, 2)])  # Just within range of the edge.
        border_rows_1 = PERCENTILES_1_IN_25.copy()
        border_rows_1[1] = 0.8
        border_rows_2 = PERCENTILES_1_IN_25.copy()
        border_rows_2[1:3] = [2./3., 14./15.]
        border_rows_3 = PERCENTILES_1_IN_25.copy()
        border_rows_3[1:3] = [2./3., 2./3.]
        expected = np.ones([15] + list(np.shape(cube.data)))
        expected = np.transpose(expected, [1, 0, 2, 3, 4])
        expected[0, :, 0, 7, 5] = PERCENTILES_1_IN_25
        expected[0, :, 0, 5:10, 4] = (
            np.transpose(np.tile(PERCENTILES_1_IN_25, (5, 1))))
        expected[0, :, 0, 5:10, 3] = (
            np.transpose(np.tile(PERCENTILES_1_IN_25, (5, 1))))
        expected[0, :, 0, 4:11, 2] = (
            np.transpose(np.tile(PERCENTILES_1_IN_25, (7, 1))))
        expected[0, :, 0, 7, 2] = border_rows_1
        expected[0, :, 0, 5:10, 1] = (
            np.transpose(np.tile(border_rows_1, (5, 1))))
        expected[0, :, 0, 7, 1] = border_rows_2
        expected[0, :, 0, 5:10, 0] = (
            np.transpose(np.tile(border_rows_2, (5, 1))))
        expected[0, :, 0, 7, 0] = border_rows_3
        radius = 6000.
        result = (
            GeneratePercentilesFromACircularNeighbourhood().run(
                cube, radius))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_adjacent_edge(self):
        """Test behaviour for a single non-zero grid cell near the edge."""
        cube = set_up_cube(
            zero_point_indices=[(0, 0, 7, 1)])  # Range 3 goes over the edge.
        border_rows_1 = PERCENTILES_1_IN_25.copy()
        border_rows_1[1] = 0.8
        border_rows_2 = PERCENTILES_1_IN_25.copy()
        border_rows_2[1:3] = [2./3., 14./15.]
        border_rows_3 = PERCENTILES_1_IN_25.copy()
        border_rows_3[1:3] = [2./3., 2./3.]
        expected = np.ones([15] + list(np.shape(cube.data)))
        expected = np.transpose(expected, [1, 0, 2, 3, 4])
        expected[0, :, 0, 7, 4] = PERCENTILES_1_IN_25
        expected[0, :, 0, 5:10, 3] = (
            np.transpose(np.tile(PERCENTILES_1_IN_25, (5, 1))))
        expected[0, :, 0, 5:10, 2] = (
            np.transpose(np.tile(PERCENTILES_1_IN_25, (5, 1))))
        expected[0, :, 0, 7, 2] = border_rows_1
        expected[0, :, 0, 4:11, 1] = (
            np.transpose(np.tile(PERCENTILES_1_IN_25, (7, 1))))
        expected[0, :, 0, 5:10, 1] = (
            np.transpose(np.tile(border_rows_1, (5, 1))))
        expected[0, :, 0, 7, 1] = border_rows_2
        expected[0, :, 0, 5:10, 0] = (
            np.transpose(np.tile(border_rows_2, (5, 1))))
        expected[0, :, 0, 7, 0] = border_rows_3
        radius = 6000.
        result = (
            GeneratePercentilesFromACircularNeighbourhood().run(
                cube, radius))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_on_edge(self):
        """Test behaviour for a non-zero grid cell on the edge.
        Note that this behaviour is 'wrong' and is a result of
        scipy.ndimage.correlate 'nearest' mode. We need to fix
        this in the future.
        """
        cube = set_up_cube(
            zero_point_indices=[(0, 0, 7, 0)])  # On the (y) edge.
        border_rows_1 = PERCENTILES_1_IN_25.copy()
        border_rows_1[1] = 0.8
        border_rows_2 = PERCENTILES_1_IN_25.copy()
        border_rows_2[1:3] = [2./3., 14./15.]
        border_rows_3 = PERCENTILES_1_IN_25.copy()
        border_rows_3[1:3] = [2./3., 2./3.]
        expected = np.ones([15] + list(np.shape(cube.data)))
        expected = np.transpose(expected, [1, 0, 2, 3, 4])
        expected[0, :, 0, 7, 3] = PERCENTILES_1_IN_25
        expected[0, :, 0, 5:10, 2] = (
            np.transpose(np.tile(PERCENTILES_1_IN_25, (5, 1))))
        expected[0, :, 0, 7, 2] = border_rows_1
        expected[0, :, 0, 5:10, 1] = (
            np.transpose(np.tile(border_rows_1, (5, 1))))
        expected[0, :, 0, 7, 1] = border_rows_2
        expected[0, :, 0, 4:11, 0] = (
            np.transpose(np.tile(PERCENTILES_1_IN_25, (7, 1))))
        expected[0, :, 0, 5:10, 0] = (
            np.transpose(np.tile(border_rows_2, (5, 1))))
        expected[0, :, 0, 7, 0] = border_rows_3
        radius = 6000.
        result = (
            GeneratePercentilesFromACircularNeighbourhood().run(
                cube, radius))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_on_corner(self):
        """Test behaviour for a single non-zero grid cell on the corner.
        Note that this behaviour is 'wrong' and is a result of
        scipy.ndimage.correlate 'nearest' mode. We need to fix
        this in the future.
        """
        cube = set_up_cube(
            zero_point_indices=[(0, 0, 0, 0)])  # Point is right on the corner.
        border_rows_1 = PERCENTILES_1_IN_25.copy()
        border_rows_1[1] = 0.8
        border_rows_3 = PERCENTILES_1_IN_25.copy()
        border_rows_3[1:3] = [2./3., 2./3.]
        border_rows_4 = PERCENTILES_1_IN_25.copy()
        border_rows_4[1:3] = [2./3., 38./45.]
        border_rows_5 = PERCENTILES_1_IN_25.copy()
        border_rows_5[1:4] = [2./3., 2./3., 8./9.]
        border_rows_6 = PERCENTILES_1_IN_25.copy()
        border_rows_6[1:3] = [2./3., 2./3.]
        border_rows_7 = PERCENTILES_1_IN_25.copy()
        border_rows_7[1:6] = [2./3., 2./3., 2./3., 8./9., 8./9.]
        expected = np.ones([15] + list(np.shape(cube.data)))
        expected = np.transpose(expected, [1, 0, 2, 3, 4])
        expected[0, :, 0, 0, 3] = PERCENTILES_1_IN_25
        expected[0, :, 0, 2, 2] = PERCENTILES_1_IN_25
        expected[0, :, 0, 1, 2] = border_rows_1
        expected[0, :, 0, 0, 2] = border_rows_3
        expected[0, :, 0, 2, 1] = border_rows_1
        expected[0, :, 0, 1, 1] = border_rows_4
        expected[0, :, 0, 0, 1] = border_rows_5
        expected[0, :, 0, 3, 0] = PERCENTILES_1_IN_25
        expected[0, :, 0, 2, 0] = border_rows_6
        expected[0, :, 0, 1, 0] = border_rows_5
        expected[0, :, 0, 0, 0] = border_rows_7
        radius = 6000.
        result = (
            GeneratePercentilesFromACircularNeighbourhood().run(
                cube, radius))
        self.assertArrayAlmostEqual(result.data, expected)


if __name__ == '__main__':
    unittest.main()
