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
"""Unit tests for the nbhood.Kernels plugin."""


import unittest

from iris.cube import Cube
from iris.tests import IrisTest
import numpy as np

from improver.nbhood import Kernels
from improver.tests.test_nbhood_neighbourhoodprocessing import (
    SINGLE_POINT_RANGE_2_CENTROID_FLAT, SINGLE_POINT_RANGE_3_CENTROID,
    SINGLE_POINT_RANGE_5_CENTROID, set_up_cube, set_up_cube_lat_long)


class Test_circular(IrisTest):

    """Test neighbourhood processing plugin on the OS National Grid."""

    RADIUS_IN_KM = 6.3  # Gives 3 grid cells worth.
    KERNEL_METHOD = "circular"

    def test_basic(self):
        """Test that the plugin returns an iris.cube.Cube."""
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        ranges = (2, 2)
        result = Kernels.circular(cube, ranges, unweighted_mode=True)
        self.assertIsInstance(result, Cube)

    def test_single_point(self):
        """Test behaviour for a single non-zero grid cell."""
        cube = set_up_cube()
        expected = np.ones_like(cube.data)
        for index, slice_ in enumerate(SINGLE_POINT_RANGE_3_CENTROID):
            expected[0][0][5 + index][5:10] = slice_
        ranges = (3, 3)
        result = Kernels.circular(cube, ranges, unweighted_mode=False)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_flat(self):
        """Test behaviour for a single non-zero grid cell, flat weighting.

        Note that this gives one more grid cell range than weighted! As the
        affected area is one grid cell more in each direction, an equivalent
        range of 2 was chosen for this test.

        """
        cube = set_up_cube()
        expected = np.ones_like(cube.data)
        for index, slice_ in enumerate(SINGLE_POINT_RANGE_2_CENTROID_FLAT):
            expected[0][0][5 + index][5:10] = slice_
        radius_in_km = 4.2  # Equivalent to a range of 2.
        ranges = (2, 2)
        result = Kernels.circular(cube, ranges, unweighted_mode=True)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_multi_point_multitimes(self):
        """Test behaviour for points over multiple times."""
        cube = set_up_cube(
            zero_point_indices=[(0, 0, 10, 10), (0, 1, 7, 7)],
            num_time_points=2
        )
        expected = np.ones_like(cube.data)
        for index, slice_ in enumerate(SINGLE_POINT_RANGE_3_CENTROID):
            expected[0][0][8 + index][8:13] = slice_
        for index, slice_ in enumerate(SINGLE_POINT_RANGE_3_CENTROID):
            expected[0][1][5 + index][5:10] = slice_
        ranges = (3, 3)
        result = Kernels.circular(cube, ranges, unweighted_mode=False)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_lat_long(self):
        """Test behaviour for a single grid cell on lat long grid."""
        cube = set_up_cube_lat_long()
        msg = "Invalid grid: projection_x/y coords required"
        expected = np.zeros_like(cube.data)
        ranges = (3, 3)
        with self.assertRaisesRegexp(ValueError, msg):
            Kernels.circular(cube, ranges, unweighted_mode=False)

    def test_single_point_masked_to_null(self):
        """Test behaviour with a masked non-zero point.

        The behaviour here is not right, as the mask is ignored.
        This comes directly from the scipy.ndimage.correlate base
        behaviour.

        """
        cube = set_up_cube()
        expected = np.ones_like(cube.data)
        mask = np.zeros_like(cube.data)
        mask[0][0][7][7] = 1
        cube.data = np.ma.masked_array(cube.data, mask=mask)
        for time_index in range(len(expected)):
            for index, slice_ in enumerate(SINGLE_POINT_RANGE_3_CENTROID):
                expected[0][time_index][5 + index][5:10] = slice_
        ranges = (3, 3)
        result = Kernels.circular(cube, ranges, unweighted_mode=False)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_masked_other_point(self):
        """Test behaviour with a non-zero point next to a masked point.

        The behaviour here is not right, as the mask is ignored.

        """
        cube = set_up_cube()
        expected = np.ones_like(cube.data)
        mask = np.zeros_like(cube.data)
        mask[0][0][6][7] = 1
        cube.data = np.ma.masked_array(cube.data, mask=mask)
        for time_index in range(len(expected)):
            for index, slice_ in enumerate(SINGLE_POINT_RANGE_3_CENTROID):
                expected[0][time_index][5 + index][5:10] = slice_
        ranges = (3, 3)
        result = Kernels.circular(cube, ranges, unweighted_mode=False)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_range_1(self):
        """Test behaviour with a non-zero point with unit range."""
        cube = set_up_cube()
        expected = np.ones_like(cube.data)
        expected[0][0][7][7] = 0.0
        radius_in_km = 2.1  # Equivalent to a range of 1 grid cell.
        ranges = (1, 1)
        result = Kernels.circular(cube, ranges, unweighted_mode=False)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_range_5(self):
        """Test behaviour with a non-zero point with a large range."""
        cube = set_up_cube()
        expected = np.ones_like(cube.data)
        for time_index in range(len(expected)):
            for index, slice_ in enumerate(SINGLE_POINT_RANGE_5_CENTROID):
                expected[0][time_index][3 + index][3:12] = slice_
        radius_in_km = 10.5  # Equivalent to a range of 5 grid cells.
        ranges = (5, 5)
        result = Kernels.circular(cube, ranges, unweighted_mode=False)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_range_5_small_domain(self):
        """Test behaviour - non-zero point, small domain, large range.

        This exhibits the undesirable edge reflection behaviour.

        """
        cube = set_up_cube(
            zero_point_indices=((0, 0, 1, 1),), num_grid_points=4)
        expected = np.array([
            [[[0.97636177, 0.97533402, 0.97636177, 0.97944502],
              [0.97533402, 0.97430627, 0.97533402, 0.97841727],
              [0.97636177, 0.97533402, 0.97636177, 0.97944502],
              [0.97944502, 0.97841727, 0.97944502, 0.98252826]]]
        ])
        radius_in_km = 10.5  # Equivalent to a range of 5 grid cells.
        ranges = (5, 5)
        result = Kernels.circular(cube, ranges, unweighted_mode=False)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_point_pair(self):
        """Test behaviour for two nearby non-zero grid cells."""
        cube = set_up_cube(
            zero_point_indices=[(0, 0, 7, 6), (0, 0, 7, 8)])
        expected_snippet = np.array([
            [0.992, 0.968, 0.952, 0.936, 0.952, 0.968, 0.992],
            [0.968, 0.944, 0.904, 0.888, 0.904, 0.944, 0.968],
            [0.96, 0.936, 0.888, 0.872, 0.888, 0.936, 0.96],
            [0.968, 0.944, 0.904, 0.888, 0.904, 0.944, 0.968],
            [0.992, 0.968, 0.952, 0.936, 0.952, 0.968, 0.992]
        ])
        expected = np.ones_like(cube.data)
        for index, slice_ in enumerate(expected_snippet):
            expected[0][0][5 + index][4:11] = slice_
        ranges = (3, 3)
        result = Kernels.circular(cube, ranges, unweighted_mode=False)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_almost_edge(self):
        """Test behaviour for a non-zero grid cell quite near the edge."""
        cube = set_up_cube(
            zero_point_indices=[
                (0, 0, 7, 2)])  # Just within range of the edge.
        expected = np.ones_like(cube.data)
        for index, slice_ in enumerate(SINGLE_POINT_RANGE_3_CENTROID):
            expected[0][0][5 + index][0:5] = slice_
        ranges = (3, 3)
        result = Kernels.circular(cube, ranges, unweighted_mode=False)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_adjacent_edge(self):
        """Test behaviour for a single non-zero grid cell near the edge."""
        cube = set_up_cube(
            zero_point_indices=[(0, 0, 7, 1)])  # Range 3 goes over the edge.
        expected = np.ones_like(cube.data)
        for index, slice_ in enumerate(SINGLE_POINT_RANGE_3_CENTROID):
            expected[0][0][5 + index][0:4] = slice_[1:]
        ranges = (3, 3)
        result = Kernels.circular(cube, ranges, unweighted_mode=False)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_on_edge(self):
        """Test behaviour for a non-zero grid cell on the edge.

        Note that this behaviour is 'wrong' and is a result of
        scipy.ndimage.correlate 'nearest' mode. We need to fix
        this in the future.

        """
        cube = set_up_cube(
            zero_point_indices=[(0, 0, 7, 0)])  # On the (y) edge.
        expected = np.ones_like(cube.data)
        expected_centroid = np.array([
            [0.92, 0.96, 0.992],
            [0.848, 0.912, 0.968],
            [0.824, 0.896, 0.96],
            [0.848, 0.912, 0.968],
            [0.92, 0.96, 0.992],
        ])
        for index, slice_ in enumerate(expected_centroid):
            expected[0][0][5 + index][0:3] = slice_
        ranges = (3, 3)
        result = Kernels.circular(cube, ranges, unweighted_mode=False)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_almost_corner(self):
        """Test behaviour for a non-zero grid cell quite near a corner."""
        cube = set_up_cube(
            zero_point_indices=[(0, 0, 2, 2)])  # Just within corner range.
        expected = np.ones_like(cube.data)
        for index, slice_ in enumerate(SINGLE_POINT_RANGE_3_CENTROID):
            expected[0][0][index][0:5] = slice_
        ranges = (3, 3)
        result = Kernels.circular(cube, ranges, unweighted_mode=False)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_adjacent_corner(self):
        """Test behaviour for a non-zero grid cell near the corner."""
        cube = set_up_cube(
            zero_point_indices=[(0, 0, 1, 1)])  # Kernel goes over the corner.
        expected = np.ones_like(cube.data)
        for index, slice_ in enumerate(SINGLE_POINT_RANGE_3_CENTROID):
            if index == 0:
                continue
            expected[0][0][index - 1][0:4] = slice_[1:]
        ranges = (3, 3)
        result = Kernels.circular(cube, ranges, unweighted_mode=False)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_on_corner(self):
        """Test behaviour for a single non-zero grid cell on the corner.

        Note that this behaviour is 'wrong' and is a result of
        scipy.ndimage.correlate 'nearest' mode. We need to fix
        this in the future.

        """
        cube = set_up_cube(
            zero_point_indices=[(0, 0, 0, 0)])  # Point is right on the corner.
        expected = np.ones_like(cube.data)
        expected_centroid = np.array([
            [0.592, 0.768, 0.92],
            [0.768, 0.872, 0.96],
            [0.92, 0.96, 0.992],
        ])
        for index, slice_ in enumerate(expected_centroid):
            expected[0][0][index][0:3] = slice_
        ranges = (3, 3)
        result = Kernels.circular(cube, ranges, unweighted_mode=False)
        self.assertArrayAlmostEqual(result.data, expected)


if __name__ == '__main__':
    unittest.main()
