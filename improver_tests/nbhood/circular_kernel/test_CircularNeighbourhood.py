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
"""Unit tests for the nbhood.circular_kernel.CircularNeighbourhood plugin."""


import unittest

import numpy as np
from iris.cube import Cube
from iris.tests import IrisTest

from improver.nbhood.circular_kernel import CircularNeighbourhood

from ..nbhood.test_BaseNeighbourhoodProcessing import (
    SINGLE_POINT_RANGE_2_CENTROID_FLAT, SINGLE_POINT_RANGE_3_CENTROID,
    SINGLE_POINT_RANGE_5_CENTROID, set_up_cube)


class Test__init__(IrisTest):

    """Test the init method."""

    def test_sum_or_fraction(self):
        """Test that a ValueError is raised if an invalid option is passed
        in for sum_or_fraction."""
        sum_or_fraction = "nonsense"
        msg = "option is invalid"
        with self.assertRaisesRegex(ValueError, msg):
            CircularNeighbourhood(sum_or_fraction=sum_or_fraction)


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(CircularNeighbourhood())
        msg = ('<CircularNeighbourhood: weighted_mode: True, '
               'sum_or_fraction: fraction>')
        self.assertEqual(str(result), msg)


class Test_apply_circular_kernel(IrisTest):

    """Test neighbourhood circular probabilities plugin."""

    def test_basic(self):
        """Test that the plugin returns an iris.cube.Cube."""
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_time_points=1,
            num_grid_points=5)
        ranges = 2
        result = (
            CircularNeighbourhood(
                weighted_mode=False).apply_circular_kernel(cube, ranges))
        self.assertIsInstance(result, Cube)

    def test_single_point(self):
        """Test behaviour for a single non-zero grid cell."""
        cube = set_up_cube()
        expected = np.ones_like(cube.data)
        for index, slice_ in enumerate(SINGLE_POINT_RANGE_3_CENTROID):
            expected[0][0][5 + index][5:10] = slice_
        ranges = 3
        result = (
            CircularNeighbourhood(
                weighted_mode=True).apply_circular_kernel(cube, ranges))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_flat(self):
        """Test behaviour for a single non-zero grid cell, flat weighting.
        Note that this gives one more grid cell range than weighted! As the
        affected area is one grid cell more in each direction, an equivalent
        range of 2 was chosen for this test."""
        cube = set_up_cube()
        expected = np.ones_like(cube.data)
        for index, slice_ in enumerate(SINGLE_POINT_RANGE_2_CENTROID_FLAT):
            expected[0][0][5 + index][5:10] = slice_
        ranges = 2
        result = (
            CircularNeighbourhood(
                weighted_mode=False).apply_circular_kernel(cube, ranges))
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
        ranges = 3
        result = (
            CircularNeighbourhood(
                weighted_mode=True).apply_circular_kernel(cube, ranges))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_masked_to_null(self):
        """Test behaviour with a masked non-zero point. The behaviour here is
        not right, as the mask is ignored. This comes directly from the
        scipy.ndimage.correlate base behaviour."""
        cube = set_up_cube()
        expected = np.ones_like(cube.data)
        mask = np.zeros_like(cube.data)
        mask[0][0][7][7] = 1
        cube.data = np.ma.masked_array(cube.data, mask=mask)
        for time_index in range(len(expected)):
            for index, slice_ in enumerate(SINGLE_POINT_RANGE_3_CENTROID):
                expected[0][time_index][5 + index][5:10] = slice_
        ranges = 3
        result = (
            CircularNeighbourhood(
                weighted_mode=True).apply_circular_kernel(cube, ranges))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_masked_other_point(self):
        """Test behaviour with a non-zero point next to a masked point.
        The behaviour here is not right, as the mask is ignored."""
        cube = set_up_cube()
        expected = np.ones_like(cube.data)
        mask = np.zeros_like(cube.data)
        mask[0][0][6][7] = 1
        cube.data = np.ma.masked_array(cube.data, mask=mask)
        for time_index in range(len(expected)):
            for index, slice_ in enumerate(SINGLE_POINT_RANGE_3_CENTROID):
                expected[0][time_index][5 + index][5:10] = slice_
        ranges = 3
        result = (
            CircularNeighbourhood(
                weighted_mode=True).apply_circular_kernel(cube, ranges))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_range_1(self):
        """Test behaviour with a non-zero point with unit range."""
        cube = set_up_cube()
        expected = np.ones_like(cube.data)
        expected[0][0][7][7] = 0.0
        ranges = 1
        result = (
            CircularNeighbourhood(
                weighted_mode=True).apply_circular_kernel(cube, ranges))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_range_5(self):
        """Test behaviour with a non-zero point with a large range."""
        cube = set_up_cube()
        expected = np.ones_like(cube.data)
        for time_index in range(len(expected)):
            for index, slice_ in enumerate(SINGLE_POINT_RANGE_5_CENTROID):
                expected[0][time_index][3 + index][3:12] = slice_
        ranges = 5
        result = (
            CircularNeighbourhood(
                weighted_mode=True).apply_circular_kernel(cube, ranges))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_range_5_small_domain(self):
        """Test behaviour - non-zero point, small domain, large range.
        This exhibits the undesirable edge reflection behaviour."""
        cube = set_up_cube(
            zero_point_indices=((0, 0, 1, 1),), num_grid_points=4)
        expected = np.array([
            [[[0.97636177, 0.97533402, 0.97636177, 0.97944502],
              [0.97533402, 0.97430627, 0.97533402, 0.97841727],
              [0.97636177, 0.97533402, 0.97636177, 0.97944502],
              [0.97944502, 0.97841727, 0.97944502, 0.98252826]]]
        ])
        ranges = 5
        result = (
            CircularNeighbourhood(
                weighted_mode=True).apply_circular_kernel(cube, ranges))
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
        ranges = 3
        result = (
            CircularNeighbourhood(
                weighted_mode=True).apply_circular_kernel(cube, ranges))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_almost_edge(self):
        """Test behaviour for a non-zero grid cell quite near the edge."""
        cube = set_up_cube(
            zero_point_indices=[
                (0, 0, 7, 2)])  # Just within range of the edge.
        expected = np.ones_like(cube.data)
        for index, slice_ in enumerate(SINGLE_POINT_RANGE_3_CENTROID):
            expected[0][0][5 + index][0:5] = slice_
        ranges = 3
        result = (
            CircularNeighbourhood(
                weighted_mode=True).apply_circular_kernel(cube, ranges))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_adjacent_edge(self):
        """Test behaviour for a single non-zero grid cell near the edge."""
        cube = set_up_cube(
            zero_point_indices=[(0, 0, 7, 1)])  # Range 3 goes over the edge.
        expected = np.ones_like(cube.data)
        for index, slice_ in enumerate(SINGLE_POINT_RANGE_3_CENTROID):
            expected[0][0][5 + index][0:4] = slice_[1:]
        ranges = 3
        result = (
            CircularNeighbourhood(
                weighted_mode=True).apply_circular_kernel(cube, ranges))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_on_edge(self):
        """Test behaviour for a non-zero grid cell on the edge."""
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
        ranges = 3
        result = (
            CircularNeighbourhood(
                weighted_mode=True).apply_circular_kernel(cube, ranges))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_almost_corner(self):
        """Test behaviour for a non-zero grid cell quite near a corner."""
        cube = set_up_cube(
            zero_point_indices=[(0, 0, 2, 2)])  # Just within corner range.
        expected = np.ones_like(cube.data)
        for index, slice_ in enumerate(SINGLE_POINT_RANGE_3_CENTROID):
            expected[0][0][index][0:5] = slice_
        ranges = 3
        result = (
            CircularNeighbourhood(
                weighted_mode=True).apply_circular_kernel(cube, ranges))
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
        ranges = 3
        result = (
            CircularNeighbourhood(
                weighted_mode=True).apply_circular_kernel(cube, ranges))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_single_point_on_corner(self):
        """Test behaviour for a single non-zero grid cell on the corner."""
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
        ranges = 3
        result = (
            CircularNeighbourhood(
                weighted_mode=True).apply_circular_kernel(cube, ranges))
        self.assertArrayAlmostEqual(result.data, expected)


class Test_run(IrisTest):

    """Test the run method on the CircularNeighbourhood class."""

    RADIUS = 6100

    def test_basic(self):
        """Test that a cube with correct data is produced by the run method"""
        data = np.array([[0.992, 0.968, 0.96, 0.968, 0.992],
                         [0.968, 0.944, 0.936, 0.944, 0.968],
                         [0.96, 0.936, 0.928, 0.936, 0.96],
                         [0.968, 0.944, 0.936, 0.944, 0.968],
                         [0.992, 0.968, 0.96, 0.968, 0.992]])

        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_grid_points=5)[0, 0]
        result = CircularNeighbourhood().run(cube, self.RADIUS)
        self.assertIsInstance(cube, Cube)
        self.assertArrayAlmostEqual(result.data, data)

    def test_mask_cube(self):
        """Test that a NotImplementedError is raised, if a mask cube is passed
        in when using a circular neighbourhood, as this option is not
        supported."""
        cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_grid_points=5)[0, 0]
        msg = ("The use of a mask cube with a circular kernel is "
               "not yet implemented.")
        with self.assertRaisesRegex(NotImplementedError, msg):
            CircularNeighbourhood().run(cube, self.RADIUS, mask_cube=cube)


if __name__ == '__main__':
    unittest.main()
