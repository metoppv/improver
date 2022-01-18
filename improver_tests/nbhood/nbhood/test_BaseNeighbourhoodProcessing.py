# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""Unit tests for the nbhood.BaseNeighbourhoodProcessing plugin."""


import unittest
from datetime import datetime

import numpy as np
from iris.cube import Cube
from iris.tests import IrisTest

from improver.nbhood.circular_kernel import CircularNeighbourhood
from improver.nbhood.nbhood import BaseNeighbourhoodProcessing as NBHood
from improver.nbhood.nbhood import SquareNeighbourhood
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_probability_cube,
    set_up_variable_cube,
)

SINGLE_POINT_RANGE_3_CENTROID = np.array(
    [
        [0.992, 0.968, 0.96, 0.968, 0.992],
        [0.968, 0.944, 0.936, 0.944, 0.968],
        [0.96, 0.936, 0.928, 0.936, 0.96],
        [0.968, 0.944, 0.936, 0.944, 0.968],
        [0.992, 0.968, 0.96, 0.968, 0.992],
    ]
)

SINGLE_POINT_RANGE_2_CENTROID_FLAT = np.array(
    [
        [1.0, 1.0, 0.92307692, 1.0, 1.0],
        [1.0, 0.92307692, 0.92307692, 0.92307692, 1.0],
        [0.92307692, 0.92307692, 0.92307692, 0.92307692, 0.92307692],
        [1.0, 0.92307692, 0.92307692, 0.92307692, 1.0],
        [1.0, 1.0, 0.92307692, 1.0, 1.0],
    ]
)

SINGLE_POINT_RANGE_5_CENTROID = np.array(
    [
        [
            1.0,
            1.0,
            0.99486125,
            0.99177801,
            0.99075026,
            0.99177801,
            0.99486125,
            1.0,
            1.0,
        ],
        [
            1.0,
            0.99280576,
            0.98766701,
            0.98458376,
            0.98355601,
            0.98458376,
            0.98766701,
            0.99280576,
            1.0,
        ],
        [
            0.99486125,
            0.98766701,
            0.98252826,
            0.97944502,
            0.97841727,
            0.97944502,
            0.98252826,
            0.98766701,
            0.99486125,
        ],
        [
            0.99177801,
            0.98458376,
            0.97944502,
            0.97636177,
            0.97533402,
            0.97636177,
            0.97944502,
            0.98458376,
            0.99177801,
        ],
        [
            0.99075026,
            0.98355601,
            0.97841727,
            0.97533402,
            0.97430627,
            0.97533402,
            0.97841727,
            0.98355601,
            0.99075026,
        ],
        [
            0.99177801,
            0.98458376,
            0.97944502,
            0.97636177,
            0.97533402,
            0.97636177,
            0.97944502,
            0.98458376,
            0.99177801,
        ],
        [
            0.99486125,
            0.98766701,
            0.98252826,
            0.97944502,
            0.97841727,
            0.97944502,
            0.98252826,
            0.98766701,
            0.99486125,
        ],
        [
            1.0,
            0.99280576,
            0.98766701,
            0.98458376,
            0.98355601,
            0.98458376,
            0.98766701,
            0.99280576,
            1.0,
        ],
        [
            1.0,
            1.0,
            0.99486125,
            0.99177801,
            0.99075026,
            0.99177801,
            0.99486125,
            1.0,
            1.0,
        ],
    ]
)


class Test__init__(IrisTest):

    """Test the __init__ method of NeighbourhoodProcessing"""

    def test_radii_varying_with_lead_time_mismatch(self):
        """
        Test that the desired error message is raised, if there is a mismatch
        between the number of radii and the number of lead times.
        """
        radii = [10000, 20000, 30000]
        lead_times = [2, 3]
        msg = "There is a mismatch in the number of radii"
        with self.assertRaisesRegex(ValueError, msg):
            neighbourhood_method = CircularNeighbourhood()
            NBHood(neighbourhood_method, radii, lead_times=lead_times)


class Test__find_radii(IrisTest):

    """Test the internal _find_radii function is working correctly."""

    def test_basic_float_cube_lead_times_is_none(self):
        """Test _find_radii returns an unaltered radius if
        the lead times are none, and this radius is a float."""
        neighbourhood_method = CircularNeighbourhood()
        radius = 6300
        plugin = NBHood(neighbourhood_method, radius)
        result = plugin._find_radii(cube_lead_times=None)
        expected_result = 6300.0
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, expected_result)

    def test_basic_array_cube_lead_times_an_array(self):
        """Test _find_radii returns an array with the correct values."""
        neighbourhood_method = CircularNeighbourhood
        fp_points = np.array([2, 3, 4])
        radii = [10000, 20000, 30000]
        lead_times = [1, 3, 5]
        plugin = NBHood(neighbourhood_method(), radii, lead_times=lead_times)
        result = plugin._find_radii(cube_lead_times=fp_points)
        expected_result = np.array([15000.0, 20000.0, 25000.0])
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_result)

    def test_interpolation(self):
        """Test that interpolation is working as expected in _find_radii."""
        neighbourhood_method = CircularNeighbourhood
        fp_points = np.array([2, 3, 4])
        radii = [10000, 30000]
        lead_times = [2, 4]
        plugin = NBHood(neighbourhood_method(), radii=radii, lead_times=lead_times)
        result = plugin._find_radii(cube_lead_times=fp_points)
        expected_result = np.array([10000.0, 20000.0, 30000.0])
        self.assertArrayAlmostEqual(result, expected_result)


class Test_process(IrisTest):

    """Tests for the process method of NeighbourhoodProcessing."""

    RADIUS = 6300  # Gives 3 grid cells worth.

    def setUp(self):
        """Set up cube."""
        data = np.ones((16, 16), dtype=np.float32)
        data[7, 7] = 0
        self.cube = set_up_variable_cube(data, spatial_grid="equalarea",)

        time_points = [
            datetime(2017, 11, 10, 2),
            datetime(2017, 11, 10, 3),
            datetime(2017, 11, 10, 4),
        ]
        self.multi_time_cube = add_coordinate(
            self.cube, coord_points=time_points, coord_name="time", is_datetime="true",
        )

    def test_basic(self):
        """Test that the plugin returns an iris.cube.Cube."""
        neighbourhood_method = CircularNeighbourhood()
        result = NBHood(neighbourhood_method, self.RADIUS)(self.cube)
        self.assertIsInstance(result, Cube)

    def test_neighbourhood_method_does_not_exist(self):
        """
        Test that desired error message is raised, if the neighbourhood method
        does not exist.
        """
        neighbourhood_method = "nonsense"
        radii = 10000
        msg = "is not valid as a neighbourhood_method"
        with self.assertRaisesRegex(ValueError, msg):
            NBHood(neighbourhood_method, radii)(self.cube)

    def test_single_point_nan(self):
        """Test behaviour for a single NaN grid cell."""
        self.cube.data[6][7] = np.NAN
        msg = "NaN detected in input cube data"
        with self.assertRaisesRegex(ValueError, msg):
            neighbourhood_method = CircularNeighbourhood
            NBHood(neighbourhood_method, self.RADIUS)(self.cube)

    def test_multiple_thresholds(self):
        """Test when the cube has a threshold dimension."""

        data = np.ones((4, 16, 16), dtype=np.float32)
        data[0, 7, 7] = 0
        cube = set_up_probability_cube(
            data,
            thresholds=np.array([278, 277, 276, 275], dtype=np.float32),
            spatial_grid="equalarea",
        )
        radii = 5600
        neighbourhood_method = CircularNeighbourhood()
        result = NBHood(neighbourhood_method, radii)(cube)
        self.assertIsInstance(result, Cube)
        expected = np.ones([4, 16, 16], dtype=np.float32)
        expected[0, 6:9, 6:9] = (
            [0.9166666, 0.875, 0.9166666],
            [0.875, 0.8333333, 0.875],
            [0.9166666, 0.875, 0.9166666],
        )
        self.assertArrayAlmostEqual(result.data, expected)

    def test_radii_varying_with_lead_time_check_data(self):
        """
        Test that the expected data is produced when the radius
        varies with lead time and that a cube is returned.
        """

        expected = np.ones_like(self.multi_time_cube.data)
        expected[0, 6:9, 6:9] = (
            [0.91666667, 0.875, 0.91666667],
            [0.875, 0.83333333, 0.875],
            [0.91666667, 0.875, 0.91666667],
        )

        expected[1, 5:10, 5:10] = SINGLE_POINT_RANGE_3_CENTROID

        expected[2, 4:11, 4:11] = (
            [1, 0.9925, 0.985, 0.9825, 0.985, 0.9925, 1],
            [0.9925, 0.98, 0.9725, 0.97, 0.9725, 0.98, 0.9925],
            [0.985, 0.9725, 0.965, 0.9625, 0.965, 0.9725, 0.985],
            [0.9825, 0.97, 0.9625, 0.96, 0.9625, 0.97, 0.9825],
            [0.985, 0.9725, 0.965, 0.9625, 0.965, 0.9725, 0.985],
            [0.9925, 0.98, 0.9725, 0.97, 0.9725, 0.98, 0.9925],
            [1, 0.9925, 0.985, 0.9825, 0.985, 0.9925, 1],
        )

        radii = [5600, 7600, 9500]
        lead_times = [2, 3, 4]
        neighbourhood_method = CircularNeighbourhood()
        plugin = NBHood(neighbourhood_method, radii, lead_times)
        result = plugin(self.multi_time_cube)
        self.assertArrayAlmostEqual(result.data, expected)
        self.assertIsInstance(result, Cube)

    def test_radii_varying_with_lead_time_multiple_thresholds(self):
        """Test that a cube is returned for the following conditions:
        1. The radius varies wtih lead time.
        2. The cube contains multiple thresholds."""

        data = np.ones((2, 16, 16), dtype=np.float32)
        data[1, 7, 7] = 0
        cube = set_up_probability_cube(
            data, thresholds=[278, 279], spatial_grid="equalarea",
        )

        time_points = [
            datetime(2017, 11, 10, 2),
            datetime(2017, 11, 10, 3),
            datetime(2017, 11, 10, 4),
        ]
        cube = add_coordinate(
            cube,
            coord_points=time_points,
            coord_name="time",
            is_datetime="true",
            order=[1, 0, 2, 3],
        )

        lead_times = [2, 3, 4]
        radii = [5600, 7600, 9500]
        expected = np.ones_like(cube.data)
        expected[1, 0, 6:9, 6:9] = (
            [0.91666667, 0.875, 0.91666667],
            [0.875, 0.83333333, 0.875],
            [0.91666667, 0.875, 0.91666667],
        )
        expected[1, 1, 5:10, 5:10] = SINGLE_POINT_RANGE_3_CENTROID
        expected[1, 2, 4:11, 4:11] = (
            [1, 0.9925, 0.985, 0.9825, 0.985, 0.9925, 1],
            [0.9925, 0.98, 0.9725, 0.97, 0.9725, 0.98, 0.9925],
            [0.985, 0.9725, 0.965, 0.9625, 0.965, 0.9725, 0.985],
            [0.9825, 0.97, 0.9625, 0.96, 0.9625, 0.97, 0.9825],
            [0.985, 0.9725, 0.965, 0.9625, 0.965, 0.9725, 0.985],
            [0.9925, 0.98, 0.9725, 0.97, 0.9725, 0.98, 0.9925],
            [1, 0.9925, 0.985, 0.9825, 0.985, 0.9925, 1],
        )
        neighbourhood_method = CircularNeighbourhood()
        plugin = NBHood(neighbourhood_method, radii, lead_times)
        result = plugin(cube)
        self.assertIsInstance(result, Cube)
        self.assertArrayEqual(result[0].data, expected[0])
        self.assertArrayAlmostEqual(result[1].data, expected[1])

    def test_radii_varying_with_lead_time_with_interpolation(self):
        """Test that a cube is returned for the following conditions:
        1. The radius varies with lead time.
        2. Linear interpolation is required to create values for the radii
        which are required but were not specified within the 'radii'
        argument."""

        radii = [10000, 30000]
        lead_times = [2, 4]
        neighbourhood_method = CircularNeighbourhood()
        plugin = NBHood(neighbourhood_method, radii, lead_times)
        result = plugin(self.multi_time_cube)
        self.assertIsInstance(result, Cube)

    def test_radii_varying_with_lead_time_with_interpolation_check_data(self):
        """Test that a cube with the correct data is returned for the
        following conditions:
        1. The radius varies with lead time.
        2. Linear interpolation is required to create values for the radii
        which are required but were not specified within the 'radii'
        argument."""

        expected = np.ones_like(self.multi_time_cube.data)
        expected[0, 6:9, 6:9] = (
            [0.91666667, 0.875, 0.91666667],
            [0.875, 0.83333333, 0.875],
            [0.91666667, 0.875, 0.91666667],
        )

        expected[1, 5:10, 5:10] = SINGLE_POINT_RANGE_3_CENTROID

        expected[2, 4:11, 4:11] = (
            [1, 0.9925, 0.985, 0.9825, 0.985, 0.9925, 1],
            [0.9925, 0.98, 0.9725, 0.97, 0.9725, 0.98, 0.9925],
            [0.985, 0.9725, 0.965, 0.9625, 0.965, 0.9725, 0.985],
            [0.9825, 0.97, 0.9625, 0.96, 0.9625, 0.97, 0.9825],
            [0.985, 0.9725, 0.965, 0.9625, 0.965, 0.9725, 0.985],
            [0.9925, 0.98, 0.9725, 0.97, 0.9725, 0.98, 0.9925],
            [1, 0.9925, 0.985, 0.9825, 0.985, 0.9925, 1],
        )

        radii = [5600, 9500]
        lead_times = [2, 4]
        neighbourhood_method = CircularNeighbourhood()
        plugin = NBHood(neighbourhood_method, radii, lead_times)
        result = plugin(self.multi_time_cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_use_mask_cube_occurrences_not_masked(self):
        """Test that the plugin returns an iris.cube.Cube with the correct
        data array if a mask cube is used and the mask cube does not mask
        out the occurrences."""
        expected = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.88888889, 0.88888889, 0.88888889, 1.0],
                [1.0, 0.88888889, 0.88888889, 0.88888889, 1.0],
                [1.0, 0.88888889, 0.88888889, 0.88888889, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )

        data = np.ones((5, 5), dtype=np.float32)
        data[2, 2] = 0
        cube = set_up_variable_cube(data, spatial_grid="equalarea",)

        mask_cube = cube.copy(data=np.ones((5, 5), dtype=np.float32))

        radius = 2000
        neighbourhood_method = SquareNeighbourhood()
        result = NBHood(neighbourhood_method, radius)(cube, mask_cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_use_mask_cube_occurrences_masked(self):
        """Test that the plugin returns an iris.cube.Cube with the correct
        data array if a mask cube is used and the mask cube does mask
        out the occurrences."""

        data = np.ones((5, 5), dtype=np.float32)
        data[2, 2] = 0
        cube = set_up_variable_cube(data, spatial_grid="equalarea",)

        expected_data = data
        mask_cube = cube.copy()

        radius = 2000
        neighbourhood_method = SquareNeighbourhood()
        result = NBHood(neighbourhood_method, radius)(cube, mask_cube)
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_use_mask_cube_occurrences_masked_irregular(self):
        """Test that the plugin returns an iris.cube.Cube with the correct
        data array if a mask cube is used and the mask cube does mask
        out the occurrences. In this case, an irregular mask is applied."""
        expected = np.array(
            [
                [1.000000, 1.000000, 1.000000, 1.000000, 1.000000],
                [1.000000, 0.000000, 0.833333, 0.000000, 1.000000],
                [1.000000, 0.000000, 0.833333, 0.875000, 1.000000],
                [1.000000, 0.857143, 0.833333, 0.857143, 1.000000],
                [1.000000, 0.000000, 1.000000, 0.000000, 0.000000],
            ]
        )

        data = np.ones((5, 5), dtype=np.float32)
        data[2, 2] = 0
        cube = set_up_variable_cube(data, spatial_grid="equalarea",)

        mask_cube = cube.copy()
        mask_cube.data = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )
        radius = 2000
        neighbourhood_method = SquareNeighbourhood()
        result = NBHood(neighbourhood_method, radius)(cube, mask_cube)
        self.assertArrayAlmostEqual(result.data, expected)


if __name__ == "__main__":
    unittest.main()
