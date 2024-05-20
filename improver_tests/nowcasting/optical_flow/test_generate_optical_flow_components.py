# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for generation of optical flow components"""

import unittest
from datetime import datetime

import iris
import numpy as np
from iris.tests import IrisTest

from improver.nowcasting.optical_flow import generate_optical_flow_components
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


class Test_generate_optical_flow_components(IrisTest):
    """Tests for the generate_optical_flow_components function. Optical flow
    velocity values are tested within the Test_optical_flow module; this class
    tests timestamps only."""

    def setUp(self):
        """Set up test input cubes"""
        self.iterations = 20
        self.ofc_box_size = 10

        dummy_cube = set_up_variable_cube(
            np.zeros((30, 30), dtype=np.float32),
            name="lwe_precipitation_rate",
            units="mm h-1",
            spatial_grid="equalarea",
            time=datetime(2018, 2, 20, 4, 0),
            frt=datetime(2018, 2, 20, 4, 0),
        )
        coord_points = 2000 * np.arange(30, dtype=np.float32)  # in metres
        dummy_cube.coord(axis="x").points = coord_points
        dummy_cube.coord(axis="y").points = coord_points

        self.first_cube = dummy_cube.copy()
        self.second_cube = dummy_cube.copy()
        # 15 minutes later, in seconds
        self.second_cube.coord("time").points = (
            self.second_cube.coord("time").points + 15 * 60
        )

        self.third_cube = dummy_cube.copy()
        # 30 minutes later, in seconds
        self.third_cube.coord("time").points = (
            self.third_cube.coord("time").points + 30 * 60
        )

        self.expected_time = self.third_cube.coord("time").points[0]

    def test_basic(self):
        """Test output is a tuple of cubes"""
        cubelist = [self.first_cube, self.second_cube, self.third_cube]
        result = generate_optical_flow_components(
            cubelist, self.ofc_box_size, self.iterations
        )
        for cube in result:
            self.assertIsInstance(cube, iris.cube.Cube)
            self.assertAlmostEqual(cube.coord("time").points[0], self.expected_time)

    def test_time_ordering(self):
        """Test output timestamps are insensitive to input cube order"""
        cubelist = [self.second_cube, self.third_cube, self.first_cube]
        result = generate_optical_flow_components(
            cubelist, self.ofc_box_size, self.iterations
        )
        for cube in result:
            self.assertAlmostEqual(cube.coord("time").points[0], self.expected_time)

    def test_fewer_inputs(self):
        """Test routine can produce output from a shorter list of inputs"""
        result = generate_optical_flow_components(
            [self.second_cube, self.third_cube], self.ofc_box_size, self.iterations
        )
        for cube in result:
            self.assertAlmostEqual(cube.coord("time").points[0], self.expected_time)


if __name__ == "__main__":
    unittest.main()
