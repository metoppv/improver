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
"""
Unit tests for the function "cube_manipulation.enforce_coordinate_ordering".
"""

import unittest

import numpy as np
from iris.cube import Cube
from iris.tests import IrisTest

from improver.metadata.constants.time_types import TIME_COORDS
from improver.utilities.cube_manipulation import enforce_coordinate_ordering

from ...set_up_test_cubes import (
    add_coordinate, set_up_probability_cube, set_up_variable_cube)


class Test_enforce_coordinate_ordering(IrisTest):

    """Test the enforce_coordinate_ordering utility."""

    def setUp(self):
        """Set up cube with non-homogeneous data to test with"""
        data = np.arange(27).reshape((3, 3, 3)) + 275
        cube = set_up_variable_cube(data.astype(np.float32))
        time_points = [cube.coord("time").points[0],
                       cube.coord("time").points[0] + 3600]
        self.cube = add_coordinate(
            cube, time_points, "time", coord_units=TIME_COORDS["time"].units,
            dtype=np.int64, order=[1, 0, 2, 3])

    def test_move_coordinate_to_start_when_already_at_start(self):
        """Test that a cube with the expected data contents is returned when
        the coordinate to be reordered is already in the desired position."""
        expected = self.cube.copy()
        enforce_coordinate_ordering(self.cube, "realization")
        self.assertEqual(self.cube.coord_dims("realization")[0], 0)
        self.assertArrayAlmostEqual(self.cube.data, expected.data)

    def test_move_coordinate_to_start(self):
        """Test that a cube with the expected data contents is returned when
        the time coordinate is reordered to be the first coordinate in the
        cube."""
        expected = self.cube.copy()
        expected.transpose([1, 0, 2, 3])
        enforce_coordinate_ordering(self.cube, "time")
        self.assertEqual(self.cube.coord_dims("time")[0], 0)
        # test associated aux coord is moved along with time dimension
        self.assertEqual(self.cube.coord_dims("forecast_period")[0], 0)
        self.assertArrayAlmostEqual(self.cube.data, expected.data)

    def test_move_coordinate_to_end(self):
        """Test that a cube with the expected data contents is returned when
        the realization coordinate is reordered to be the last coordinate in
        the cube."""
        expected = self.cube.copy()
        expected.transpose([1, 2, 3, 0])
        enforce_coordinate_ordering(
            self.cube, "realization", anchor_start=False)
        self.assertEqual(self.cube.coord_dims("realization")[0], 3)
        self.assertArrayAlmostEqual(self.cube.data, expected.data)

    def test_move_coordinate_to_start_with_list(self):
        """Test that a cube with the expected data contents is returned when
        the time coordinate is reordered to be the first coordinate in the
        cube."""
        expected = self.cube.copy()
        expected.transpose([1, 0, 2, 3])
        enforce_coordinate_ordering(self.cube, ["time"])
        self.assertEqual(self.cube.coord_dims("time")[0], 0)
        self.assertArrayAlmostEqual(self.cube.data, expected.data)

    def test_move_multiple_coordinate_to_start_with_list(self):
        """Test that a cube with the expected data contents is returned when
        the time and realization coordinates are reordered to be the first
        coordinates in the cube."""
        expected = self.cube.copy()
        expected.transpose([1, 0, 2, 3])
        enforce_coordinate_ordering(
            self.cube, ["time", "realization"])
        self.assertEqual(self.cube.coord_dims("time")[0], 0)
        self.assertEqual(self.cube.coord_dims("realization")[0], 1)
        self.assertArrayAlmostEqual(self.cube.data, expected.data)

    def test_move_multiple_coordinate_to_end_with_list(self):
        """Test that a cube with the expected data contents is returned when
        the time and realization coordinates are reordered to be the last
        coordinates in the cube. The coordinate name to be reordered is
        specified as a list."""
        expected = self.cube.copy()
        expected.transpose([2, 3, 1, 0])
        enforce_coordinate_ordering(
            self.cube, ["time", "realization"], anchor_start=False)
        self.assertEqual(self.cube.coord_dims("time")[0], 2)
        self.assertEqual(self.cube.coord_dims("realization")[0], 3)
        self.assertArrayAlmostEqual(self.cube.data, expected.data)

    def test_full_reordering(self):
        """Test that a cube with the expected data contents is returned when
        all the coordinates within the cube are reordered into the order
        specified by the names within the input list."""
        expected = self.cube.copy()
        expected.transpose([2, 0, 3, 1])
        enforce_coordinate_ordering(
            self.cube, ["latitude", "realization", "longitude", "time"])
        self.assertEqual(self.cube.coord_dims("latitude")[0], 0)
        self.assertEqual(self.cube.coord_dims("realization")[0], 1)
        self.assertEqual(self.cube.coord_dims("longitude")[0], 2)
        self.assertEqual(self.cube.coord_dims("time")[0], 3)
        self.assertArrayAlmostEqual(self.cube.data, expected.data)

    def test_include_extra_coordinates(self):
        """Test that a cube with the expected data contents is returned when
        extra coordinates are passed in for reordering but these coordinates
        are not present within the cube."""
        expected = self.cube.copy()
        expected.transpose([1, 0, 2, 3])
        enforce_coordinate_ordering(
            self.cube, ["time", "realization", "nonsense"])
        self.assertEqual(self.cube.coord_dims("time")[0], 0)
        self.assertEqual(self.cube.coord_dims("realization")[0], 1)
        self.assertArrayAlmostEqual(self.cube.data, expected.data)

    def test_no_impact_scalar(self):
        """Test that a cube with the expected data contents is returned when
        reordered on a scalar coordinate."""
        cube = self.cube[0, :, :, :]
        expected = cube.copy()
        enforce_coordinate_ordering(cube, "realization")
        self.assertFalse(cube.coord_dims("realization"))
        self.assertArrayAlmostEqual(cube.data, expected.data)

    def test_handles_threshold(self):
        """Test a probability cube is correctly handled"""
        thresholds = np.array([278, 279, 280], dtype=np.float32)
        data = 0.03*np.arange(27).reshape((3, 3, 3))
        cube = set_up_probability_cube(data.astype(np.float32), thresholds)
        enforce_coordinate_ordering(
            cube, ["threshold"], anchor_start=False)
        self.assertEqual(cube.coord_dims("air_temperature")[0], 2)


if __name__ == '__main__':
    unittest.main()
