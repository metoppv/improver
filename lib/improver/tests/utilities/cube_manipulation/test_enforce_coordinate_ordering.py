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

from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest

from improver.utilities.cube_manipulation import enforce_coordinate_ordering

from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import set_up_temperature_cube


class Test_enforce_coordinate_ordering(IrisTest):

    """Test the enforce_coordinate_ordering utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()

    def test_basic(self):
        """Test that the function returns an iris.cube.Cube."""
        result = enforce_coordinate_ordering(self.cube, "realization")
        self.assertIsInstance(result, Cube)

    def test_move_coordinate_to_start_when_already_at_start(self):
        """Test that a cube with the expected data contents is returned when
        the coordinate to be reordered is already in the desired position."""
        result = enforce_coordinate_ordering(self.cube, "realization")
        self.assertEqual(result.coord_dims("realization")[0], 0)
        self.assertArrayAlmostEqual(result.data, self.cube.data)

    def test_move_coordinate_to_start(self):
        """Test that a cube with the expected data contents is returned when
        the time coordinate is reordered to be the first coordinate in the
        cube."""
        expected = self.cube.copy()
        expected.transpose([1, 0, 2, 3])
        cube = self.cube.copy()
        result = enforce_coordinate_ordering(cube, "time")
        self.assertEqual(result.coord_dims("time")[0], 0)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_move_coordinate_to_end(self):
        """Test that a cube with the expected data contents is returned when
        the realization coordinate is reordered to be the last coordinate in
        the cube."""
        expected = self.cube.copy()
        expected.transpose([1, 2, 3, 0])
        cube = self.cube.copy()
        result = enforce_coordinate_ordering(cube, "realization", anchor="end")
        self.assertEqual(result.coord_dims("realization")[0], 3)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_move_coordinate_to_start_with_list(self):
        """Test that a cube with the expected data contents is returned when
        the time coordinate is reordered to be the first coordinate in the
        cube. The coordinate name to be reordered is specified as a list."""
        expected = self.cube.copy()
        expected.transpose([1, 0, 2, 3])
        cube = self.cube.copy()
        result = enforce_coordinate_ordering(cube, ["time"])
        self.assertEqual(result.coord_dims("time")[0], 0)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_move_multiple_coordinate_to_start_with_list(self):
        """Test that a cube with the expected data contents is returned when
        the time and realization coordinates are reordered to be the first
        coordinates in the cube. The coordinate name to be reordered is
        specified as a list."""
        expected = self.cube.copy()
        expected.transpose([1, 0, 2, 3])
        cube = self.cube.copy()
        result = enforce_coordinate_ordering(cube, ["time", "realization"])
        self.assertEqual(result.coord_dims("time")[0], 0)
        self.assertEqual(result.coord_dims("realization")[0], 1)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_move_multiple_coordinate_to_end_with_list(self):
        """Test that a cube with the expected data contents is returned when
        the time and realization coordinates are reordered to be the last
        coordinates in the cube. The coordinate name to be reordered is
        specified as a list."""
        expected = self.cube.copy()
        expected.transpose([2, 3, 1, 0])
        cube = self.cube.copy()
        result = enforce_coordinate_ordering(
            cube, ["time", "realization"], anchor="end")

        self.assertEqual(result.coord_dims("time")[0], 2)
        self.assertEqual(result.coord_dims("realization")[0], 3)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_full_reordering(self):
        """Test that a cube with the expected data contents is returned when
        all the coordinates within the cube are reordered into the order
        specified by the names within the input list."""
        expected = self.cube.copy()
        expected.transpose([2, 0, 3, 1])
        cube = self.cube.copy()
        result = enforce_coordinate_ordering(
            cube, ["latitude", "realization", "longitude", "time"])
        self.assertEqual(result.coord_dims("latitude")[0], 0)
        self.assertEqual(result.coord_dims("realization")[0], 1)
        self.assertEqual(result.coord_dims("longitude")[0], 2)
        self.assertEqual(result.coord_dims("time")[0], 3)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_partial_names(self):
        """Test that a cube with the expected data contents is returned when
        the names provided are partial matches of the names of the coordinates
        within the cube."""
        expected = self.cube.copy()
        expected.transpose([1, 0, 2, 3])
        cube = self.cube.copy()
        result = enforce_coordinate_ordering(cube, ["tim", "realiz"])
        self.assertEqual(result.coord_dims("time")[0], 0)
        self.assertEqual(result.coord_dims("realization")[0], 1)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_partial_names_multiple_matches_exception(self):
        """Test that the expected exception is raised when the names provided
        are partial matches of the names of multiple coordinates within the
        cube."""
        expected = self.cube.copy()
        expected.transpose([2, 3, 0, 1])
        cube = self.cube.copy()
        msg = "More than 1 coordinate"
        with self.assertRaisesRegex(ValueError, msg):
            enforce_coordinate_ordering(cube, ["l", "e"])

    def test_include_extra_coordinates(self):
        """Test that a cube with the expected data contents is returned when
        extra coordinates are passed in for reordering but these coordinates
        are not present within the cube."""
        expected = self.cube.copy()
        expected.transpose([1, 0, 2, 3])
        cube = self.cube.copy()
        result = enforce_coordinate_ordering(
            cube, ["time", "realization", "nonsense"])
        self.assertEqual(result.coord_dims("time")[0], 0)
        self.assertEqual(result.coord_dims("realization")[0], 1)
        self.assertArrayAlmostEqual(result.data, expected.data)

    def test_force_promotion_of_scalar(self):
        """Test that a cube with the expected data contents is returned when
        the probabilistic dimension is a scalar coordinate, which is promoted
        to a dimension coordinate."""
        cube = self.cube[0, :, :, :]
        result = enforce_coordinate_ordering(
            cube, "realization", promote_scalar=True)
        self.assertEqual(result.coord_dims("realization")[0], 0)
        self.assertArrayAlmostEqual(result.data, [cube.data])

    def test_do_not_promote_scalar(self):
        """Test that a cube with the expected data contents is returned when
        the probabilistic dimension is a scalar coordinate, which is not
        promoted to a dimension coordinate."""
        cube = self.cube[0, :, :, :]
        result = enforce_coordinate_ordering(cube, "realization")
        self.assertFalse(result.coord_dims("realization"))
        self.assertArrayAlmostEqual(result.data, cube.data)

    def test_coordinate_raise_exception(self):
        """Test that the expected error message is raised when the required
        probabilistic dimension is not available in the cube."""
        cube = self.cube[0, :, :, :]
        cube.remove_coord("realization")
        msg = "The requested coordinate"
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            enforce_coordinate_ordering(
                cube, "realization", raise_exception=True)


if __name__ == '__main__':
    unittest.main()
