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
Unit tests for the function "cube_manipulation.sort_coord_in_cube".
"""

import unittest

import iris
from iris.tests import IrisTest
from iris.coords import AuxCoord
import numpy as np

from improver.utilities.cube_manipulation import sort_coord_in_cube

from improver.tests.utilities.test_mathematical_operations import (
    set_up_height_cube)
from improver.utilities.warnings_handler import ManageWarnings


class Test_sort_coord_in_cube(IrisTest):
    """Class to test the sort_coord_in_cube function."""

    def setUp(self):
        """Set up a cube."""
        self.ascending_height_points = np.array([5., 10., 20.])
        cube = set_up_height_cube(self.ascending_height_points)[:, 0, :, :, :]
        data = np.zeros(cube.shape)
        data[0] = np.ones(cube[0].shape, dtype=np.int32)
        data[1] = np.full(cube[1].shape, 2, dtype=np.int32)
        data[2] = np.full(cube[2].shape, 3, dtype=np.int32)
        cube.data = data
        self.ascending_cube = cube
        descending_cube = cube.copy()
        self.descending_height_points = np.array([20., 10., 5.])
        descending_cube.coord("height").points = self.descending_height_points
        self.descending_cube = descending_cube

    def test_ascending_then_ascending(self):
        """Test that the sorting successfully sorts the cube based
        on the points within the given coordinate. The points in the resulting
        cube should now be in ascending order."""
        expected_data = np.array(
            [[[[1.00, 1.00, 1.00],
               [1.00, 1.00, 1.00],
               [1.00, 1.00, 1.00]]],
             [[[2.00, 2.00, 2.00],
               [2.00, 2.00, 2.00],
               [2.00, 2.00, 2.00]]],
             [[[3.00, 3.00, 3.00],
               [3.00, 3.00, 3.00],
               [3.00, 3.00, 3.00]]]])
        coord_name = "height"
        result = sort_coord_in_cube(self.ascending_cube, coord_name)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(self.ascending_cube.coord_dims(coord_name),
                         result.coord_dims(coord_name))
        self.assertArrayAlmostEqual(
            self.ascending_height_points,
            result.coord(coord_name).points)
        self.assertDictEqual(
            self.ascending_cube.coord(coord_name).attributes,
            {"positive": "up"})
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_auxcoord(self):
        """Test that the above sorting is successful when an AuxCoord is
        used."""
        expected_data = np.array(
            [[[[1.00, 1.00, 1.00],
               [1.00, 1.00, 1.00],
               [1.00, 1.00, 1.00]]],
             [[[2.00, 2.00, 2.00],
               [2.00, 2.00, 2.00],
               [2.00, 2.00, 2.00]]],
             [[[3.00, 3.00, 3.00],
               [3.00, 3.00, 3.00],
               [3.00, 3.00, 3.00]]]])
        coord_name = "height_aux"
        height_coord = self.ascending_cube.coord('height')
        height_coord_index, = self.ascending_cube.coord_dims('height')
        new_coord = AuxCoord(height_coord.points, long_name=coord_name)
        self.ascending_cube.add_aux_coord(new_coord, height_coord_index)
        result = sort_coord_in_cube(self.ascending_cube, coord_name)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(self.ascending_cube.coord_dims(coord_name),
                         result.coord_dims(coord_name))
        self.assertArrayAlmostEqual(
            self.ascending_height_points,
            result.coord(coord_name).points)
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_ascending_then_descending(self):
        """Test that the sorting successfully sorts the cube based
        on the points within the given coordinate. The points in the resulting
        cube should now be in descending order."""
        expected_data = np.array(
            [[[[3.00, 3.00, 3.00],
               [3.00, 3.00, 3.00],
               [3.00, 3.00, 3.00]]],
             [[[2.00, 2.00, 2.00],
               [2.00, 2.00, 2.00],
               [2.00, 2.00, 2.00]]],
             [[[1.00, 1.00, 1.00],
               [1.00, 1.00, 1.00],
               [1.00, 1.00, 1.00]]]])
        coord_name = "height"
        result = sort_coord_in_cube(
            self.ascending_cube, coord_name, order="descending")
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(self.descending_cube.coord_dims(coord_name),
                         result.coord_dims(coord_name))
        self.assertArrayAlmostEqual(
            self.descending_height_points, result.coord(coord_name).points)
        self.assertDictEqual(
            result.coord(coord_name).attributes, {"positive": "down"})
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_descending_then_ascending(self):
        """Test that the sorting successfully sorts the cube based
        on the points within the given coordinate. The points in the resulting
        cube should now be in ascending order."""
        expected_data = np.array(
            [[[[3.00, 3.00, 3.00],
               [3.00, 3.00, 3.00],
               [3.00, 3.00, 3.00]]],
             [[[2.00, 2.00, 2.00],
               [2.00, 2.00, 2.00],
               [2.00, 2.00, 2.00]]],
             [[[1.00, 1.00, 1.00],
               [1.00, 1.00, 1.00],
               [1.00, 1.00, 1.00]]]])
        coord_name = "height"
        result = sort_coord_in_cube(self.descending_cube, coord_name)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(self.ascending_cube.coord_dims(coord_name),
                         result.coord_dims(coord_name))
        self.assertArrayAlmostEqual(
            self.ascending_height_points, result.coord(coord_name).points)
        self.assertDictEqual(
            result.coord(coord_name).attributes, {"positive": "up"})
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_descending_then_descending(self):
        """Test that the sorting successfully sorts the cube based
        on the points within the given coordinate. The points in the resulting
        cube should now be in descending order."""
        expected_data = np.array(
            [[[[1.00, 1.00, 1.00],
               [1.00, 1.00, 1.00],
               [1.00, 1.00, 1.00]]],
             [[[2.00, 2.00, 2.00],
               [2.00, 2.00, 2.00],
               [2.00, 2.00, 2.00]]],
             [[[3.00, 3.00, 3.00],
               [3.00, 3.00, 3.00],
               [3.00, 3.00, 3.00]]]])
        coord_name = "height"
        result = sort_coord_in_cube(
            self.descending_cube, coord_name, order="descending")
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(self.descending_cube.coord_dims(coord_name),
                         result.coord_dims(coord_name))
        self.assertArrayAlmostEqual(
            self.descending_height_points, result.coord(coord_name).points)
        self.assertDictEqual(
            result.coord(coord_name).attributes, {"positive": "down"})
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_latitude(self):
        """Test that the sorting successfully sorts the cube based
        on the points within the given coordinate (latitude).
        The points in the resulting cube should now be in descending order."""
        expected_data = np.array(
            [[[[1.00, 1.00, 1.00],
               [1.00, 1.00, 1.00],
               [6.00, 1.00, 1.00]]],
             [[[2.00, 2.00, 2.00],
               [2.00, 2.00, 2.00],
               [6.00, 2.00, 2.00]]],
             [[[3.00, 3.00, 3.00],
               [3.00, 3.00, 3.00],
               [6.00, 3.00, 3.00]]]])
        self.ascending_cube.data[:, :, 0, 0] = 6.0
        expected_points = np.array([45., 0., -45])
        coord_name = "latitude"
        result = sort_coord_in_cube(
            self.ascending_cube, coord_name, order="descending")
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(
            self.ascending_cube.coord_dims(coord_name),
            result.coord_dims(coord_name))
        self.assertArrayAlmostEqual(
            expected_points, result.coord(coord_name).points)
        self.assertArrayAlmostEqual(result.data, expected_data)

    @ManageWarnings(record=True)
    def test_warn_raised_for_circular_coordinate(self, warning_list=None):
        """Test that a warning is successfully raised when circular
        coordinates are sorted."""
        self.ascending_cube.data[:, :, 0, 0] = 6.0
        coord_name = "latitude"
        self.ascending_cube.coord(coord_name).circular = True
        result = sort_coord_in_cube(
            self.ascending_cube, coord_name, order="descending")
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        warning_msg = "The latitude coordinate is circular."
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertIsInstance(result, iris.cube.Cube)


if __name__ == '__main__':
    unittest.main()
