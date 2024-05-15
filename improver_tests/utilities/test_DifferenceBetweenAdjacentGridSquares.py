# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
""" Tests of DifferenceBetweenAdjacentGridSquares plugin."""

import unittest

import iris
import numpy as np
from iris.coords import CellMethod
from iris.cube import Cube
from iris.tests import IrisTest
from numpy import ma

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.spatial import DifferenceBetweenAdjacentGridSquares


class Test_create_difference_cube(IrisTest):

    """Test the create_difference_cube method."""

    def setUp(self):
        """Set up cube."""
        data = np.array([[1, 2, 3], [2, 4, 6], [5, 10, 15]])
        self.diff_in_y_array = np.array([[1, 2, 3], [3, 6, 9]])
        self.cube = set_up_variable_cube(
            data, name="wind_speed", units="m s-1", spatial_grid="equalarea",
        )
        self.plugin = DifferenceBetweenAdjacentGridSquares()

    def test_y_dimension(self):
        """Test differences calculated along the y dimension."""
        points = self.cube.coord(axis="y").points
        expected_y = (points[1:] + points[:-1]) / 2
        result = self.plugin.create_difference_cube(
            self.cube, "projection_y_coordinate", self.diff_in_y_array
        )
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.coord(axis="y").points, expected_y)
        self.assertArrayEqual(result.data, self.diff_in_y_array)

    def test_x_dimension(self):
        """Test differences calculated along the x dimension."""
        diff_array = np.array([[1, 1], [2, 2], [5, 5]])
        points = self.cube.coord(axis="x").points
        expected_x = (points[1:] + points[:-1]) / 2
        result = self.plugin.create_difference_cube(
            self.cube, "projection_x_coordinate", diff_array
        )
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.coord(axis="x").points, expected_x)
        self.assertArrayEqual(result.data, diff_array)

    def test_othercoords(self):
        """Test that other coords are transferred properly"""
        time_coord = self.cube.coord("time")
        proj_x_coord = self.cube.coord(axis="x")
        result = self.plugin.create_difference_cube(
            self.cube, "projection_y_coordinate", self.diff_in_y_array
        )
        self.assertEqual(result.coord(axis="x"), proj_x_coord)
        self.assertEqual(result.coord("time"), time_coord)


class Test_calculate_difference(IrisTest):

    """Test the calculate_difference method."""

    def setUp(self):
        """Set up cube."""
        data = np.array([[1, 2, 3], [2, 4, 6], [5, 10, 15]])
        self.cube = set_up_variable_cube(
            data, name="wind_speed", units="m s-1", spatial_grid="equalarea",
        )
        self.plugin = DifferenceBetweenAdjacentGridSquares()

    def test_x_dimension(self):
        """Test differences calculated along the x dimension."""
        expected = np.array([[1, 1], [2, 2], [5, 5]])
        result = self.plugin.calculate_difference(
            self.cube, self.cube.coord(axis="x").name()
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, expected)

    def test_y_dimension(self):
        """Test differences calculated along the y dimension."""
        expected = np.array([[1, 2, 3], [3, 6, 9]])
        result = self.plugin.calculate_difference(
            self.cube, self.cube.coord(axis="y").name()
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, expected)

    def test_missing_data(self):
        """Test that the result is as expected when data is missing."""
        data = np.array([[1, 2, 3], [np.nan, 4, 6], [5, 10, 15]], dtype=np.float32)
        self.cube.data = data
        expected = np.array([[np.nan, 2, 3], [np.nan, 6, 9]])
        result = self.plugin.calculate_difference(
            self.cube, self.cube.coord(axis="y").name()
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected)

    def test_masked_data(self):
        """Test that the result is as expected when data is masked."""
        data = ma.array(
            [[1, 2, 3], [2, 4, 6], [5, 10, 15]], mask=[[0, 0, 0], [1, 0, 0], [0, 0, 0]]
        )
        self.cube.data = data
        expected = ma.array([[1, 2, 3], [3, 6, 9]], mask=[[1, 0, 0], [1, 0, 0]])
        result = self.plugin.calculate_difference(
            self.cube, self.cube.coord(axis="y").name()
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, expected)
        self.assertArrayEqual(result.mask, expected.mask)


class Test_process(IrisTest):

    """Test the process method."""

    def setUp(self):
        """Set up cube."""
        data = np.array([[1, 2, 3], [2, 4, 6], [5, 10, 15]])
        self.cube = set_up_variable_cube(
            data,
            name="wind_speed",
            units="m s-1",
            spatial_grid="equalarea",
            realizations=np.array([1, 2]),
        )
        self.plugin = DifferenceBetweenAdjacentGridSquares()

    def test_basic(self):
        """Test that differences are calculated along both the x and
        y dimensions and returned as separate cubes."""
        expected_x = np.array([[1, 1], [2, 2], [5, 5]])
        expected_y = np.array([[1, 2, 3], [3, 6, 9]])
        result = self.plugin.process(self.cube)
        self.assertIsInstance(result[0], Cube)
        self.assertArrayEqual(result[0].data, expected_x)
        self.assertIsInstance(result[1], Cube)
        self.assertArrayEqual(result[1].data, expected_y)

    def test_metadata(self):
        """Test the resulting metadata is correct."""
        cell_method_x = CellMethod(
            "difference", coords=["projection_x_coordinate"], intervals="1 grid length"
        )
        cell_method_y = CellMethod(
            "difference", coords=["projection_y_coordinate"], intervals="1 grid length"
        )

        result = self.plugin.process(self.cube)
        for cube, cm in zip(result, [cell_method_x, cell_method_y]):
            self.assertEqual(cube.cell_methods[0], cm)
            self.assertEqual(
                cube.attributes["form_of_difference"], "forward_difference"
            )
            self.assertEqual(cube.name(), "difference_of_wind_speed")

    def test_3d_cube(self):
        """Test the differences are calculated along both the x and
        y dimensions and returned as separate cubes when a 3d cube is input."""
        data = np.array(
            [[[1, 2, 3], [2, 4, 6], [5, 10, 15]], [[1, 2, 3], [2, 2, 6], [5, 10, 20]]]
        )
        expected_x = np.array([[[1, 1], [2, 2], [5, 5]], [[1, 1], [0, 4], [5, 10]]])
        expected_y = np.array([[[1, 2, 3], [3, 6, 9]], [[1, 0, 3], [3, 8, 14]]])
        cube = set_up_variable_cube(
            data,
            name="wind_speed",
            units="m s-1",
            spatial_grid="equalarea",
            realizations=np.array([1, 2]),
        )
        result = self.plugin.process(cube)
        self.assertIsInstance(result[0], iris.cube.Cube)
        self.assertArrayEqual(result[0].data, expected_x)
        self.assertIsInstance(result[1], iris.cube.Cube)
        self.assertArrayEqual(result[1].data, expected_y)


if __name__ == "__main__":
    unittest.main()
