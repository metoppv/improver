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
""" Tests to support utilities."""

import unittest

import iris
import numpy as np
from cf_units import Unit
from iris.coords import AuxCoord, CellMethod, DimCoord
from iris.cube import Cube
from iris.tests import IrisTest
from numpy import ma

from improver.utilities.spatial import DifferenceBetweenAdjacentGridSquares


def set_up_cube(data, phenomenon_standard_name, phenomenon_units,
                realizations=np.array([0]), timesteps=1,
                y_dimension_length=3, x_dimension_length=3):
    """Create a cube containing multiple realizations."""
    coord_placer = 0
    cube = Cube(data, standard_name=phenomenon_standard_name,
                units=phenomenon_units)
    if len(realizations) > 1:
        realizations = DimCoord(realizations, "realization")
        cube.add_dim_coord(realizations, coord_placer)
        coord_placer = 1
    else:
        cube.add_aux_coord(AuxCoord(realizations, 'realization',
                                    units='1'))
    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"
    tunit = Unit(time_origin, calendar)
    cube.add_aux_coord(AuxCoord(np.linspace(402192.5, 402292.5, timesteps),
                                "time", units=tunit))
    cube.add_dim_coord(DimCoord(np.linspace(0, 10000, y_dimension_length),
                                'projection_y_coordinate', units='m'),
                       coord_placer)
    cube.add_dim_coord(DimCoord(np.linspace(0, 10000, x_dimension_length),
                                'projection_x_coordinate', units='m'),
                       coord_placer+1)
    return cube


class Test_create_difference_cube(IrisTest):

    """Test the create_difference_cube method."""

    def setUp(self):
        """Set up cube."""
        data = np.array([[1, 2, 3],
                         [2, 4, 6],
                         [5, 10, 15]])
        self.cube = set_up_cube(data, "wind_speed", "m s-1")
        self.plugin = DifferenceBetweenAdjacentGridSquares()

    def test_y_dimension(self):
        """Test differences calculated along the y dimension."""
        diff_array = np.array([[1, 2, 3],
                               [3, 6, 9]])
        result = self.plugin.create_difference_cube(
            self.cube, "projection_y_coordinate", diff_array)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, diff_array)

    def test_x_dimension(self):
        """Test differences calculated along the x dimension."""
        diff_array = np.array([[1, 1],
                               [2, 2],
                               [5, 5]])
        result = self.plugin.create_difference_cube(
            self.cube, "projection_x_coordinate", diff_array)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, diff_array)

    def test_metadata(self):
        """Test that the result has the expected metadata."""
        diff_array = np.array([[1, 2, 3],
                               [3, 6, 9]])
        cell_method = CellMethod(
            "difference", coords=["projection_y_coordinate"],
            intervals='1 grid length')
        result = self.plugin.create_difference_cube(
            self.cube, "projection_y_coordinate", diff_array)
        self.assertEqual(
            result.cell_methods[0], cell_method)
        self.assertEqual(
            result.attributes["form_of_difference"],
            "forward_difference")
        self.assertEqual(result.name(), 'difference_of_wind_speed')

    def test_othercoords(self):
        """Test that other coords are transferred properly"""
        diff_array = np.array([[1, 2, 3],
                               [3, 6, 9]])
        time_coord = self.cube.coord('time')
        proj_x_coord = self.cube.coord(axis='x')
        result = self.plugin.create_difference_cube(
            self.cube, "projection_y_coordinate", diff_array)
        self.assertEqual(result.coord(axis='x'), proj_x_coord)
        self.assertEqual(result.coord('time'), time_coord)

    def test_gradient(self):
        """Test that the correct metadata is set if the desired output is a
        gradient not a difference (except name)"""
        plugin = DifferenceBetweenAdjacentGridSquares(gradient=True)
        diff_array = np.array([[1, 2, 3],
                               [3, 6, 9]])
        result = plugin.create_difference_cube(
            self.cube, "projection_y_coordinate", diff_array)
        self.assertNotIn("form_of_difference", result.attributes)
        self.assertFalse(result.cell_methods)


class Test_calculate_difference(IrisTest):

    """Test the calculate_difference method."""

    def setUp(self):
        """Set up cube."""
        data = np.array([[1, 2, 3],
                         [2, 4, 6],
                         [5, 10, 15]])
        self.cube = set_up_cube(data, "wind_speed", "m s-1")
        self.plugin = DifferenceBetweenAdjacentGridSquares()

    def test_x_dimension(self):
        """Test differences calculated along the x dimension."""
        expected = np.array([[1, 1],
                             [2, 2],
                             [5, 5]])
        result = self.plugin.calculate_difference(self.cube, "x")
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_y_dimension(self):
        """Test differences calculated along the y dimension."""
        expected = np.array([[1, 2, 3],
                             [3, 6, 9]])
        result = self.plugin.calculate_difference(self.cube, "y")
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_missing_data(self):
        """Test that the result is as expected when data is missing."""
        data = np.array([[1, 2, 3],
                         [np.nan, 4, 6],
                         [5, 10, 15]])
        cube = set_up_cube(data, "wind_speed", "m s-1")
        expected = np.array([[np.nan, 2, 3],
                             [np.nan, 6, 9]])
        result = self.plugin.calculate_difference(cube, "y")
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_masked_data(self):
        """Test that the result is as expected when data is masked."""
        data = ma.array([[1, 2, 3],
                         [2, 4, 6],
                         [5, 10, 15]],
                        mask=[[0, 0, 0],
                              [1, 0, 0],
                              [0, 0, 0]])
        cube = set_up_cube(data, "wind_speed", "m s-1")
        expected = ma.array([[1, 2, 3],
                             [3, 6, 9]],
                            mask=[[1, 0, 0],
                                  [1, 0, 0]])
        result = self.plugin.calculate_difference(cube, "y")
        self.assertIsInstance(result, Cube)
        self.assertArrayEqual(result.data, expected)
        self.assertArrayEqual(result.data.mask, expected.mask)


class Test_gradient_from_diff(IrisTest):

    """Test for correct behaviour when calculating a gradient"""

    def setUp(self):
        """Set up cube."""
        data = np.array([[1, 2, 3],
                         [2, 4, 2],
                         [5, 10, 15]])
        x_coord = DimCoord(2.*np.arange(3), "projection_x_coordinate")
        y_coord = DimCoord(2.*np.arange(3), "projection_y_coordinate")
        self.cube = iris.cube.Cube(data, "wind_speed", units="m s-1",
                                   dim_coords_and_dims=[(y_coord, 0),
                                                        (x_coord, 1)])
        self.plugin = DifferenceBetweenAdjacentGridSquares(gradient=True)

    def test_basic(self):
        """Test contents and metadata"""
        expected = np.array([[0.5, 0.5, 0.5],
                             [2.0, 0.0, -2.0],
                             [2.5, 2.5, 2.5]])
        xdiff = self.plugin.calculate_difference(self.cube, "x")
        result = self.plugin.gradient_from_diff(xdiff, self.cube, "x")
        self.assertArrayAlmostEqual(result.data, expected)
        self.assertEqual(result.name(), "gradient_of_wind_speed")


class Test_process(IrisTest):

    """Test the process method."""

    def setUp(self):
        """Set up cube."""
        data = np.array([[1, 2, 3],
                         [2, 4, 6],
                         [5, 10, 15]])
        self.cube = set_up_cube(data, "wind_speed", "m s-1")
        self.plugin = DifferenceBetweenAdjacentGridSquares()

    def test_basic(self):
        """Test that differences are calculated along both the x and
        y dimensions and returned as separate cubes."""
        expected_x = np.array([[1, 1],
                               [2, 2],
                               [5, 5]])
        expected_y = np.array([[1, 2, 3],
                               [3, 6, 9]])
        result = self.plugin.process(self.cube)
        self.assertIsInstance(result[0], Cube)
        self.assertArrayAlmostEqual(result[0].data, expected_x)
        self.assertIsInstance(result[1], Cube)
        self.assertArrayAlmostEqual(result[1].data, expected_y)

    def test_3d_cube(self):
        """Test the differences are calculated along both the x and
        y dimensions and returned as separate cubes when a 3d cube is input."""
        data = np.array([[[1, 2, 3],
                          [2, 4, 6],
                          [5, 10, 15]],
                         [[1, 2, 3],
                          [2, 2, 6],
                          [5, 10, 20]]])
        expected_x = np.array([[[1, 1],
                                [2, 2],
                                [5, 5]],
                               [[1, 1],
                                [0, 4],
                                [5, 10]]])
        expected_y = np.array([[[1, 2, 3],
                                [3, 6, 9]],
                               [[1, 0, 3],
                                [3, 8, 14]]])
        cube = set_up_cube(data, "wind_speed", "m s-1",
                           realizations=np.array([1, 2]))
        result = self.plugin.process(cube)
        self.assertIsInstance(result[0], iris.cube.Cube)
        self.assertArrayAlmostEqual(result[0].data, expected_x)
        self.assertIsInstance(result[1], iris.cube.Cube)
        self.assertArrayAlmostEqual(result[1].data, expected_y)


if __name__ == '__main__':
    unittest.main()
