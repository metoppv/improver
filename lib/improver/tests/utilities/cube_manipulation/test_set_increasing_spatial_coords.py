# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
Unit tests for the function "cube_manipulation.set_increasing_spatial_coords".
"""

import unittest
import numpy as np

import iris
from iris.coords import DimCoord
from iris.tests import IrisTest

from improver.utilities.cube_manipulation import set_increasing_spatial_coords


class Test_set_increasing_spatial_coords(IrisTest):
    """Test function to ensure spatial coordinates are in the increasing
    direction"""

    def setUp(self):
        """Create test cubes"""
        data = np.arange(9).reshape(3, 3)
        x_coord = DimCoord([0, 1, 2], "projection_x_coordinate", "km")
        y_coord = DimCoord([0, 1, 2], "projection_y_coordinate", "km")
        y_coord_reversed = DimCoord([2, 1, 0], "projection_y_coordinate", "km")

        self.cube = iris.cube.Cube(data, "air_temperature", "celcius",
                                   dim_coords_and_dims=[(y_coord_reversed, 0),
                                                        (x_coord, 1)])
        self.normal_cube = iris.cube.Cube(data, "air_temperature", "celcius",
                                          dim_coords_and_dims=[(y_coord, 0),
                                                               (x_coord, 1)])

    def test_basic(self):
        """Test cube remains a cube"""
        set_increasing_spatial_coords(self.cube)
        self.assertIsInstance(self.cube, iris.cube.Cube)

    def test_points(self):
        """Test coordinate point values and data are reversed"""
        expected_data = np.array([[6, 7, 8],
                                  [3, 4, 5],
                                  [0, 1, 2]])
        set_increasing_spatial_coords(self.cube)
        self.assertArrayEqual(self.cube.data, expected_data)
        self.assertArrayEqual(self.cube.coord(axis='y').points, [0, 1, 2])

    def test_bounds(self):
        """Test coordinate bounds are reversed"""
        self.cube.coord(axis='y').bounds = np.array([
            [2.5, 1.5], [1.5, 0.5], [0.5, -0.5]])
        set_increasing_spatial_coords(self.cube)
        expected_bounds = np.array([
            [-0.5, 0.5], [0.5, 1.5], [1.5, 2.5]])
        self.assertArrayEqual(
            self.cube.coord(axis='y').bounds, expected_bounds)

    def test_no_impact(self):
        """Test no impact on cube where coordinates are increasing"""
        expected_data = self.cube.data
        set_increasing_spatial_coords(self.normal_cube)
        self.assertArrayEqual(self.normal_cube.data, expected_data)


if __name__ == '__main__':
    unittest.main()
