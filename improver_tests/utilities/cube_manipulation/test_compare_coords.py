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
Unit tests for the function "cube_manipulation.compare_coords".
"""

import unittest

import iris
import numpy as np
from iris.coords import AuxCoord, DimCoord

from improver.utilities.cube_manipulation import compare_coords
from improver.utilities.warnings_handler import ManageWarnings

from ...set_up_test_cubes import set_up_variable_cube


class Test_compare_coords(unittest.TestCase):
    """Test the compare_coords utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        data = 275*np.ones((3, 3, 3), dtype=np.float32)
        self.cube = set_up_variable_cube(data)
        self.extra_dim_coord = DimCoord(
            np.array([5.0], dtype=np.float32),
            standard_name="height", units="m")
        self.extra_aux_coord = AuxCoord(
            ['uk_det', 'uk_ens', 'gl_ens'], long_name='model', units='no_unit')

    def test_basic(self):
        """Test that the utility returns a list."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cubelist = iris.cube.CubeList([cube1, cube2])
        result = compare_coords(cubelist)
        self.assertIsInstance(result, list)

    @ManageWarnings(record=True)
    def test_catch_warning(self, warning_list=None):
        """Test warning is raised if the input is cubelist of length 1."""
        cube = self.cube.copy()
        result = compare_coords(iris.cube.CubeList([cube]))
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        warning_msg = "Only a single cube so no differences will be found "
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertEqual(result, [])

    def test_first_cube_has_extra_dimension_coordinates(self):
        """Test for comparing coordinate between cubes, where the first
        cube in the list has extra dimension coordinates."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube1.add_aux_coord(self.extra_dim_coord)
        cube1 = iris.util.new_axis(cube1, "height")
        cubelist = iris.cube.CubeList([cube1, cube2])
        result = compare_coords(cubelist)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(len(result[1]), 0)
        self.assertEqual(result[0]["height"]["coord"], self.extra_dim_coord)
        self.assertEqual(result[0]["height"]["data_dims"], 0)
        self.assertEqual(result[0]["height"]["aux_dims"], None)

    def test_second_cube_has_extra_dimension_coordinates(self):
        """Test for comparing coordinate between cubes, where the second
        cube in the list has extra dimension coordinates."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube2.add_aux_coord(self.extra_dim_coord)
        cube2 = iris.util.new_axis(cube2, "height")
        cubelist = iris.cube.CubeList([cube1, cube2])
        result = compare_coords(cubelist)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result[0]), 0)
        self.assertEqual(len(result[1]), 1)
        self.assertEqual(result[1]["height"]["coord"], self.extra_dim_coord)
        self.assertEqual(result[1]["height"]["data_dims"], 0)
        self.assertEqual(result[1]["height"]["aux_dims"], None)

    def test_first_cube_has_extra_auxiliary_coordinates(self):
        """Test for comparing coordinate between cubes, where the first
        cube in the list has extra auxiliary coordinates."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube1.add_aux_coord(self.extra_aux_coord, data_dims=0)
        cubelist = iris.cube.CubeList([cube1, cube2])
        result = compare_coords(cubelist)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(len(result[1]), 0)
        self.assertEqual(result[0]["model"]["coord"], self.extra_aux_coord)
        self.assertEqual(result[0]["model"]["data_dims"], None)
        self.assertEqual(result[0]["model"]["aux_dims"], 0)

    def test_second_cube_has_extra_auxiliary_coordinates(self):
        """Test for comparing coordinate between cubes, where the second
        cube in the list has extra auxiliary coordinates."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube2.add_aux_coord(self.extra_aux_coord, data_dims=0)
        cubelist = iris.cube.CubeList([cube1, cube2])
        result = compare_coords(cubelist)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result[0]), 0)
        self.assertEqual(len(result[1]), 1)
        self.assertEqual(result[1]["model"]["coord"], self.extra_aux_coord)
        self.assertEqual(result[1]["model"]["data_dims"], None)
        self.assertEqual(result[1]["model"]["aux_dims"], 0)


if __name__ == '__main__':
    unittest.main()
