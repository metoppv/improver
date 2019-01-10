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
Unit tests for the function "cube_manipulation._equalise_cube_coords".
"""

import unittest

import iris
from iris.coords import DimCoord
from iris.tests import IrisTest
import numpy as np

from improver.utilities.cube_manipulation import _equalise_cube_coords

from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import (
        set_up_temperature_cube,
        set_up_probability_above_threshold_temperature_cube)

from improver.utilities.warnings_handler import ManageWarnings


class Test__equalise_cube_coords(IrisTest):

    """Test the_equalise_cube_coords utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()

    @ManageWarnings(record=True)
    def test_basic(self, warning_list=None):
        """Test that the utility returns an iris.cube.CubeList."""
        result = _equalise_cube_coords(iris.cube.CubeList([self.cube]))
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        warning_msg = "Only a single cube so no differences will be found "
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_threshold_exception(self):
        """Test that an exception is raised if a threshold coordinate is
        unmatched."""
        cube = set_up_probability_above_threshold_temperature_cube()
        cube1 = cube.copy()
        cube2 = cube.copy()
        cube2.remove_coord("threshold")
        cubes = iris.cube.CubeList([cube1, cube2])
        msg = "threshold coordinates must match to merge"
        with self.assertRaisesRegex(ValueError, msg):
            _equalise_cube_coords(cubes)

    def test_model_id_without_realization(self):
        """Test that if model_id is an unmatched coordinate, and the cubes
        do not have a realization coordinate the code does not try and
        add realization coordinate."""
        cube1 = self.cube.copy()[0]
        cube2 = self.cube.copy()[0]
        cube1.remove_coord("realization")
        cube2.remove_coord("realization")
        model_id_coord = DimCoord(
            np.array([1000*1], np.int), long_name='model_id')
        cube1.add_aux_coord(model_id_coord)
        cube1 = iris.util.new_axis(cube1)
        cubes = iris.cube.CubeList([cube1, cube2])
        result = _equalise_cube_coords(cubes)
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(len(result), 2)
        self.assertFalse(result[0].coords("realization"))
        self.assertFalse(result[1].coords("realization"))
        self.assertTrue(result[0].coords("model_id"))

    def test_model_id_with_realization_exception(self):
        """Test that an exception is raised if a cube has multiple model_id
        points."""
        cube1 = self.cube.copy()
        model_id_coord = DimCoord(
            np.array([1000], np.int), long_name='model_id')
        cube1.add_aux_coord(model_id_coord)
        cube1 = iris.util.new_axis(cube1, "model_id")
        cube2 = cube1.copy()
        cube2.coord("model_id").points = 200
        cube1 = iris.cube.CubeList([cube1, cube2]).concatenate_cube()
        cube2 = self.cube.copy()[0]
        cube2.remove_coord("realization")
        cubes = iris.cube.CubeList([cube1, cube2])
        msg = "Model_id has more than one point"
        with self.assertRaisesRegex(ValueError, msg):
            _equalise_cube_coords(cubes)

    def test_model_id_with_realization_in_cube(self):
        """Test if model_id is an unmatched coordinate, a cube has a
        realization coordinate and the cube being inspected has a realization
        coordinate."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()[0]
        cube2.remove_coord("realization")
        model_id_coord = DimCoord(
            np.array([1000*1], np.int), long_name='model_id')
        cube1.add_aux_coord(model_id_coord)
        cube1 = iris.util.new_axis(cube1, "model_id")
        cubes = iris.cube.CubeList([cube1, cube2])
        result = _equalise_cube_coords(cubes)
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(len(result), 4)
        self.assertTrue(result[0].coords("realization"))
        self.assertFalse(result[3].coords("realization"))
        self.assertTrue(result[0].coords("model_id"))

    def test_model_id_with_realization_not_in_cube(self):
        """Test if model_id is an unmatched coordinate, a cube has a
        realization coordinate and the cube being inspected does not have a
        realization coordinate."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube1.remove_coord("realization")
        model_id_coord = DimCoord(
            np.array([1000*1], np.int), long_name='model_id')
        cube2.add_aux_coord(model_id_coord)
        cube2 = iris.util.new_axis(cube2, "model_id")
        cubes = iris.cube.CubeList([cube1, cube2])
        result = _equalise_cube_coords(cubes)
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(len(result), 4)
        self.assertFalse(result[0].coords("realization"))
        self.assertTrue(result[1].coords("realization"))
        self.assertTrue(result[1].coords("model_id"))


if __name__ == '__main__':
    unittest.main()
