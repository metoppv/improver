# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
Unit tests for the "cube_manipulation.ConcatenateCubes" plugin.
"""

import unittest
from datetime import datetime as dt

import iris
import numpy as np
from iris.coords import DimCoord
from iris.tests import IrisTest

from improver.utilities.cube_manipulation import (
    ConcatenateCubes,
    enforce_coordinate_ordering,
)

from ...set_up_test_cubes import set_up_variable_cube


class Test__init__(IrisTest):
    """Test the __init__ method"""

    def test_default(self):
        """Test default settings"""
        plugin = ConcatenateCubes()
        self.assertFalse(plugin.coords_to_associate)
        self.assertFalse(plugin.coords_to_slice_over)

    def test_arguments(self):
        """Test custom arguments"""
        plugin = ConcatenateCubes(coords_to_slice_over=["time"])
        self.assertDictEqual(plugin.coords_to_associate, {"time": "forecast_period"})
        self.assertSequenceEqual(plugin.coords_to_slice_over, ["time"])


class Test__slice_over_coordinate(IrisTest):
    """Test the _slice_over_coordinate method"""

    def setUp(self):
        """Set up default plugin and test cube with dimensions:
        realization (3), latitude (3), longitude (3)"""
        self.plugin = ConcatenateCubes(coords_to_slice_over="time")
        data = 275.0 * np.ones((3, 3, 3), dtype=np.float32)
        self.cube = set_up_variable_cube(
            data, time=dt(2017, 1, 10, 3), frt=dt(2017, 1, 10, 0)
        )

    def test_basic(self):
        """Test that the utility returns an iris.cube.CubeList when given an
        iris.cube.CubeList instance."""
        cubelist = iris.cube.CubeList([self.cube])
        result = self.plugin._slice_over_coordinate(cubelist, "time")
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_basic_cube(self):
        """Test that the utility returns an iris.cube.CubeList when given an
        iris.cube.Cube instance."""
        result = self.plugin._slice_over_coordinate(self.cube, "time")
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_association(self):
        """Test that with the default setup, forecast period is associated with
        the new time dimension"""
        (result_cube,) = self.plugin._slice_over_coordinate(self.cube, "time")
        self.assertEqual(result_cube.coord_dims("time")[0], 0)
        self.assertEqual(result_cube.coord_dims("forecast_period")[0], 0)

    def test_reordering(self):
        """Test that the sliced coordinate is the first dimension in each
        output cube in the list, regardless of input dimension order"""
        enforce_coordinate_ordering(self.cube, "realization", anchor_start=False)
        plugin = ConcatenateCubes(coords_to_slice_over="realization")
        result = plugin._slice_over_coordinate(self.cube, "realization")
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].coord_dims("realization")[0], 0)


class Test_process(IrisTest):
    """Test the process method (see also test_concatenate_cubes.py)"""

    def setUp(self):
        """Set up default plugin and test cubes."""
        self.plugin = ConcatenateCubes(coords_to_slice_over="realization")
        data = 275.0 * np.ones((3, 3, 3), dtype=np.float32)
        cube1 = set_up_variable_cube(
            data, time=dt(2017, 1, 10, 3), frt=dt(2017, 1, 10, 0)
        )
        cube2 = cube1.copy(data=data + 1.0)
        cube2.coord("realization").points = [0, 4, 5]
        cube1.coord("realization").points = [1, 2, 3]
        self.cubelist = iris.cube.CubeList([cube1, cube2])
        self.cube = cube1.copy()

    def test_basic(self):
        """Test that the utility returns an iris.cube.Cube."""
        result = self.plugin.process(self.cubelist)
        self.assertIsInstance(result, iris.cube.Cube)

    def test_single_cube(self):
        """Test a single cube is returned unchanged"""
        result = self.plugin.process(self.cube)
        self.assertArrayAlmostEqual(result.data, self.cube.data)
        self.assertEqual(result.metadata, self.cube.metadata)

    def test_single_item_cubelist(self):
        """Test a single item cubelist returns the cube unchanged"""
        result = self.plugin.process([self.cube])
        self.assertArrayAlmostEqual(result.data, self.cube.data)
        self.assertEqual(result.metadata, self.cube.metadata)

    def test_identical_cubes(self):
        """
        Test that the utility returns the expected error message,
        if an attempt is made to concatenate identical cubes.
        """
        cubes = iris.cube.CubeList([self.cube, self.cube])
        msg = "An unexpected problem prevented concatenation."
        with self.assertRaisesRegex(iris.exceptions.ConcatenateError, msg):
            self.plugin.process(cubes)

    def test_values(self):
        """
        Test that the utility returns an iris.cube.Cube with the expected
        concatenated coordinate order and expected data
        """
        data1 = next(self.cubelist[0].slices_over("realization")).data
        data2 = next(self.cubelist[1].slices_over("realization")).data
        expected_data = np.array([data2, data1, data1, data1, data2, data2])
        result = self.plugin.process(self.cubelist)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertArrayEqual(result.coord("realization").points, np.arange(6))

    def test_cubelist_different_number_of_realizations(self):
        """
        Test that the utility returns an iris.cube.Cube with the expected
        realizations, if a CubeList containing cubes with different numbers
        of realizations are passed in as the input.
        """
        cube1 = next(self.cubelist[1].slices_over("realization"))
        result = self.plugin.process([cube1, self.cube])
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.coord("realization").points, np.arange(4))

    def test_cubelist_different_number_of_realizations_time(self):
        """
        Test that the utility returns the expected error message, if a
        CubeList containing cubes with different numbers of realizations are
        passed in as the input, but inputs are only sliced over time.
        """
        plugin = ConcatenateCubes(coords_to_slice_over=["time"])
        cube1 = next(self.cubelist[1].slices_over("realization"))
        msg = "failed to concatenate into a single cube"
        with self.assertRaisesRegex(iris.exceptions.ConcatenateError, msg):
            plugin.process([cube1, self.cube])

    def test_cubelist_concatenate_time(self):
        """
        Test that the utility returns an iris.cube.Cube with the expected
        time and associated forecast period coordinates, if a CubeList
        containing cubes with different timesteps is passed in as the input.
        """
        plugin = ConcatenateCubes(coords_to_slice_over=["time"])

        cube2 = self.cube.copy()
        cube2.coord("time").points = cube2.coord("time").points + 3600
        cube2.coord("forecast_period").points = (
            cube2.coord("forecast_period").points - 3600
        )

        expected_time_points = [
            self.cube.coord("time").points[0],
            cube2.coord("time").points[0],
        ]
        expected_fp_points = [
            self.cube.coord("forecast_period").points[0],
            cube2.coord("forecast_period").points[0],
        ]

        result = plugin.process([self.cube, cube2])
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.coord("time").points, expected_time_points)
        self.assertArrayEqual(
            result.coord("forecast_period").points, expected_fp_points
        )

    def test_cubelist_different_var_names(self):
        """
        Test that the utility returns an iris.cube.Cube, if a CubeList
        containing non-identical cubes is passed in as the input.
        """
        self.cubelist[0].coord("time").var_name = "time_0"
        self.cubelist[1].coord("time").var_name = "time_1"
        result = self.plugin.process(self.cubelist)
        self.assertIsInstance(result, iris.cube.Cube)


if __name__ == "__main__":
    unittest.main()
