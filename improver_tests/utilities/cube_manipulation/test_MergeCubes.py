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
Unit tests for the "cube_manipulation.MergeCubes" plugin.
"""

import unittest
from datetime import datetime as dt

import iris
import numpy as np
from iris.exceptions import DuplicateDataError
from iris.tests import IrisTest

from improver.utilities.cube_manipulation import MergeCubes

from ...set_up_test_cubes import set_up_probability_cube, set_up_variable_cube


class Test__init__(IrisTest):
    """Test the __init__ method"""

    def test_basic(self):
        """Test default parameters"""
        plugin = MergeCubes()
        self.assertSequenceEqual(plugin.silent_attributes,
                                 ["history", "title", "mosg__grid_version"])


class Test__equalise_cell_methods(IrisTest):
    """Test the _equalise_cell_methods method"""

    def setUp(self):
        """Use temperature probability cube to test with."""
        data = np.array(
            [0.9*np.ones((3, 3)), 0.5*np.ones((3, 3)), 0.1*np.ones((3, 3))],
            dtype=np.float32)
        thresholds = np.array([273., 275., 277.], dtype=np.float32)
        self.cube = set_up_probability_cube(data.copy(), thresholds)
        self.cell_method1 = iris.coords.CellMethod("mean", "realization")
        self.cell_method2 = iris.coords.CellMethod("mean", "time")
        self.cell_method3 = iris.coords.CellMethod("max", "neighbourhood")
        self.plugin = MergeCubes()

    def test_basic(self):
        """Test returns an iris.cube.CubeList."""
        cubelist = iris.cube.CubeList([self.cube, self.cube])
        self.plugin._equalise_cell_methods(cubelist)
        self.assertIsInstance(cubelist, iris.cube.CubeList)
        self.assertEqual(len(cubelist), 2)
        self.assertTrue(cubelist[0].is_compatible(cubelist[1]))

    def test_different_cell_methods(self):
        """Test returns an iris.cube.CubeList with matching cell methods."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube3 = self.cube.copy()
        cube1.cell_methods = tuple([self.cell_method1, self.cell_method2])
        cube2.cell_methods = tuple([self.cell_method1, self.cell_method2,
                                    self.cell_method3])
        cube3.cell_methods = tuple([self.cell_method1, self.cell_method3])
        cubelist = iris.cube.CubeList([cube1, cube2, cube3])
        self.plugin._equalise_cell_methods(cubelist)
        self.assertIsInstance(cubelist, iris.cube.CubeList)
        self.assertEqual(len(cubelist[0].cell_methods), 1)
        check = cubelist[1].cell_methods[0] == self.cell_method1
        self.assertTrue(check)


class Test__check_time_bounds_ranges(IrisTest):
    """Test the _check_time_bounds_ranges method"""

    def setUp(self):
        """Set up some cubes with different time bounds ranges"""
        frt = dt(2017, 11, 9, 21, 0)
        times = [dt(2017, 11, 10, 3, 0),
                 dt(2017, 11, 10, 4, 0),
                 dt(2017, 11, 10, 5, 0)]
        time_bounds = np.array([
            [dt(2017, 11, 10, 2, 0), dt(2017, 11, 10, 3, 0)],
            [dt(2017, 11, 10, 3, 0), dt(2017, 11, 10, 4, 0)],
            [dt(2017, 11, 10, 4, 0), dt(2017, 11, 10, 5, 0)]])

        cubes = iris.cube.CubeList([])
        for tpoint, tbounds in zip(times, time_bounds):
            cube = set_up_probability_cube(
                0.6*np.ones((2, 3, 3), dtype=np.float32),
                np.array([278., 280.], dtype=np.float32),
                time=tpoint, frt=frt, time_bounds=tbounds)
            cubes.append(cube)
        self.matched_cube = cubes.merge_cube()

        time_bounds[2, 0] = dt(2017, 11, 10, 2, 0)
        cubes = iris.cube.CubeList([])
        for tpoint, tbounds in zip(times, time_bounds):
            cube = set_up_probability_cube(
                0.6*np.ones((2, 3, 3), dtype=np.float32),
                np.array([278., 280.], dtype=np.float32),
                time=tpoint, frt=frt, time_bounds=tbounds)
            cubes.append(cube)
        self.unmatched_cube = cubes.merge_cube()
        self.plugin = MergeCubes()

    def test_basic(self):
        """Test no error when bounds match"""
        self.plugin._check_time_bounds_ranges(self.matched_cube)

    def test_inverted(self):
        """Test no error when bounds ranges match but bounds are in the wrong
        order"""
        inverted_bounds = np.flip(
            self.matched_cube.coord("time").bounds.copy(), axis=1)
        self.matched_cube.coord("time").bounds = inverted_bounds
        self.plugin._check_time_bounds_ranges(self.matched_cube)

    def test_error(self):
        """Test error when bounds do not match"""
        msg = 'Cube with mismatching time bounds ranges'
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin._check_time_bounds_ranges(self.unmatched_cube)

    def test_no_error_missing_coord(self):
        """Test missing time or forecast period coordinate does not raise
        error"""
        self.matched_cube.remove_coord("forecast_period")
        self.plugin._check_time_bounds_ranges(self.matched_cube)


class Test_process(IrisTest):
    """Test the process method (see also test_merge_cubes.py)"""

    def setUp(self):
        """Use temperature exceedance probability cubes to test with."""
        data = np.ones((2, 3, 3), dtype=np.float32)
        thresholds = np.array([274, 275], dtype=np.float32)
        time_point = dt(2015, 11, 23, 7)

        # set up some UKV cubes with 4, 5 and 6 hour forecast periods and
        # different histories
        self.cube_ukv = set_up_probability_cube(
            data.copy(), thresholds.copy(), standard_grid_metadata='uk_det',
            time=time_point, frt=dt(2015, 11, 23, 3),
            attributes={'history': 'something'})

        self.cube_ukv_t1 = set_up_probability_cube(
            data.copy(), thresholds.copy(), standard_grid_metadata='uk_det',
            time=time_point, frt=dt(2015, 11, 23, 2),
            attributes={'history': 'different'})

        self.cube_ukv_t2 = set_up_probability_cube(
            data.copy(), thresholds.copy(), standard_grid_metadata='uk_det',
            time=time_point, frt=dt(2015, 11, 23, 1),
            attributes={'history': 'entirely'})

        self.plugin = MergeCubes()

    def test_basic(self):
        """Test that the utility returns an iris.cube.Cube"""
        result = self.plugin.process([self.cube_ukv, self.cube_ukv_t1])
        self.assertIsInstance(result, iris.cube.Cube)

    def test_null(self):
        """Test single cube is returned unmodified"""
        cube = self.cube_ukv.copy()
        result = self.plugin.process(cube)
        self.assertArrayAlmostEqual(result.data, self.cube_ukv.data)
        self.assertEqual(result.metadata, self.cube_ukv.metadata)

    def test_single_item_list(self):
        """Test cube from single item list is returned unmodified"""
        cubelist = iris.cube.CubeList([self.cube_ukv.copy()])
        result = self.plugin.process(cubelist)
        self.assertArrayAlmostEqual(result.data, self.cube_ukv.data)
        self.assertEqual(result.metadata, self.cube_ukv.metadata)

    def test_unmatched_attributes(self):
        """Test that unmatched attributes are removed without modifying the
        input cubes"""
        result = self.plugin.process([self.cube_ukv, self.cube_ukv_t1])
        self.assertNotIn("history", result.attributes.keys())
        self.assertEqual(self.cube_ukv.attributes['history'], 'something')
        self.assertEqual(self.cube_ukv_t1.attributes['history'], 'different')

    def test_identical_cubes(self):
        """Test that merging identical cubes fails."""
        cubes = iris.cube.CubeList([self.cube_ukv, self.cube_ukv])
        msg = "failed to merge into a single cube"
        with self.assertRaisesRegex(DuplicateDataError, msg):
            self.plugin.process(cubes)

    def test_lagged_ukv(self):
        """Test lagged UKV merge OK (forecast periods in seconds)"""
        expected_fp_points = 3600*np.array([6, 5, 4], dtype=np.int32)
        cubes = iris.cube.CubeList([self.cube_ukv,
                                    self.cube_ukv_t1,
                                    self.cube_ukv_t2])
        result = self.plugin.process(cubes)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.coord("forecast_period").points, expected_fp_points)

    def test_check_time_bounds_ranges(self):
        """Test optional failure when time bounds ranges are not matched
        (eg if merging cubes with different accumulation periods)"""
        time_point = dt(2015, 11, 23, 7)
        time_bounds = [dt(2015, 11, 23, 4), time_point]
        cube1 = set_up_variable_cube(
            self.cube_ukv.data.copy(), standard_grid_metadata='uk_det',
            time=time_point, frt=dt(2015, 11, 23, 3), time_bounds=time_bounds)
        cube2 = cube1.copy()
        cube2.coord("forecast_reference_time").points = (
            cube2.coord("forecast_reference_time").points + 3600)
        cube2.coord("time").bounds = [
            cube2.coord("time").bounds[0, 0] + 3600,
            cube2.coord("time").bounds[0, 1]]
        cube2.coord("forecast_period").bounds = [
            cube2.coord("forecast_period").bounds[0, 0] + 3600,
            cube2.coord("forecast_period").bounds[0, 1]]
        msg = "Cube with mismatching time bounds ranges cannot be blended"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin.process([cube1, cube2], check_time_bounds_ranges=True)


if __name__ == '__main__':
    unittest.main()
