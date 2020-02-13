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
"""Unit tests for the cube_combiner.CubeCombiner plugin."""
import unittest
from datetime import datetime

import iris
import numpy as np
from iris.cube import Cube
from iris.tests import IrisTest

from improver.cube_combiner import CubeCombiner

from ..set_up_test_cubes import set_up_probability_cube


class Test__init__(IrisTest):

    """Test the __init__ method."""

    def test_basic(self):
        """Test that the __init__ sets things up correctly"""
        plugin = CubeCombiner('+')
        self.assertEqual(plugin.operation, '+')

    def test_raise_error_wrong_operation(self):
        """Test __init__ raises a ValueError for invalid operation"""
        msg = 'Unknown operation '
        with self.assertRaisesRegex(ValueError, msg):
            CubeCombiner('%')


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(CubeCombiner('+'))
        msg = '<CubeCombiner: operation=+, warnings_on = False>'
        self.assertEqual(result, msg)


class CombinerTest(IrisTest):
    """Set up a common set of test cubes for subsequent test classes."""

    def setUp(self):
        """ Set up cubes for testing. """
        data = np.full((1, 2, 2), 0.5, dtype=np.float32)
        self.cube1 = set_up_probability_cube(
            data, np.array([0.001], dtype=np.float32),
            variable_name="lwe_thickness_of_precipitation_amount",
            time=datetime(2015, 11, 19, 0),
            time_bounds=(datetime(2015, 11, 18, 23),
                         datetime(2015, 11, 19, 0)),
            frt=datetime(2015, 11, 18, 22))

        data = np.full((1, 2, 2), 0.6, dtype=np.float32)
        self.cube2 = set_up_probability_cube(
            data, np.array([0.001], dtype=np.float32),
            variable_name="lwe_thickness_of_precipitation_amount",
            time=datetime(2015, 11, 19, 1),
            time_bounds=(datetime(2015, 11, 19, 0),
                         datetime(2015, 11, 19, 1)),
            frt=datetime(2015, 11, 18, 22))

        data = np.full((1, 2, 2), 0.1, dtype=np.float32)
        self.cube3 = set_up_probability_cube(
            data, np.array([0.001], dtype=np.float32),
            variable_name="lwe_thickness_of_precipitation_amount",
            time=datetime(2015, 11, 19, 1),
            time_bounds=(datetime(2015, 11, 19, 0),
                         datetime(2015, 11, 19, 1)),
            frt=datetime(2015, 11, 18, 22))


class Test__get_expanded_coord_names(CombinerTest):
    """Test method to determine coordinates for expansion"""

    def test_basic(self):
        """Test correct names are returned for scalar coordinates with
        different values"""
        expected_coord_set = {'time', 'forecast_period'}
        result = CubeCombiner('+')._get_expanded_coord_names([
            self.cube1, self.cube2, self.cube3])
        self.assertIsInstance(result, list)
        self.assertSetEqual(set(result), expected_coord_set)

    def test_identical_inputs(self):
        """Test no coordinates are returned if inputs are identical"""
        result = CubeCombiner('+')._get_expanded_coord_names([
            self.cube1, self.cube1, self.cube1])
        self.assertFalse(result)

    def test_unmatched_coords_ignored(self):
        """Test coordinates that are not present on all cubes are ignored,
        regardless of input order"""
        expected_coord_set = {'time', 'forecast_period'}
        height = iris.coords.AuxCoord([1.5], "height", units="m")
        self.cube1.add_aux_coord(height)
        result = CubeCombiner('+')._get_expanded_coord_names([
            self.cube1, self.cube2, self.cube3])
        self.assertSetEqual(set(result), expected_coord_set)
        result = CubeCombiner('+')._get_expanded_coord_names([
            self.cube3, self.cube2, self.cube1])
        self.assertSetEqual(set(result), expected_coord_set)


class Test_process(CombinerTest):

    """Test the plugin combines the cubelist into a cube."""

    def test_basic(self):
        """Test that the plugin returns a Cube. """
        plugin = CubeCombiner('+')
        cubelist = iris.cube.CubeList([self.cube1, self.cube2])
        result = plugin.process(cubelist, 'new_cube_name')
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), 'new_cube_name')
        expected_data = np.full((1, 2, 2), 1.1, dtype=np.float32)
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_mean(self):
        """Test that the plugin calculates the mean correctly. """
        plugin = CubeCombiner('mean')
        cubelist = iris.cube.CubeList([self.cube1, self.cube2])
        result = plugin.process(cubelist, 'new_cube_name')
        expected_data = np.full((1, 2, 2), 0.55, dtype=np.float32)
        self.assertEqual(result.name(), 'new_cube_name')
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_bounds_expansion(self):
        """Test that the plugin calculates the sum of the input cubes
        correctly and expands the time coordinate bounds on the
        resulting output."""
        plugin = CubeCombiner('add')
        cubelist = iris.cube.CubeList([self.cube1, self.cube2])
        result = plugin.process(cubelist, 'new_cube_name')
        expected_data = np.full((1, 2, 2), 1.1, dtype=np.float32)
        self.assertEqual(result.name(), 'new_cube_name')
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertEqual(result.coord('time').points[0], 1447894800)
        self.assertArrayEqual(result.coord('time').bounds,
                              [[1447887600, 1447894800]])

    def test_bounds_expansion_midpoint(self):
        """Test option to use the midpoint between the bounds as the time
        coordinate point, rather than the (default) maximum."""
        plugin = CubeCombiner('add')
        cubelist = iris.cube.CubeList([self.cube1, self.cube2])
        result = plugin.process(cubelist, 'new_cube_name', use_midpoint=True)
        self.assertEqual(result.name(), 'new_cube_name')
        self.assertEqual(result.coord('time').points[0], 1447891200)
        self.assertArrayEqual(result.coord('time').bounds,
                              [[1447887600, 1447894800]])

    def test_unmatched_scalar_coords(self):
        """Test a scalar coordinate that is present on the first cube is
        present unmodified on the output; and if present on a later cube is
        not present on the output."""
        height = iris.coords.AuxCoord([1.5], "height", units="m")
        self.cube1.add_aux_coord(height)
        result = CubeCombiner('add').process(
            [self.cube1, self.cube2], 'new_cube_name')
        self.assertEqual(result.coord("height"), height)
        result = CubeCombiner('add').process(
            [self.cube2, self.cube1], 'new_cube_name')
        result_coords = [coord.name() for coord in result.coords()]
        self.assertNotIn("height", result_coords)

    def test_mean_multi_cube(self):
        """Test that the plugin calculates the mean for three cubes."""
        plugin = CubeCombiner('mean')
        cubelist = iris.cube.CubeList([self.cube1,
                                       self.cube2,
                                       self.cube3])
        result = plugin.process(cubelist, 'new_cube_name')
        expected_data = np.full((1, 2, 2), 0.4, dtype=np.float32)
        self.assertEqual(result.name(), 'new_cube_name')
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_with_mask(self):
        """Test that the plugin preserves the mask if any of the inputs are
        masked"""
        expected_data = np.full((1, 2, 2), 1.2, dtype=np.float32)
        mask = [[[False, True], [False, False]]]
        self.cube1.data = np.ma.MaskedArray(self.cube1.data, mask=mask)
        plugin = CubeCombiner('add')
        result = plugin.process(
            [self.cube1, self.cube2, self.cube3], 'new_cube_name')
        self.assertIsInstance(result.data, np.ma.MaskedArray)
        self.assertArrayAlmostEqual(result.data.data, expected_data)
        self.assertArrayEqual(result.data.mask, mask)

    def test_exception_mismatched_dimensions(self):
        """Test an error is raised if dimension coordinates do not match"""
        self.cube2.coord("lwe_thickness_of_precipitation_amount").rename(
            "snow_depth")
        plugin = CubeCombiner('+')
        msg = "Cannot combine cubes with different dimensions"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process([self.cube1, self.cube2], 'new_cube_name')

    def test_exception_for_single_entry_cubelist(self):
        """Test that the plugin raises an exception if a cubelist containing
        only one cube is passed in."""
        plugin = CubeCombiner('-')
        msg = "Expecting 2 or more cubes in cube_list"
        cubelist = iris.cube.CubeList([self.cube1])
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(cubelist, 'new_cube_name')


if __name__ == '__main__':
    unittest.main()
