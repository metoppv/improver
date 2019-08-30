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
from improver.tests.set_up_test_cubes import set_up_probability_cube

TIME_UNIT = 'seconds since 1970-01-01 00:00:00'
CALENDAR = 'gregorian'


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


class set_up_cubes(IrisTest):
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
            frt=datetime(2015, 11, 18, 22),
            attributes={'attribute_to_update': 'first_value'})

        data = np.full((1, 2, 2), 0.6, dtype=np.float32)
        self.cube2 = set_up_probability_cube(
            data, np.array([0.001], dtype=np.float32),
            variable_name="lwe_thickness_of_precipitation_amount",
            time=datetime(2015, 11, 19, 1),
            time_bounds=(datetime(2015, 11, 19, 0),
                         datetime(2015, 11, 19, 1)),
            frt=datetime(2015, 11, 18, 22),
            attributes={'attribute_to_update': 'first_value'})

        data = np.full((1, 2, 2), 0.1, dtype=np.float32)
        self.cube3 = set_up_probability_cube(
            data, np.array([0.001], dtype=np.float32),
            variable_name="lwe_thickness_of_precipitation_amount",
            time=datetime(2015, 11, 19, 1),
            time_bounds=(datetime(2015, 11, 19, 0),
                         datetime(2015, 11, 19, 1)),
            frt=datetime(2015, 11, 18, 22),
            attributes={'attribute_to_update': 'first_value'})


class Test_combine(set_up_cubes):

    """Test the combine method."""

    def test_basic(self):
        """Test that the function returns a Cube. """
        operation = '*'
        plugin = CubeCombiner(operation)
        result = plugin.combine(self.cube1, self.cube2)
        self.assertIsInstance(result, Cube)
        expected_data = np.full((1, 2, 2), 0.3, dtype=np.float32)
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_add(self):
        """Test combine adds the cubes correctly. """
        operation = '+'
        plugin = CubeCombiner(operation)
        result = plugin.combine(self.cube1, self.cube2)
        expected_data = np.full((1, 2, 2), 1.1, dtype=np.float32)
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_minus(self):
        """Test combine minus the cubes correctly. """
        operation = '-'
        plugin = CubeCombiner(operation)
        result = plugin.combine(self.cube1, self.cube2)
        expected_data = np.full((1, 2, 2), -0.1, dtype=np.float32)
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_max(self):
        """Test combine finds the max of the cubes correctly."""
        operation = 'max'
        plugin = CubeCombiner(operation)
        result = plugin.combine(self.cube1, self.cube2)
        expected_data = np.full((1, 2, 2), 0.6, dtype=np.float32)
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_min(self):
        """Test combine finds the min of the cubes correctly."""
        operation = 'min'
        plugin = CubeCombiner(operation)
        result = plugin.combine(self.cube1, self.cube2)
        expected_data = np.full((1, 2, 2), 0.5, dtype=np.float32)
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_mean(self):
        """Test that the function adds the cubes correctly for mean."""
        operation = 'mean'
        plugin = CubeCombiner(operation)
        result = plugin.combine(self.cube1, self.cube2)
        expected_data = np.full((1, 2, 2), 1.1, dtype=np.float32)
        self.assertArrayAlmostEqual(result.data, expected_data)


class Test_process(set_up_cubes):

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
        correctly and expands the requested coordinate bounds in the
        resulting output."""
        plugin = CubeCombiner('add')
        expanded_coord = {'time': 'upper'}
        cubelist = iris.cube.CubeList([self.cube1, self.cube2])
        result = plugin.process(cubelist, 'new_cube_name',
                                expanded_coord=expanded_coord)
        expected_data = np.full((1, 2, 2), 1.1, dtype=np.float32)
        self.assertEqual(result.name(), 'new_cube_name')
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertEqual(result.coord('time').points[0], 1447894800)
        self.assertArrayEqual(result.coord('time').bounds,
                              [[1447887600, 1447894800]])

    def test_mean_multi_cube(self):
        """Test that the plugin calculates the mean for three cubes. """
        plugin = CubeCombiner('mean')
        cubelist = iris.cube.CubeList([self.cube1,
                                       self.cube2,
                                       self.cube3])
        result = plugin.process(cubelist, 'new_cube_name')
        expected_data = np.full((1, 2, 2), 0.4, dtype=np.float32)
        self.assertEqual(result.name(), 'new_cube_name')
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_exception_for_cube_passed_in(self):
        """Test that the plugin raises an exception if something other than a
        cubelist is passed in."""
        plugin = CubeCombiner('-')
        msg = "Expecting data to be an instance of"

        with self.assertRaisesRegex(TypeError, msg):
            plugin.process(self.cube1, 'new_cube_name')

    def test_exception_for_single_entry_cubelist(self):
        """Test that the plugin raises an exception if a cubelist containing
        only one cube is passed in."""
        plugin = CubeCombiner('-')
        msg = "Expecting 2 or more cubes in cube_list"

        cubelist = iris.cube.CubeList([self.cube1])
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(cubelist, 'new_cube_name')

    def test_revised_coords(self):
        """Test that the plugin passes through the relevant dictionary to
        modify a coordinate and that these modifications are present in the
        returned cube."""
        plugin = CubeCombiner('mean')
        revised_coords = {'lwe_thickness_of_precipitation_amount': {
            'points': [2.0]}}
        cubelist = iris.cube.CubeList([self.cube1, self.cube2])
        result = plugin.process(cubelist, 'new_cube_name',
                                revised_coords=revised_coords)
        self.assertEqual(
            result.coord('lwe_thickness_of_precipitation_amount').points[0],
            2.0)

    def test_revised_attributes(self):
        """Test that the plugin passes through the relevant dictionary to
        modify an attribute and that these modifications are present in the
        returned cube."""
        plugin = CubeCombiner('mean')
        revised_attributes = {'attribute_to_update': 'second_value'}
        cubelist = iris.cube.CubeList([self.cube1, self.cube2])
        result = plugin.process(cubelist, 'new_cube_name',
                                revised_attributes=revised_attributes)
        self.assertEqual(result.attributes['attribute_to_update'],
                         'second_value')


if __name__ == '__main__':
    unittest.main()
