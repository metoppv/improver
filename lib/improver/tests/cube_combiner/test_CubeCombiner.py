# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
"""Unit tests for the windgust_diagnostic.WindGustDiagnostic plugin."""
import unittest

import numpy as np

import iris
from iris.tests import IrisTest
from iris.cube import Cube
from iris.coords import DimCoord
from cf_units import Unit

from improver.cube_combiner import CubeCombiner


def create_cube_with_threshold(data=None,
                               long_name=None,
                               threshold_values=None,
                               units=None):
    """Create a cube with threshold coord."""
    if threshold_values is None:
        threshold_values = [1.0]
    if data is None:
        data = np.zeros((len(threshold_values), 2, 2, 2))
        data[:, 0, :, :] = 0.5
        data[:, 1, :, :] = 0.6
    if long_name is None:
        long_name = "probability_of_rainfall_rate"
    if units is None:
        units = "m s^-1"

    cube = Cube(data, long_name=long_name, units='1')
    cube.add_dim_coord(DimCoord(np.linspace(-45.0, 45.0, 2), 'latitude',
                                units='degrees'), 2)
    cube.add_dim_coord(DimCoord(np.linspace(120, 180, 2), 'longitude',
                                units='degrees'), 3)
    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"
    tunit = Unit(time_origin, calendar)
    cube.add_dim_coord(DimCoord([402192.5, 402193.5],
                                "time", units=tunit), 1)
    cube.add_dim_coord(DimCoord(threshold_values,
                                long_name='threshold',
                                units=units), 0)
    return cube


class Test__init__(IrisTest):

    """Test the __init__ method."""

    def test_basic(self):
        """Test that the __init__ sets things up correctly"""
        plugin = CubeCombiner('+')
        self.assertEqual(plugin.operation, '+')

    def test_raise_error_wrong_operation(self):
        """Test __init__ raises a ValueError for invalid operation"""
        msg = 'Unknown operation '
        with self.assertRaisesRegexp(ValueError, msg):
            CubeCombiner('%')


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(CubeCombiner('+'))
        msg = '<CubeCombiner: operation=+>'
        self.assertEqual(result, msg)


class Test_resolve_metadata_diff(IrisTest):

    """Test the resolve_metadata_diff method."""

    def test_basic(self):
        """Test that the function returns a tuple of Cubes. """
        plugin = CubeCombiner('-')
        cube1 = create_cube_with_threshold()
        cube2 = cube1.copy()
        result = plugin.resolve_metadata_diff(cube1, cube2)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], Cube)
        self.assertIsInstance(result[1], Cube)


class Test_combine(IrisTest):

    """Test the combine method."""

    def setUp(self):
        """ Set up cubes for testing. """
        self.cube1 = create_cube_with_threshold()
        data = np.zeros((1, 2, 2, 2))
        data[0, 0, :, :] = 0.1
        data[0, 1, :, :] = 0.4
        self.cube2 = create_cube_with_threshold(data=data)
        data2 = np.zeros((1, 2, 2, 2))
        data2[0, 0, :, :] = 0.1
        data2[0, 1, :, :] = 0.8
        self.cube3 = create_cube_with_threshold(data=data2)

    def test_basic(self):
        """Test that the function returns a Cube. """
        operation = '*'
        plugin = CubeCombiner(operation)
        cube1 = self.cube1
        cube2 = cube1.copy()
        result = plugin.combine(cube1, cube2, operation,
                                'new_cube_name')
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), 'new_cube_name')
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[0, 0, :, :] = 0.25
        expected_data[0, 1, :, :] = 0.36
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_add(self):
        """Test combine adds the cubes correctly. """
        operation = '+'
        plugin = CubeCombiner(operation)
        result = plugin.combine(self.cube1, self.cube2,
                                operation,
                                'new_cube_name')
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[0, 0, :, :] = 0.6
        expected_data[0, 1, :, :] = 1.0
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_minus(self):
        """Test combine minus the cubes correctly. """
        operation = '-'
        plugin = CubeCombiner(operation)
        result = plugin.combine(self.cube1, self.cube2,
                                operation,
                                'new_cube_name')
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[0, 0, :, :] = 0.4
        expected_data[0, 1, :, :] = 0.2
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_max(self):
        """Test combine finds the max of the cubes correctly."""
        operation = 'max'
        plugin = CubeCombiner(operation)
        result = plugin.combine(self.cube1, self.cube3,
                                operation,
                                'new_cube_name')
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[0, 0, :, :] = 0.5
        expected_data[0, 1, :, :] = 0.8
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_min(self):
        """Test combine finds the min of the cubes correctly."""
        operation = 'min'
        plugin = CubeCombiner(operation)
        result = plugin.combine(self.cube1, self.cube3,
                                operation,
                                'new_cube_name')
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[0, 0, :, :] = 0.1
        expected_data[0, 1, :, :] = 0.6
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_mean(self):
        """Test that the function adds the cubes correctly for mean."""
        operation = '+'
        plugin = CubeCombiner(operation)
        result = plugin.combine(self.cube1, self.cube3,
                                operation,
                                'new_cube_name')
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[0, 0, :, :] = 0.6
        expected_data[0, 1, :, :] = 1.4
        self.assertArrayAlmostEqual(result.data, expected_data)


class Test_process(IrisTest):

    """Test the plugin combines the cubelist into a cube."""

    def setUp(self):
        """ Set up cubes for testing. """
        self.cube1 = create_cube_with_threshold()
        data = np.zeros((1, 2, 2, 2))
        data[0, 0, :, :] = 0.1
        data[0, 1, :, :] = 0.4
        self.cube2 = create_cube_with_threshold(data=data)
        data2 = np.zeros((1, 2, 2, 2))
        data2[0, 0, :, :] = 0.9
        data2[0, 1, :, :] = 0.2
        self.cube3 = create_cube_with_threshold(data=data2)

    def test_basic(self):
        """Test that the plugin returns a Cube. """
        plugin = CubeCombiner('+')
        cubelist = iris.cube.CubeList([self.cube1, self.cube1])
        result = plugin.process(cubelist, 'new_cube_name')
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), 'new_cube_name')
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[:, 0, :, :] = 1.0
        expected_data[:, 1, :, :] = 1.2
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_mean(self):
        """Test that the plugin calculates the mean correctly. """
        plugin = CubeCombiner('mean')
        cubelist = iris.cube.CubeList([self.cube1, self.cube2])
        result = plugin.process(cubelist, 'new_cube_name')
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[:, 0, :, :] = 0.3
        expected_data[:, 1, :, :] = 0.5
        self.assertEqual(result.name(), 'new_cube_name')
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_mean_multi_cube(self):
        """Test that the plugin calculates the mean for three cubes. """
        plugin = CubeCombiner('mean')
        cubelist = iris.cube.CubeList([self.cube1,
                                       self.cube2,
                                       self.cube3])
        result = plugin.process(cubelist, 'new_cube_name')
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[:, 0, :, :] = 0.5
        expected_data[:, 1, :, :] = 0.4
        self.assertEqual(result.name(), 'new_cube_name')
        self.assertArrayAlmostEqual(result.data, expected_data)


if __name__ == '__main__':
    unittest.main()
