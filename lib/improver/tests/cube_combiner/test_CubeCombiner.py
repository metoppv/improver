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

import numpy as np
from cf_units import date2num
from datetime import datetime as dt

import iris
from iris.tests import IrisTest
from iris.cube import Cube

from improver.cube_combiner import CubeCombiner
from improver.tests.utilities.test_cube_metadata import (
    create_cube_with_threshold)
from improver.utilities.warnings_handler import ManageWarnings
from improver.tests.set_up_test_cubes import set_up_variable_cube

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


class Test_expand_bounds(IrisTest):

    """Test expand_bounds method"""

    def setUp(self):
        """Set up a cubelist for testing"""

        data = 275.5*np.ones((3, 3), dtype=np.float32)
        frt = dt(2015, 11, 19, 0)
        time_points = [dt(2015, 11, 19, 1), dt(2015, 11, 19, 3)]
        time_bounds = [[dt(2015, 11, 19, 0), dt(2015, 11, 19, 2)],
                       [dt(2015, 11, 19, 1), dt(2015, 11, 19, 3)]]

        self.cubelist = iris.cube.CubeList([])
        for tpoint, tbounds in zip(time_points, time_bounds):
            cube = set_up_variable_cube(
                data, frt=frt, time=tpoint, time_bounds=tbounds)
            self.cubelist.append(cube)

        self.expected_bounds_seconds = [
            date2num(dt(2015, 11, 19, 0), TIME_UNIT,
                     CALENDAR).astype(np.int64),
            date2num(dt(2015, 11, 19, 3), TIME_UNIT,
                     CALENDAR).astype(np.int64)]

        self.expected_bounds_hours = [
            date2num(dt(2015, 11, 19, 0), 'hours since 1970-01-01 00:00:00',
                     CALENDAR),
            date2num(dt(2015, 11, 19, 3), 'hours since 1970-01-01 00:00:00',
                     CALENDAR)]

    def test_basic_time_mid(self):
        """Test that expand_bound produces sensible bounds
        when given arg 'mid' for times in seconds"""
        time_point = np.around(date2num(dt(2015, 11, 19, 1, 30), TIME_UNIT,
                                        CALENDAR)).astype(np.int64)
        expected_result = iris.coords.DimCoord(
            [time_point], bounds=self.expected_bounds_seconds,
            standard_name='time', units=TIME_UNIT)
        result = CubeCombiner.expand_bounds(
            self.cubelist[0], self.cubelist, 'time', 'mid')
        self.assertEqual(result.coord('time'), expected_result)
        self.assertEqual(result.coord('time').dtype, np.int64)

    def test_time_mid_data_precision(self):
        """Test that expand_bound does not escalate precision when input is
        of dtype int32"""
        expected_result = iris.coords.DimCoord(
            np.array([5400], dtype=np.int32),
            bounds=np.array([0, 10800], dtype=np.int32),
            standard_name='forecast_period', units='seconds')
        result = CubeCombiner.expand_bounds(
            self.cubelist[0], self.cubelist, 'forecast_period', 'mid')
        self.assertEqual(result.coord('forecast_period'), expected_result)
        self.assertEqual(result.coord('forecast_period').dtype, np.int32)

    def test_float_time_mid(self):
        """Test that expand_bound produces sensible bounds
        when given arg 'mid' for times in hours"""
        time_unit = 'hours since 1970-01-01 00:00:00'
        for cube in self.cubelist:
            cube.coord("time").convert_units(time_unit)
        time_point = date2num(dt(2015, 11, 19, 1, 30), time_unit, CALENDAR)
        expected_result = iris.coords.DimCoord(
            [time_point], bounds=self.expected_bounds_hours,
            standard_name='time', units=time_unit)
        result = CubeCombiner.expand_bounds(
            self.cubelist[0], self.cubelist, 'time', 'mid')
        self.assertEqual(result.coord('time'), expected_result)

    def test_basic_time_upper(self):
        """Test that expand_bound produces sensible bounds
        when given arg 'upper'"""
        time_point = np.around(date2num(dt(2015, 11, 19, 3), TIME_UNIT,
                                        CALENDAR)).astype(np.int64)
        expected_result = iris.coords.DimCoord(
            [time_point], bounds=self.expected_bounds_seconds,
            standard_name='time', units=TIME_UNIT)
        result = CubeCombiner.expand_bounds(
            self.cubelist[0], self.cubelist, 'time', 'upper')
        self.assertEqual(result.coord('time'), expected_result)

    def test_basic_no_time_bounds(self):
        """Test that it creates appropriate bounds if there are no time bounds
        """
        for cube in self.cubelist:
            cube.coord('time').bounds = None

        time_point = np.around(date2num(dt(2015, 11, 19, 2), TIME_UNIT,
                                        CALENDAR)).astype(np.int64)
        time_bounds = [
            np.around(date2num(dt(2015, 11, 19, 1), TIME_UNIT,
                               CALENDAR)).astype(np.int64),
            np.around(date2num(dt(2015, 11, 19, 3), TIME_UNIT,
                               CALENDAR)).astype(np.int64)]
        expected_result = iris.coords.DimCoord(
            time_point, bounds=time_bounds,
            standard_name='time', units=TIME_UNIT)

        result = CubeCombiner.expand_bounds(
            self.cubelist[0], self.cubelist, 'time', 'mid')
        self.assertEqual(result.coord('time'), expected_result)

    def test_fails_with_multi_point_coord(self):
        """Test that if an error is raised if a coordinate with more than
        one point is given"""
        emsg = 'the expand bounds function should only be used on a'
        with self.assertRaisesRegex(ValueError, emsg):
            CubeCombiner.expand_bounds(
                self.cubelist[0], self.cubelist, 'latitude', 'mid')


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
        result = plugin.combine(cube1, cube2, operation)
        self.assertIsInstance(result, Cube)
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[0, 0, :, :] = 0.25
        expected_data[0, 1, :, :] = 0.36
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_add(self):
        """Test combine adds the cubes correctly. """
        operation = '+'
        plugin = CubeCombiner(operation)
        result = plugin.combine(self.cube1, self.cube2,
                                operation)
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[0, 0, :, :] = 0.6
        expected_data[0, 1, :, :] = 1.0
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_minus(self):
        """Test combine minus the cubes correctly. """
        operation = '-'
        plugin = CubeCombiner(operation)
        result = plugin.combine(self.cube1, self.cube2,
                                operation)
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[0, 0, :, :] = 0.4
        expected_data[0, 1, :, :] = 0.2
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_max(self):
        """Test combine finds the max of the cubes correctly."""
        operation = 'max'
        plugin = CubeCombiner(operation)
        result = plugin.combine(self.cube1, self.cube3,
                                operation)
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[0, 0, :, :] = 0.5
        expected_data[0, 1, :, :] = 0.8
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_min(self):
        """Test combine finds the min of the cubes correctly."""
        operation = 'min'
        plugin = CubeCombiner(operation)
        result = plugin.combine(self.cube1, self.cube3,
                                operation)
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[0, 0, :, :] = 0.1
        expected_data[0, 1, :, :] = 0.6
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_mean(self):
        """Test that the function adds the cubes correctly for mean."""
        operation = '+'
        plugin = CubeCombiner(operation)
        result = plugin.combine(self.cube1, self.cube3,
                                operation)
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

    @ManageWarnings(record=True)
    def test_warnings_on(self, warning_list=None):
        """Test that the plugin raises warnings and updates metadata. """
        plugin = CubeCombiner('-', warnings_on=True)
        cubelist = iris.cube.CubeList([self.cube1, self.cube1])
        attributes = {'relative_to_threshold': 'between'}
        warning_msg = "Adding or updating attribute"
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[0, 0, :, :] = 0.0
        expected_data[0, 1, :, :] = 0.0
        result = plugin.process(cubelist, 'new_cube_name',
                                revised_attributes=attributes)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertEqual(result.name(), 'new_cube_name')
        self.assertArrayAlmostEqual(result.data, expected_data)


if __name__ == '__main__':
    unittest.main()
