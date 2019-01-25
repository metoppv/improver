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
"""Unit tests for temporal utilities."""

import datetime
from datetime import timedelta
import unittest
import numpy as np
from cf_units import Unit

import iris
from iris.coords import DimCoord
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest
from iris.cube import Cube

from improver.utilities.temporal_interpolation import TemporalInterpolation
from improver.tests.set_up_test_cubes import (
    set_up_variable_cube, add_coordinate)


class Test__init__(IrisTest):

    """Test the __init__ method."""

    def test_raises_error_with_no_keyword_args(self):
        """Test __init__ raises a ValueError if both keywords are unset."""
        msg = "TemporalInterpolation: One of"
        with self.assertRaisesRegex(ValueError, msg):
            TemporalInterpolation()

    def test_unknown_method(self):
        """Test __init__ raises a ValueError if method unknown."""
        msg = "TemporalInterpolation: Unknown interpolation method"
        with self.assertRaisesRegex(ValueError, msg):
            TemporalInterpolation(interval_in_minutes=60,
                                  interpolation_method='invalid')


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(TemporalInterpolation(interval_in_minutes=60))
        msg = ('<TemporalInterpolation: interval_in_minutes: 60,'
               ' times: None,'
               ' method: linear>')
        self.assertEqual(result, msg)

    def test_solar(self):
        """Test that the __repr__ returns the expected string for solar."""
        result = str(TemporalInterpolation(interval_in_minutes=60,
                                           interpolation_method='solar'))
        msg = ('<TemporalInterpolation: interval_in_minutes: 60,'
               ' times: None,'
               ' method: solar>')
        self.assertEqual(result, msg)

    def test_daynight(self):
        """Test that the __repr__ returns the expected string for daynight."""
        result = str(TemporalInterpolation(interval_in_minutes=60,
                                           interpolation_method='daynight'))
        msg = ('<TemporalInterpolation: interval_in_minutes: 60,'
               ' times: None,'
               ' method: daynight>')
        self.assertEqual(result, msg)


class Test_construct_time_list(IrisTest):

    """Test construction of time lists suitable for iris interpolation using
    this function."""

    def setUp(self):
        """Set up the test inputs."""
        self.time_0 = datetime.datetime(2017, 11, 1, 3)
        self.time_1 = datetime.datetime(2017, 11, 1, 9)
        self.times = []
        for i in range(4, 9):
            self.times.append(datetime.datetime(2017, 11, 1, i))
        self.expected = [('time', [time for time in self.times])]

    def test_return_type(self):
        """Test that a list is returned."""

        result = (
            TemporalInterpolation(interval_in_minutes=60).construct_time_list(
                self.time_0, self.time_1))
        self.assertIsInstance(result, list)

    def test_list_from_interval_in_minutes(self):
        """Test generating a list between two times using the
        interval_in_minutes keyword to define the spacing."""

        result = (
            TemporalInterpolation(interval_in_minutes=60).construct_time_list(
                self.time_0, self.time_1))
        self.assertEqual(self.expected, result)

    def test_non_equally_divisible_interval_in_minutes(self):
        """Test an exception is raised when trying to generate a list of times
        that would not divide the time range equally."""

        msg = 'interval_in_minutes provided to time_interpolate does not'
        with self.assertRaisesRegex(ValueError, msg):
            TemporalInterpolation(interval_in_minutes=61).construct_time_list(
                self.time_0, self.time_1)

    def test_list_from_existing_list(self):
        """Test generating an iris interpolation suitable list from an
        existing list of datetime objects."""

        result = (
            TemporalInterpolation(times=self.times).construct_time_list(
                self.time_0, self.time_1))
        self.assertEqual(self.expected, result)

    def test_time_list_out_of_bounds(self):
        """Test an exception is raised when trying to generate a list of times
        using a pre-existing list that includes times outside the range of the
        initial and final times."""

        self.times.append(datetime.datetime(2017, 11, 1, 10))

        msg = 'List of times falls outside the range given by'
        with self.assertRaisesRegex(ValueError, msg):
            TemporalInterpolation(times=self.times).construct_time_list(
                self.time_0, self.time_1)


class Test_check_cube_coords(IrisTest):

    """Test construction of time lists suitable for iris interpolation using
    this function."""

    def setUp(self):
        """Set up the test inputs."""
        time_start = datetime.datetime(2017, 11, 1, 3)
        time_mid = datetime.datetime(2017, 11, 1, 6)
        time_end = datetime.datetime(2017, 11, 1, 9)
        self.extra_coord = np.array([1.0, 4.0, 7.0], dtype=np.float32)
        self.bad_extra_coord = np.array([1.1, 4.3, 6.9], dtype=np.float64)
        self.npoints = 10
        data_time_0 = np.ones((self.npoints, self.npoints), dtype=np.float32)
        cube_time_0 = set_up_variable_cube(data_time_0,
                                           time=time_start,
                                           frt=time_start)
        cube_times = add_coordinate(cube_time_0,
                                    [time_start, time_mid, time_end],
                                    'time', is_datetime=True)
        self.cube = add_coordinate(cube_times, self.extra_coord,
                                   'extra_coord')
        self.bad_coords = self.cube.copy()
        self.bad_coords.coord('time').points = (
            self.bad_coords.coord('time').points.astype(np.float64))
        self.bad_coords.coord('forecast_period').points = (
            self.bad_coords.coord('forecast_period').points.astype(np.int64))
        self.bad_coords.coord('extra_coord').points = (
            self.bad_extra_coord)

    def test_check_all_type(self):
        """Test that a cube is returned.with the right types"""
        plugin = TemporalInterpolation(interval_in_minutes=60)
        result = plugin.check_cube_coords_dtype(self.cube,
                                                self.bad_coords)
        self.assertIsInstance(result, iris.cube.Cube)
        # All coords reverted back to the dtypes of the
        # original cube and points in extra_coord cube have been rounded
        self.assertEqual(result.coord('time'),
                         self.cube.coord('time'))
        self.assertEqual(str(result.coord('time').points.dtype),
                         'int64')
        self.assertEqual(result.coord('forecast_period'),
                         self.cube.coord('forecast_period'))
        self.assertEqual(str(result.coord('forecast_period').points.dtype),
                         'int32')
        self.assertEqual(result.coord('extra_coord'),
                         self.cube.coord('extra_coord'))
        self.assertEqual(str(result.coord('extra_coord').points.dtype),
                         'float32')

    def test_time_types(self):
        """Test that only the coords associated with time have been updated."""
        plugin = TemporalInterpolation(interval_in_minutes=60)
        result = plugin.check_cube_coords_dtype(self.cube,
                                                self.bad_coords,
                                                dim_coords=['time'])
        # Time and forecast_period reverted back to original type
        self.assertEqual(result.coord('time'),
                         self.cube.coord('time'))
        self.assertEqual(str(result.coord('time').points.dtype),
                         'int64')
        self.assertEqual(result.coord('forecast_period'),
                         self.cube.coord('forecast_period'))
        self.assertEqual(str(result.coord('forecast_period').points.dtype),
                         'int32')
        # Extra coordinate left as changed.
        self.assertEqual(result.coord('extra_coord'),
                         self.bad_coords.coord('extra_coord'))
        self.assertEqual(str(result.coord('extra_coord').points.dtype),
                         'float64')


class Test_daynight_interpolation(IrisTest):

    """Test daynight interpolation."""

    def setUp(self):
        """Set up the test inputs."""
        self.time_0 = datetime.datetime(2017, 11, 1, 0)
        self.time_mid = datetime.datetime(2017, 11, 1, 2)
        self.time_1 = datetime.datetime(2017, 11, 1, 4)
        self.npoints = 10
        self.daynight_mask = np.array([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                       [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                       [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                       [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                       [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                       [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                       [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                                       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])

        data_time_0 = np.ones((self.npoints, self.npoints), dtype=np.float32)
        data_time_1 = np.ones((self.npoints, self.npoints),
                              dtype=np.float32) * 7
        data_time_mid = np.ones((self.npoints, self.npoints),
                                dtype=np.float32) * 4
        cube_time_0 = set_up_variable_cube(data_time_0,
                                           time=self.time_0,
                                           frt=self.time_0)
        cube_time_1 = set_up_variable_cube(data_time_1,
                                           time=self.time_1,
                                           frt=self.time_0)
        cubes = iris.cube.CubeList([cube_time_0, cube_time_1])
        self.cube = cubes.merge_cube()
        interp_cube = set_up_variable_cube(data_time_mid,
                                           time=self.time_mid,
                                           frt=self.time_0)
        self.interpolated_cube = iris.util.new_axis(interp_cube, 'time')

    def test_return_type(self):
        """Test that an iris cubelist is returned."""

        plugin = TemporalInterpolation(interpolation_method='daynight',
                                       times=[self.time_mid])
        result = plugin.daynight_interpolate(self.cube, self.interpolated_cube)
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_daynight_interpolation(self):
        """Test interpolating to the a point where the daynight
           mask is not all zero."""

        expected_data = np.ones((self.npoints, self.npoints))*4
        index = np.where(self.daynight_mask == 0)
        expected_data[index] = 0.0
        expected_time = (self.time_0 + timedelta(hours=2)).timestamp()
        expected_fp = 2 * 3600
        plugin = TemporalInterpolation(interpolation_method='daynight',
                                       times=[self.time_mid])
        result, = plugin.daynight_interpolate(self.cube,
                                              self.interpolated_cube)
        self.assertArrayAlmostEqual(expected_data, result.data)
        self.assertArrayAlmostEqual(result.coord('time').points,
                                    expected_time)
        self.assertAlmostEqual(result.coord('forecast_period').points[0],
                               expected_fp)


class Test_process(IrisTest):

    """Test interpolation of cubes to intermediate times using the plugin."""

    def setUp(self):
        """Set up the test inputs."""
        self.time_0 = datetime.datetime(2017, 11, 1, 3)
        self.time_extra = datetime.datetime(2017, 11, 1, 6)
        self.time_1 = datetime.datetime(2017, 11, 1, 9)
        self.npoints = 10
        data_time_0 = np.ones((self.npoints, self.npoints), dtype=np.float32)
        data_time_1 = np.ones((self.npoints, self.npoints),
                              dtype=np.float32) * 7
        self.cube_time_0 = set_up_variable_cube(data_time_0,
                                                time=self.time_0,
                                                frt=self.time_0)
        self.cube_time_1 = set_up_variable_cube(data_time_1,
                                                time=self.time_1,
                                                frt=self.time_0)

    def test_return_type(self):
        """Test that an iris cubelist is returned."""

        result = TemporalInterpolation(interval_in_minutes=180).process(
            self.cube_time_0, self.cube_time_1)
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_valid_single_interpolation(self):
        """Test interpolating to the mid point of the time range. Expect the
        data to be half way between, and the time coordinate should be at
        06Z November 11th 2017."""

        expected_data = np.ones((self.npoints, self.npoints)) * 4
        expected_time = (self.time_0 + timedelta(hours=3)).timestamp()
        expected_fp = 3 * 3600
        result, = TemporalInterpolation(interval_in_minutes=180).process(
            self.cube_time_0, self.cube_time_1)

        self.assertArrayAlmostEqual(expected_data, result.data)
        self.assertArrayAlmostEqual(result.coord('time').points, expected_time)
        self.assertAlmostEqual(result.coord('forecast_period').points[0],
                               expected_fp)

    def test_valid_multiple_interpolations(self):
        """Test interpolating to every hour between the two input cubes.
        Check the data increments as expected and the time coordinates are also
        set correctly.

        NB Interpolation in iris is prone to float precision errors of order
        10E-6, hence the need to use AlmostEqual below."""

        result = TemporalInterpolation(interval_in_minutes=60).process(
            self.cube_time_0, self.cube_time_1)
        for i, cube in enumerate(result):
            expected_data = np.ones((self.npoints, self.npoints)) * i + 2
            expected_time = (self.time_0 + timedelta(hours=(i+1))).timestamp()

            self.assertArrayAlmostEqual(expected_data, cube.data)
            self.assertArrayAlmostEqual(
                cube.coord('time').points, expected_time, decimal=5)
            self.assertAlmostEqual(cube.coord('forecast_period').points[0],
                                   (i+1)*3600)

    def test_valid_interpolation_from_given_list(self):
        """Test interpolating to a point defined in a list between the two
        input cube validity times. Check the data increments as expected and
        the time coordinates are also set correctly.

        NB Interpolation in iris is prone to float precision errors of order
        10E-6, hence the need to use AlmostEqual below."""

        result, = TemporalInterpolation(times=[self.time_extra]).process(
            self.cube_time_0, self.cube_time_1)
        expected_data = np.ones((self.npoints, self.npoints)) * 4
        expected_time = self.time_extra.timestamp()
        expected_fp = 3 * 3600

        self.assertArrayAlmostEqual(expected_data, result.data)
        self.assertArrayAlmostEqual(
            result.coord('time').points, expected_time, decimal=5)
        self.assertAlmostEqual(result.coord('forecast_period').points[0],
                               expected_fp)

    def test_input_cube_without_time_coordinate(self):
        """Test that an exception is raised if a cube is provided without a
        time coordiate."""

        self.cube_time_0.remove_coord('time')

        msg = 'Cube provided to time_interpolate contains no time coordinate'
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            TemporalInterpolation(interval_in_minutes=180).process(
                self.cube_time_0, self.cube_time_1)

    def test_input_cubes_in_incorrect_time_order(self):
        """Test that an exception is raised if the cube representing the
        initial time has a validity time that is after the cube representing
        the final time."""

        msg = 'time_interpolate input cubes ordered incorrectly'
        with self.assertRaisesRegex(ValueError, msg):
            TemporalInterpolation(interval_in_minutes=180).process(
                self.cube_time_1, self.cube_time_0)

    def test_input_cube_with_multiple_times(self):
        """Test that an exception is raised if a cube is provided that has
        multiple validity times, e.g. a multi-entried time dimension."""

        second_time = self.cube_time_0.copy()
        second_time.coord('time').points = self.time_extra.timestamp()
        cube = iris.cube.CubeList([self.cube_time_0, second_time])
        cube = cube.merge_cube()

        msg = 'Cube provided to time_interpolate contains multiple'
        with self.assertRaisesRegex(ValueError, msg):
            TemporalInterpolation(interval_in_minutes=180).process(
                cube, self.cube_time_1)

    def test_input_cubelists_raises_exception(self):
        """Test that providing cubelists instead of cubes raises an
        exception."""

        cubes = iris.cube.CubeList([self.cube_time_1])

        msg = 'Inputs to TemporalInterpolation are not of type '
        with self.assertRaisesRegex(TypeError, msg):
            TemporalInterpolation(interval_in_minutes=180).process(
                cubes, self.cube_time_0)


if __name__ == '__main__':
    unittest.main()
