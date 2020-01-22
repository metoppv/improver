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
import unittest

import iris
import numpy as np
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest

from improver.utilities.temporal_interpolation import TemporalInterpolation

from ..set_up_test_cubes import add_coordinate, set_up_variable_cube


class Test__init__(IrisTest):

    """Test the __init__ method."""

    def test_raises_error_with_no_keyword_args(self):
        """Test __init__ raises a ValueError if both interval_in_minutes
        and times keywords are unset."""
        msg = "TemporalInterpolation: One of"
        with self.assertRaisesRegex(ValueError, msg):
            TemporalInterpolation()

    def test_raises_error_with_both_keyword_args(self):
        """Test __init__ raises a ValueError if both interval_in_minutes
        and times keywords are both set."""
        msg = "TemporalInterpolation: Only one of"
        with self.assertRaisesRegex(ValueError, msg):
            TemporalInterpolation(interval_in_minutes=60,
                                  times=[datetime.datetime(2017, 11, 1, 9)])

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

        msg = 'interval_in_minutes of'
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


class Test_enforce_time_coords_dtype(IrisTest):

    """Test that the datatypes and units of the time, forecast_reference_time
    and forecast_period coordinates have been enforced."""

    def setUp(self):
        """Set up the test inputs."""
        time_start = datetime.datetime(2017, 11, 1, 3)
        time_mid = datetime.datetime(2017, 11, 1, 6)
        time_end = datetime.datetime(2017, 11, 1, 9)
        self.npoints = 10
        data_time_0 = np.ones((self.npoints, self.npoints), dtype=np.float32)
        cube_time_0 = set_up_variable_cube(data_time_0,
                                           time=time_start,
                                           frt=time_start)
        cube_times = add_coordinate(cube_time_0.copy(),
                                    [time_start, time_mid, time_end],
                                    'time', is_datetime=True)
        # Convert units and datatypes, so that they are non-standard.
        cube_times.coord("time").convert_units(
            "hours since 1970-01-01 00:00:00")
        cube_times.coord("time").points = (
            cube_times.coord("time").points.astype(np.int32))
        cube_times.coord("forecast_reference_time").convert_units(
            "hours since 1970-01-01 00:00:00")
        cube_times.coord("forecast_reference_time").points = (
            cube_times.coord(
                "forecast_reference_time").points.astype(np.int32))
        cube_times.coord("forecast_period").convert_units("hours")
        cube_times.coord("forecast_period").points.astype(np.float32)
        self.cube = cube_times

        self.coord_dtypes = {
            'time': np.int64,
            'forecast_reference_time': np.int64,
            'forecast_period': np.int32,
            }
        self.coord_units = {
            'time': "seconds since 1970-01-01 00:00:00",
            'forecast_reference_time': "seconds since 1970-01-01 00:00:00",
            'forecast_period': "seconds",
            }

    def test_check_points(self):
        """Test that a cube is returned with the desired types for the points
        of the time, forecast_reference_time and forecast_period
        coordinates."""
        plugin = TemporalInterpolation(interval_in_minutes=60)
        result = plugin.enforce_time_coords_dtype(self.cube.copy())
        self.assertIsInstance(result, iris.cube.Cube)
        # All coordinates converted to the desired units and datatypes.
        # Check time coordinate.
        self.assertEqual(result.coord('time').points.dtype,
                         self.coord_dtypes['time'])
        self.assertEqual(result.coord('time').units, self.coord_units['time'])
        # Check forecast_reference_time.
        self.assertEqual(
            result.coord('forecast_reference_time').points.dtype,
            self.coord_dtypes["forecast_reference_time"])
        self.assertEqual(
            result.coord('forecast_reference_time').units,
            self.coord_units["forecast_reference_time"])
        # Check forecast_period.
        self.assertEqual(result.coord('forecast_period').points.dtype,
                         self.coord_dtypes['forecast_period'])
        self.assertEqual(result.coord('forecast_period').units,
                         self.coord_units['forecast_period'])

    def test_check_bounds(self):
        """Test that a cube is returned with the desired types when
        the time and forecast_period coordinates have bounds."""
        plugin = TemporalInterpolation(interval_in_minutes=60)
        cube = self.cube
        # Use of guess_bounds converts datatype to float64.
        cube.coord("time").guess_bounds()
        cube.coord("forecast_period").guess_bounds()

        result = plugin.enforce_time_coords_dtype(cube.copy())
        self.assertIsInstance(result, iris.cube.Cube)
        # All coordinates including bounds converted to the
        # desired units and datatypes.
        # Check time coordinate.
        self.assertEqual(result.coord('time').points.dtype,
                         self.coord_dtypes['time'])
        self.assertEqual(result.coord('time').bounds.dtype,
                         self.coord_dtypes['time'])
        self.assertEqual(result.coord('time').units, self.coord_units['time'])
        # Check forecast_reference_time coordinate.
        self.assertEqual(
            result.coord('forecast_reference_time').points.dtype,
            self.coord_dtypes["forecast_reference_time"])
        self.assertEqual(
            result.coord('forecast_reference_time').units,
            self.coord_units["forecast_reference_time"])
        # Check forecast_period coordinate.
        self.assertEqual(result.coord('forecast_period').points.dtype,
                         self.coord_dtypes["forecast_period"])
        self.assertEqual(result.coord('forecast_period').bounds.dtype,
                         self.coord_dtypes["forecast_period"])
        self.assertEqual(result.coord('forecast_period').units,
                         self.coord_units['forecast_period'])


class Test_calc_sin_phi(IrisTest):

    """Test Calculate sin of solar elevation."""

    def test_sin_phi(self):
        """Test that the function returns the values expected."""
        latitudes = np.array([50.0, 50.0, 50.0])
        longitudes = np.array([-5.0, 0.0, 5.0])
        dtval = datetime.datetime(2017, 1, 11, 8)
        expected_array = np.array([-0.05481607, -0.00803911, 0.03659632])
        plugin = TemporalInterpolation(interval_in_minutes=60,
                                       interpolation_method='solar')
        result = plugin.calc_sin_phi(dtval,
                                     latitudes,
                                     longitudes)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_array)


class Test_calc_lats_lons(IrisTest):

    """Test Calculate lats and lons."""

    def setUp(self):
        """Set up the test inputs."""
        time_start = datetime.datetime(2017, 11, 1, 3)
        time_mid = datetime.datetime(2017, 11, 1, 6)
        time_end = datetime.datetime(2017, 11, 1, 9)
        self.npoints = 3
        data_time_0 = np.ones((self.npoints, self.npoints), dtype=np.float32)
        cube_time_0 = set_up_variable_cube(data_time_0,
                                           time=time_start,
                                           frt=time_start)
        self.cube = add_coordinate(cube_time_0,
                                   [time_start, time_mid, time_end],
                                   'time', is_datetime=True)
        cube_time_0_equalarea = set_up_variable_cube(data_time_0,
                                                     time=time_start,
                                                     frt=time_start,
                                                     spatial_grid='equalarea')
        self.cube_equalarea = add_coordinate(cube_time_0_equalarea,
                                             [time_start, time_mid, time_end],
                                             'time', is_datetime=True)

    def test_lat_lon(self):
        """Test that the function returns the lats and lons expected."""
        expected_lons = np.array([[-20.0, 0.0, 20.0],
                                  [-20.0, 0.0, 20.0],
                                  [-20.0, 0.0, 20.0]])
        expected_lats = np.array([[40.0, 40.0, 40.0],
                                  [60.0, 60.0, 60.0],
                                  [80.0, 80.0, 80.0]])
        plugin = TemporalInterpolation(interval_in_minutes=60,
                                       interpolation_method='solar')
        result_lats, result_lons = plugin.calc_lats_lons(self.cube)
        self.assertIsInstance(result_lats, np.ndarray)
        self.assertEqual(result_lats.shape, (3, 3))
        self.assertIsInstance(result_lons, np.ndarray)
        self.assertEqual(result_lons.shape, (3, 3))
        self.assertArrayAlmostEqual(result_lats, expected_lats)
        self.assertArrayAlmostEqual(result_lons, expected_lons)

    def test_x_y(self):
        """Test that the function returns the lats and lons expected."""
        expected_lats = np.array([[53.84618597, 53.99730779, 53.93247526],
                                  [56.82670954, 56.99111356, 56.9205672],
                                  [59.8045105,  59.98499383, 59.90752513]])

        expected_lons = np.array([[-8.58580705, -3.51660018, 1.56242662],
                                  [-9.06131306, -3.59656346, 1.88105082],
                                  [-9.63368459, -3.69298822, 2.26497216]])

        plugin = TemporalInterpolation(interval_in_minutes=60,
                                       interpolation_method='solar')
        result_lats, result_lons = plugin.calc_lats_lons(self.cube_equalarea)
        self.assertIsInstance(result_lats, np.ndarray)
        self.assertEqual(result_lats.shape, (3, 3))
        self.assertIsInstance(result_lons, np.ndarray)
        self.assertEqual(result_lons.shape, (3, 3))
        self.assertArrayAlmostEqual(result_lats, expected_lats)
        self.assertArrayAlmostEqual(result_lons, expected_lons)


class Test_solar_interpolation(IrisTest):

    """Test Solar interpolation."""

    def setUp(self):
        """Set up the test inputs spanning sunrise."""
        self.time_0 = datetime.datetime(2017, 11, 1, 6)
        self.time_mid = datetime.datetime(2017, 11, 1, 8)
        self.time_1 = datetime.datetime(2017, 11, 1, 10)
        self.npoints = 5
        self.expected = np.array([
            [0.02358028, 0.15887623, 0.2501732, 0.32049885, 0.3806127],
            [0., 0.09494493, 0.21051247, 0.2947393, 0.36431003],
            [0., 0., 0.11747278, 0.23689085, 0.32841164],
            [0., 0., 0., 0., 0.15872595],
            [0., 0., 0., 0., 0.]])

        data_time_0 = np.zeros((self.npoints, self.npoints), dtype=np.float32)
        data_time_1 = np.ones((self.npoints, self.npoints),
                              dtype=np.float32)
        data_time_mid = np.ones((self.npoints, self.npoints),
                                dtype=np.float32)
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
        data_time_0_ens = np.zeros((3, self.npoints, self.npoints),
                                   dtype=np.float32)
        data_time_1_ens = np.ones((3, self.npoints, self.npoints),
                                  dtype=np.float32)
        data_time_mid_ens = np.ones((3, self.npoints, self.npoints),
                                    dtype=np.float32)
        cube_time_0_ens = set_up_variable_cube(data_time_0_ens,
                                               time=self.time_0,
                                               frt=self.time_0,
                                               realizations=[0, 1, 2])
        cube_time_1_ens = set_up_variable_cube(data_time_1_ens,
                                               time=self.time_1,
                                               frt=self.time_0,
                                               realizations=[0, 1, 2])
        interp_cube_ens = set_up_variable_cube(data_time_mid_ens,
                                               time=self.time_mid,
                                               frt=self.time_0,
                                               realizations=[0, 1, 2])
        self.interpolated_cube_ens = iris.util.new_axis(interp_cube_ens,
                                                        'time')
        cubes_ens = iris.cube.CubeList([cube_time_0_ens, cube_time_1_ens])
        self.cube_ens = cubes_ens.merge_cube()

    def test_return_type(self):
        """Test that an iris cubelist is returned."""

        plugin = TemporalInterpolation(interpolation_method='solar',
                                       times=[self.time_mid])
        result = plugin.solar_interpolate(self.cube, self.interpolated_cube)
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_solar_interpolation(self):
        """Test interpolating using solar method works correctly."""

        expected_time = [1509523200]
        expected_fp = 2 * 3600
        plugin = TemporalInterpolation(interpolation_method='solar',
                                       times=[self.time_mid])
        result, = plugin.solar_interpolate(self.cube,
                                           self.interpolated_cube)
        self.assertArrayAlmostEqual(result.data, self.expected)
        self.assertArrayAlmostEqual(result.coord('time').points,
                                    expected_time)
        self.assertAlmostEqual(result.coord('forecast_period').points[0],
                               expected_fp)

    def test_solar_interpolation_shape(self):
        """Test interpolating using solar method with len(shape) >= 3
         works correctly."""

        expected_time = [1509523200]
        expected_fp = 2 * 3600
        plugin = TemporalInterpolation(interpolation_method='solar',
                                       times=[self.time_mid])
        result, = plugin.solar_interpolate(self.cube_ens,
                                           self.interpolated_cube_ens)

        self.assertArrayEqual(result.data.shape, (3, 5, 5))
        self.assertArrayAlmostEqual(result.data[0], self.expected)
        self.assertArrayAlmostEqual(result.coord('time').points,
                                    expected_time)
        self.assertAlmostEqual(result.coord('forecast_period').points[0],
                               expected_fp)


class Test_daynight_interpolation(IrisTest):

    """Test daynight interpolation."""

    def setUp(self):
        """Set up the test inputs spanning sunrise."""
        self.time_0 = datetime.datetime(2017, 11, 1, 6)
        self.time_mid = datetime.datetime(2017, 11, 1, 8)
        self.npoints = 10
        self.daynight_mask = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                       [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                       [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                       [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                                       [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                                       [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                       [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        data_time_mid = np.ones((self.npoints, self.npoints),
                                dtype=np.float32) * 4

        interp_cube = set_up_variable_cube(data_time_mid,
                                           time=self.time_mid,
                                           frt=self.time_0)
        self.interpolated_cube = iris.util.new_axis(interp_cube, 'time')

        data_time_mid_ens = np.ones((3, self.npoints, self.npoints),
                                    dtype=np.float32)*4
        interp_cube_ens = set_up_variable_cube(data_time_mid_ens,
                                               time=self.time_mid,
                                               frt=self.time_0,
                                               realizations=[0, 1, 2])
        self.interpolated_cube_ens = iris.util.new_axis(interp_cube_ens,
                                                        'time')

    def test_return_type(self):
        """Test that an iris cubelist is returned."""

        plugin = TemporalInterpolation(interpolation_method='daynight',
                                       times=[self.time_mid])
        result = plugin.daynight_interpolate(self.interpolated_cube)
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_daynight_interpolation(self):
        """Test interpolating to the a point where the daynight
           mask is not all zero."""

        expected_data = np.ones((self.npoints, self.npoints))*4
        index = np.where(self.daynight_mask == 0)
        expected_data[index] = 0.0
        expected_time = [1509523200]
        expected_fp = 2 * 3600
        plugin = TemporalInterpolation(interpolation_method='daynight',
                                       times=[self.time_mid])
        result, = plugin.daynight_interpolate(self.interpolated_cube)
        self.assertArrayAlmostEqual(expected_data, result.data)
        self.assertArrayAlmostEqual(result.coord('time').points,
                                    expected_time)
        self.assertAlmostEqual(result.coord('forecast_period').points[0],
                               expected_fp)

    def test_daynight_interpolation_ens(self):
        """Test interpolating to the a point where the daynight
           mask is not all zero and the len(shape) of the cube > 2."""

        expected_data_grid = np.ones((self.npoints, self.npoints))*4
        index = np.where(self.daynight_mask == 0)
        expected_data_grid[index] = 0.0
        expected_data = np.repeat(expected_data_grid[np.newaxis, :, :],
                                  3, axis=0)
        expected_time = [1509523200]
        expected_fp = 2 * 3600
        plugin = TemporalInterpolation(interpolation_method='daynight',
                                       times=[self.time_mid])
        result, = plugin.daynight_interpolate(self.interpolated_cube_ens)
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
        expected_time = [1509516000]
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
            expected_time = [1509508800 + i * 3600]

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
        expected_time = [1509516000]
        expected_fp = 3 * 3600

        self.assertArrayAlmostEqual(expected_data, result.data)
        self.assertArrayAlmostEqual(
            result.coord('time').points, expected_time, decimal=5)
        self.assertAlmostEqual(result.coord('forecast_period').points[0],
                               expected_fp)
        self.assertEqual(result.data.dtype, np.float32)
        self.assertEqual(str(result.coord('time').points.dtype), 'int64')
        self.assertEqual(str(result.coord('forecast_period').points.dtype),
                         'int32')

    def test_solar_interpolation_from_given_list(self):
        """Test solar interpolating to a point defined in a list
        between the two input cube validity times.
        Check the data increments as expected and
        the time coordinates are also set correctly."""

        plugin = TemporalInterpolation(times=[self.time_extra],
                                       interpolation_method='solar')
        result, = plugin.process(self.cube_time_0,
                                 self.cube_time_1)
        expected_time = [1509516000]
        expected_fp = 3 * 3600

        self.assertArrayAlmostEqual(
            result.coord('time').points, expected_time, decimal=5)
        self.assertAlmostEqual(result.coord('forecast_period').points[0],
                               expected_fp)
        self.assertEqual(result.data.dtype, np.float32)
        self.assertEqual(str(result.coord('time').points.dtype), 'int64')
        self.assertEqual(str(result.coord('forecast_period').points.dtype),
                         'int32')

    def test_daynight_interpolation_from_given_list(self):
        """Test daynight interpolating to a point defined in a list
        between the two input cube validity times.
        Check the data increments as expected and
        the time coordinates are also set correctly."""

        plugin = TemporalInterpolation(times=[self.time_extra],
                                       interpolation_method='daynight')
        result, = plugin.process(self.cube_time_0, self.cube_time_1)
        expected_data = np.zeros((self.npoints, self.npoints))
        expected_data[:2, 7:] = 4.
        expected_data[2, 8:] = 4.
        expected_data[3, 9] = 4.
        expected_time = [1509516000]
        expected_fp = 3 * 3600

        self.assertArrayAlmostEqual(expected_data, result.data)
        self.assertArrayAlmostEqual(
            result.coord('time').points, expected_time, decimal=5)
        self.assertAlmostEqual(result.coord('forecast_period').points[0],
                               expected_fp)
        self.assertEqual(result.data.dtype, np.float32)
        self.assertEqual(str(result.coord('time').points.dtype), 'int64')
        self.assertEqual(str(result.coord('forecast_period').points.dtype),
                         'int32')

    def test_input_cube_without_time_coordinate(self):
        """Test that an exception is raised if a cube is provided without a
        time coordiate."""

        self.cube_time_0.remove_coord('time')

        msg = ('Cube provided to TemporalInterpolation '
               'contains no time coordinate')
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            TemporalInterpolation(interval_in_minutes=180).process(
                self.cube_time_0, self.cube_time_1)

    def test_input_cubes_in_incorrect_time_order(self):
        """Test that an exception is raised if the cube representing the
        initial time has a validity time that is after the cube representing
        the final time."""

        msg = 'TemporalInterpolation input cubes ordered incorrectly'
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

        msg = 'Cube provided to TemporalInterpolation contains multiple'
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
