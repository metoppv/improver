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

import unittest
from datetime import datetime

import iris
import numpy as np
from iris.cube import Cube, CubeList
from iris.tests import IrisTest
from iris.time import PartialDateTime

from improver.utilities.temporal import (
    cycletime_to_datetime, cycletime_to_number, datetime_constraint,
    datetime_to_cycletime, datetime_to_iris_time, extract_cube_at_time,
    extract_nearest_time_point, iris_time_to_datetime)
from improver.utilities.warnings_handler import ManageWarnings

from ..set_up_test_cubes import add_coordinate, set_up_variable_cube


class Test_cycletime_to_datetime(IrisTest):

    """Test that a cycletime of a format such as YYYYMMDDTHHMMZ is converted
    into a datetime object."""

    def test_basic(self):
        """Test that a datetime object is returned of the expected value."""
        cycletime = "20171122T0100Z"
        dt = datetime(2017, 11, 22, 1, 0)
        result = cycletime_to_datetime(cycletime)
        self.assertIsInstance(result, datetime)
        self.assertEqual(result, dt)

    def test_define_cycletime_format(self):
        """Test when a cycletime is defined."""
        cycletime = "201711220100"
        dt = datetime(2017, 11, 22, 1, 0)
        result = cycletime_to_datetime(
            cycletime, cycletime_format="%Y%m%d%H%M")
        self.assertEqual(result, dt)


class Test_datetime_to_cycletime(IrisTest):

    """Test that a datetime object can be converted into a cycletime
    of a format such as YYYYMMDDTHHMMZ."""

    def test_basic(self):
        """Test that a datetime object is returned of the expected value."""
        dt = datetime(2017, 11, 22, 1, 0)
        cycletime = "20171122T0100Z"
        result = datetime_to_cycletime(dt)
        self.assertIsInstance(result, str)
        self.assertEqual(result, cycletime)

    def test_define_cycletime_format(self):
        """Test when a cycletime is defined."""
        dt = datetime(2017, 11, 22, 1, 0)
        cycletime = "201711220100"
        result = datetime_to_cycletime(dt, cycletime_format="%Y%m%d%H%M")
        self.assertEqual(result, cycletime)

    def test_define_cycletime_format_with_seconds(self):
        """Test when a cycletime is defined with seconds."""
        dt = datetime(2017, 11, 22, 1, 0)
        cycletime = "20171122010000"
        result = datetime_to_cycletime(dt, cycletime_format="%Y%m%d%H%M%S")
        self.assertEqual(result, cycletime)


class Test_cycletime_to_number(IrisTest):

    """Test that a cycletime of a format such as YYYYMMDDTHHMMZ is converted
      into a numeric time value."""

    def test_basic(self):
        """Test that a number is returned of the expected value."""
        cycletime = "20171122T0000Z"
        dt = 419808.0
        result = cycletime_to_number(cycletime)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, dt)

    def test_cycletime_format_defined(self):
        """Test when a cycletime is defined."""
        cycletime = "201711220000"
        dt = 419808.0
        result = cycletime_to_number(
            cycletime, cycletime_format="%Y%m%d%H%M")
        self.assertAlmostEqual(result, dt)

    def test_alternative_units_defined(self):
        """Test when alternative units are defined. The result is cast as
        an integer as seconds should be of this type and compared as such.
        There are small precision errors in the 7th decimal place of the
        returned float."""
        cycletime = "20171122T0000Z"
        dt = 1511308800
        result = cycletime_to_number(
            cycletime, time_unit="seconds since 1970-01-01 00:00:00")
        self.assertEqual(int(np.round(result)), dt)

    def test_alternative_calendar_defined(self):
        """Test when an alternative calendar is defined."""
        cycletime = "20171122T0000Z"
        dt = 419520.0
        result = cycletime_to_number(
            cycletime, calendar="365_day")
        self.assertAlmostEqual(result, dt)


class Test_iris_time_to_datetime(IrisTest):
    """ Test iris_time_to_datetime """

    def setUp(self):
        """Set up an input cube"""
        self.cube = set_up_variable_cube(
            np.ones((3, 3), dtype=np.float32),
            time=datetime(2017, 2, 17, 6, 0),
            frt=datetime(2017, 2, 17, 3, 0))

    def test_basic(self):
        """Test iris_time_to_datetime returns list of datetime """
        result = iris_time_to_datetime(self.cube.coord('time'))
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, datetime)
        self.assertEqual(result[0], datetime(2017, 2, 17, 6, 0))

    def test_bounds(self):
        """Test iris_time_to_datetime returns list of datetimes calculated
        from the coordinate bounds."""
        # Assign time bounds equivalent to [
        # datetime(2017, 2, 17, 5, 0),
        # datetime(2017, 2, 17, 6, 0)]
        self.cube.coord('time').bounds = [1487307600, 1487311200]

        result = iris_time_to_datetime(
            self.cube.coord('time'), point_or_bound="bound")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)
        for item in result[0]:
            self.assertIsInstance(item, datetime)
        self.assertEqual(result[0][0], datetime(2017, 2, 17, 5, 0))
        self.assertEqual(result[0][1], datetime(2017, 2, 17, 6, 0))

    def test_input_cube_unmodified(self):
        """Test that an input cube with unexpected coordinate units is not
        modified"""
        self.cube.coord("time").convert_units(
            "hours since 1970-01-01 00:00:00")
        self.cube.coord("time").points = (
            self.cube.coord("time").points.astype(np.int64))
        reference_coord = self.cube.coord("time").copy()
        iris_time_to_datetime(self.cube.coord("time"))
        self.assertArrayEqual(self.cube.coord("time").points,
                              reference_coord.points)
        self.assertArrayEqual(self.cube.coord("time").units,
                              reference_coord.units)
        self.assertEqual(self.cube.coord("time").dtype, np.int64)


class Test_datetime_to_iris_time(IrisTest):

    """Test the datetime_to_iris_time function."""

    def setUp(self):
        """Define datetime for use in tests."""
        self.dt_in = datetime(2017, 2, 17, 6, 0)

    def test_seconds(self):
        """Test datetime_to_iris_time returns float with expected value
        in seconds"""
        result = datetime_to_iris_time(self.dt_in)
        expected = 1487311200.0
        self.assertIsInstance(result, np.int64)
        self.assertEqual(result, expected)


class Test_datetime_constraint(IrisTest):
    """
    Test construction of an iris.Constraint from a python datetime object.
    """
    def setUp(self):
        """Set up test cubes"""
        cube = set_up_variable_cube(
            np.ones((12, 12), dtype=np.float32),
            time=datetime(2017, 2, 17, 6, 0),
            frt=datetime(2017, 2, 17, 6, 0))
        cube.remove_coord("forecast_period")
        self.time_points = np.arange(
            1487311200, 1487354400, 3600).astype(np.int64)
        self.cube = add_coordinate(
            cube, self.time_points, "time", dtype=np.int64,
            coord_units="seconds since 1970-01-01 00:00:00")

    def test_constraint_list_equality(self):
        """Check a list of constraints is as expected."""
        plugin = datetime_constraint
        time_start = datetime(2017, 2, 17, 6, 0)
        time_limit = datetime(2017, 2, 17, 18, 0)
        dt_constraint = plugin(time_start, time_max=time_limit)
        result = self.cube.extract(dt_constraint)
        self.assertEqual(result.shape, (12, 12, 12))
        self.assertArrayEqual(result.coord('time').points, self.time_points)

    def test_constraint_type(self):
        """Check type is iris.Constraint."""
        plugin = datetime_constraint
        dt_constraint = plugin(datetime(2017, 2, 17, 6, 0))
        self.assertIsInstance(dt_constraint, iris.Constraint)

    def test_valid_constraint(self):
        """Test use of constraint at a time valid within the cube."""
        plugin = datetime_constraint
        dt_constraint = plugin(datetime(2017, 2, 17, 6, 0))
        result = self.cube.extract(dt_constraint)
        self.assertIsInstance(result, Cube)

    def test_invalid_constraint(self):
        """Test use of constraint at a time invalid within the cube."""
        plugin = datetime_constraint
        dt_constraint = plugin(datetime(2017, 2, 17, 18, 0))
        result = self.cube.extract(dt_constraint)
        self.assertNotIsInstance(result, Cube)


class Test_extract_cube_at_time(IrisTest):
    """
    Test wrapper for iris cube extraction at desired times.
    """
    def setUp(self):
        """Set up a test cube with several time points"""
        cube = set_up_variable_cube(
            np.ones((12, 12), dtype=np.float32),
            time=datetime(2017, 2, 17, 6, 0),
            frt=datetime(2017, 2, 17, 6, 0))
        cube.remove_coord("forecast_period")
        self.time_points = np.arange(
            1487311200, 1487354400, 3600).astype(np.int64)
        self.cube = add_coordinate(
            cube, self.time_points, "time", dtype=np.int64,
            coord_units="seconds since 1970-01-01 00:00:00")
        self.time_dt = datetime(2017, 2, 17, 6, 0)
        self.time_constraint = iris.Constraint(
            time=lambda cell: cell.point == PartialDateTime(
                self.time_dt.year, self.time_dt.month,
                self.time_dt.day, self.time_dt.hour))

    def test_valid_time(self):
        """Case for a time that is available within the diagnostic cube."""
        plugin = extract_cube_at_time
        cubes = CubeList([self.cube])
        result = plugin(cubes, self.time_dt, self.time_constraint)
        self.assertIsInstance(result, Cube)

    def test_valid_time_for_coord_with_bounds(self):
        """Case for a time that is available within the diagnostic cube.
           Test it still works for coordinates with bounds."""
        plugin = extract_cube_at_time
        self.cube.coord("time").guess_bounds()
        cubes = CubeList([self.cube])
        result = plugin(cubes, self.time_dt, self.time_constraint)
        self.assertIsInstance(result, Cube)

    @ManageWarnings(record=True)
    def test_invalid_time(self, warning_list=None):
        """Case for a time that is unavailable within the diagnostic cube."""
        plugin = extract_cube_at_time
        time_dt = datetime(2017, 2, 18, 6, 0)
        time_constraint = iris.Constraint(time=PartialDateTime(
            time_dt.year, time_dt.month, time_dt.day, time_dt.hour))
        cubes = CubeList([self.cube])
        plugin(cubes, time_dt, time_constraint)
        warning_msg = "Forecast time"
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))


class Test_extract_nearest_time_point(IrisTest):

    """Test the extract_nearest_time_point function."""

    def setUp(self):
        """Set up a cube for the tests."""
        cube = set_up_variable_cube(
            np.ones((1, 7, 7), dtype=np.float32),
            time=datetime(2015, 11, 23, 7, 0),
            frt=datetime(2015, 11, 23, 3, 0))
        cube.remove_coord("forecast_period")
        time_points = [1448262000, 1448265600]
        self.cube = add_coordinate(
            cube, time_points, "time", dtype=np.int64,
            coord_units="seconds since 1970-01-01 00:00:00",
            order=[1, 0, 2, 3])

    def test_time_coord(self):
        """Test that the nearest time point within the time coordinate is
        extracted."""
        expected = self.cube[:, 0, :, :]
        time_point = datetime(2015, 11, 23, 6, 31)
        result = extract_nearest_time_point(self.cube, time_point,
                                            allowed_dt_difference=1800)
        self.assertEqual(result, expected)

    def test_time_coord_lower_case(self):
        """Test that the nearest time point within the time coordinate is
        extracted, when a time of 07:30 is requested."""
        expected = self.cube[:, 0, :, :]
        time_point = datetime(2015, 11, 23, 7, 30)
        result = extract_nearest_time_point(self.cube, time_point,
                                            allowed_dt_difference=1800)
        self.assertEqual(result, expected)

    def test_time_coord_upper_case(self):
        """Test that the nearest time point within the time coordinate is
        extracted, when a time of 07:31 is requested."""
        expected = self.cube[:, 1, :, :]
        time_point = datetime(2015, 11, 23, 7, 31)
        result = extract_nearest_time_point(self.cube, time_point,
                                            allowed_dt_difference=1800)
        self.assertEqual(result, expected)

    def test_forecast_reference_time_coord(self):
        """Test that the nearest time point within the forecast_reference_time
        coordinate is extracted."""
        later_frt = self.cube.copy()
        later_frt.coord('forecast_reference_time').points = (
            later_frt.coord('forecast_reference_time').points + 3600)
        cubes = iris.cube.CubeList([self.cube, later_frt])
        cube = cubes.merge_cube()
        expected = self.cube
        time_point = datetime(2015, 11, 23, 3, 29)
        result = extract_nearest_time_point(
            cube, time_point, time_name="forecast_reference_time",
            allowed_dt_difference=1800)
        self.assertEqual(result, expected)

    def test_exception_using_allowed_dt_difference(self):
        """Test that an exception is raised, if the time point is outside of
        the allowed difference specified in seconds."""
        time_point = datetime(2017, 11, 23, 6, 0)
        msg = "is not available within the input cube"
        with self.assertRaisesRegex(ValueError, msg):
            extract_nearest_time_point(self.cube, time_point,
                                       allowed_dt_difference=3600)

    def test_time_name_exception(self):
        """Test that an exception is raised, if an invalid time name
        is specified."""
        time_point = datetime(2017, 11, 23, 6, 0)
        msg = ("The time_name must be either "
               "'time' or 'forecast_reference_time'")
        with self.assertRaisesRegex(ValueError, msg):
            extract_nearest_time_point(self.cube, time_point,
                                       time_name="forecast_period")


if __name__ == '__main__':
    unittest.main()
