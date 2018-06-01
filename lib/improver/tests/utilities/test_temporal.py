# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
from datetime import time
from datetime import timedelta
import unittest
import numpy as np

import iris
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest
from iris.cube import Cube, CubeList
from iris.time import PartialDateTime

from improver.utilities.temporal import (
    cycletime_to_datetime, cycletime_to_number, forecast_period_coord,
    iris_time_to_datetime, dt_to_utc_hours, datetime_constraint,
    extract_cube_at_time, set_utc_offset, get_forecast_times)
from improver.tests.ensemble_calibration.ensemble_calibration.helper_functions\
    import add_forecast_reference_time_and_forecast_period
from improver.tests.nbhood.nbhood.test_NeighbourhoodProcessing import (
    set_up_cube)
from improver.tests.spotdata.spotdata.test_common_functions import (
    Test_common_functions)
from improver.utilities.warnings_handler import ManageWarnings


class Test_cycletime_to_datetime(IrisTest):

    """Test that a cycletime of a format such as YYYYMMDDTHHMMZ is converted
    into a datetime object."""

    def test_basic(self):
        """Test that a datetime object is returned of the expected value."""
        cycletime = "20171122T0100Z"
        dt = datetime.datetime(2017, 11, 22, 1, 0)
        result = cycletime_to_datetime(cycletime)
        self.assertIsInstance(result, datetime.datetime)
        self.assertEqual(result, dt)

    def test_define_cycletime_format(self):
        """Test when a cycletime is defined."""
        cycletime = "201711220100"
        dt = datetime.datetime(2017, 11, 22, 1, 0)
        result = cycletime_to_datetime(
            cycletime, cycletime_format="%Y%m%d%H%M")
        self.assertEqual(result, dt)


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
        """Test when alternative units are defined."""
        cycletime = "20171122T0000Z"
        dt = 1511308800.0
        result = cycletime_to_number(
            cycletime, time_unit="seconds since 1970-01-01 00:00:00")
        self.assertAlmostEqual(result, dt)

    def test_alternative_calendar_defined(self):
        """Test when an alternative calendar is defined."""
        cycletime = "20171122T0000Z"
        dt = 419520.0
        result = cycletime_to_number(
            cycletime, calendar="365_day")
        self.assertAlmostEqual(result, dt)


class Test_forecast_period_coord(IrisTest):

    """Test determining of the lead times present within the input cube."""

    def test_basic(self):
        """Test that an iris.coord.DimCoord is returned."""
        cube = add_forecast_reference_time_and_forecast_period(set_up_cube())
        result = forecast_period_coord(cube)
        self.assertIsInstance(result, iris.coords.DimCoord)

    def test_basic_AuxCoord(self):
        """Test that an iris.coord.AuxCoord is returned."""
        cube = add_forecast_reference_time_and_forecast_period(set_up_cube())
        cube.remove_coord('forecast_period')
        result = forecast_period_coord(cube, force_lead_time_calculation=True)
        self.assertIsInstance(result, iris.coords.AuxCoord)

    def test_check_coordinate(self):
        """Test that the data within the coord is as expected with the
        expected units, when the input cube has a forecast_period coordinate.
        """
        cube = add_forecast_reference_time_and_forecast_period(set_up_cube())
        fp_coord = cube.coord("forecast_period").copy()
        fp_coord.convert_units("seconds")
        expected_points = fp_coord.points
        expected_units = str(fp_coord.units)
        result = forecast_period_coord(cube)
        self.assertArrayAlmostEqual(result.points, expected_points)
        self.assertEqual(str(result.units), expected_units)

    def test_check_coordinate_force_lead_time_calculation(self):
        """Test that the data within the coord is as expected with the
        expected units, when the input cube has a forecast_period coordinate.
        """
        cube = add_forecast_reference_time_and_forecast_period(set_up_cube())
        fp_coord = cube.coord("forecast_period").copy()
        fp_coord.convert_units("seconds")
        expected_points = fp_coord.points
        expected_units = str(fp_coord.units)
        result = forecast_period_coord(
            cube, force_lead_time_calculation=True)
        self.assertArrayAlmostEqual(result.points, expected_points)
        self.assertEqual(result.units, expected_units)

    def test_check_coordinate_without_forecast_period(self):
        """Test that the data within the coord is as expected with the
        expected units, when the input cube has a time coordinate and a
        forecast_reference_time coordinate.
        """
        cube = add_forecast_reference_time_and_forecast_period(set_up_cube())
        fp_coord = cube.coord("forecast_period").copy()
        fp_coord.convert_units("seconds")
        expected_result = fp_coord
        cube.remove_coord("forecast_period")
        result = forecast_period_coord(cube)
        self.assertEqual(result, expected_result)

    def test_check_time_unit_conversion(self):
        """Test that the data within the coord is as expected with the
        expected units, when the input cube has a time coordinate with units
        other than the usual units of hours since 1970-01-01 00:00:00.
        """
        cube = add_forecast_reference_time_and_forecast_period(set_up_cube())
        fp_coord = cube.coord("forecast_period").copy()
        fp_coord.convert_units("seconds")
        expected_result = fp_coord
        cube.coord("time").convert_units("seconds since 1970-01-01 00:00:00")
        result = forecast_period_coord(cube, force_lead_time_calculation=True)
        self.assertEqual(result, expected_result)

    @ManageWarnings(record=True)
    def test_negative_forecast_periods_warning(self, warning_list=None):
        """Test that a warning is raised if the point within the
        time coordinate is prior to the point within the
        forecast_reference_time, and therefore the forecast_period values that
        have been generated are negative.
        """
        cube = add_forecast_reference_time_and_forecast_period(set_up_cube())
        cube.remove_coord("forecast_period")
        cube.coord("forecast_reference_time").points = 402295.0
        cube.coord("time").points = 402192.5
        msg = "The values for the time"
        forecast_period_coord(cube)
        self.assertTrue(len(warning_list) == 1)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(msg in str(warning_list[0]))

    def test_exception_raised(self):
        """Test that a CoordinateNotFoundError exception is raised if the
        forecast_period, or the time and forecast_reference_time,
        are not present.
        """
        cube = set_up_cube()
        msg = "The forecast period coordinate is not available"
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            forecast_period_coord(cube)


class Test_iris_time_to_datetime(Test_common_functions):
    """ Test iris_time_to_datetime """
    def test_basic(self):
        """Test iris_time_to_datetime returns list of datetime """
        result = iris_time_to_datetime(self.cube.coord('time'))
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], datetime.datetime(2017, 2, 17, 6, 0))


class Test_dt_to_utc_hours(IrisTest):
    """ Test dt_to_utc_hours """
    def test_basic(self):
        """Test dt_to_utc_hours returns float with expected value """
        dt_in = datetime.datetime(2017, 2, 17, 6, 0)
        result = dt_to_utc_hours(dt_in)
        expected = 413142.0
        self.assertIsInstance(result, float)
        self.assertEqual(result, expected)


class Test_datetime_constraint(Test_common_functions):
    """
    Test construction of an iris.Constraint from a python.datetime.datetime
    object.
    """

    def test_constraint_list_equality(self):
        """Check a list of constraints is as expected."""
        plugin = datetime_constraint
        time_start = datetime.datetime(2017, 2, 17, 6, 0)
        time_limit = datetime.datetime(2017, 2, 17, 18, 0)
        expected_times = list(range(1487311200, 1487354400, 3600))
        dt_constraint = plugin(time_start, time_max=time_limit)
        result = self.long_cube.extract(dt_constraint)
        self.assertEqual(result.shape, (12, 12, 12))
        self.assertArrayEqual(result.coord('time').points,
                              expected_times)

    def test_constraint_type(self):
        """Check type is iris.Constraint."""
        plugin = datetime_constraint
        dt_constraint = plugin(datetime.datetime(2017, 2, 17, 6, 0))
        self.assertIsInstance(dt_constraint, iris.Constraint)

    def test_valid_constraint(self):
        """Test use of constraint at a time valid within the cube."""
        plugin = datetime_constraint
        dt_constraint = plugin(datetime.datetime(2017, 2, 17, 6, 0))
        result = self.cube.extract(dt_constraint)
        self.assertIsInstance(result, Cube)

    def test_invalid_constraint(self):
        """Test use of constraint at a time invalid within the cube."""
        plugin = datetime_constraint
        dt_constraint = plugin(datetime.datetime(2017, 2, 17, 18, 0))
        result = self.cube.extract(dt_constraint)
        self.assertNotIsInstance(result, Cube)


class Test_extract_cube_at_time(Test_common_functions):
    """
    Test wrapper for iris cube extraction at desired times.

    """

    def test_valid_time(self):
        """Case for a time that is available within the diagnostic cube."""
        plugin = extract_cube_at_time
        cubes = CubeList([self.cube])
        result = plugin(cubes, self.time_dt, self.time_extract)
        self.assertIsInstance(result, Cube)

    def test_valid_time_for_coord_with_bounds(self):
        """Case for a time that is available within the diagnostic cube.
           Test it still works for coordinates with bounds."""
        plugin = extract_cube_at_time
        self.long_cube.coord("time").guess_bounds()
        cubes = CubeList([self.long_cube])
        result = plugin(cubes, self.time_dt, self.time_extract)
        self.assertIsInstance(result, Cube)

    @ManageWarnings(record=True)
    def test_invalid_time(self, warning_list=None):
        """Case for a time that is unavailable within the diagnostic cube."""
        plugin = extract_cube_at_time
        time_dt = datetime.datetime(2017, 2, 18, 6, 0)
        time_extract = iris.Constraint(time=PartialDateTime(
            time_dt.year, time_dt.month, time_dt.day, time_dt.hour))
        cubes = CubeList([self.cube])
        plugin(cubes, time_dt, time_extract)
        self.assertTrue(len(warning_list), 1)
        self.assertTrue(issubclass(warning_list[0].category, UserWarning))
        self.assertTrue("Forecast time" in str(warning_list[0]))


class Test_set_utc_offset(IrisTest):
    """
    Test setting of UTC_offsets with longitudes using crude 15 degree bins.

    """

    def test_output(self):
        """
        Test full span of crude timezones from UTC-12 to UTC+12. Note the
        degeneracy at +-180.

        """
        longitudes = np.arange(-180, 185, 15)
        expected = np.arange(-12, 13, 1)
        result = set_utc_offset(longitudes)
        self.assertArrayEqual(expected, result)


class Test_get_forecast_times(IrisTest):

    """Test the generation of forecast time using the function."""

    def test_all_data_provided(self):
        """Test setting up a forecast range when start date, start hour and
        forecast length are all provided."""

        forecast_start = datetime.datetime(2017, 6, 1, 9, 0)
        forecast_date = forecast_start.strftime("%Y%m%d")
        forecast_time = int(forecast_start.strftime("%H"))
        forecast_length = 300
        forecast_end = forecast_start + timedelta(hours=forecast_length)
        result = get_forecast_times(forecast_length,
                                    forecast_date=forecast_date,
                                    forecast_time=forecast_time)
        self.assertEqual(forecast_start, result[0])
        self.assertEqual(forecast_end, result[-1])
        self.assertEqual(timedelta(hours=1), result[1] - result[0])
        self.assertEqual(timedelta(hours=3), result[-1] - result[-2])

    def test_no_data_provided(self):
        """Test setting up a forecast range when no data is provided. Expect a
        range of times starting from last hour before now that was an interval
        of 6 hours. Length set to 7 days (168 hours).

        Note: this could fail if time between forecast_start being set and
        reaching the get_forecast_times call bridges a 6-hour time
        (00, 06, 12, 18). As such it is allowed two goes before
        reporting a failure (slightly unconventional I'm afraid)."""

        second_chance = 0
        while second_chance < 2:
            forecast_start = datetime.datetime.utcnow()
            expected_date = forecast_start.date()
            expected_hour = time(divmod(forecast_start.hour, 6)[0]*6)
            forecast_date = None
            forecast_time = None
            forecast_length = 168
            result = get_forecast_times(forecast_length,
                                        forecast_date=forecast_date,
                                        forecast_time=forecast_time)

            check1 = (expected_date == result[0].date())
            check2 = (expected_hour.hour == result[0].hour)
            check3 = (timedelta(hours=168) == (result[-1] - result[0]))

            if not all([check1, check2, check3]):
                second_chance += 1
                continue
            else:
                break

        self.assertTrue(check1)
        self.assertTrue(check2)
        self.assertTrue(check3)

    def test_partial_data_provided(self):
        """Test setting up a forecast range when start hour and forecast length
        are both provided, but no start date."""

        forecast_start = datetime.datetime(2017, 6, 1, 15, 0)
        forecast_date = None
        forecast_time = int(forecast_start.strftime("%H"))
        forecast_length = 144
        expected_date = datetime.datetime.utcnow().date()
        expected_start = datetime.datetime.combine(expected_date,
                                                   time(forecast_time))
        expected_end = expected_start + timedelta(hours=144)
        result = get_forecast_times(forecast_length,
                                    forecast_date=forecast_date,
                                    forecast_time=forecast_time)

        self.assertEqual(expected_start, result[0])
        self.assertEqual(expected_end, result[-1])
        self.assertEqual(timedelta(hours=1), result[1] - result[0])
        self.assertEqual(timedelta(hours=3), result[-1] - result[-2])

    def test_invalid_date_format(self):
        """Test error is raised when a date is provided in an unexpected
        format."""

        forecast_date = '17MARCH2017'
        msg = 'Date .* is in unexpected format'
        with self.assertRaisesRegex(ValueError, msg):
            get_forecast_times(144, forecast_date=forecast_date,
                               forecast_time=6)


if __name__ == '__main__':
    unittest.main()
