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
from cf_units import Unit
import unittest
import numpy as np

import iris
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest
from iris.cube import Cube, CubeList
from iris.time import PartialDateTime

from improver.utilities.temporal import (
    cycletime_to_datetime, cycletime_to_number, forecast_period_coord,
    iris_time_to_datetime, datetime_constraint,
    extract_cube_at_time, set_utc_offset, get_forecast_times,
    unify_forecast_reference_time, find_latest_cycletime,
    extract_nearest_time_point, datetime_to_iris_time)

from improver.tests.set_up_test_cubes import (
    set_up_variable_cube, add_coordinate)
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


class Test_forecast_period_coord(IrisTest):

    """Test determining of the lead times present within the input cube."""

    def setUp(self):
        """Set up a test cube with a forecast period scalar coordinate"""
        self.cube = set_up_variable_cube(np.ones((1, 3, 3), dtype=np.float32))

    def test_basic(self):
        """Test that an iris.coord.DimCoord is returned."""
        result = forecast_period_coord(self.cube)
        self.assertIsInstance(result, iris.coords.DimCoord)

    def test_basic_AuxCoord(self):
        """Test that an iris.coord.AuxCoord is returned."""
        self.cube.remove_coord('forecast_period')
        result = forecast_period_coord(
            self.cube, force_lead_time_calculation=True)
        self.assertIsInstance(result, iris.coords.AuxCoord)

    def test_check_coordinate(self):
        """Test that the data within the coord is as expected with the
        expected units, when the input cube has a forecast_period coordinate.
        """
        fp_coord = self.cube.coord("forecast_period").copy()
        expected_points = fp_coord.points
        expected_units = str(fp_coord.units)
        result = forecast_period_coord(self.cube)
        self.assertArrayEqual(result.points, expected_points)
        self.assertEqual(str(result.units), expected_units)

    def test_check_coordinate_force_lead_time_calculation(self):
        """Test that the data within the coord is as expected with the
        expected units, when the input cube has a forecast_period coordinate.
        """
        fp_coord = self.cube.coord("forecast_period").copy()
        expected_points = fp_coord.points
        expected_units = str(fp_coord.units)
        result = forecast_period_coord(
            self.cube, force_lead_time_calculation=True)
        self.assertArrayEqual(result.points, expected_points)
        self.assertEqual(result.units, expected_units)

    def test_check_coordinate_in_hours_force_lead_time_calculation(self):
        """Test that the data within the coord is as expected with the
        expected units, when the input cube has a forecast_period coordinate.
        """
        fp_coord = self.cube.coord("forecast_period").copy()
        fp_coord.convert_units("hours")
        expected_points = fp_coord.points
        expected_units = str(fp_coord.units)
        result = forecast_period_coord(
            self.cube, force_lead_time_calculation=True,
            result_units=fp_coord.units)
        self.assertArrayEqual(result.points, expected_points)
        self.assertEqual(result.units, expected_units)

    def test_check_coordinate_without_forecast_period(self):
        """Test that the data within the coord is as expected with the
        expected units, when the input cube has a time coordinate and a
        forecast_reference_time coordinate.
        """
        fp_coord = self.cube.coord("forecast_period").copy()
        expected_result = fp_coord
        self.cube.remove_coord("forecast_period")
        result = forecast_period_coord(self.cube)
        self.assertEqual(result, expected_result)

    def test_check_time_unit_conversion(self):
        """Test that the data within the coord is as expected with the
        expected units, when the input cube has a time coordinate with units
        other than the usual units of seconds since 1970-01-01 00:00:00.
        """
        expected_result = self.cube.coord("forecast_period")
        self.cube.coord("time").convert_units(
            "hours since 1970-01-01 00:00:00")
        result = forecast_period_coord(
            self.cube, force_lead_time_calculation=True)
        self.assertEqual(result, expected_result)

    def test_check_time_unit_has_bounds(self):
        """Test that the forecast_period coord has bounds if time has bounds.
        """
        cube = set_up_variable_cube(
            np.ones((3, 3), dtype=np.float32),
            time=datetime.datetime(2018, 3, 12, 20, 0),
            frt=datetime.datetime(2018, 3, 12, 15, 0),
            time_bounds=[datetime.datetime(2018, 3, 12, 19, 0),
                         datetime.datetime(2018, 3, 12, 20, 0)])
        expected_result = cube.coord("forecast_period").copy()
        expected_result.bounds = [[14400, 18000]]
        result = forecast_period_coord(cube, force_lead_time_calculation=True)
        self.assertEqual(result, expected_result)

    @ManageWarnings(record=True)
    def test_negative_forecast_periods_warning(self, warning_list=None):
        """Test that a warning is raised if the point within the
        time coordinate is prior to the point within the
        forecast_reference_time, and therefore the forecast_period values that
        have been generated are negative.
        """
        cube = set_up_variable_cube(np.ones((3, 3), dtype=np.float32))
        cube.remove_coord("forecast_period")
        # default cube has a 4 hour forecast period, so add 5 hours to frt
        cube.coord("forecast_reference_time").points = (
            cube.coord("forecast_reference_time").points + 5*3600)
        warning_msg = "The values for the time"
        forecast_period_coord(cube)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))

    def test_exception_raised(self):
        """Test that a CoordinateNotFoundError exception is raised if the
        forecast_period, or the time and forecast_reference_time,
        are not present.
        """
        self.cube.remove_coord("forecast_reference_time")
        self.cube.remove_coord("forecast_period")
        msg = "The forecast period coordinate is not available"
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            forecast_period_coord(self.cube)


class Test_iris_time_to_datetime(IrisTest):
    """ Test iris_time_to_datetime """
    def test_basic(self):
        """Test iris_time_to_datetime returns list of datetime """
        cube = set_up_variable_cube(
            np.ones((3, 3), dtype=np.float32),
            time=datetime.datetime(2017, 2, 17, 6, 0),
            frt=datetime.datetime(2017, 2, 17, 3, 0))
        result = iris_time_to_datetime(cube.coord('time'))
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], datetime.datetime(2017, 2, 17, 6, 0))


class Test_datetime_to_iris_time(IrisTest):

    """Test the datetime_to_iris_time function."""

    def setUp(self):
        """Define datetime for use in tests."""
        self.dt_in = datetime.datetime(2017, 2, 17, 6, 0)

    def test_hours(self):
        """Test datetime_to_iris_time returns float with expected value
        in hours"""
        result = datetime_to_iris_time(self.dt_in)
        expected = 413142.0
        self.assertIsInstance(result, float)
        self.assertEqual(result, expected)

    def test_minutes(self):
        """Test datetime_to_iris_time returns float with expected value
        in minutes"""
        result = datetime_to_iris_time(self.dt_in, time_units="minutes")
        expected = 24788520.0
        self.assertIsInstance(result, float)
        self.assertEqual(result, expected)

    def test_seconds(self):
        """Test datetime_to_iris_time returns float with expected value
        in seconds"""
        result = datetime_to_iris_time(self.dt_in, time_units="seconds")
        expected = 1487311200.0
        self.assertIsInstance(result, float)
        self.assertEqual(result, expected)

    def test_seconds_from_origin(self):
        """Test datetime_to_iris_time returns float with expected value
        in seconds when an origin is supplied."""
        result = datetime_to_iris_time(
            self.dt_in, time_units="seconds since 1970-01-01 00:00:00")
        expected = 1487311200.0
        self.assertIsInstance(result, float)
        self.assertEqual(result, expected)

    def test_exception_raised(self):
        """Test an exception is raised if the if the time unit does not
        contain hours, minutes or seconds."""
        msg = "The time unit must contain 'hours', 'minutes' or 'seconds'"
        with self.assertRaisesRegex(ValueError, msg):
            datetime_to_iris_time(self.dt_in, time_units="days")


class Test_datetime_constraint(IrisTest):
    """
    Test construction of an iris.Constraint from a python.datetime.datetime
    object.
    """
    def setUp(self):
        """Set up test cubes"""
        cube = set_up_variable_cube(
            np.ones((12, 12), dtype=np.float32),
            time=datetime.datetime(2017, 2, 17, 6, 0),
            frt=datetime.datetime(2017, 2, 17, 6, 0))
        cube.remove_coord("forecast_period")
        self.time_points = np.arange(
            1487311200, 1487354400, 3600).astype(np.int64)
        self.cube = add_coordinate(
            cube, self.time_points, "time", dtype=np.int64,
            coord_units="seconds since 1970-01-01 00:00:00")

    def test_constraint_list_equality(self):
        """Check a list of constraints is as expected."""
        plugin = datetime_constraint
        time_start = datetime.datetime(2017, 2, 17, 6, 0)
        time_limit = datetime.datetime(2017, 2, 17, 18, 0)
        dt_constraint = plugin(time_start, time_max=time_limit)
        result = self.cube.extract(dt_constraint)
        self.assertEqual(result.shape, (12, 12, 12))
        self.assertArrayEqual(result.coord('time').points, self.time_points)

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


class Test_extract_cube_at_time(IrisTest):
    """
    Test wrapper for iris cube extraction at desired times.
    """
    def setUp(self):
        """Set up a test cube with several time points"""
        cube = set_up_variable_cube(
            np.ones((12, 12), dtype=np.float32),
            time=datetime.datetime(2017, 2, 17, 6, 0),
            frt=datetime.datetime(2017, 2, 17, 6, 0))
        cube.remove_coord("forecast_period")
        self.time_points = np.arange(
            1487311200, 1487354400, 3600).astype(np.int64)
        self.cube = add_coordinate(
            cube, self.time_points, "time", dtype=np.int64,
            coord_units="seconds since 1970-01-01 00:00:00")
        self.time_dt = datetime.datetime(2017, 2, 17, 6, 0)
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
        time_dt = datetime.datetime(2017, 2, 18, 6, 0)
        time_constraint = iris.Constraint(time=PartialDateTime(
            time_dt.year, time_dt.month, time_dt.day, time_dt.hour))
        cubes = CubeList([self.cube])
        plugin(cubes, time_dt, time_constraint)
        warning_msg = "Forecast time"
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))


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


class Test_extract_nearest_time_point(IrisTest):

    """Test the extract_nearest_time_point function."""

    def setUp(self):
        """Set up a cube for the tests."""
        cube = set_up_variable_cube(
            np.ones((1, 7, 7), dtype=np.float32),
            time=datetime.datetime(2015, 11, 23, 7, 0),
            frt=datetime.datetime(2015, 11, 23, 3, 0))
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
        time_point = datetime.datetime(2015, 11, 23, 6, 0)
        result = extract_nearest_time_point(self.cube, time_point)
        self.assertEqual(result, expected)

    def test_time_coord_lower_case(self):
        """Test that the nearest time point within the time coordinate is
        extracted, when a time of 07:30 is requested."""
        expected = self.cube[:, 0, :, :]
        time_point = datetime.datetime(2015, 11, 23, 7, 30)
        result = extract_nearest_time_point(self.cube, time_point)
        self.assertEqual(result, expected)

    def test_time_coord_upper_case(self):
        """Test that the nearest time point within the time coordinate is
        extracted, when a time of 07:31 is requested."""
        expected = self.cube[:, 1, :, :]
        time_point = datetime.datetime(2015, 11, 23, 7, 31)
        result = extract_nearest_time_point(self.cube, time_point)
        self.assertEqual(result, expected)

    def test_forecast_reference_time_coord(self):
        """Test that the nearest time point within the forecast_reference_time
        coordinate is extracted."""
        expected = self.cube
        time_point = datetime.datetime(2015, 11, 23, 6, 0)
        result = extract_nearest_time_point(
            self.cube, time_point, time_name="forecast_reference_time")
        self.assertEqual(result, expected)

    def test_exception_using_allowed_dt_difference(self):
        """Test that an exception is raised, if the time point is outside of
        the allowed difference specified in seconds."""
        time_point = datetime.datetime(2017, 11, 23, 6, 0)
        msg = "is not available within the input cube"
        with self.assertRaisesRegex(ValueError, msg):
            extract_nearest_time_point(self.cube, time_point,
                                       allowed_dt_difference=3600)

    def test_time_name_exception(self):
        """Test that an exception is raised, if an invalid time name
        is specified."""
        time_point = datetime.datetime(2017, 11, 23, 6, 0)
        msg = ("The time_name must be either "
               "'time' or 'forecast_reference_time'")
        with self.assertRaisesRegex(ValueError, msg):
            extract_nearest_time_point(self.cube, time_point,
                                       time_name="forecast_period")


class Test_unify_forecast_reference_time(IrisTest):

    """Test the unify_forecast_reference_time function."""

    def setUp(self):
        """Set up a UK deterministic cube for testing."""
        self.cycletime = datetime.datetime(2017, 1, 10, 6, 0)
        cube_uk_det = set_up_variable_cube(
            np.full((4, 4), 273.15, dtype=np.float32),
            time=self.cycletime, frt=datetime.datetime(2017, 1, 10, 3, 0))

        cube_uk_det.remove_coord("forecast_period")
        # set up forecast periods of 6, 8 and 10 hours
        time_points = [1484038800, 1484046000, 1484053200]
        cube_uk_det = add_coordinate(
            cube_uk_det, time_points, "time", dtype=np.int64,
            coord_units="seconds since 1970-01-01 00:00:00")
        fp_coord = forecast_period_coord(cube_uk_det)
        cube_uk_det.add_aux_coord(fp_coord, data_dims=0)

        self.cube_uk_det = add_coordinate(cube_uk_det, [1000], "model_id")
        self.cube_uk_det.add_aux_coord(
            iris.coords.AuxCoord(["uk_det"], long_name="model_configuration"))

    def test_cubelist_input(self):
        """Test when supplying a cubelist as input containing cubes
        representing UK deterministic and UK ensemble model configuration
        and unifying the forecast_reference_time, so that both model
        configurations have a common forecast_reference_time."""
        cube_uk_ens = set_up_variable_cube(
            np.full((3, 4, 4), 273.15, dtype=np.float32),
            time=self.cycletime, frt=datetime.datetime(2017, 1, 10, 4, 0))

        cube_uk_ens.remove_coord("forecast_period")
        # set up forecast periods of 5, 7 and 9 hours
        time_points = [1484031600, 1484038800, 1484046000]
        cube_uk_ens = add_coordinate(
            cube_uk_ens, time_points, "time", dtype=np.int64,
            coord_units="seconds since 1970-01-01 00:00:00")
        fp_coord = forecast_period_coord(cube_uk_ens)
        cube_uk_ens.add_aux_coord(fp_coord, data_dims=0)

        expected_uk_det = self.cube_uk_det.copy()
        frt_units = expected_uk_det.coord('forecast_reference_time').units
        frt_points = [
            np.round(frt_units.date2num(self.cycletime)).astype(np.int64)]
        expected_uk_det.coord("forecast_reference_time").points = frt_points
        expected_uk_det.coord("forecast_period").points = (
            np.array([3, 5, 7]) * 3600)
        expected_uk_ens = cube_uk_ens.copy()
        expected_uk_ens.coord("forecast_reference_time").points = frt_points
        expected_uk_ens.coord("forecast_period").points = (
            np.array([1, 3, 5]) * 3600)
        expected = iris.cube.CubeList([expected_uk_det, expected_uk_ens])

        cubes = iris.cube.CubeList([self.cube_uk_det, cube_uk_ens])
        result = unify_forecast_reference_time(cubes, self.cycletime)

        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(result, expected)

    def test_cube_input(self):
        """Test when supplying a cube representing a UK deterministic model
        configuration only. This effectively updates the
        forecast_reference_time on the cube to the specified cycletime."""
        expected_uk_det = self.cube_uk_det.copy()
        frt_units = expected_uk_det.coord('forecast_reference_time').units
        frt_points = [
            np.round(frt_units.date2num(self.cycletime)).astype(np.int64)]
        expected_uk_det.coord("forecast_reference_time").points = frt_points
        expected_uk_det.coord("forecast_period").points = (
            np.array([3, 5, 7]) * 3600)
        result = unify_forecast_reference_time(
            self.cube_uk_det, self.cycletime)
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(result[0], expected_uk_det)

    def test_cube_input_no_forecast_period_coordinate(self):
        """Test when supplying a cube representing a UK deterministic model
        configuration only. This forces a forecast_period coordinate to be
        created from a forecast_reference_time coordinate and a time
        coordinate."""
        expected_uk_det = self.cube_uk_det.copy()
        frt_units = expected_uk_det.coord('forecast_reference_time').units
        frt_points = [
            np.round(frt_units.date2num(self.cycletime)).astype(np.int64)]
        expected_uk_det.coord("forecast_reference_time").points = frt_points
        expected_uk_det.coord("forecast_period").points = (
            np.array([3, 5, 7]) * 3600)
        cube_uk_det = self.cube_uk_det.copy()
        cube_uk_det.remove_coord("forecast_period")
        result = unify_forecast_reference_time(cube_uk_det, self.cycletime)
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertEqual(result[0], expected_uk_det)


class Test_find_latest_cycletime(IrisTest):

    """Test the find_latest_cycletime function."""

    def setUp(self):
        """Set up a template cubes with scalar time, forecast_reference_time
           and forecast_period coordinates"""
        self.input_cube = set_up_variable_cube(
            np.full((7, 7), 273.15, dtype=np.float32),
            time=datetime.datetime(2015, 11, 23, 6, 0),
            frt=datetime.datetime(2015, 11, 23, 3, 0))
        self.input_cube2 = self.input_cube.copy()
        self.input_cube2.coord("forecast_reference_time").points = np.array(
            self.input_cube2.coord("forecast_reference_time").points[0] + 3600)
        self.input_cubelist = iris.cube.CubeList(
            [self.input_cube, self.input_cube2])

    def test_basic(self):
        """Test the type of the output and that the input is unchanged."""
        original_cubelist = iris.cube.CubeList(
            [self.input_cube.copy(), self.input_cube2.copy()])
        cycletime = find_latest_cycletime(self.input_cubelist)
        self.assertEqual(self.input_cubelist[0], original_cubelist[0])
        self.assertEqual(self.input_cubelist[1], original_cubelist[1])
        self.assertIsInstance(cycletime, datetime.datetime)

    def test_returns_latest(self):
        """Test the returned cycle time is the latest in the input cubelist."""
        cycletime = find_latest_cycletime(self.input_cubelist)
        expected_datetime = datetime.datetime(2015, 11, 23, 4, 0, 0)
        self.assertEqual(timedelta(hours=0, seconds=0),
                         cycletime - expected_datetime)

    def test_two_cubes_same_reference_time(self):
        """Test the a cycletime is still found when two cubes have the same
           cycletime."""
        input_cubelist = iris.cube.CubeList(
            [self.input_cube, self.input_cube.copy()])
        cycletime = find_latest_cycletime(input_cubelist)
        expected_datetime = datetime.datetime(2015, 11, 23, 3, 0, 0)
        self.assertEqual(timedelta(hours=0, seconds=0),
                         cycletime - expected_datetime)

    def test_one_input_cube(self):
        """Test the a cycletime is still found when only one input cube."""
        input_cubelist = iris.cube.CubeList([self.input_cube])
        cycletime = find_latest_cycletime(input_cubelist)
        expected_datetime = datetime.datetime(2015, 11, 23, 3, 0, 0)
        self.assertEqual(timedelta(hours=0, seconds=0),
                         cycletime - expected_datetime)

    def test_different_units(self):
        """Test the right cycletime is still the coords have different
           units."""
        self.input_cube2.coord("forecast_reference_time").convert_units(
            'minutes since 1970-01-01 00:00:00')
        cycletime = find_latest_cycletime(self.input_cubelist)
        expected_datetime = datetime.datetime(2015, 11, 23, 4, 0, 0)
        self.assertEqual(timedelta(hours=0, seconds=0),
                         cycletime - expected_datetime)

    def test_raises_error(self):
        """Test the error is raised if time is dimensional"""
        input_cube2 = iris.util.new_axis(
            self.input_cube2, "forecast_reference_time")
        input_cubelist = iris.cube.CubeList([self.input_cube, input_cube2])
        msg = "Expecting scalar forecast_reference_time for each input cube"
        with self.assertRaisesRegex(ValueError, msg):
            find_latest_cycletime(input_cubelist)


if __name__ == '__main__':
    unittest.main()
