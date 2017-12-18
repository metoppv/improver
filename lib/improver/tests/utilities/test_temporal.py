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
"""Unit tests for temporal utilities."""

import datetime
import unittest
import warnings

import iris
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest

from improver.utilities.temporal import (
    cycletime_to_datetime, cycletime_to_number, forecast_period_coord)
from improver.tests.ensemble_calibration.ensemble_calibration.helper_functions\
    import add_forecast_reference_time_and_forecast_period
from improver.tests.nbhood.nbhood.test_NeighbourhoodProcessing import (
    set_up_cube)


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

    def test_negative_forecast_periods_warning(self):
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
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
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
        with self.assertRaisesRegexp(CoordinateNotFoundError, msg):
            forecast_period_coord(cube)


if __name__ == '__main__':
    unittest.main()
