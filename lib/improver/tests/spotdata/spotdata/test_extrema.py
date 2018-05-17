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
"""Unit tests for spotdata.extrema"""


import unittest
import numpy as np
import cf_units
from iris import Constraint
from iris.coords import (DimCoord,
                         AuxCoord)
from iris.cube import Cube
from iris.tests import IrisTest
from time import mktime
from datetime import datetime as dt

from improver.spotdata.extrema import ExtractExtrema as Plugin
from improver.spotdata.extrema import make_local_time_cube
from improver.spotdata.extrema import get_datetime_limits
from improver.utilities.warnings_handler import ManageWarnings


class Test_extrema(IrisTest):
    """Setup tests for plugin and functions in extrema.py"""

    def setUp(self):
        """Set up the initial conditions for tests."""

        times = list(range(414048, 414097, 1))  # Hours since 00Z-01-01-1970
        n_times, n_data = len(times), 27
        indices = DimCoord(np.arange(n_data), long_name='index',
                           units='1')
        time_units = cf_units.Unit('hours since 1970-01-01 00:00:00',
                                   calendar='gregorian')
        time_coord = DimCoord(times, standard_name='time',
                              units=time_units)
        forecast_ref_time = time_coord[0].copy()
        forecast_ref_time.rename('forecast_reference_time')

        # UTC offset arranged to sequence -12 --> 14.
        utc_offsets = list(range(-12, 15)) * int(n_data/24)
        utc_offset = AuxCoord(utc_offsets, long_name='utc_offset',
                              units='hours')
        # Data arranged to ascend 0 --> n_data, so each site shows a constant
        # temperature over time; this will need to be modified to check
        # analysis collapse method.
        data = np.array([list(range(n_data))*n_times])
        data.resize(n_times, n_data)

        cube = Cube(data,
                    long_name="air_temperature",
                    dim_coords_and_dims=[(time_coord, 0), (indices, 1)],
                    aux_coords_and_dims=[(utc_offset, 1)],
                    units="K")
        cube.add_aux_coord(forecast_ref_time)
        self.cube = cube
        self.time_coord = time_coord
        self.n_data = n_data


class Test_ExtractExtrema(Test_extrema):
    """Test the extraction of maxima/minima values in given periods, where the
    periods are in local time.

    Imagine that 27 sites sample all timeszones from UTC-12 to UTC+14."""

    def test_repr(self):
        """Test return from __repr__ in class."""
        expected = '<ExtractExtrema: period: 24, start_hour: 9>'
        self.assertEqual(expected, Plugin(24).__repr__())
        expected = '<ExtractExtrema: period: 12, start_hour: 9>'
        self.assertEqual(expected, Plugin(12).__repr__())
        expected = '<ExtractExtrema: period: 12, start_hour: 12>'
        self.assertEqual(expected, Plugin(12, start_hour=12).__repr__())

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_time_coordinates_24_hour(self):
        """Time coordinate should be a series of mid points calculated from the
        start hour + half the period. Each should have an associated pair of
        bounds that show the range over which the extrema values have been
        calculated.

        The first day is the first day for which any site has valid data. The
        UTC time coord starts at 00 UTC on 27th April 2017, so the first day
        in which any data falls is 26th April 2017 (for any UTC-N sites).
        The first 24 hours starting at 00 therefore runs 00 26th to 00 27th.
        Subsequent 24 hour periods are then expected to the latest day for
        which any site has data; a UTC+14 site.

        Input data spans 48 hours, this will spread to three days with timezone
        adjustments."""

        n_periods = 72//24
        mid_start = mktime(dt(2017, 3, 26, 12).utctimetuple())/3600.
        lower_bound = mktime(dt(2017, 3, 26, 00).utctimetuple())/3600.
        upper_bound = mktime(dt(2017, 3, 27, 00).utctimetuple())/3600.

        result = Plugin(24, start_hour=0).process(self.cube)
        result = result.extract(Constraint(name='air_temperature_max'))

        # Expected time coordinate values.
        for i in range(n_periods):
            mid_time = mid_start + (i*24.)
            low_bound = lower_bound + (i*24.)
            up_bound = upper_bound + (i*24.)

            self.assertEqual(result[i].coord('time').points, [mid_time])
            self.assertEqual(result[i].coord('time').bounds[0, 0], [low_bound])
            self.assertEqual(result[i].coord('time').bounds[0, 1], [up_bound])

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_time_coordinates_9_hour(self):
        """Time coordinate should be a series of mid points calculated from the
        start hour + half the period. Each should have an associated pair of
        bounds that show the range over which the extrema values have been
        calculated.

        The first day is the first day for which any site has valid data. The
        UTC time coord starts at 00 UTC on 27th April 2017, so the first day
        in which any data falls is 26th April 2017 (for any UTC-N sites).
        The first 9 hours to contain any data are 09-18, so the first period
        runs 09 26th to 18 26th. Subsequent 9 hour periods are then expected
        to the latest day for which any site has data; a UTC+14 site.

        Input data spans 48 hours, this will spread to three days with timezone
        adjustments."""

        # -1 as no data falls in the first 9 hour period.
        n_periods = 72//9 - 1

        mid_start = mktime(dt(2017, 3, 26, 13, 30).utctimetuple())/3600.
        lower_bound = mktime(dt(2017, 3, 26, 9).utctimetuple())/3600.
        upper_bound = mktime(dt(2017, 3, 26, 18).utctimetuple())/3600.

        result = Plugin(9, start_hour=0).process(self.cube)
        result = result.extract(Constraint(name='air_temperature_max'))

        # Expected time coordinate values.
        for i in range(n_periods):
            mid_time = mid_start + (i*9.)
            low_bound = lower_bound + (i*9.)
            up_bound = upper_bound + (i*9.)
            self.assertEqual(result[i].coord('time').points, [mid_time])
            self.assertEqual(result[i].coord('time').bounds[0, 0], [low_bound])
            self.assertEqual(result[i].coord('time').bounds[0, 1], [up_bound])

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_data_arrays_day1(self):
        """Test extraction of maxima and minima values from the time localised
        cube in the first 24 hours. The first day is the first day for which
        any site has valid data. The UTC time coord starts at 00 UTC on 27th
        April 2017, so the first day in which any data falls is 26th April
        2017 (for any UTC-N sites). The first 24 hours starting at 00
        therefore runs 00 26th to 00 27th. Any sites UTC+N will be have no
        valid data for this first day. That the correct sites return valid data
        is tested here."""

        # Expected time coordinate values.
        mid_time = mktime(dt(2017, 3, 26, 12).utctimetuple())/3600.
        lower_bound = mktime(dt(2017, 3, 26, 00).utctimetuple())/3600.
        upper_bound = mktime(dt(2017, 3, 27, 00).utctimetuple())/3600.

        # Expected data array.
        expected = np.full(self.n_data, np.nan)
        expected[0:12] = list(range(12))
        expected = np.ma.masked_invalid(expected)

        result = Plugin(24, start_hour=0).process(self.cube)
        result = result.extract(Constraint(name='air_temperature_max'))
        self.assertArrayEqual(expected, result[0].data)
        self.assertEqual(result[0].coord('time').points, [mid_time])
        self.assertEqual(result[0].coord('time').bounds[0, 0], [lower_bound])
        self.assertEqual(result[0].coord('time').bounds[0, 1], [upper_bound])

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_data_arrays_day2(self):
        """Test extraction of maxima and minima values from the time localised
        cube in the second 24 hours. All sites should return valid data during
        day 2 which runs 00 27th to 00 28th."""

        # Expected time coordinate values.
        mid_time = mktime(dt(2017, 3, 27, 12).utctimetuple())/3600.
        lower_bound = mktime(dt(2017, 3, 27, 00).utctimetuple())/3600.
        upper_bound = mktime(dt(2017, 3, 28, 00).utctimetuple())/3600.

        # Expected data array.
        expected = np.arange(0, 27)

        result = Plugin(24, start_hour=0).process(self.cube)
        result = result.extract(Constraint(name='air_temperature_max'))
        self.assertArrayEqual(expected, result[2].data)
        self.assertEqual(result[1].coord('time').points, [mid_time])
        self.assertEqual(result[1].coord('time').bounds[0, 0], [lower_bound])
        self.assertEqual(result[1].coord('time').bounds[0, 1], [upper_bound])

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_extrema_values_day1(self):
        """Test the actual values returned by the collapse method to ensure it
        is successfully extracting the maximum/minimum temperatures in the
        defined period.

        UTC times : 00  01  02  ... 10 11 12 13 14
        UTC offset:-12 -11 -10  ... -2 -1  0  1  2
        site index:  0   1   2  ... 10 11 12 13 14
        Local time: 12  13  14  ... 22 23 00 01 02

        site_index 12 at time_index 0 should be adjusted to fall at 00 27th
        April 2017 in local time, so is expected to fall outside day 1. Thus
        setting a high value for this site should leave the maximum unset for
        site 12.

        site_index 2 at time_index 9 (09Z 27 April 2017) will fall at 23 26th
        April 2017 local time, so setting this to 40 should modify the maximum
        for site 2 on day 1."""

        self.cube.data[9, 2] = 40
        self.cube.data[0, 12] = 40

        # Expected data array.
        expected = np.arange(0, 27).astype(float)
        expected[2] = 40.

        result = Plugin(24, start_hour=0).process(self.cube)
        result = result.extract(Constraint(name='air_temperature_max'))
        self.assertTrue(result[0].data[12].mask)
        self.assertArrayEqual(result[0].data, expected)


class Test_get_datetime_limits(Test_extrema):
    """Test extraction of day min and max and hour setting."""

    def test_get_datetime_limits_6(self):
        """Given an iris time coord, check that this function returns the day
        min and maxima with the provided hour appended."""

        expected_start = dt(2017, 3, 27, 6)
        expected_end = dt(2017, 3, 29, 6)
        result_start, result_end = get_datetime_limits(self.time_coord,
                                                       start_hour=6)
        self.assertEqual(expected_start, result_start)
        self.assertEqual(expected_end, result_end)

    def test_get_datetime_limits_non_int_hour(self):
        """Check an error is raised if a non-integer hour is provided."""

        msg = "integer argument expected, got float"
        with self.assertRaisesRegex(TypeError, msg):
            get_datetime_limits(self.time_coord, start_hour=6.2)


class Test_make_local_time_cube(Test_extrema):
    """Test time localisation function."""

    def test_time_coord(self):
        """Test that a new time_coord has been constructed that spans all local
        times. UTC-12 to UTC+14 is the maximum range of timezones."""

        expected_first_time = self.time_coord.points[0] - 12
        expected_last_time = self.time_coord.points[-1] + 14
        result = make_local_time_cube(self.cube)
        self.assertEqual(result.coord('time').points[0],
                         expected_first_time)
        self.assertEqual(result.coord('time').points[-1],
                         expected_last_time)
        self.assertEqual(result.coord('forecast_reference_time').points[0],
                         self.time_coord.points[0])

    def test_time_shifting(self):
        """Test that "temperature" data has been shifted to the correct local
        time accounting for the sites UTC_offset."""

        result = make_local_time_cube(self.cube)
        n_times, n_data = result.data.shape

        # Generate masked array that shifts from a single valid initial value
        # to a filled array of valid values. (Far western site initially only
        # one valid, before reaching local times for which all sites have valid
        # data).
        for i in range(n_times//2 + 12):
            values = list(range(min(n_data, i+1)))
            expected = np.full(n_data, np.nan)
            expected[0:i+1] = values
            expected = np.ma.masked_invalid(expected)
            self.assertArrayEqual(expected, result.data[i])

        # Generate masked array that shifts from being full of valid entries
        # to ever more masked values, finishing with a single valid data point.
        # (All sites have valid data at initial local time, but by the end of
        # the sequence only far eastern sites still have valid data).
        for ii, i in enumerate(range(n_times//2 + 12, n_times)):
            base = list(range(0, 27))
            values = base[ii+1:]
            expected = np.full(n_data, np.nan)
            expected[ii+1:] = values
            expected = np.ma.masked_invalid(expected)
            self.assertArrayEqual(expected, result.data[i])


if __name__ == '__main__':
    unittest.main()
