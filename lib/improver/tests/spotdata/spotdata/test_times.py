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
"""Unit tests for spotdata.times."""


import unittest
from iris.tests import IrisTest
from datetime import datetime as dt
from datetime import timedelta
from datetime import time

from improver.spotdata.times import get_forecast_times as Function


class Test_get_forecast_times(IrisTest):

    """Test the generation of forecast time using the function."""

    def test_all_data_provided(self):
        """Test setting up a forecast range when start date, start hour and
        forecast length are all provided."""

        forecast_start = dt(2017, 6, 1, 9, 0)
        forecast_date = forecast_start.strftime("%Y%m%d")
        forecast_time = int(forecast_start.strftime("%H"))
        forecast_length = 300
        forecast_end = forecast_start + timedelta(hours=forecast_length)
        result = Function(forecast_date, forecast_time, forecast_length)
        self.assertEqual(forecast_start, result[0])
        self.assertEqual(forecast_end, result[-1])
        self.assertEqual(timedelta(hours=1), result[1] - result[0])
        self.assertEqual(timedelta(hours=3), result[-1] - result[-2])

    def test_no_data_provided(self):
        """Test setting up a forecast range when no data is provided. Expect a
        range of times starting from last hour before now that was an interval
        of 6 hours, on today's date and going out to T+144.

        Note: this could fail if time between forecast_start being set and
        reaching the function call bridges a 6-hour time (00, 06, 12, 18). As
        such it is allowed two goes before reporting a failure (slightly
        unconventional I'm afraid)."""

        second_chance = 0
        while second_chance < 2:
            forecast_start = dt.utcnow()
            expected_date = forecast_start.date()
            expected_hour = time(divmod(forecast_start.hour, 6)[0]*6)
            forecast_date = None
            forecast_time = None
            forecast_length = None
            result = Function(forecast_date, forecast_time, forecast_length)

            check1 = (expected_date == result[0].date())
            check2 = (expected_hour.hour == result[0].hour)
            check3 = (timedelta(hours=144) == (result[-1] - result[0]))

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

        forecast_start = dt(2017, 6, 1, 15, 0)
        forecast_date = None
        forecast_time = int(forecast_start.strftime("%H"))
        forecast_length = 144
        expected_date = dt.utcnow().date()
        expected_start = dt.combine(expected_date, time(forecast_time))
        expected_end = expected_start + timedelta(hours=144)
        result = Function(forecast_date, forecast_time, forecast_length)

        self.assertEqual(expected_start, result[0])
        self.assertEqual(expected_end, result[-1])
        self.assertEqual(timedelta(hours=1), result[1] - result[0])
        self.assertEqual(timedelta(hours=3), result[-1] - result[-2])

    def test_invalid_date_format(self):
        """Test error is raised when a date is provided in an unexpected
        format."""

        forecast_date = '17MARCH2017'
        msg = 'Date .* is in unexpected format'
        with self.assertRaisesRegexp(ValueError, msg):
            Function(forecast_date, 6, 144)


if __name__ == '__main__':
    unittest.main()
