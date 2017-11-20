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
"""Plugins written for the Improver site specific process chain."""
import re

from datetime import datetime as dt
from datetime import time
from datetime import timedelta


def get_forecast_times(forecast_length, forecast_date=None,
                       forecast_time=None):
    """
    Generate a list of python datetime objects specifying the desired forecast
    times. This list will be created from input specifications if provided.
    Otherwise defaults are to start today at the most recent 6-hourly interval
    (00, 06, 12, 18) and to run out to T+144 hours.

    Args:
        forecast_length (int):
            An integer giving the desired length of the forecast output in
            hours (e.g. 48 for a two day forecast period).

        forecast_date (string (YYYYMMDD)):
            A string of format YYYYMMDD defining the start date for which
            forecasts are required. If unset it defaults to today in UTC.

        forecast_time (int):
            An integer giving the hour on the forecast_date at which to start
            the forecast output; 24hr clock such that 17 = 17Z for example. If
            unset it defaults to the latest 6 hour cycle as a start time.

    Returns:
        forecast_times (list of datetime.datetime objects):
            A list of python datetime.datetime objects that represent the
            times at which diagnostic data should be extracted.

    Raises:
        ValueError : raised if the input date is not in the expected format.

    """
    date_format = re.compile('[0-9]{8}')

    if forecast_date is None:
        start_date = dt.utcnow().date()
    else:
        if date_format.match(forecast_date) and len(forecast_date) == 8:
            start_date = dt.strptime(forecast_date, "%Y%m%d").date()
        else:
            raise ValueError('Date {} is in unexpected format; should be '
                             'YYYYMMDD.'.format(forecast_date))

    if forecast_time is None:
        # If no start hour provided, go back to the nearest multiple of 6
        # hours (e.g. utcnow = 11Z --> 06Z).
        forecast_start_time = dt.combine(
            start_date, time(divmod(dt.utcnow().hour, 6)[0]*6))
    else:
        forecast_start_time = dt.combine(start_date, time(forecast_time))

    # Generate forecast times. Hourly to T+48, 3 hourly to T+forecast_length.
    forecast_times = [forecast_start_time + timedelta(hours=x) for x in
                      range(min(forecast_length, 49))]
    forecast_times = (forecast_times +
                      [forecast_start_time + timedelta(hours=x) for x in
                       range(51, forecast_length+1, 3)])

    return forecast_times
