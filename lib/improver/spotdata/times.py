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

"""
Plugins written for the Improver site specific process chain.

"""

from datetime import datetime as dt
from datetime import time
from datetime import timedelta


def get_forecast_times(forecast_date=None, forecast_time=None,
                       forecast_length=None):
    '''
    Generate a list of python datetime objects specifying the desired forecast
    times. This list will be created from input specifications if provided.
    Otherwise defaults are to start today at the most recent 6-hourly interval
    (00, 06, 12, 18) and to run out to T+144 hours.

    '''
    if forecast_date is not None:
        start_date = dt.strptime(forecast_date, "%Y%m%d").date()
    else:
        start_date = dt.utcnow().date()

    if forecast_time is not None:
        forecast_start_time = dt.combine(start_date, time(forecast_time))
    else:
        # If no start hour provided, go back to the nearest multiple of 6
        # hours (e.g. utcnow = 11Z --> 06Z).
        forecast_start_time = dt.combine(
            start_date, time(divmod(dt.utcnow().hour, 6)[0]*6))

    if forecast_length is None:
        forecast_length = 144

    # Generate forecast times. Hourly to T+48, 3 hourly to T+144.
    forecast_times = [forecast_start_time + timedelta(hours=x) for x in
                      range(min(forecast_length, 49))]
    forecast_times = (forecast_times +
                      [forecast_start_time + timedelta(hours=x) for x in
                       range(51, min(forecast_length, 144), 3)])

    return forecast_times
