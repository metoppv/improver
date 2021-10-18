# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""Unit tests for ModalWeatherCode class."""

import pytest
from calendar import timegm
from datetime import timedelta
from datetime import datetime as dt
from iris.cube import Cube
import numpy as np
from improver.wxcode.modal_code import ModalWeatherCode

from . import set_up_wxcube

START_TIME = dt(2020, 6, 15, 7)


@pytest.fixture(name="wxcode_series")
def wxcode_series_fixture(data) -> Cube:
    """Generate a time series of weather code cubes for combination to create
    a period representative code."""

    time = START_TIME
    ntimes = len(data)
    wxcubes = []

    for i in range(ntimes):
        wxtime = time + timedelta(hours=i)
        wxbounds = [wxtime - timedelta(hours=1), wxtime]
        wxfrt = time - timedelta(hours=6)
        wxdata = np.ones((2, 2), dtype=np.int8)
        wxdata[0, 0] = data[i]

        wxcubes.append(
            set_up_wxcube(data=wxdata, time=wxtime, time_bounds=wxbounds, frt=wxfrt)
        )

    return wxcubes


@pytest.mark.parametrize(
    "data, expected",
    (
        # Sunny day (1), one rain code (15)that is in the minority, expect sun
        # code.
        ([1, 1, 1, 15], 1),
        # Short period with an equal split. The most significant weather
        # (hail, 21) should be returned.
        ([1, 21], 21),
        # A single time is provided in which sleet is falling (18). We expect
        # the cube to be returned unchanged as it already represents the period
        # of interest.
        ([18], 18),
        # Equal split in day codes, but a night code corresponding to one
        # of the day types means a valid mode can be calculated. We expect the
        # day code (10) to be returned.
        ([1, 1, 10, 10, 9], 10),
        # No clear representative code. Falls back to grouping, which
        # consolidates the codes containing rain (10, 12, 14, 15) and yields
        # the least significant of these that is present (10).
        ([1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15], 10),
        # No clear representative code. Falls back to grouping, which
        # consolidates the codes containing rain (10, 12, 14, 15) and yields
        # the least significant of these which is present (12); the light
        # shower code (10) is not present, so will not be picked.
        ([1, 3, 4, 5, 6, 7, 8, 16, 11, 12, 14, 15], 12),
        # An extreme edge case in which all the codes across time for a site
        # are different. All the codes fall into different groups and cannot be
        # consolidated. In this case the most significant weather from the whole
        # set is returned. In this case that is a light snow shower (23).
        ([1, 3, 4, 5, 6, 7, 8, 10, 11, 17, 20, 23], 23),
    ),
)
def test_expected_values(wxcode_series, expected):
    """Test that the expected period representative symbol is returned."""
    result = ModalWeatherCode()(wxcode_series)
    assert result.data[0, 0] == expected


@pytest.mark.parametrize("data", [np.ones((12))])
def test_metadata(wxcode_series):
    """Check that the returned metadata is correct. In this case we expect a
    time coordinate with bounds that describe the full period over which the
    representative symbol has been calculated."""
    def as_utc_timestamp(time):
        return timegm(time.utctimetuple())

    result = ModalWeatherCode()(wxcode_series)
    expected_time = START_TIME + timedelta(hours=11)
    expected_bounds = [START_TIME - timedelta(hours=1),
                       START_TIME + timedelta(hours=11)]

    assert result.coord("time").points[0] == as_utc_timestamp(expected_time)
    assert result.coord("time").bounds[0][0] == as_utc_timestamp(expected_bounds[0])
    assert result.coord("time").bounds[0][1] == as_utc_timestamp(expected_bounds[1])
