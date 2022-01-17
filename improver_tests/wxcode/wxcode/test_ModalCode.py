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

from calendar import timegm
from datetime import datetime as dt
from datetime import timedelta
from typing import Tuple

import numpy as np
import pytest
from iris.cube import CubeList

from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.synthetic_data.set_up_test_cubes import construct_scalar_time_coords
from improver.wxcode.modal_code import ModalWeatherCode

from . import set_up_wxcube

MODEL_ID_ATTR = "mosg__model_configuration"
TARGET_TIME = dt(2020, 6, 15, 18)


@pytest.fixture(name="wxcode_series")
def wxcode_series_fixture(
    data, cube_type, offset_reference_times: bool, model_id_attr: bool,
) -> Tuple[bool, CubeList]:
    """Generate a time series of weather code cubes for combination to create
    a period representative code. When offset_reference_times is set, each
    successive cube will have a reference time one hour older."""

    time = TARGET_TIME

    ntimes = len(data)
    wxcubes = CubeList()

    for i in range(ntimes):
        wxtime = time - timedelta(hours=i)
        wxbounds = [wxtime - timedelta(hours=1), wxtime]
        if offset_reference_times:
            wxfrt = time - timedelta(hours=18) - timedelta(hours=i)
        else:
            wxfrt = time - timedelta(hours=18)
        wxdata = np.ones((2, 2), dtype=np.int8)
        wxdata[0, 0] = data[i]

        if cube_type == "gridded":
            wxcubes.append(
                set_up_wxcube(data=wxdata, time=wxtime, time_bounds=wxbounds, frt=wxfrt)
            )
        else:
            time_coords = construct_scalar_time_coords(wxtime, wxbounds, wxfrt)
            time_coords = [crd for crd, _ in time_coords]
            latitudes = np.array([50, 52, 54, 56])
            longitudes = np.array([-4, -2, 0, 2])
            altitudes = wmo_ids = unique_site_id = np.arange(4)
            unique_site_id_key = "met_office_site_id"
            wxcubes.append(
                build_spotdata_cube(
                    wxdata.flatten(),
                    "weather_code",
                    1,
                    altitudes,
                    latitudes,
                    longitudes,
                    wmo_ids,
                    unique_site_id=unique_site_id,
                    unique_site_id_key=unique_site_id_key,
                    scalar_coords=time_coords,
                )
            )
        if model_id_attr:
            [c.attributes.update({MODEL_ID_ATTR: "uk_ens"}) for c in wxcubes]
            wxcubes[0].attributes.update({MODEL_ID_ATTR: "uk_det uk_ens"})
    return model_id_attr, wxcubes


@pytest.mark.parametrize("model_id_attr", [False, True])
@pytest.mark.parametrize("offset_reference_times", [False, True])
@pytest.mark.parametrize("cube_type", ["gridded", "spot"])
@pytest.mark.parametrize(
    "data, expected",
    (
        # Sunny day (1), one rain code (15) that is in the minority, expect sun
        # code (1).
        ([1, 1, 1, 15], 1),
        # Short period with an equal split. The most significant weather
        # (hail, 21) should be returned.
        ([1, 21], 21),
        # A single time is provided in which a sleet shower is forecast (16).
        # We expect the cube to be returned with the night code changed to a
        # day code (17).
        ([16], 17),
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
    _, wxcode_cubes = wxcode_series
    result = ModalWeatherCode()(wxcode_cubes)
    assert result.data.flatten()[0] == expected


@pytest.mark.parametrize("model_id_attr", [False, True])
@pytest.mark.parametrize("offset_reference_times", [False, True])
@pytest.mark.parametrize("cube_type", ["gridded", "spot"])
@pytest.mark.parametrize("data", [np.ones(12), np.ones(1)])
def test_metadata(wxcode_series):
    """Check that the returned metadata is correct. In this case we expect a
    time coordinate with bounds that describe the full period over which the
    representative symbol has been calculated while the forecast_reference_time
    will be the latest of those input and the forecast_period will be the
    difference between the forecast_reference_time and time.

    A single data point is tested which means a single cube is passed in. This
    ensures the metadata is consistent whether or not the input data has passed
    through the modal aggregator."""

    def as_utc_timestamp(time):
        return timegm(time.utctimetuple())

    model_id_attr, wxcode_cubes = wxcode_series

    if model_id_attr:
        kwargs = {"model_id_attr": MODEL_ID_ATTR}
    else:
        kwargs = {}

    result = ModalWeatherCode(**kwargs)(wxcode_cubes)

    n_times = len(wxcode_cubes)
    expected_time = TARGET_TIME
    expected_bounds = [TARGET_TIME - timedelta(hours=n_times), TARGET_TIME]
    expected_reference_time = TARGET_TIME - timedelta(hours=18)
    expected_forecast_period = (expected_time - expected_reference_time).total_seconds()
    expected_forecast_period_bounds = [
        expected_forecast_period - n_times * 3600,
        expected_forecast_period,
    ]
    expected_cell_method = ["mode", "time"]

    assert result.coord("time").points[0] == as_utc_timestamp(expected_time)
    assert result.coord("time").bounds[0][0] == as_utc_timestamp(expected_bounds[0])
    assert result.coord("time").bounds[0][1] == as_utc_timestamp(expected_bounds[1])
    assert result.coord("forecast_reference_time").points[0] == as_utc_timestamp(
        expected_reference_time
    )
    assert not result.coord("forecast_reference_time").has_bounds()
    assert result.coord("forecast_period").points[0] == expected_forecast_period
    assert np.allclose(
        result.coord("forecast_period").bounds[0], expected_forecast_period_bounds
    )
    assert result.cell_methods[0].method == expected_cell_method[0]
    assert result.cell_methods[0].coord_names[0] == expected_cell_method[1]
    if model_id_attr:
        assert result.attributes[MODEL_ID_ATTR] == "uk_det uk_ens"
    else:
        assert MODEL_ID_ATTR not in result.attributes.keys()
