# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
"""Unit tests for the ApplyDzRescaling plugin."""
from typing import List

import iris
import numpy as np
import pandas as pd
import pytest
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube

from improver.calibration.dz_rescaling import ApplyDzRescaling
from improver.constants import SECONDS_IN_HOUR
from improver.metadata.constants.time_types import TIME_COORDS
from improver.spotdata.build_spotdata_cube import build_spotdata_cube

altitude = np.zeros(2)
latitude = np.zeros(2)
longitude = np.zeros(2)
wmo_id = ["00001", "00002"]


def _create_forecasts(
    forecast_reference_time: str,
    validity_time: str,
    forecast_period: float,
    forecast_percs: List[float],
) -> Cube:
    """Create site forecast cube for testing.

    Args:
        forecast_reference_time: Timestamp e.g. "20170101T0000Z".
        validity_time: Timestamp e.g. "20170101T0600Z".
        forecast_period: Forecast period in hours.
        forecast_percs: Forecast wind speed at 10th, 50th and 90th percentile.

    Returns:
        Forecast cube containing three percentiles and two sites.
    """
    data = np.array(forecast_percs).repeat(2).reshape(3, 2)

    perc_coord = DimCoord(
        np.array([10, 50, 90], dtype=np.float32), long_name="percentile", units="%",
    )
    fp_coord = AuxCoord(
        np.array(
            forecast_period * SECONDS_IN_HOUR,
            dtype=TIME_COORDS["forecast_period"].dtype,
        ),
        "forecast_period",
        units=TIME_COORDS["forecast_period"].units,
    )
    time_coord = AuxCoord(
        np.array(
            pd.Timestamp(validity_time).timestamp(), dtype=TIME_COORDS["time"].dtype,
        ),
        "time",
        units=TIME_COORDS["time"].units,
    )
    frt_coord = AuxCoord(
        np.array(
            pd.Timestamp(forecast_reference_time).timestamp(),
            dtype=TIME_COORDS["forecast_reference_time"].dtype,
        ),
        "forecast_reference_time",
        units=TIME_COORDS["forecast_reference_time"].units,
    )

    cube = build_spotdata_cube(
        data,
        "wind_speed_at_10m",
        "m s-1",
        altitude,
        latitude,
        longitude,
        wmo_id,
        scalar_coords=[fp_coord, time_coord, frt_coord],
        additional_dims=[perc_coord],
    )
    return cube


def _create_scaling_factor_cube(
    frt_hour: int, forecast_period_hour: int, scaling_factor: float
) -> Cube:
    """Create a scaling factor cube containing forecast_reference_time_hours of 3 and 9 and
    forecast_period_hours of 6, 12, 18 and 24 and two sites.
    All scaling factors are 1 except at the specified [frt_hour, forecast_period_hour], where
    scaling_factor is used for the first site only.

    Returns:
        Scaling factor cube.
    """
    cubelist = iris.cube.CubeList()
    for ref_hour in [3, 9]:
        for forecast_period in [6, 12, 18, 24]:
            if ref_hour == frt_hour and forecast_period == forecast_period_hour:
                data = np.array((scaling_factor, 1), dtype=np.float32)
            else:
                data = np.ones(2, dtype=np.float32)
            fp_coord = AuxCoord(
                np.array(
                    forecast_period * SECONDS_IN_HOUR,
                    dtype=TIME_COORDS["forecast_period"].dtype,
                ),
                "forecast_period",
                units=TIME_COORDS["forecast_period"].units,
            )
            frth_coord = AuxCoord(
                np.array(
                    ref_hour * SECONDS_IN_HOUR,
                    dtype=TIME_COORDS["forecast_period"].dtype,
                ),
                long_name="forecast_reference_time_hour",
                units=TIME_COORDS["forecast_period"].units,
            )
            cube = build_spotdata_cube(
                data,
                "scaled_vertical_displacement",
                "1",
                altitude,
                latitude,
                longitude,
                wmo_id,
                scalar_coords=[fp_coord, frth_coord],
            )
            cubelist.append(cube)
    return cubelist.merge_cube()


@pytest.mark.parametrize("wmo_id", [True, False])
@pytest.mark.parametrize("forecast_period", [6, 18])
@pytest.mark.parametrize("frt_hour", [3, 9])
@pytest.mark.parametrize("scaling_factor", [0.99, 1.01])
@pytest.mark.parametrize("forecast_period_offset", [0, -1, -5])
@pytest.mark.parametrize("frt_hour_offset", [0, 1, 2])
def test_apply_dz_rescaling(
    wmo_id,
    forecast_period,
    frt_hour,
    forecast_period_offset,
    scaling_factor,
    frt_hour_offset,
):
    """Test the ApplyDzRescaling plugin.
    wmo_id checks that the plugin site_id_coord behaves correctly.
    forecast_period and frt_hour (hours) control which element of scaling_factor cube
    contains the scaling_factor value.
    forecast_period_offset (hours) adjusts the forecast period coord on the forecast
    cube to ensure the plugin always snaps to the next largest forecast_time when the
    precise point is not available.
    frt_hour_offset (hours) alters the forecast reference time hour within the forecast
    whilst the forececast reference time hour of the scaling factor remains the same.
    This checks that the a mismatch in the forecast reference time hour can still
    result in a match, if a leniency is specified.
    """
    forecast_reference_time = f"20170101T{frt_hour-frt_hour_offset:02d}00Z"
    forecast = [10.0, 20.0, 30.0]
    expected_data = np.array(forecast).repeat(2).reshape(3, 2)
    expected_data[:, 0] *= scaling_factor

    validity_time = (
        pd.Timestamp(forecast_reference_time)
        + pd.Timedelta(hours=forecast_period + forecast_period_offset)
    ).strftime("%Y%m%dT%H%MZ")

    forecast = _create_forecasts(
        forecast_reference_time,
        validity_time,
        forecast_period + forecast_period_offset,
        forecast,
    )
    scaling_factor = _create_scaling_factor_cube(
        frt_hour, forecast_period, scaling_factor
    )

    kwargs = {}
    if not wmo_id:
        forecast.coord("wmo_id").rename("station_id")
        scaling_factor.coord("wmo_id").rename("station_id")
        kwargs["site_id_coord"] = "station_id"

    kwargs["frt_hour_leniency"] = abs(frt_hour_offset)
    plugin = ApplyDzRescaling(**kwargs)

    result = plugin(forecast, scaling_factor)
    assert isinstance(result, Cube)

    np.testing.assert_allclose(result.data, expected_data, atol=1e-4, rtol=1e-4)


def test_mismatching_sites():
    """Test an exception is raised if the sites mismatch."""
    forecast_period = 6
    forecast_reference_time = "20170101T0300Z"

    validity_time = (
        pd.Timestamp(forecast_reference_time) + pd.Timedelta(hours=forecast_period)
    ).strftime("%Y%m%dT%H%MZ")

    forecast = _create_forecasts(
        forecast_reference_time, validity_time, forecast_period, [10, 20, 30]
    )
    scaling_factor = _create_scaling_factor_cube(3, forecast_period, 1.0)

    with pytest.raises(ValueError, match="The mismatched sites are: {'00002'}"):
        ApplyDzRescaling()(forecast, scaling_factor[..., :1])


@pytest.mark.parametrize(
    "forecast_period,frt_hour,exception",
    [
        (25, 3, "forecast period greater than or equal to 25"),
        (7, 1, "forecast reference time hour equal to 1"),
    ],
)
def test_no_appropriate_scaled_dz(forecast_period, frt_hour, exception):
    """Test an exception is raised if no appropriate scaled version of the difference
    in altitude is available."""
    forecast_reference_time = f"20170101T{frt_hour:02}00Z"

    validity_time = (
        pd.Timestamp(forecast_reference_time) + pd.Timedelta(hours=forecast_period)
    ).strftime("%Y%m%dT%H%MZ")

    forecast = _create_forecasts(
        forecast_reference_time, validity_time, forecast_period, [10, 20, 30]
    )
    scaling_factor = _create_scaling_factor_cube(3, forecast_period, 1.0)

    with pytest.raises(ValueError, match=exception):
        ApplyDzRescaling()(forecast, scaling_factor)
