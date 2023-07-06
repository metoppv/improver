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

import iris
import numpy as np
import pandas as pd
import pytest
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube

from improver.calibration.dz_rescaling import ApplyDzRescaling
from improver.metadata.constants.time_types import TIME_COORDS
from improver.spotdata.build_spotdata_cube import build_spotdata_cube


def _create_forecasts(
    forecast_reference_time: str, validity_time: str, forecast_period: float
) -> Cube:
    """Create site forecasts for testing.

    Args:
        forecast_reference_time: Timestamp e.g. "20170101T0000Z".
        validity_time: Timestamp e.g. "20170101T0600Z".
        forecast_period: Forecast period in hours.

    Returns:
        Forecast cube.
    """
    data = np.array([[0, 15, 5, 10], [0, 20, 10, 15], [5, 25, 15, 20]])
    altitude = np.array([0, 100, 20, 50])
    latitude = np.array([0, 1, 2, 3])
    longitude = np.array([0, 1, 2, 3])
    wmo_id = ["00001", "00002", "00003", "00004"]

    perc_coord = DimCoord(
        np.array([10, 50, 90], dtype=np.float32), long_name="percentile", units="%",
    )
    fp_coord = AuxCoord(
        np.array(forecast_period * 3600, dtype=TIME_COORDS["forecast_period"].dtype,),
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


def _create_scaling_factor_cube():
    """Create a scaling factor cube.

    Returns:
        Scaling factor cube.
    """
    cubelist = iris.cube.CubeList()
    for frt_hour, multiplier in zip([3, 9], [0.1, 0.15]):
        for fp_index, forecast_period in enumerate([6, 12, 18, 24]):
            data = np.array([1, 1.1, 1, 1.05]) + fp_index * multiplier
            altitude = np.array([0, 100, 20, 50])
            latitude = np.array([0, 1, 2, 3])
            longitude = np.array([0, 1, 2, 3])
            wmo_id = ["00001", "00002", "00003", "00004"]
            fp_coord = AuxCoord(
                np.array(
                    forecast_period * 3600, dtype=TIME_COORDS["forecast_period"].dtype,
                ),
                "forecast_period",
                units=TIME_COORDS["forecast_period"].units,
            )
            frth_coord = AuxCoord(
                np.array(frt_hour * 3600, dtype=np.int32,),
                long_name="forecast_reference_time_hour",
                units="seconds",
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
@pytest.mark.parametrize(
    "forecast_period,expected_data",
    [
        (5, [[0.0, 16.5, 5, 10.5], [0, 22, 10, 15.75], [5, 27.5, 15, 21]]),
        (7, [[0.0, 18, 5.5, 11.5], [0, 24, 11, 17.25], [5.5, 30, 16.5, 23]]),
        (11, [[0.0, 18, 5.5, 11.5], [0, 24, 11, 17.25], [5.5, 30, 16.5, 23]]),
        (12, [[0.0, 18, 5.5, 11.5], [0, 24, 11, 17.25], [5.5, 30, 16.5, 23]]),
    ],
)
def test_apply_dz_rescaling(wmo_id, forecast_period, expected_data):
    """Test the ApplyDzRescaling plugin."""
    forecast_reference_time = "20170101T0300Z"

    validity_time = (
        pd.Timestamp(forecast_reference_time) + pd.Timedelta(hours=forecast_period)
    ).strftime("%Y%m%dT%H%MZ")

    forecast = _create_forecasts(
        forecast_reference_time, validity_time, forecast_period
    )
    scaling_factor = _create_scaling_factor_cube()

    if wmo_id:
        plugin = ApplyDzRescaling()
    else:
        forecast.coord("wmo_id").rename("station_id")
        scaling_factor.coord("wmo_id").rename("station_id")
        plugin = ApplyDzRescaling(site_id_coord="station_id")

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
        forecast_reference_time, validity_time, forecast_period
    )
    scaling_factor = _create_scaling_factor_cube()

    with pytest.raises(ValueError, match="The mismatched sites are: {'00004'}"):
        ApplyDzRescaling()(forecast, scaling_factor[..., :3])


@pytest.mark.parametrize(
    "forecast_period,frt_hour,exception",
    [
        (100, 3, "forecast period greater than or equal to 100"),
        (7, 2, "forecast reference time hour equal to 2"),
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
        forecast_reference_time, validity_time, forecast_period
    )
    scaling_factor = _create_scaling_factor_cube()

    with pytest.raises(ValueError, match=exception):
        ApplyDzRescaling()(forecast, scaling_factor)
