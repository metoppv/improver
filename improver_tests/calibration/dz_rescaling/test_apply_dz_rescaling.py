# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the ApplyDzRescaling plugin."""

from datetime import datetime as dt
from typing import List

import iris
import numpy as np
import pandas as pd
import pytest
from iris.coords import AuxCoord
from iris.cube import Cube

from improver.calibration.dz_rescaling import ApplyDzRescaling
from improver.constants import SECONDS_IN_HOUR
from improver.metadata.constants.time_types import DT_FORMAT, TIME_COORDS
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.synthetic_data.set_up_test_cubes import set_up_spot_percentile_cube

WMO_ID = ["00001", "00002"]


def _create_forecasts(
    forecast_reference_time: str, validity_time: str, forecast_percs: List[float]
) -> Cube:
    """Create site forecast cube for testing.

    Args:
        forecast_reference_time: Timestamp e.g. "20170101T0000Z".
        validity_time: Timestamp e.g. "20170101T0600Z".
        forecast_percs: Forecast wind speed at 10th, 50th and 90th percentile.

    Returns:
        Forecast cube containing three percentiles and two sites.
    """
    data = np.array(forecast_percs, dtype=np.float32).repeat(2).reshape(3, 2)
    percentiles = [10, 50, 90]

    cube = set_up_spot_percentile_cube(
        data,
        percentiles,
        name="wind_speed_at_10m",
        units="m s-1",
        wmo_ids=WMO_ID,
        time=dt.strptime(validity_time, DT_FORMAT),
        frt=dt.strptime(forecast_reference_time, DT_FORMAT),
    )
    return cube


def _create_scaling_factor_cube(
    frt_hour: int, forecast_period_hour: int, scaling_factor: float
) -> Cube:
    """Create a scaling factor cube containing forecast_reference_time_hours of 3 and 12 and
    forecast_period_hours of 6, 12, 18 and 24 and two sites.
    All scaling factors are 1 except at the specified [frt_hour, forecast_period_hour], where
    scaling_factor is used for the first site only.

    Returns:
        Scaling factor cube.
    """
    altitude = np.zeros(2)
    latitude = np.zeros(2)
    longitude = np.zeros(2)

    cubelist = iris.cube.CubeList()
    for ref_hour in [3, 12]:
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
                WMO_ID,
                scalar_coords=[fp_coord, frth_coord],
            )
            cubelist.append(cube)
    return cubelist.merge_cube()


@pytest.mark.parametrize("wmo_id", [True, False])
@pytest.mark.parametrize("forecast_period", [6, 18, 30])
@pytest.mark.parametrize("frt_hour", [3, 12])
@pytest.mark.parametrize("scaling_factor", [0.99, 1.01])
@pytest.mark.parametrize("forecast_period_offset", [0, -1, -5])
@pytest.mark.parametrize("frt_hour_offset", [0, 1, 4])
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
    precise point is not available except when the forecast period of the forecast
    exceeds all forecast periods within the scaling factor cube. In this case, the
    last forecast period within the scaling factor cube will be used.
    frt_hour_offset (hours) alters the forecast reference time hour within the forecast
    whilst the forecast reference time hour of the scaling factor remains the same.
    This checks that the a mismatch in the forecast reference time hour can still
    result in a match, if a leniency is specified.
    """
    forecast_reference_time = f"20170101T{(frt_hour-frt_hour_offset) % 24:02d}00Z"
    forecast = [10.0, 20.0, 30.0]
    expected_data = np.array(forecast).repeat(2).reshape(3, 2)
    expected_data[:, 0] *= scaling_factor

    validity_time = (
        pd.Timestamp(forecast_reference_time)
        + pd.Timedelta(hours=forecast_period + forecast_period_offset)
    ).strftime(DT_FORMAT)

    forecast = _create_forecasts(forecast_reference_time, validity_time, forecast)
    # Use min(fp, 24) here to ensure that the scaling cube contains
    # the scaling factor for the last forecast_period if the specified
    # forecast period is beyond the T+24 limit of the scaling cube.
    scaling_factor = _create_scaling_factor_cube(
        frt_hour, min(forecast_period, 24), scaling_factor
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


def test_use_correct_time():
    """Test the ApplyDzRescaling plugin uses the exact forecast reference time
    if it is available, rather than selecting another time within the leniency
    range.

    In this test a large leniency is used that could select the 03Z FRT, but
    the 12Z FRT should be used. The scaling factors for the two FRTs are
    different, so the data test ensures that the 12Z scaling factor has been
    used.
    """
    forecast_reference_time = "20170101T1200Z"
    forecast_period = 6
    forecast = [10.0, 20.0, 30.0]
    scaling_factor = 0.99
    expected_data = np.array(forecast).repeat(2).reshape(3, 2)
    expected_data[:, 0] *= scaling_factor

    validity_time = (
        pd.Timestamp(forecast_reference_time) + pd.Timedelta(hours=forecast_period)
    ).strftime(DT_FORMAT)

    forecast = _create_forecasts(forecast_reference_time, validity_time, forecast)
    scaling_factor = _create_scaling_factor_cube(12, forecast_period, scaling_factor)
    scaling_factor.data[0, 0, 0] = scaling_factor.data[0, 0, 0].copy() + 0.01

    kwargs = {}
    kwargs["frt_hour_leniency"] = abs(9)
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
    ).strftime(DT_FORMAT)

    forecast = _create_forecasts(forecast_reference_time, validity_time, [10, 20, 30])
    scaling_factor = _create_scaling_factor_cube(3, forecast_period, 1.0)

    with pytest.raises(ValueError, match="The mismatched sites are: {'00002'}"):
        ApplyDzRescaling()(forecast, scaling_factor[..., :1])


@pytest.mark.parametrize(
    "forecast_period,frt_hour,exception",
    [(7, 1, "forecast reference time hour equal to 1")],
)
def test_no_appropriate_scaled_dz(forecast_period, frt_hour, exception):
    """Test an exception is raised if no appropriate scaled version of the difference
    in altitude is available."""
    forecast_reference_time = f"20170101T{frt_hour:02}00Z"

    validity_time = (
        pd.Timestamp(forecast_reference_time) + pd.Timedelta(hours=forecast_period)
    ).strftime(DT_FORMAT)

    forecast = _create_forecasts(forecast_reference_time, validity_time, [10, 20, 30])
    scaling_factor = _create_scaling_factor_cube(3, forecast_period, 1.0)

    with pytest.raises(ValueError, match=exception):
        ApplyDzRescaling()(forecast, scaling_factor)
