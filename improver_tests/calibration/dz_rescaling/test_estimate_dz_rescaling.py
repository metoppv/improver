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
"""Unit tests for the EstimateDzRescaling plugin."""

from typing import List

import iris
import numpy as np
import pandas as pd
import pytest
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube

from improver.calibration.dz_rescaling import EstimateDzRescaling
from improver.metadata.constants.time_types import TIME_COORDS
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.spotdata.neighbour_finding import NeighbourSelection

# Define four spots
altitude_grid = np.array([0, 50, 20, 50])
altitude_spot = [0, 20, 100, 80]
latitude = np.arange(4)
longitude = np.zeros(4)
wmo_id = ["00001", "00002", "00003", "00004"]
# Set forecast and truth data so that the two spots that are higher than the grid have slightly
# higher truth values.
forecast_data = np.array([0, 20, 10, 15])
truth_data = np.array([0, 20, 10.2, 15.1])


def _create_forecasts(
    forecast_reference_times: List[str], forecast_periods: List[float],
) -> Cube:
    """Create site forecasts for testing.

    Args:
        forecast_reference_time: Timestamp e.g. "20170101T0000Z".
        forecast_period: Forecast period in hours.

    Returns:
        Forecast cube containing four spots and specified time coordinates
    """
    perc_coord = AuxCoord(
        np.array(50, dtype=np.float32), long_name="percentile", units="%",
    )
    fp_coord = DimCoord(
        np.array(
            [fp * 3600 for fp in forecast_periods],
            dtype=TIME_COORDS["forecast_period"].dtype,
        ),
        "forecast_period",
        units=TIME_COORDS["forecast_period"].units,
    )

    validity_times = []
    for frt in forecast_reference_times:
        for fp in forecast_periods:
            validity_times.append(
                (pd.Timestamp(frt) + pd.Timedelta(hours=fp)).timestamp()
            )

    validity_times = np.reshape(
        validity_times, (len(forecast_reference_times), len(forecast_periods))
    )
    time_coord = AuxCoord(
        np.array(validity_times, dtype=TIME_COORDS["time"].dtype,),
        "time",
        units=TIME_COORDS["time"].units,
    )

    frts = [pd.Timestamp(frt).timestamp() for frt in forecast_reference_times]
    frt_coord = DimCoord(
        np.array(frts, dtype=TIME_COORDS["forecast_reference_time"].dtype,),
        "forecast_reference_time",
        units=TIME_COORDS["forecast_reference_time"].units,
    )

    data = np.reshape(forecast_data, (1, 1, len(wmo_id)))
    data = np.tile(data, (len(forecast_reference_times), len(forecast_periods), 1))

    cube = build_spotdata_cube(
        data,
        "wind_speed_at_10m",
        "m s-1",
        altitude_spot,
        latitude,
        longitude,
        wmo_id,
        scalar_coords=[perc_coord],
        additional_dims=[frt_coord, fp_coord],
    )
    cube.add_aux_coord(time_coord, data_dims=(0, 1))
    return cube


def _create_truths(
    forecast_reference_times: List[str], forecast_periods: List[float],
) -> Cube:
    """Create site truths for testing. The truth data here shows an example where the
    wind speed is slightly greater at the sites with higher altitude_grid.

    Args:
        forecast_reference_times: Timestamp e.g. "20170101T0600Z".
        forecast_periods: list of forecast period values in hours

    Returns:
        Truth cube containing four spots and specified time coordinates
    """
    validity_times = []
    for frt in forecast_reference_times:
        for fp in forecast_periods:
            validity_times.append(
                (pd.Timestamp(frt) + pd.Timedelta(hours=fp)).timestamp()
            )

    time_coord = DimCoord(
        np.array(validity_times, dtype=TIME_COORDS["time"].dtype,),
        "time",
        units=TIME_COORDS["time"].units,
    )
    data = np.reshape(truth_data, (1, len(wmo_id)))
    data = np.tile(data, (len(validity_times), 1))
    cube = build_spotdata_cube(
        data,
        "wind_speed_at_10m",
        "m s-1",
        altitude_spot,
        latitude,
        longitude,
        wmo_id,
        additional_dims=[time_coord],
    )
    return cube


def _create_neighbour_cube() -> Cube:
    """Use the NeighbourSelection functionality to create a cube containing the
    most appropriate neighbouring grid point for the four sites.

    Returns:
        Neighbour cube.
    """
    land_data = np.zeros((9, 9))
    land_data[0:2, 4] = 1
    land_data[4, 4] = 1
    orography_data = np.zeros((9, 9))
    orography_data[4:8, 4] = altitude_grid

    # Global coordinates and cubes
    projection = iris.coord_systems.GeogCS(6371229.0)
    xcoord = iris.coords.DimCoord(
        np.linspace(-4, 4, 9),
        standard_name="longitude",
        units="degrees",
        coord_system=projection,
        circular=True,
    )
    xcoord.guess_bounds()
    ycoord = iris.coords.DimCoord(
        np.linspace(-4, 4, 9),
        standard_name="latitude",
        units="degrees",
        coord_system=projection,
        circular=False,
    )
    ycoord.guess_bounds()

    global_land_mask = iris.cube.Cube(
        land_data,
        standard_name="land_binary_mask",
        units=1,
        dim_coords_and_dims=[(ycoord, 1), (xcoord, 0)],
    )
    global_orography = iris.cube.Cube(
        orography_data,
        standard_name="surface_altitude",
        units="m",
        dim_coords_and_dims=[(ycoord, 1), (xcoord, 0)],
    )
    global_sites = [
        {"altitude": alt, "latitude": lat, "longitude": lon, "wmo_id": int(site)}
        for alt, lat, lon, site in zip(altitude_spot, latitude, longitude, wmo_id)
    ]
    plugin = NeighbourSelection()
    cube = plugin.process(global_sites, global_orography, global_land_mask)
    return cube


@pytest.mark.parametrize("n_frts", [1, 2])
@pytest.mark.parametrize("wmo_id", [True, False])
@pytest.mark.parametrize(
    "forecast_periods,dz_lower_bound,dz_upper_bound,truth_adjustment,expected_data",
    [
        ([6], -200, 200, 0, [1.0, 1.0043, 1.0218, 1.0174]),
        ([6], None, None, 0, [1.0, 1.0043, 1.0218, 1.0174]),
        ([6], -75, 75, 0, [1.0, 1.0, 1.0, 1.0]),
        ([6], -200, 200, -1, [1.0, 0.9930, 0.9657, 0.9724]),
        ([6], -75, 75, -1, [1.0, 0.9747, 0.9083, 0.9083]),
        ([6], -200, 200, 1, [1.0, 1.0142, 1.0731, 1.0580]),
        ([6], -75, 75, 1, [1.0, 1.0247, 1.0958, 1.0958]),
        ([6, 12], -200, 200, 0, [1.0, 1.0043, 1.0218, 1.0174],),
        ([6, 12], -75, 75, 0, [1.0, 1.0, 1.0, 1.0],),
    ],
)
def test_estimate_dz_rescaling(
    n_frts,
    wmo_id,
    forecast_periods,
    dz_lower_bound,
    dz_upper_bound,
    truth_adjustment,
    expected_data,
):
    """Test the EstimateDzRescaling plugin with a range of options.
    n_frts controls how many forecast_reference_time_hours are included.
    wmo_ids checks that a non-standard site id coordinate can be used.
    forecast_periods is included to ensure that this can be scalar or dimensional.
    truth_adjustment is applied to the truth data to show how larger and inverted
    data (w.r.t. height) behave.
    """
    if n_frts > 1 and len(forecast_periods) > 1:
        return

    forecast_reference_times = [f"201701{day+1:02}T0000Z" for day in range(n_frts)]

    forecasts = _create_forecasts(forecast_reference_times, forecast_periods)
    truths = _create_truths(forecast_reference_times, forecast_periods)

    neighbour_cube = _create_neighbour_cube()
    truths.data = np.clip(truths.data + truth_adjustment, 0, None)

    kwargs = {}
    if not wmo_id:
        forecasts.coord("wmo_id").rename("station_id")
        truths.coord("wmo_id").rename("station_id")
        neighbour_cube.coord("wmo_id").rename("station_id")
        kwargs["site_id_coord"] = "station_id"
    plugin = EstimateDzRescaling(
        forecast_period=forecast_periods[0],
        dz_lower_bound=dz_lower_bound,
        dz_upper_bound=dz_upper_bound,
        **kwargs,
    )

    result = plugin(forecasts, truths, neighbour_cube)
    assert isinstance(result, Cube)
    assert result.name() == "scaled_vertical_displacement"
    assert result.units == "1"
    assert result.coord("forecast_reference_time_hour").points == 0
    assert result.coord("forecast_reference_time_hour").units == "seconds"
    assert result.coord("forecast_reference_time_hour").points.dtype == np.int32
    np.testing.assert_allclose(result.data, expected_data, atol=1e-4, rtol=1e-4)
