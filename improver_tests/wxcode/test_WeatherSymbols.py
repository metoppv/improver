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
"""Unit tests for WeatherSymbols class."""

from datetime import datetime as dt
from datetime import timedelta
from typing import List

import iris
import numpy as np
import pytest
from cf_units import Unit
from iris.coords import AuxCoord
from iris.cube import CubeList
from iris.tests import IrisTest

from improver.metadata.probabilistic import (
    find_threshold_coordinate,
    get_threshold_coord_name_from_probability_name,
)
from improver.synthetic_data.set_up_test_cubes import set_up_probability_cube
from improver.wxcode.utilities import WX_DICT
from improver.wxcode.weather_symbols import WeatherSymbols

from improver_tests.wxcode import wxcode_decision_tree


@pytest.fixture(name="wxcode_inputs")
def wxcode_inputs_fixture(
    cloud: List[float],
    hail_accum: List[float],
    hail_rate: List[float],
    lightning: List[float],
    low_cloud: List[float],
    rain: List[float],
    rain_vic: List[float],
    shower_condition: List[float],
    sleet: List[float],
    sleet_vic: List[float],
    snow: List[float],
    snow_vic: List[float],
    vis: List[float],
    hour: int,
):
    """
    Set up cubes and constraints required for Weather Symbols. Each cube has one spatial point.
    """

    time = dt(2017, 10, 10, hour, 0)
    time_bounds = [time - timedelta(hours=1), time]
    frt = dt(2017, 10, 9, 18, 0)
    standard_kwargs = {"time": time, "frt": frt}
    time_window_kwargs = standard_kwargs.copy()
    time_window_kwargs.update({"time_bounds": time_bounds})

    precip_kwargs = time_window_kwargs.copy()
    precip_kwargs.update(
        {
            "thresholds": np.array([0.03e-03, 0.1e-03, 1.0e-03], dtype=np.float32),
            "threshold_units": "m",
        }
    )

    snowfall_cube = set_up_probability_cube(
        np.array(snow).reshape(3, 1, 1).astype(np.float32),
        variable_name="lwe_thickness_of_snowfall_amount",
        **precip_kwargs,
    )

    snowfall_vicinity_cube = set_up_probability_cube(
        np.array(snow_vic).reshape(3, 1, 1).astype(np.float32),
        variable_name="lwe_thickness_of_snowfall_amount_in_vicinity",
        **precip_kwargs,
    )

    sleetfall_cube = set_up_probability_cube(
        np.array(sleet).reshape(3, 1, 1).astype(np.float32),
        variable_name="lwe_thickness_of_sleetfall_amount",
        **precip_kwargs,
    )

    sleetfall_vicinity_cube = set_up_probability_cube(
        np.array(sleet_vic).reshape(3, 1, 1).astype(np.float32),
        variable_name="lwe_thickness_of_sleetfall_amount_in_vicinity",
        **precip_kwargs,
    )

    rainfall_cube = set_up_probability_cube(
        np.array(rain).reshape(3, 1, 1).astype(np.float32),
        variable_name="thickness_of_rainfall_amount",
        **precip_kwargs,
    )

    rainfall_vicinity_cube = set_up_probability_cube(
        np.array(rain_vic).reshape(3, 1, 1).astype(np.float32),
        variable_name="thickness_of_rainfall_amount_in_vicinity",
        **precip_kwargs,
    )

    precip_data = (
        np.maximum.reduce([snow, sleet, rain]).reshape((3, 1, 1)).astype(np.float32)
    )

    precip_cube = set_up_probability_cube(
        precip_data,
        variable_name="lwe_thickness_of_precipitation_amount",
        **precip_kwargs,
    )

    precip_data = (
        np.maximum.reduce([snow_vic, sleet_vic, rain_vic])
        .reshape((3, 1, 1))
        .astype(np.float32)
    )

    precip_vicinity_cube = set_up_probability_cube(
        precip_data,
        variable_name="lwe_thickness_of_precipitation_amount_in_vicinity",
        **precip_kwargs,
    )

    hail_rate_cube = set_up_probability_cube(
        np.array(hail_rate).reshape(1, 1, 1).astype(np.float32),
        thresholds=np.array([2.777777e-07], dtype=np.float32),
        variable_name="lwe_graupel_and_hail_fall_rate_in_vicinity",
        threshold_units="m s-1",
        **time_window_kwargs,
    )

    hail_accum_cube = set_up_probability_cube(
        np.array(hail_accum).reshape(1, 1, 1).astype(np.float32),
        thresholds=np.array([0.1e-03], dtype=np.float32),
        variable_name="lwe_thickness_of_graupel_and_hail_fall_amount",
        threshold_units="m",
        **time_window_kwargs,
    )

    cloud_cube = set_up_probability_cube(
        np.array(cloud).reshape(2, 1, 1).astype(np.float32),
        thresholds=np.array([0.1875, 0.8125], dtype=np.float32),
        variable_name="low_and_medium_type_cloud_area_fraction",
        threshold_units="1",
        **standard_kwargs,
    )

    cloud_low_cube = set_up_probability_cube(
        np.array(low_cloud).reshape(1, 1, 1).astype(np.float32),
        thresholds=np.array([0.85], dtype=np.float32),
        variable_name="low_type_cloud_area_fraction",
        threshold_units="1",
        **standard_kwargs,
    )

    visibility_cube = set_up_probability_cube(
        np.array(vis).reshape(2, 1, 1).astype(np.float32),
        thresholds=np.array([1000.0, 5000.0], dtype=np.float32),
        variable_name="visibility_in_air",
        threshold_units="m",
        spp__relative_to_threshold="below",
        **standard_kwargs,
    )

    lightning_cube = set_up_probability_cube(
        np.array(lightning).reshape(1, 1, 1).astype(np.float32),
        thresholds=np.array([0.0], dtype=np.float32),
        variable_name="number_of_lightning_flashes_per_unit_area_in_vicinity",
        threshold_units="m-2",
        **time_window_kwargs,
    )

    shower_condition_cube = set_up_probability_cube(
        np.array(shower_condition).reshape(1, 1, 1).astype(np.float32),
        thresholds=np.array([1.0], dtype=np.float32),
        variable_name="shower_condition",
        threshold_units="1",
        **standard_kwargs,
    )

    cubes = CubeList(
        [
            snowfall_cube,
            snowfall_vicinity_cube,
            sleetfall_cube,
            sleetfall_vicinity_cube,
            rainfall_cube,
            rainfall_vicinity_cube,
            precip_cube,
            precip_vicinity_cube,
            cloud_cube,
            cloud_low_cube,
            visibility_cube,
            shower_condition_cube,
            lightning_cube,
            hail_rate_cube,
            hail_accum_cube,
        ]
    )
    return cubes


def run_wxcode_test(
    expected: str, wxcode_inputs: CubeList, day_night: str = "Day"
) -> None:
    """Runs the WeatherSymbols plugin with the supplied inputs and asserts that the resulting
    weather code matches the expected symbol

    Args:
        expected:
            Weather symbol meaning that we expect the WeatherSymbols plugin to produce
        wxcode_inputs:
            All the input cubes to give to the plugin
        day_night:
            Fills {day_night} in expected string (Also changes Sunny to Clear)
    """
    result = WeatherSymbols(wxtree=wxcode_decision_tree(), target_period=3600)(
        wxcode_inputs
    )
    assert WX_DICT[int(result.data)] == expected.format(day_night=day_night).replace(
        "Sunny_Night", "Clear_Night"
    )

    assert isinstance(result, iris.cube.Cube)
    assert all(result.attributes["weather_code"] == list(WX_DICT.keys()))
    assert result.attributes["weather_code_meaning"] == " ".join(WX_DICT.values())
    assert result.dtype == np.int32


@pytest.mark.parametrize("hour, day_night", ((12, "Day"), (0, "Night")))
@pytest.mark.parametrize(
    "hail_accum, hail_rate, lightning, rain, rain_vic, shower_condition, sleet, sleet_vic, snow, snow_vic",
    ((0, 0, 0, [0, 0, 0], [0, 0, 0], 0, [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],),),
)
@pytest.mark.parametrize(
    "expected, cloud, low_cloud, vis",
    (
        ("Sunny_{day_night}", [0, 0], 0, [0, 0]),
        ("Partly_Cloudy_{day_night}", [1, 0], 0, [0, 0]),
        ("Cloudy", [1, 1], 0, [0, 0]),
        ("Overcast", [1, 1], 1, [0, 0]),
        ("Mist", [1, 1], 1, [0, 1]),
        ("Fog", [1, 1], 1, [1, 1]),
    ),
)
def test_dry_routes(wxcode_inputs, day_night, expected):
    """Tests that each route to a non-precipitating symbol can be traversed"""
    run_wxcode_test(expected, wxcode_inputs, day_night=day_night)


@pytest.mark.parametrize("hour, day_night", ((12, "Day"), (0, "Night")))
@pytest.mark.parametrize(
    "cloud, low_cloud, rain_vic, sleet_vic, snow_vic, vis",
    (([0, 0], 0, [1, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0],),),
)
@pytest.mark.parametrize(
    "expected, hail_accum, hail_rate, lightning, rain, shower_condition, sleet, snow",
    (
        ("Light_Rain", 0, 0, 0, [1, 1, 0], 0, [0, 0, 0], [0, 0, 0]),
        ("Hail", 1, 1, 0, [0, 0, 0], 0, [0, 0, 0], [0, 0, 0]),
        ("Light_Rain", 0.6, 0.6, 0, [0.7, 0.7, 0], 0, [0, 0, 0], [0, 0, 0]),
        ("Sleet", 0.6, 0.6, 0, [0, 0, 0], 0, [0.7, 0.7, 0], [0, 0, 0]),
        ("Light_Snow", 0.6, 0.6, 0, [0, 0, 0], 0, [0, 0, 0], [0.7, 0.7, 0]),
        ("Hail_Shower_{day_night}", 1, 1, 0, [0, 0, 0], 1, [0, 0, 0], [0, 0, 0]),
        ("Thunder", 0, 0, 1, [0, 0, 0], 0, [0, 0, 0], [0, 0, 0]),
        ("Thunder_Shower_{day_night}", 0, 0, 1, [0, 0, 0], 1, [0, 0, 0], [0, 0, 0]),
    ),
)
def test_lightning_and_hail_routes(wxcode_inputs, day_night, expected):
    """Tests that each route through the lightning and hail tree can be traversed.
    Note that the background state is light-precip-in-vicinity and that there are three non-hail
    tests where hail is present but not dominant."""
    run_wxcode_test(expected, wxcode_inputs, day_night=day_night)


@pytest.mark.parametrize("hour", (12, 0))
@pytest.mark.parametrize(
    "cloud, hail_accum, hail_rate, lightning, rain_vic, shower_condition, sleet_vic, snow_vic",
    (([1, 1], 0, 0, 0, [0, 0, 0], 0, [0, 0, 0], [0, 0, 0],),),
)
@pytest.mark.parametrize(
    "expected, low_cloud, rain, sleet, snow, vis",
    (
        ("Cloudy", 0, [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0]),
        ("Drizzle", 0, [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1]),
        ("Drizzle", 1, [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0]),
        ("Overcast", 1, [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0]),
        ("Overcast", 1, [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0]),
    ),
)
def test_drizzle_routes(wxcode_inputs, expected):
    """Tests that each route to a drizzle symbol can be traversed. Background condition is Cloudy.
    The Overcast tests are for very light precip that is not rain."""
    run_wxcode_test(expected, wxcode_inputs)


@pytest.mark.parametrize("hour, day_night", ((12, "Day"), (0, "Night")))
@pytest.mark.parametrize(
    "cloud, hail_accum, hail_rate, lightning, low_cloud, rain, rain_vic, sleet, sleet_vic, vis",
    (([0, 0], 0, 0, 0, 0, [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0],),),
)
@pytest.mark.parametrize(
    "expected, shower_condition, snow, snow_vic",
    (
        ("Sunny_{day_night}", 0, [0, 0, 0], [0, 0, 0],),
        ("Heavy_Snow", 0, [1, 1, 1], [1, 1, 1],),
        ("Heavy_Snow_Shower_{day_night}", 1, [1, 1, 1], [1, 1, 1],),
        ("Heavy_Snow", 0, [1, 1, 0], [1, 1, 1],),
        ("Light_Snow", 0, [1, 1, 0], [1, 1, 0],),
        ("Heavy_Snow_Shower_{day_night}", 1, [1, 1, 0], [1, 1, 1],),
        ("Light_Snow_Shower_{day_night}", 1, [1, 1, 0], [1, 1, 0],),
    ),
)
def test_snow_routes(wxcode_inputs, day_night, expected):
    """Tests that each route to a snow symbol can be traversed."""
    run_wxcode_test(expected, wxcode_inputs, day_night=day_night)


@pytest.mark.parametrize("hour, day_night", ((12, "Day"), (0, "Night")))
@pytest.mark.parametrize(
    "cloud, hail_accum, hail_rate, lightning, low_cloud, sleet, sleet_vic, snow, snow_vic, vis",
    (([0, 0], 0, 0, 0, 0, [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0],),),
)
@pytest.mark.parametrize(
    "expected, shower_condition, rain, rain_vic",
    (
        ("Sunny_{day_night}", 0, [0, 0, 0], [0, 0, 0],),
        ("Heavy_Rain", 0, [1, 1, 1], [1, 1, 1],),
        ("Heavy_Shower_{day_night}", 1, [1, 1, 1], [1, 1, 1],),
        ("Heavy_Rain", 0, [1, 1, 0], [1, 1, 1],),
        ("Light_Rain", 0, [1, 1, 0], [1, 1, 0],),
        ("Heavy_Shower_{day_night}", 1, [1, 1, 0], [1, 1, 1],),
        ("Light_Shower_{day_night}", 1, [1, 1, 0], [1, 1, 0],),
    ),
)
def test_rain_routes(wxcode_inputs, day_night, expected):
    """Tests that each route to a rain symbol can be traversed."""
    run_wxcode_test(expected, wxcode_inputs, day_night=day_night)


@pytest.mark.parametrize("hour, day_night", ((12, "Day"), (0, "Night")))
@pytest.mark.parametrize(
    "cloud, hail_accum, hail_rate, lightning, low_cloud, rain, rain_vic, snow, snow_vic, vis",
    (([0, 0], 0, 0, 0, 0, [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0],),),
)
@pytest.mark.parametrize(
    "expected, shower_condition, sleet, sleet_vic",
    (
        ("Sunny_{day_night}", 0, [0, 0, 0], [0, 0, 0],),
        ("Sleet_Shower_{day_night}", 1, [1, 1, 1], [1, 1, 1],),
        ("Sleet", 0, [1, 1, 0], [1, 1, 0],),
        ("Sleet_Shower_{day_night}", 1, [1, 1, 0], [1, 1, 0],),
    ),
)
def test_sleet_routes(wxcode_inputs, day_night, expected):
    """Tests that each route to a sleet symbol can be traversed."""
    run_wxcode_test(expected, wxcode_inputs, day_night=day_night)
