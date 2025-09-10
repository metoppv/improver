# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the various utility functions within samos_calibration.py."""

from datetime import datetime
from typing import Dict, Optional

import cf_units
import numpy as np
import pandas as pd
import pytest
from iris.cube import Cube
from pandas.testing import assert_frame_equal

from improver.calibration.samos_calibration import (
    convert_dataframe_to_cube,
    prepare_data_for_gam,
)
from improver.synthetic_data.set_up_test_cubes import (
    set_up_spot_variable_cube,
    set_up_variable_cube,
)
from improver_tests.calibration.samos_calibration.helper_functions import (
    create_simple_cube,
)


@pytest.fixture
def gridded_dataframe(spatial_grid: str):
    """Fixture for creating the expected dataframe of gridded data"""
    time = datetime(2017, 11, 10, 4, 0, 0)
    time = cf_units.date2num(
        time, "seconds since 1970-01-01 00:00:00", cf_units.CALENDAR_STANDARD
    )
    time = cf_units.num2date(
        time, "seconds since 1970-01-01 00:00:00", cf_units.CALENDAR_STANDARD
    )

    frt = datetime(2017, 11, 10, 0, 0, 0)
    frt = cf_units.date2num(
        frt, "seconds since 1970-01-01 00:00:00", cf_units.CALENDAR_STANDARD
    )
    frt = cf_units.num2date(
        frt, "seconds since 1970-01-01 00:00:00", cf_units.CALENDAR_STANDARD
    )

    if spatial_grid == "latlon":
        data = {
            "realization": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
            "latitude": np.array([-5.0, -5.0, 5.0, 5.0] * 2, dtype=np.float32),
            "longitude": np.array([-5.0, 5.0] * 4, dtype=np.float32),
            "air_temperature": np.array([305.0] * 8, dtype=np.float32),
            "forecast_period": np.array([14400] * 8, dtype=np.int32),
            "forecast_reference_time": np.array([frt] * 8),
            "time": np.array([time] * 8),
        }
    elif spatial_grid == "equalarea":
        data = {
            "realization": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
            "projection_y_coordinate": np.array(
                [-1000.0, -1000.0, 1000.0, 1000.0] * 2, dtype=np.float32
            ),
            "projection_x_coordinate": np.array(
                [-1000.0, 1000.0] * 4, dtype=np.float32
            ),
            "air_temperature": np.array([305.0] * 8, dtype=np.float32),
            "forecast_period": np.array([14400] * 8, dtype=np.int32),
            "forecast_reference_time": [frt] * 8,
            "time": [time] * 8,
        }

    return pd.DataFrame(data=data)


@pytest.fixture
def spot_dataframe():
    """Fixture for creating the expected dataframe of spot data"""
    time = datetime(2017, 11, 10, 4, 0, 0)
    time = cf_units.date2num(
        time, "seconds since 1970-01-01 00:00:00", cf_units.CALENDAR_STANDARD
    )
    time = cf_units.num2date(
        time, "seconds since 1970-01-01 00:00:00", cf_units.CALENDAR_STANDARD
    )

    frt = datetime(2017, 11, 10, 0, 0, 0)
    frt = cf_units.date2num(
        frt, "seconds since 1970-01-01 00:00:00", cf_units.CALENDAR_STANDARD
    )
    frt = cf_units.num2date(
        frt, "seconds since 1970-01-01 00:00:00", cf_units.CALENDAR_STANDARD
    )

    data = {
        "realization": np.array([0, 0, 1, 1], dtype=np.int32),
        "spot_index": [0, 1] * 2,
        "air_temperature": np.array([305.0] * 4, dtype=np.float32),
        "forecast_period": np.array([14400] * 4, dtype=np.int32),
        "forecast_reference_time": np.array([frt] * 4),
        "time": np.array([time] * 4),
        "altitude": np.array([1.0] * 4, dtype=np.float32),
        "latitude": np.array([50.0, 60.0] * 2, dtype=np.float32),
        "longitude": np.array([-5.0, 5.0] * 2, dtype=np.float32),
        "wmo_id": ["00000", "00001"] * 2,
    }

    return pd.DataFrame(data=data)


def altitude_cube(forecast_type: str, set_up_kwargs: Optional[Dict] = None) -> Cube:
    """Function for creating an altitude cube ancillary."""
    if set_up_kwargs is None:
        set_up_kwargs = {}
    if forecast_type == "gridded":
        data = np.array([[10, 20], [20, 10]], dtype=np.float32)
        output = set_up_variable_cube(
            data=data, name="surface_altitude", **set_up_kwargs
        )
    elif forecast_type == "spot":
        data = np.array([10, 20], dtype=np.float32)
        output = set_up_spot_variable_cube(
            data=data, name="surface_altitude", **set_up_kwargs
        )

    return output


def land_fraction_cube(
    forecast_type: str, set_up_kwargs: Optional[Dict] = None
) -> Cube:
    """Fixture for creating a land fraction cube ancillary."""
    if set_up_kwargs is None:
        set_up_kwargs = {}
    if forecast_type == "gridded":
        data = np.array([[0.0, 0.1, 0.2, 0.3], [0.3, 0.2, 0.1, 0.0]], dtype=np.float32)
        output = set_up_variable_cube(data=data, name="land_fraction", **set_up_kwargs)
    if forecast_type == "spot":
        data = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
        output = set_up_spot_variable_cube(
            data=data, name="land_fraction", **set_up_kwargs
        )

    return output


@pytest.mark.parametrize("include_altitude", [False, True])
@pytest.mark.parametrize("include_land_fraction", [False, True])
@pytest.mark.parametrize("spatial_grid", ["latlon", "equalarea"])
def test_prepare_data_for_gam_gridded(
    include_altitude, include_land_fraction, spatial_grid, gridded_dataframe
):
    """Test that this method correctly creates a dataframe from the input gridded data
    cubes."""
    set_up_kwargs = {"spatial_grid": spatial_grid}
    input_cube = create_simple_cube(
        forecast_type="gridded",
        n_spatial_points=2,
        n_realizations=2,
        n_times=1,
        fill_value=305.0,
        set_up_kwargs=set_up_kwargs,
    )

    additional_cubes = []
    if include_altitude:
        additional_cubes.append(altitude_cube("gridded", set_up_kwargs))
        surface_altitude = np.array([10.0, 20.0, 20.0, 10.0] * 2, dtype=np.float32)
        gridded_dataframe["surface_altitude"] = surface_altitude
    if include_land_fraction:
        additional_cubes.append(land_fraction_cube("gridded", set_up_kwargs))
        land_fraction = np.array([0.1, 0.2, 0.2, 0.1] * 2, dtype=np.float32)
        gridded_dataframe["land_fraction"] = land_fraction

    result = prepare_data_for_gam(input_cube, additional_cubes)

    assert np.all(result.columns == gridded_dataframe.columns)
    assert_frame_equal(result, gridded_dataframe)


@pytest.mark.parametrize("include_altitude", [False, True])
@pytest.mark.parametrize("include_land_fraction", [False, True])
def test_prepare_data_for_gam_spot(
    include_altitude, include_land_fraction, spot_dataframe
):
    """Test that this method correctly creates a dataframe from the input spot data
    cubes."""
    input_cube = create_simple_cube(
        forecast_type="spot",
        n_spatial_points=2,
        n_realizations=2,
        n_times=1,
        fill_value=305.0,
    )

    additional_cubes = []
    if include_altitude:
        additional_cubes.append(altitude_cube("spot"))
        surface_altitude = np.array([10.0, 20.0] * 2, dtype=np.float32)
        spot_dataframe["surface_altitude"] = surface_altitude
    if include_land_fraction:
        additional_cubes.append(land_fraction_cube("spot"))
        land_fraction = np.array([0.0, 0.3] * 2, dtype=np.float32)
        spot_dataframe["land_fraction"] = land_fraction

    result = prepare_data_for_gam(input_cube, additional_cubes)

    assert np.all(result.columns == spot_dataframe.columns)
    assert_frame_equal(result, spot_dataframe)


@pytest.mark.parametrize("spatial_grid", ["latlon", "equalarea"])
def test_convert_dataframe_to_cube_gridded(spatial_grid, gridded_dataframe):
    """Test that this method correctly creates a cube from the input dataframe of
    gridded data."""
    set_up_kwargs = {"spatial_grid": spatial_grid}
    expected_cube = create_simple_cube(
        forecast_type="gridded",
        n_spatial_points=2,
        n_realizations=2,
        n_times=1,
        fill_value=305.0,
        set_up_kwargs=set_up_kwargs,
    )
    template_cube = expected_cube.copy(data=np.zeros_like(expected_cube.data))

    # Change forecast data so that realization zero is equal to 305.0 across domain and
    # realization 1 is equal to 306.0 across domain.
    gridded_dataframe["air_temperature"] = np.array(
        [305.0] * 4 + [306.0] * 4, dtype=np.float32
    )
    expected_cube.data = np.array(
        [
            [
                [305.0, 305.0],
                [305.0, 305.0],
            ],
            [[306.0, 306.0], [306.0, 306.0]],
        ],
        dtype=np.float32,
    )

    result = convert_dataframe_to_cube(gridded_dataframe, template_cube)

    assert result == expected_cube


def test_convert_dataframe_to_cube_spot(spot_dataframe):
    """Test that this method correctly creates a cube from the input dataframe of
    spot data."""
    expected_cube = create_simple_cube(
        forecast_type="spot",
        n_spatial_points=2,
        n_realizations=2,
        n_times=1,
        fill_value=305.0,
    )
    template_cube = expected_cube.copy(data=np.zeros_like(expected_cube.data))

    # Change forecast data so that realization zero is equal to 305.0 across domain and
    # realization 1 is equal to 306.0 across domain.
    spot_dataframe["air_temperature"] = np.array(
        [305.0, 305.0, 306.0, 306.0], dtype=np.float32
    )
    expected_cube.data = np.array([[305.0, 305.0], [306.0, 306.0]], dtype=np.float32)

    result = convert_dataframe_to_cube(spot_dataframe, template_cube)

    assert result == expected_cube
