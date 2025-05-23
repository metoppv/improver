# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Helper functions for SAMOS unit tests.
"""
from datetime import datetime, timedelta
import iris.cube
import numpy as np
import pandas as pd
import pytest
from iris.cube import Cube, CubeList
from iris.coords import CellMethod
from improver.synthetic_data.set_up_test_cubes import (
    set_up_spot_variable_cube,
    set_up_variable_cube,
)
from typing import Optional, Dict
import cf_units


@pytest.fixture
def gridded_dataframe(spatial_grid: str):
    """Fixture for creating the expected dataframe of gridded data"""
    time = datetime(2017, 11, 10, 4, 0, 0)
    time = cf_units.date2num(time, 'seconds since 1970-01-01 00:00:00', cf_units.CALENDAR_STANDARD)
    time = cf_units.num2date(time, 'seconds since 1970-01-01 00:00:00', cf_units.CALENDAR_STANDARD)

    frt = datetime(2017, 11, 10, 0, 0, 0)
    frt = cf_units.date2num(frt, 'seconds since 1970-01-01 00:00:00', cf_units.CALENDAR_STANDARD)
    frt = cf_units.num2date(frt, 'seconds since 1970-01-01 00:00:00', cf_units.CALENDAR_STANDARD)

    if spatial_grid is "latlon":
        data = {
            "realization": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
            "latitude": np.array([-5.0, -5.0, 5.0, 5.0] * 2, dtype=np.float32),
            "longitude": np.array([-5.0, 5.0] * 4, dtype=np.float32),
            "air_temperature": np.array([305.0] * 8, dtype=np.float32),
            "forecast_period": np.array([14400] * 8, dtype=np.int32),
            "forecast_reference_time": np.array([frt] * 8),
            "time": np.array([time] * 8),
        }
    elif spatial_grid is "equalarea":
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
    time = cf_units.date2num(time, 'seconds since 1970-01-01 00:00:00', cf_units.CALENDAR_STANDARD)
    time = cf_units.num2date(time, 'seconds since 1970-01-01 00:00:00', cf_units.CALENDAR_STANDARD)

    frt = datetime(2017, 11, 10, 0, 0, 0)
    frt = cf_units.date2num(frt, 'seconds since 1970-01-01 00:00:00', cf_units.CALENDAR_STANDARD)
    frt = cf_units.num2date(frt, 'seconds since 1970-01-01 00:00:00', cf_units.CALENDAR_STANDARD)

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


def altitude_cube(forecast_type, set_up_kwargs: Optional[Dict] = None) -> Cube:
    """Function for creating an altitude cube ancillary."""
    if set_up_kwargs is None:
        set_up_kwargs = {}
    if forecast_type is "gridded":
        data = np.array([[10, 20], [20, 10]], dtype=np.float32)
        output = set_up_variable_cube(
            data=data, name="surface_altitude", **set_up_kwargs
        )
    elif forecast_type is "spot":
        data = np.array([10, 20], dtype=np.float32)
        output = set_up_spot_variable_cube(
            data=data, name="surface_altitude", **set_up_kwargs
        )

    return output


def land_fraction_cube(forecast_type, set_up_kwargs: Optional[Dict] = None) -> Cube:
    """Fixture for creating a land fraction cube ancillary."""
    if set_up_kwargs is None:
        set_up_kwargs = {}
    if forecast_type is "gridded":
        data = np.array(
            [[0.0, 0.1, 0.2, 0.3], [0.3, 0.2, 0.1, 0.0]], dtype=np.float32
        )
        output = set_up_variable_cube(data=data, name="land_fraction", **set_up_kwargs)
    if forecast_type is "spot":
        data = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
        output = set_up_spot_variable_cube(
            data=data, name="land_fraction", **set_up_kwargs
        )

    return output


def create_simple_cube(
    forecast_type,
    n_spatial_points,
    realizations,
    times,
    fill_value,
    set_up_kwargs: Optional[Dict] = None,
    fixed_forecast_period = False
) -> Cube:
    """Function for creating a cube of temperature data."""
    if set_up_kwargs is None:
        set_up_kwargs = {}

    initial_dt = datetime(2017, 11, 10, 4, 0)
    initial_frt = datetime(2017, 11, 10, 0, 0)
    result = iris.cube.CubeList()

    if forecast_type == "gridded":
        data_shape = [n_spatial_points, n_spatial_points]  # lat, lon
        plugin = set_up_variable_cube
    elif forecast_type == "spot":
        data_shape = [n_spatial_points]  # no of sites
        plugin = set_up_spot_variable_cube

    if realizations > 1:
        data_shape.insert(0, realizations)

    for i in range(times):
        dt = initial_dt + timedelta(days=i)
        frt = initial_frt + timedelta(days=i) if fixed_forecast_period else initial_frt

        data = np.full(data_shape, fill_value, dtype=np.float32)
        new_cube = plugin(data=data, time=dt, frt=frt, **set_up_kwargs)
        result.append(new_cube.copy())

    return result.merge_cube()


def create_cubes_for_gam_fitting(
        n_spatial_points,
        n_realizations,
        n_times,
        include_altitude,
        include_land_fraction,
        fixed_forecast_period=False,
):
    """Function to create a temperature cube with data which varies spatially.
    Optionally, may also produce altitude and/or land fraction cubes whilst
    simultaneously modifying the temperature cube so that these are useful predictors
    of the temperature.
    """
    input_cube = create_simple_cube(
        forecast_type="gridded",
        n_spatial_points=n_spatial_points,
        realizations=n_realizations,
        times=n_times,
        fill_value=273.15,
        fixed_forecast_period=fixed_forecast_period
    )
    # Create array of data to add to cube which increases with x and y, so that
    # these features are useful in the GAMs.
    lat_addition = np.linspace(
        start=0, stop=15, num=n_spatial_points
    ).reshape([n_spatial_points, 1])
    lon_addition = np.linspace(
        start=0, stop=15, num=n_spatial_points
    ).reshape([1, n_spatial_points])
    addition = lat_addition + lon_addition  # 10x10 array
    addition = np.broadcast_to(
        addition,
        shape=input_cube.data.shape
    )
    # Create array of random noise which increases with x and y, so that there is
    # some variance in the data to model in the standard deviation GAM.
    noise = np.random.normal(loc=0.0, scale=addition/30)
    input_cube.data = input_cube.data + addition + noise

    additional_cubes = []
    if include_altitude:
        # Create an altitude cube with small values in the centre of the domain and
        # large values around the outside of the domain, with a smooth gradient in
        # between.
        altitude_cube = create_simple_cube(
            forecast_type="gridded",
            n_spatial_points=n_spatial_points,
            realizations=1,
            times=1,
            fill_value=1000.0
        )
        altitude_cube.rename("surface_altitude")

        lat_multiplier = np.abs(np.linspace(
            start=-1, stop=1, num=n_spatial_points
        ).reshape([n_spatial_points, 1]))  # 1 at ends, close to 0 in the middle.
        lon_multiplier = np.abs(np.linspace(
            start=-1, stop=1, num=n_spatial_points
        ).reshape([1, n_spatial_points]) - 1)  # 1 at ends, close to 0 in the middle.
        altitude_multiplier = lat_multiplier * lon_multiplier

        altitude_cube.data = altitude_cube.data * altitude_multiplier
        additional_cubes.append(altitude_cube)

        # Subtract values from input_cube data which increase with altitude.
        altitude_multiplier = np.broadcast_to(
            altitude_multiplier,
            shape=input_cube.data.shape
        )
        input_cube.data = input_cube.data - (5.0 * altitude_multiplier)

    if include_land_fraction:
        # Create a land fraction cube with full land in the top left corner of the
        # domain, full sea in the bottom right, and a smooth gradient of fractions in
        # between.
        lf_cube = create_simple_cube(
            forecast_type="gridded",
            n_spatial_points=n_spatial_points,
            realizations=1,
            times=1,
            fill_value=1
        )
        lf_cube.rename("land_fraction")

        lat_multiplier = np.linspace(
            start=1.0, stop=0.0, num=n_spatial_points
        ).reshape([n_spatial_points, 1])
        lon_multiplier = np.linspace(
            start=1.0, stop=0.0, num=n_spatial_points
        ).reshape([1, n_spatial_points])
        lf_multiplier = lat_multiplier * lon_multiplier

        lf_cube.data = lf_cube.data * lf_multiplier
        additional_cubes.append(lf_cube)

        # Add values to input_cube data which increase with land fraction.
        lf_multiplier = np.broadcast_to(
            lf_multiplier,
            shape=input_cube.data.shape
        )
        input_cube.data = input_cube.data + (2.0 * lf_multiplier)

    return input_cube, additional_cubes
