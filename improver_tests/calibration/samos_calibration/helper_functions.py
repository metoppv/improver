# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Helper functions for SAMOS unit tests."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import iris.cube
import numpy as np
from iris.cube import Cube

from improver.synthetic_data.set_up_test_cubes import (
    set_up_spot_variable_cube,
    set_up_variable_cube,
)

FORECAST_ATTRIBUTES = {
    "title": "MOGREPS-UK Forecast",
    "source": "Met Office Unified Model",
    "institution": "Met Office",
}


def create_simple_cube(
    forecast_type: str,
    n_spatial_points: int,
    n_realizations: int,
    n_times: int,
    fill_value: float,
    set_up_kwargs: Optional[Dict] = None,
    fixed_forecast_period=False,
) -> Cube:
    """Function for creating a cube of temperature data.

    Args:
        forecast_type:
            Either "gridded" or "spot".
        n_spatial_points:
            The desired number of spatial points. For a gridded cube, this is the number
            of points in the x and y directions. For a spot cube, this is the number of
            sites.
        n_realizations:
            The desired number of realizations.
        n_times:
            The desired number of times.
        fill_value:
            The single value to use for all the data.
        set_up_kwargs:
            A dictionary of keyword arguments accepted by set_up_variable_cube or
            set_up_spot_variable_cube.
        fixed_forecast_period:
            If true, the returned cube will have a forecast_reference_time coordinate
            of the same length as the time coordinate in order to allow the cube to
            have a single forecast period. If false, the returned cube will have a
            scalar forecast_reference_time coordinate and forecast_period coordinate
            of the same length as the time coordinate.

    Returns:
        A cube with all data equal to fill_value and metadata according to the given
        inputs.
    """
    if set_up_kwargs is None:
        set_up_kwargs = {}

    initial_dt = datetime(2017, 11, 10, 4, 0)
    initial_frt = datetime(2017, 11, 10, 0, 0)
    result = iris.cube.CubeList()

    if forecast_type == "gridded":
        data_shape = [n_spatial_points, n_spatial_points]  # Latitude, Longitude.
        plugin = set_up_variable_cube
    elif forecast_type == "spot":
        data_shape = [n_spatial_points]  # Number of sites.
        plugin = set_up_spot_variable_cube

    if n_realizations > 1:
        data_shape.insert(0, n_realizations)

    for i in range(n_times):
        dt = initial_dt + timedelta(days=i)
        frt = initial_frt + timedelta(days=i) if fixed_forecast_period else initial_frt

        data = np.full(data_shape, fill_value, dtype=np.float32)
        new_cube = plugin(data=data, time=dt, frt=frt, **set_up_kwargs)
        result.append(new_cube.copy())

    return result.merge_cube()


def create_cubes_for_gam_fitting(
    n_spatial_points: int,
    n_realizations: int,
    n_times: int,
    include_altitude: bool,
    fixed_forecast_period=False,
) -> Tuple[Cube, List[Cube]]:
    """Function to create a temperature cube with data which varies spatially.
    Optionally, may also produce an altitude cube whilst simultaneously modifying the
    temperature cube so that this is a useful predictor of the temperature.

    Args:
        n_spatial_points:
            The desired number of spatial points in the x and y directions.
        n_realizations:
            The desired number of realizations.
        n_times:
            The desired number of times.
        include_altitude:
            If True, the data in the returned cube is modified so that altitude is a
            useful predictor of the temperature.
        fixed_forecast_period:
            If true, the returned cube will have a forecast_reference_time coordinate
            of the same length as the time coordinate in order to allow the cube to
            have a single forecast period. If false, the returned cube will have a
            scalar forecast_reference_time coordinate and forecast_period coordinate
            of the same length as the time coordinate.

    Returns:
        A tuple with first element a cube of temperature data with metadata determined
        by the inputs and second element a list. The list will be empty if
        include_altitude is False, otherwise the list will contain a surface_altitude
        cube which is a useful predictor of data in the temperature cube.
    """
    input_cube = create_simple_cube(
        forecast_type="gridded",
        n_spatial_points=n_spatial_points,
        n_realizations=n_realizations,
        n_times=n_times,
        fill_value=273.15,
        fixed_forecast_period=fixed_forecast_period,
    )
    # Create array of data to add to cube which increases with x and y, so that
    # these features are useful in the GAMs.
    lat_addition = np.linspace(start=0, stop=15, num=n_spatial_points).reshape(
        [n_spatial_points, 1]
    )
    lon_addition = np.linspace(start=0, stop=15, num=n_spatial_points).reshape(
        [1, n_spatial_points]
    )
    addition = lat_addition + lon_addition  # 10x10 array
    addition = np.broadcast_to(addition, shape=input_cube.data.shape)
    # Create array of random noise which increases with x and y, so that there is
    # some variance in the data to model in the standard deviation GAM.
    rng = np.random.RandomState(1)
    noise = rng.normal(loc=0.0, scale=0.05 + (addition / 30))
    input_cube.data = input_cube.data + addition + noise

    additional_cubes = []
    if include_altitude:
        # Create an altitude cube with small values in the centre of the domain and
        # large values around the outside of the domain, with a smooth gradient in
        # between.
        altitude_cube = create_simple_cube(
            forecast_type="gridded",
            n_spatial_points=n_spatial_points,
            n_realizations=1,
            n_times=1,
            fill_value=1000.0,
        )
        altitude_cube.rename("surface_altitude")

        lat_multiplier = np.abs(
            np.linspace(start=-1, stop=1, num=n_spatial_points).reshape(
                [n_spatial_points, 1]
            )
        )  # 1 at ends, close to 0 in the middle.
        lon_multiplier = np.abs(
            np.linspace(start=-1, stop=1, num=n_spatial_points).reshape(
                [1, n_spatial_points]
            )
            - 1
        )  # 1 at ends, close to 0 in the middle.
        altitude_multiplier = lat_multiplier * lon_multiplier

        altitude_cube.data = altitude_cube.data * altitude_multiplier
        additional_cubes.append(altitude_cube)

        # Subtract values from input_cube data which increase with altitude.
        altitude_multiplier = np.broadcast_to(
            altitude_multiplier, shape=input_cube.data.shape
        )
        input_cube.data = input_cube.data - (5.0 * altitude_multiplier)

    return input_cube, additional_cubes
