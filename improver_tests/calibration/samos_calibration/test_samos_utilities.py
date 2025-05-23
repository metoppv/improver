# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the various utility functions within samos_calibration.py. Also
defines some broadly applicable utility functions for SAMOS unit tests.
"""
import iris.pandas
import numpy as np
import pytest
from improver.calibration.samos_calibration import (
    prepare_data_for_gam,
    convert_dataframe_to_cube
)
from improver_tests.calibration.samos_calibration.helper_functions import (
    create_simple_cube,
    altitude_cube,
    land_fraction_cube,
    gridded_dataframe,
    spot_dataframe
)
from pandas.testing import assert_frame_equal
from cftime import datetime


@pytest.mark.parametrize("include_altitude", [False, True])
@pytest.mark.parametrize("include_land_fraction", [False, True])
@pytest.mark.parametrize("spatial_grid", ["latlon", "equalarea"])
def test_prepare_data_for_gam_gridded(
    include_altitude,
    include_land_fraction,
    spatial_grid,
    gridded_dataframe
):
    """Test that this method correctly creates a dataframe from the input gridded data
    cubes."""
    set_up_kwargs = {"spatial_grid": spatial_grid}
    input_cube = create_simple_cube(
        forecast_type="gridded",
        n_spatial_points=2,
        realizations=2,
        times=1,
        fill_value=305.0,
        set_up_kwargs=set_up_kwargs
    )

    additional_cubes = []
    if include_altitude:
        additional_cubes.append(altitude_cube("gridded", set_up_kwargs))
        surface_altitude = np.array([10.0, 20.0, 20.0, 10.0] * 2, dtype=np.float32)
        gridded_dataframe['surface_altitude'] = surface_altitude
    if include_land_fraction:
        additional_cubes.append(land_fraction_cube("gridded", set_up_kwargs))
        land_fraction = np.array([0.1, 0.2, 0.2, 0.1] * 2, dtype=np.float32)
        gridded_dataframe['land_fraction'] = land_fraction

    result = prepare_data_for_gam(input_cube, additional_cubes)

    assert np.all(result.columns == gridded_dataframe.columns)
    assert_frame_equal(result, gridded_dataframe)


@pytest.mark.parametrize("include_altitude", [False, True])
@pytest.mark.parametrize("include_land_fraction", [False, True])
def test_prepare_data_for_gam_spot(
        include_altitude,
        include_land_fraction,
        spot_dataframe
):
    """Test that this method correctly creates a dataframe from the input spot data
    cubes."""
    input_cube = create_simple_cube(
        forecast_type="spot",
        n_spatial_points=2,
        realizations=2,
        times=1,
        fill_value=305.0
    )

    additional_cubes = []
    if include_altitude:
        additional_cubes.append(altitude_cube("spot"))
        surface_altitude = np.array([10.0, 20.0] * 2, dtype=np.float32)
        spot_dataframe['surface_altitude'] = surface_altitude
    if include_land_fraction:
        additional_cubes.append(land_fraction_cube("spot"))
        land_fraction = np.array([0.0, 0.3] * 2, dtype=np.float32)
        spot_dataframe['land_fraction'] = land_fraction

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
        realizations=2,
        times=1,
        fill_value=305.0,
        set_up_kwargs=set_up_kwargs
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
            [
                [306.0, 306.0],
                [306.0, 306.0]
            ]
        ],
        dtype=np.float32
    )

    result = convert_dataframe_to_cube(
        gridded_dataframe, template_cube
    )

    assert result == expected_cube


def test_convert_dataframe_to_cube_spot(spot_dataframe):
    """Test that this method correctly creates a cube from the input dataframe of
    spot data."""
    expected_cube = create_simple_cube(
        forecast_type="spot",
        n_spatial_points=2,
        realizations=2,
        times=1,
        fill_value=305.0,
    )
    template_cube = expected_cube.copy(data=np.zeros_like(expected_cube.data))

    # Change forecast data so that realization zero is equal to 305.0 across domain and
    # realization 1 is equal to 306.0 across domain.
    spot_dataframe["air_temperature"] = np.array(
        [305.0, 305.0, 306.0, 306.0], dtype=np.float32
    )
    expected_cube.data = np.array(
        [[305.0, 305.0], [306.0, 306.0]], dtype=np.float32
    )

    result = convert_dataframe_to_cube(
        spot_dataframe, template_cube
    )

    assert result == expected_cube
