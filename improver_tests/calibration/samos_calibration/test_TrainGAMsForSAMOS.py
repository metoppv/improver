# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the TrainGAMsForSAMOS class within samos_calibration.py"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from improver.calibration.samos_calibration import TrainGAMsForSAMOS
from improver.synthetic_data.set_up_test_cubes import (
    set_up_spot_variable_cube,
    set_up_variable_cube,
)


@pytest.fixture
def gridded_cube():
    """Fixture for creating a cube of gridded data."""
    data = np.full((2, 2, 2), 305, dtype=np.float32)
    return set_up_variable_cube(data=data)


@pytest.fixture
def spot_cube():
    """Fixture for creating a cube of spot data."""
    data = np.full((2, 2), 305, dtype=np.float32)
    return set_up_spot_variable_cube(data=data)


def altitude_cube(forecast_type):
    """Function for creating an altitude cube ancillary."""
    if forecast_type is "gridded":
        data = np.array([[10, 20], [20, 10]], dtype=np.float32)
        output = set_up_variable_cube(data=data, name="surface_altitude")
    elif forecast_type is "spot":
        data = np.array([10, 20], dtype=np.float32)
        output = set_up_spot_variable_cube(data=data, name="surface_altitude")

    return output


def land_fraction_cube(forecast_type):
    """Fixture for creating a land fraction cube ancillary."""
    if forecast_type is "gridded":
        data = np.array([[0.0, 0.1, 0.2, 0.3], [0.3, 0.2, 0.1, 0.0]], dtype=np.float32)
        output = set_up_variable_cube(data=data, name="land_fraction")
    if forecast_type is "spot":
        data = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
        output = set_up_spot_variable_cube(data=data, name="land_fraction")

    return output


@pytest.fixture
def gridded_dataframe():
    """Fixture for creating the expected dataframe of gridded data"""
    data = {
        "realization": [0, 0, 0, 0, 1, 1, 1, 1],
        "latitude": [-5.0, -5.0, 5.0, 5.0, -5.0, -5.0, 5.0, 5.0],
        "longitude": [-5.0, 5.0, -5.0, 5.0, -5.0, 5.0, -5.0, 5.0],
        "forecast_period": [timedelta(hours=4)] * 8,
        "forecast_reference_time": [datetime(2017, 11, 10, 0, 0)] * 8,
        "time": [datetime(2017, 11, 10, 4, 0)] * 8,
        "air_temperature": [305.0] * 8,
    }
    return pd.DataFrame(data=data)


@pytest.fixture
def spot_dataframe():
    """Fixture for creating the expected dataframe of spot data"""
    data = {
        "realization": [0, 0, 1, 1],
        "spot_index": [0, 1, 0, 1],
        "forecast_period": [timedelta(hours=4)] * 4,
        "forecast_reference_time": [datetime(2017, 11, 10, 0, 0)] * 4,
        "time": [datetime(2017, 11, 10, 4, 0)] * 4,
        "altitude": [1.0, 1.0, 1.0, 1.0],
        "latitude": [50.0, 60.0, 50.0, 60.0],
        "longitude": [-5.0, 5.0, -5.0, 5.0],
        "wmo_id": ["00000", "00001", "00000", "00001"],
        "air_temperature": [305.0] * 4,
    }
    return pd.DataFrame(data=data)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "model_specification": [["l", [0], {}]]
        },  # define a model specification but leave all other inputs as default
        {
            "model_specification": [["l", [0], {}]],
            "max_iter": 200,
            "tol": 0.1,
        },  # check that inputs related to model fitting are initialised correctly
        {
            "model_specification": [["l", [0], {}]],
            "distribution": "gamma",
            "link": "inverse",
            "fit_intercept": False,
        },  # check that inputs related to the model design are initialised correctly
    ],
)
def test__init__(kwargs):
    """Test that the class initializes variables correctly."""
    # Skip test if pyGAM not available.
    pytest.importorskip("pygam")

    # Define the default, then update with any differently specified inputs
    expected = {
        "model_specification": None,
        "max_iter": 100,
        "tol": 0.0001,
        "distribution": "normal",
        "link": "identity",
        "fit_intercept": True,
    }
    expected.update(kwargs)
    result = TrainGAMsForSAMOS(**kwargs)

    for key in [key for key in kwargs.keys()]:
        assert getattr(result, key) == kwargs[key]


@pytest.mark.parametrize("include_altitude", [False, True])
@pytest.mark.parametrize("include_land_fraction", [False, True])
def test_prepare_data_for_gam_gridded(
        include_altitude,
        include_land_fraction,
        gridded_cube,
        gridded_dataframe
):
    """Test that this method correctly creates a dataframe from the input cubes."""
    additional_cubes = []
    if include_altitude:
        additional_cubes.append(altitude_cube("gridded"))
        surface_altitude = [10.0, 20.0, 20.0, 10.0, 10.0, 20.0, 20.0, 10.0]
        gridded_dataframe['surface_altitude'] = surface_altitude
    if include_land_fraction:
        additional_cubes.append(land_fraction_cube("gridded"))
        land_fraction = [0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1]
        gridded_dataframe['land_fraction'] = land_fraction

    result = TrainGAMsForSAMOS.prepare_data_for_gam(gridded_cube, additional_cubes)

    # Split columns containing floats from those that don't when checking results
    result_columns = set(result.columns)
    non_float_columns = {'forecast_period', 'forecast_reference_time', 'time'}
    float_columns = result_columns - non_float_columns
    non_float_columns = list(non_float_columns)
    float_columns = list(float_columns)

    assert np.all(result.columns == gridded_dataframe.columns)
    np.testing.assert_array_almost_equal(
        result[float_columns].values,
        gridded_dataframe[float_columns].values
    )
    assert np.all(
        result[non_float_columns].values == gridded_dataframe[non_float_columns].values
    )


@pytest.mark.parametrize("include_altitude", [False, True])
@pytest.mark.parametrize("include_land_fraction", [False, True])
def test_prepare_data_for_gam_spot(
        include_altitude,
        include_land_fraction,
        spot_cube,
        spot_dataframe
):
    """Test that this method correctly creates a dataframe from the input cubes."""
    additional_cubes = []
    if include_altitude:
        additional_cubes.append(altitude_cube("spot"))
        surface_altitude = [10.0, 20.0, 10.0, 20.0]
        spot_dataframe['surface_altitude'] = surface_altitude
    if include_land_fraction:
        additional_cubes.append(land_fraction_cube("spot"))
        land_fraction = [0.0, 0.3, 0.0, 0.3]
        spot_dataframe['land_fraction'] = land_fraction

    result = TrainGAMsForSAMOS.prepare_data_for_gam(spot_cube, additional_cubes)

    # Split columns containing floats from those that don't when checking results
    result_columns = set(result.columns)
    non_float_columns = {'forecast_period', 'forecast_reference_time', 'time', 'wmo_id'}
    float_columns = result_columns - non_float_columns
    non_float_columns = list(non_float_columns)
    float_columns = list(float_columns)

    assert np.all(result.columns == spot_dataframe.columns)
    np.testing.assert_array_almost_equal(
        result[float_columns].values,
        spot_dataframe[float_columns].values
    )
    assert np.all(
        result[non_float_columns].values == spot_dataframe[non_float_columns].values
    )


def test_no_realization_dimension_exception():
    """Test that the correct exception is raised when the input cube does not contain a
    realization dimension."""

    assert True