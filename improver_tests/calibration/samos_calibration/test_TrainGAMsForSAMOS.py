# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the TrainGAMsForSAMOS class within samos_calibration.py"""

from datetime import datetime, timedelta

import iris.cube
import numpy as np
import pandas as pd
import pytest
from iris.cube import CubeList
from iris.coords import CellMethod
from improver.calibration.samos_calibration import TrainGAMsForSAMOS
from improver.synthetic_data.set_up_test_cubes import (
    set_up_spot_variable_cube,
    set_up_variable_cube,
)


@pytest.fixture
def gridded_dataframe():
    """Fixture for creating the expected dataframe of gridded data"""
    data = {
        "realization": [0, 0, 0, 0, 1, 1, 1, 1],
        "latitude": [-5.0, -5.0, 5.0, 5.0, -5.0, -5.0, 5.0, 5.0],
        "longitude": [-5.0, 5.0, -5.0, 5.0, -5.0, 5.0, -5.0, 5.0],
        "forecast_period": [timedelta(hours=4)] * 8,
        "forecast_reference_time": [datetime(
            2017, 11, 10, 0, 0)
        ] * 8,
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
        "forecast_reference_time": [datetime(
            2017, 11, 10, 0, 0
        )] * 4,
        "time": [datetime(2017, 11, 10, 4, 0)] * 4,
        "altitude": [1.0, 1.0, 1.0, 1.0],
        "latitude": [50.0, 60.0, 50.0, 60.0],
        "longitude": [-5.0, 5.0, -5.0, 5.0],
        "wmo_id": ["00000", "00001", "00000", "00001"],
        "air_temperature": [305.0] * 4,
    }
    return pd.DataFrame(data=data)


@pytest.fixture
def model_specification():
    """Fixture for creating a model specification as used in SAMOS plugins."""
    return [["l", [0], {}], ["s", [1], {}]]


def create_cube(forecast_type, realizations, times, fill_value=305):
    """Fixture for creating a cube of gridded data."""
    initial_dt = datetime(2017, 11, 10, 4, 0)
    result = iris.cube.CubeList()

    if forecast_type == "gridded":
        data_shape = [2, 2]  # lat, lon
        plugin = set_up_variable_cube
    elif forecast_type == "spot":
        data_shape = [2]  # no of sites
        plugin = set_up_spot_variable_cube

    if realizations > 1:
        data_shape.insert(0, realizations)

    for i in range(times):
        dt = initial_dt + timedelta(days=i)
        data = np.full(data_shape, fill_value, dtype=np.float32)
        new_cube = plugin(data=data, time=dt)
        result.append(new_cube.copy())

    return result.merge_cube()


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
        data = np.array(
            [[0.0, 0.1, 0.2, 0.3], [0.3, 0.2, 0.1, 0.0]], dtype=np.float32
        )
        output = set_up_variable_cube(data=data, name="land_fraction")
    if forecast_type is "spot":
        data = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
        output = set_up_spot_variable_cube(data=data, name="land_fraction")

    return output


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


@pytest.mark.parametrize("forecast_type", ["gridded", "spot"])
@pytest.mark.parametrize("realizations,times", [[2, 1], [2, 2], [1, 2]])
def test_calculate_cube_statistics(forecast_type, realizations, times):
    """Test that this method correctly calculates the mean and standard deviation of
    the input cube."""
    create_cube_kwargs = {
        "forecast_type": forecast_type, "realizations": realizations, "times": times
    }
    expected_cube_kwargs = {
        "forecast_type": forecast_type, "realizations": 1, "times": times
    }

    input_cube = create_cube(**create_cube_kwargs)

    # create cubelist containing expected mean and standard deviations cubes
    expected_mean = create_cube(fill_value=305, **expected_cube_kwargs)
    expected_sd = create_cube(fill_value=0, **expected_cube_kwargs)
    if realizations > 1:
        # Expect statistics to be calculated over the realization dimension.
        expected_mean.add_cell_method(CellMethod("mean", coords="realization"))
        expected_sd.add_cell_method(
            CellMethod("standard_deviation", coords="realization")
        )
    else:
        # Expect statistics to be calculated over the time dimension.
        expected_mean.add_cell_method(CellMethod("mean", coords="time"))
        expected_sd.add_cell_method(
            CellMethod("standard_deviation", coords="time")
        )
    expected = CubeList([expected_mean, expected_sd])

    result = TrainGAMsForSAMOS.calculate_cube_statistics(input_cube)

    assert expected == result


@pytest.mark.parametrize("include_altitude", [False, True])
@pytest.mark.parametrize("include_land_fraction", [False, True])
def test_prepare_data_for_gam_gridded(
    include_altitude,
    include_land_fraction,
    gridded_dataframe
):
    """Test that this method correctly creates a dataframe from the input gridded data
    cubes."""
    input_cube = create_cube(forecast_type="gridded", realizations=2, times=1)

    additional_cubes = []
    if include_altitude:
        additional_cubes.append(altitude_cube("gridded"))
        surface_altitude = [10.0, 20.0, 20.0, 10.0, 10.0, 20.0, 20.0, 10.0]
        gridded_dataframe['surface_altitude'] = surface_altitude
    if include_land_fraction:
        additional_cubes.append(land_fraction_cube("gridded"))
        land_fraction = [0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1]
        gridded_dataframe['land_fraction'] = land_fraction

    result = TrainGAMsForSAMOS.prepare_data_for_gam(input_cube, additional_cubes)

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
    spot_dataframe
):
    """Test that this method correctly creates a dataframe from the input spot data
    cubes."""
    input_cube = create_cube(forecast_type="spot", realizations=2, times=1)

    additional_cubes = []
    if include_altitude:
        additional_cubes.append(altitude_cube("spot"))
        surface_altitude = [10.0, 20.0, 10.0, 20.0]
        spot_dataframe['surface_altitude'] = surface_altitude
    if include_land_fraction:
        additional_cubes.append(land_fraction_cube("spot"))
        land_fraction = [0.0, 0.3, 0.0, 0.3]
        spot_dataframe['land_fraction'] = land_fraction

    result = TrainGAMsForSAMOS.prepare_data_for_gam(input_cube, additional_cubes)

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


@pytest.mark.parametrize("exception", ["no_time_coord", "single_point_time_coord"])
def test_missing_required_coordinates_exception(
    exception,
    model_specification
):
    """Test that the correct exceptions are raised when the input cube does not contain
    suitable realization or time coordinates."""
    # create a cube with no realization coordinate and a single point time coordinate
    input_cube = create_cube(forecast_type="gridded", realizations=1, times=1)
    features = ["latitude", "longitude"]

    if exception is "no_time_coord":
        input_cube.remove_coord("time")
        msg = ("The input cube must contain at least one of a realization or time "
               "coordinate in order to allow the calculation of means and standard "
               "deviations.")
    elif exception is "single_point_time_coord":
        msg = ("The input cube does not contain a realization coordinate. In order to "
               "calculate means and standard deviations the time coordinate must "
               "contain more than one point.")

    with pytest.raises(ValueError, match=msg):
        TrainGAMsForSAMOS(model_specification).process(input_cube, features)
