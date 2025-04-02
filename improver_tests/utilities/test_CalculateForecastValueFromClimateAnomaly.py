# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the CalculateForecastValueFromClimateAnomaly plugin within
mathematical_operations.py"""

from datetime import datetime

import iris
import numpy as np
import pytest
from iris.cube import CubeList

from improver.constants import SECONDS_IN_HOUR
from improver.synthetic_data.set_up_test_cubes import (
    set_up_spot_variable_cube,
    set_up_variable_cube,
)
from improver.utilities.mathematical_operations import (
    CalculateClimateAnomalies,
    CalculateForecastValueFromClimateAnomaly,
)

# Importing the add_reference_epoch_metadata class static method to use in the
# fixtures
add_reference_epoch_metadata = CalculateClimateAnomalies._add_reference_epoch_metadata
remove_reference_epoch_metadata = (
    CalculateForecastValueFromClimateAnomaly._remove_reference_epoch_metadata
)


@pytest.fixture
def time_bounds():
    """Fixture for creating time bounds for a cube."""
    return {"mean_and_std": (datetime(2024, 9, 16, 0, 0), datetime(2024, 10, 16, 1, 0))}


@pytest.fixture
def validity_time():
    """Fixture for creating validity times for a cube."""
    return {
        "anomaly_basic": datetime(2024, 10, 16, 0, 0),
        "anomaly_multiple": [
            datetime(2024, 10, 15, 23, 0),
            datetime(2024, 10, 16, 0, 0),
            datetime(2024, 10, 16, 1, 0),
        ],
        "mean_and_std": datetime(2024, 10, 16, 0, 0),
    }


@pytest.fixture
def forecast_reference_time():
    """Fixture for creating forecast reference times for a cube."""
    return datetime(2024, 10, 15, 18, 0)


@pytest.fixture
def mean_cube(validity_time, time_bounds):
    """Fixture for creating a mean cube."""
    data = np.array([298], dtype=np.float32).reshape(1, 1, 1)
    cube = set_up_variable_cube(
        data=data,
        time=validity_time["mean_and_std"],
        time_bounds=time_bounds["mean_and_std"],
        units="K",
    )
    cell_method = iris.coords.CellMethod(method="mean", coords="time")
    cube.add_cell_method(cell_method)
    return cube


@pytest.fixture
def std_cube(validity_time, time_bounds):
    """Fixture for creating a std cube"""
    data = np.array([4], dtype=np.float32).reshape(1, 1, 1)
    cube = set_up_variable_cube(
        data=data,
        time=validity_time["mean_and_std"],
        time_bounds=time_bounds["mean_and_std"],
        units="1",
    )
    cell_method = iris.coords.CellMethod(method="standard_deviation", coords="time")
    cube.add_cell_method(cell_method)
    return cube


# Gridded cube fixtures
@pytest.fixture
def unstandardized_anomaly_cube(validity_time, forecast_reference_time, mean_cube):
    """Fixture for creating an unstandarized gridded anomaly cube."""
    data = np.array([7], dtype=np.float32).reshape(1, 1, 1)
    cube = set_up_variable_cube(
        data=data,
        time=validity_time["anomaly_basic"],
        frt=forecast_reference_time,
        units="K",
    )
    cube.standard_name = "air_temperature_anomaly"
    add_reference_epoch_metadata(cube, mean_cube)
    return cube


@pytest.fixture
def unstandardized_anomaly_cube_multiple_time_points(
    validity_time, forecast_reference_time, mean_cube
):
    """
    Fixture for creating an unstandarized gridded anomaly cube with multiple
    time points.add_reference_epoch_metadata
    """
    cubes = CubeList()
    for time, value in zip(validity_time["anomaly_multiple"], [2, 0, -2]):
        data = np.full((1, 1), value, dtype=np.float32)
        cubes.append(
            set_up_variable_cube(
                data=data, time=time, frt=forecast_reference_time, units="K"
            )
        )
    cube = cubes.merge_cube()
    cube.standard_name = "air_temperature_anomaly"
    add_reference_epoch_metadata(cube, mean_cube)
    return cube


@pytest.fixture
def standardized_anomaly_cube(validity_time, forecast_reference_time, mean_cube):
    """Fixture for creating a standardized gridded anomaly cube."""
    data = np.array([1.75], dtype=np.float32).reshape(1, 1, 1)
    cube = set_up_variable_cube(
        data=data,
        time=validity_time["anomaly_basic"],
        frt=forecast_reference_time,
        units="1",
    )
    cube.long_name = "air_temperature_standardized_anomaly"
    add_reference_epoch_metadata(cube, mean_cube)
    return cube


@pytest.fixture
def standardized_anomaly_cube_multiple_time_points(
    validity_time, forecast_reference_time, mean_cube
):
    """
    Fixture for creating a standardized gridded anomaly cube with multiple
    time points.
    """
    cubes = CubeList()
    for time, value in zip(validity_time["anomaly_multiple"], [0.5, 0.0, -0.5]):
        data = np.full((1, 1), value, dtype=np.float32)
        cubes.append(
            set_up_variable_cube(
                data=data, time=time, frt=forecast_reference_time, units="1"
            )
        )
    cube = cubes.merge_cube()
    cube.long_name = "air_temperature_standardized_anomaly"
    add_reference_epoch_metadata(cube, mean_cube)
    return cube


# Site cube fixtures
# Site anomaly cubes with mutliple time points are not created for testing as the
# plugin's functionality is not expected to change, and so is implicitly tested by the
# gridded anomaly cubes with multiple time points.


@pytest.fixture
def site_cubes(validity_time, time_bounds, forecast_reference_time, mean_cube):
    """Fixture for creating site cubes."""
    # Unstandardized anomaly site cube
    data = np.array([7.0], dtype=np.float32)
    unstandardized_anomaly_site_cube = set_up_spot_variable_cube(
        data=data,
        time=validity_time["anomaly_basic"],
        frt=forecast_reference_time,
        units="K",
    )
    unstandardized_anomaly_site_cube.standard_name = "air_temperature_anomaly"
    add_reference_epoch_metadata(unstandardized_anomaly_site_cube, mean_cube)

    # Standardized anomaly site cube
    data = np.array([1.75], dtype=np.float32)
    standardized_anomaly_site_cube = set_up_spot_variable_cube(
        data=data,
        time=validity_time["anomaly_basic"],
        frt=forecast_reference_time,
        units="1",
    )
    standardized_anomaly_site_cube.long_name = "air_temperature_standardized_anomaly"
    add_reference_epoch_metadata(standardized_anomaly_site_cube, mean_cube)

    # Mean site cube
    data = np.array([298], dtype=np.float32)
    mean_site_cube = set_up_spot_variable_cube(
        data=data,
        time=validity_time["mean_and_std"],
        time_bounds=time_bounds["mean_and_std"],
        units="K",
    )
    cell_method = iris.coords.CellMethod(method="mean", coords="time")
    mean_site_cube.add_cell_method(cell_method)

    # Standard deviation site cube
    data = np.array([4], dtype=np.float32)
    std_site_cube = set_up_spot_variable_cube(
        data=data,
        time=validity_time["mean_and_std"],
        time_bounds=time_bounds["mean_and_std"],
        units="1",
    )
    cell_method = iris.coords.CellMethod(method="std", coords="time")
    std_site_cube.add_cell_method(cell_method)

    return (
        unstandardized_anomaly_site_cube,
        standardized_anomaly_site_cube,
        mean_site_cube,
        std_site_cube,
    )


# Testing the plugin's basic functionality
@pytest.mark.parametrize(
    "fixture_name",
    ["unstandardized_anomaly_cube", "unstandardized_anomaly_cube_multiple_time_points"],
)
def test_calculate_unstandardized_forecasts_gridded_data(
    request, fixture_name, mean_cube
):
    """Test that the plugin calculates forecast values from unstandardized anomalies
    correctly."""

    anomaly_cube = request.getfixturevalue(fixture_name)
    plugin = CalculateForecastValueFromClimateAnomaly()
    if fixture_name == "unstandardized_anomaly_cube":
        expected_values = np.array([305], dtype=np.float32).reshape(1, 1, 1)
    elif fixture_name == "unstandardized_anomaly_cube_multiple_time_points":
        expected_values = np.array([300, 298, 296], dtype=np.float32).reshape(3, 1, 1)
    result = plugin.process(anomaly_cube, mean_cube)
    np.testing.assert_allclose(result.data, expected_values, rtol=1e-5)
    assert result.name() == "air_temperature"
    assert result.units == "K"
    assert "reference_epoch" not in [coord.name() for coord in result.aux_coords]
    assert "anomaly" not in tuple(cm for cm in result.cell_methods)


@pytest.mark.parametrize(
    "fixture_name",
    ["standardized_anomaly_cube", "standardized_anomaly_cube_multiple_time_points"],
)
def test_calculate_standardized_forecasts_gridded_data(
    request, fixture_name, mean_cube, std_cube
):
    """Test that the plugin calculates forecast values from standardized anomalies
    correctly."""

    anomaly_cube = request.getfixturevalue(fixture_name)
    plugin = CalculateForecastValueFromClimateAnomaly()
    if fixture_name == "standardized_anomaly_cube":
        expected_values = np.full((1, 1, 1), 305, dtype=np.float32)
    elif fixture_name == "standardized_anomaly_cube_multiple_time_points":
        expected_values = np.array([300, 298, 296], dtype=np.float32).reshape(3, 1, 1)
    result = plugin.process(anomaly_cube, mean_cube, std_cube)
    np.testing.assert_allclose(result.data, expected_values, rtol=1e-5)
    assert result.name() == "air_temperature"
    assert result.units == "K"
    assert "reference_epoch" not in [coord.name() for coord in result.aux_coords]
    assert "anomaly" not in tuple(cm for cm in result.cell_methods)


def test_calculate_unstandardized_anomalies_site_data(site_cubes):
    """Test that the plugin calculates forecast values from unstandardized anomalies
    correctly for site data."""
    plugin = CalculateForecastValueFromClimateAnomaly()
    unstandardized_anomaly_site_cube, _, mean_site_cube, _ = site_cubes
    result = plugin.process(unstandardized_anomaly_site_cube, mean_site_cube)
    expected_anomalies = np.array([305], dtype=np.float32)
    np.testing.assert_allclose(result.data, expected_anomalies, rtol=1e-5)
    assert result.name() == "air_temperature"
    assert result.units == "K"
    assert "reference_epoch" not in [coord.name() for coord in result.aux_coords]
    assert "anomaly" not in tuple(cm for cm in result.cell_methods)


def test_calculate_standardized_anomalies_site_data(site_cubes):
    """Test that the plugin calculates forecast values from standardized anomalies
    correctly for site data."""
    plugin = CalculateForecastValueFromClimateAnomaly()
    _, standardized_anomaly_site_cube, mean_site_cube, std_site_cube = site_cubes
    result = plugin.process(
        standardized_anomaly_site_cube, mean_site_cube, std_site_cube
    )
    expected_anomalies = np.array([305.0], dtype=np.float32)
    np.testing.assert_allclose(result.data, expected_anomalies, rtol=1e-5)
    assert result.name() == "air_temperature"
    assert result.units == "K"
    assert "reference_epoch" not in [coord.name() for coord in result.aux_coords]
    assert "anomaly" not in tuple(cm for cm in result.cell_methods)


def test_ignore_temporal_mismatch(standardized_anomaly_cube, mean_cube, std_cube):
    """Test that the plugin handles requests to ignore temporal mismatch
    between anomaly cube and mean/std cube.
    """
    standardized_anomaly_cube.coord("time").points = (
        standardized_anomaly_cube.coord("time").points + 10 * SECONDS_IN_HOUR
    )  # Moves anomaly bounds outside mean bounds

    plugin = CalculateForecastValueFromClimateAnomaly(ignore_temporal_mismatch=True)
    result = plugin.process(standardized_anomaly_cube, mean_cube, std_cube)

    assert result.name() == "air_temperature"
    assert result.units == "K"
    assert "reference_epoch" not in [coord.name() for coord in result.aux_coords]
    assert "anomaly" not in tuple(cm for cm in result.cell_methods)


# Testing the plugin's internal verification checks


@pytest.mark.parametrize(
    "error_to_check",
    ["standardized_anomaly_no_std_cube", "unstandardized_anomaly_and_std_cube"],
)
def test_error_inputs_mismatch(
    unstandardized_anomaly_cube, standardized_anomaly_cube, std_cube, error_to_check
):
    """Test that the plugin raises a ValueError if the inputs are incorrect for the
    type of anomaly data (standardised or unstandardised) input"""

    plugin = CalculateForecastValueFromClimateAnomaly()
    if error_to_check == "standardized_anomaly_no_std_cube":
        with pytest.raises(
            ValueError,
            match="The standard deviation cube must be provided to calculate "
            "the forecast value from a standardized anomaly.",
        ):
            plugin.verify_inputs_for_forecast(standardized_anomaly_cube)
    else:
        with pytest.raises(
            ValueError,
            match="The standard deviation cube should not be provided to calculate "
            "the forecast value from an unstandardized anomaly.",
        ):
            plugin.verify_inputs_for_forecast(unstandardized_anomaly_cube, std_cube)


@pytest.mark.parametrize(
    "error_to_check", ["mean_and_anomaly_check", "std_and_anomaly_check"]
)
def test_error_units_mismatch(
    unstandardized_anomaly_cube,
    standardized_anomaly_cube,
    mean_cube,
    std_cube,
    error_to_check,
):
    """Test that the plugin raises a ValueError if the units of the anomaly and
    another cube mismatch"""
    plugin = CalculateForecastValueFromClimateAnomaly()
    if error_to_check == "mean_and_anomaly_check":
        mean_cube.units = "C"  # The units should be K ordinarily
        with pytest.raises(
            ValueError,
            match="The mean cube must have the same units as the unstandardized "
            "anomaly cube.",
        ):
            plugin.verify_units_match(unstandardized_anomaly_cube, mean_cube)
    else:
        std_cube.units = "C"  # The units should be K ordinarily
        with pytest.raises(
            ValueError,
            match="The standard deviation cube, if supplied, must have the same units ",
        ):
            plugin.verify_units_match(standardized_anomaly_cube, mean_cube, std_cube)


def test_error_spatial_coords_mismatch_gridded_data(
    standardized_anomaly_cube, mean_cube, std_cube
):
    """Test that the plugin raises a ValueError if the spatial coordinates of the
    anomaly_cube and another cube mismatch"""
    mean_cube.coord("latitude").points = mean_cube.coord("latitude").points + 20
    mean_cube.coord("longitude").points = mean_cube.coord("longitude").points + 20
    plugin = CalculateForecastValueFromClimateAnomaly()
    with pytest.raises(ValueError, match="The spatial coordinates must match."):
        plugin.verify_spatial_coords_match(
            standardized_anomaly_cube, mean_cube, std_cube
        )


@pytest.mark.parametrize(
    "error_to_raise", ["more_than_one_spot_index", "mismatching_spot_index_points"]
)
def test_error_spatial_coords_mismatch_site_data(site_cubes, mean_cube, error_to_raise):
    _, standardized_anomaly_site_cube, mean_site_cube, std_site_cube = site_cubes
    """Test that the plugin raises a ValueError if the spatial coordinates of the
    mean and std cubes have different bounds."""
    plugin = CalculateForecastValueFromClimateAnomaly()
    if error_to_raise == "more_than_one_spot_index":
        # Uses a cube without the spot_index coordinate (gridded mean_cube)
        # to trigger the error
        with pytest.raises(
            ValueError,
            match="The cubes must all have the same spatial coordinates. Some cubes "
            "contain spot_index coordinates and some do not.",
        ):
            plugin.verify_spatial_coords_match(
                standardized_anomaly_site_cube, mean_cube, std_site_cube
            )
    else:
        standardized_anomaly_site_cube.coord("spot_index").points = (
            standardized_anomaly_site_cube.coord("spot_index").points + 1
        )
        with pytest.raises(
            ValueError,
            match="Mismatching spot_index coordinates were found on the input cubes.",
        ):
            plugin.verify_spatial_coords_match(
                standardized_anomaly_site_cube, mean_site_cube, std_site_cube
            )


@pytest.mark.parametrize(
    "check", ["reference_epoch_to_mean_bounds", "reference_epoch_to_mean_points"]
)
def test_error_time_coords_mismatch(
    standardized_anomaly_cube, mean_cube, std_cube, check
):
    """Test that the plugin raises a ValueError if the time coordinates of the
    anomaly_cube and the mean cube mismatch."""

    plugin = CalculateForecastValueFromClimateAnomaly(ignore_temporal_mismatch=False)
    if check == "reference_epoch_to_mean_bounds":
        mean_cube.coord("time").bounds = mean_cube.coord("time").bounds + (
            1 * SECONDS_IN_HOUR,
            1 * SECONDS_IN_HOUR,
        )  # Moves mean bounds outside std bounds
        with pytest.raises(
            ValueError,
            match="The reference epoch coordinate of the anomaly cube must ",
        ):
            plugin.verify_time_coords_match(
                standardized_anomaly_cube, mean_cube, std_cube
            )
    else:
        mean_cube.coord("time").points = mean_cube.coord("time").points + (
            10 * SECONDS_IN_HOUR,
        )  # Moves mean bounds outside std bounds
        with pytest.raises(
            ValueError, match="The reference epoch coordinate of the anomaly cube must "
        ):
            plugin.verify_time_coords_match(
                standardized_anomaly_cube, mean_cube, std_cube
            )
