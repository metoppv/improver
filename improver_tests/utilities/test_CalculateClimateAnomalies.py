# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the CalculateClimateAnomalies plugin within mathematical_operations.py"""

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
)


@pytest.fixture
def time_bounds():
    """Fixture for creating time bounds for a cube."""
    return {
        "mean_and_variance": (datetime(2024, 9, 16, 0, 0), datetime(2024, 10, 16, 1, 0))
    }


@pytest.fixture
def validity_time():
    """Fixture for creating validity times for a cube."""
    return {
        "diagnostic_basic": datetime(2024, 10, 16, 0, 0),
        "diagnostic_multiple": [
            datetime(2024, 10, 15, 23, 0),
            datetime(2024, 10, 16, 0, 0),
            datetime(2024, 10, 16, 1, 0),
        ],
        "mean_and_variance": datetime(2024, 10, 16, 0, 0),
    }


@pytest.fixture
def forecast_reference_time():
    """Fixture for creating forecast reference times for a cube."""
    return datetime(2024, 10, 15, 18, 0)


@pytest.fixture
def diagnostic_cube(validity_time, forecast_reference_time):
    """Fixture for creating a diagnostic cube"""
    data = np.full((1, 1, 1), 305, dtype=np.float32)
    return set_up_variable_cube(
        data=data, time=validity_time["diagnostic_basic"], frt=forecast_reference_time
    )


@pytest.fixture
def diagnostic_cube_multiple_time_points(validity_time, forecast_reference_time):
    """Fixture for creating a diagnostic cube with multiple time points"""
    cubes = CubeList()
    for time, value in zip(validity_time["diagnostic_multiple"], [300, 298, 296]):
        data = np.full((1, 1), value, dtype=np.float32)
        cubes.append(
            set_up_variable_cube(data=data, time=time, frt=forecast_reference_time)
        )
    cube = cubes.merge_cube()
    return cube


@pytest.fixture
def mean_cube(validity_time, time_bounds):
    """Fixture for creating a mean cube."""
    data = np.array([298], dtype=np.float32).reshape(1, 1, 1)
    cube = set_up_variable_cube(
        data=data,
        time=validity_time["mean_and_variance"],
        time_bounds=time_bounds["mean_and_variance"],
    )
    cell_method = iris.coords.CellMethod(method="mean", coords="time")
    cube.add_cell_method(cell_method)
    return cube


@pytest.fixture
def variance_cube(validity_time, time_bounds):
    """Fixture for creating a variance cube"""
    data = np.array([4], dtype=np.float32).reshape(1, 1, 1)
    cube = set_up_variable_cube(
        data=data,
        time=validity_time["mean_and_variance"],
        time_bounds=time_bounds["mean_and_variance"],
        units="K2",
    )
    cell_method = iris.coords.CellMethod(method="variance", coords="time")
    cube.add_cell_method(cell_method)
    return cube


@pytest.fixture
def site_cubes(validity_time, time_bounds, forecast_reference_time):
    """Fixture for creating site cubes."""
    site_cube_diagnostic_data = np.array([305], dtype=np.float32)
    site_cube_diagnostic = set_up_spot_variable_cube(
        site_cube_diagnostic_data,
        time=validity_time["diagnostic_basic"],
        frt=forecast_reference_time,
    )

    site_cube_mean_data = np.array([298], dtype=np.float32)
    site_cube_mean = set_up_spot_variable_cube(
        site_cube_mean_data,
        time=validity_time["mean_and_variance"],
        time_bounds=time_bounds["mean_and_variance"],
    )
    cell_method = iris.coords.CellMethod(method="mean", coords="time")
    site_cube_mean.add_cell_method(cell_method)

    site_cube_variance_data = np.array([4], dtype=np.float32)
    site_cube_variance = set_up_spot_variable_cube(
        site_cube_variance_data,
        time=validity_time["mean_and_variance"],
        time_bounds=time_bounds["mean_and_variance"],
        units="K2",
    )
    cell_method = iris.coords.CellMethod(method="variance", coords="time")
    site_cube_variance.add_cell_method(cell_method)
    return site_cube_diagnostic, site_cube_mean, site_cube_variance


## Testing the plugin's basic functionality
@pytest.mark.parametrize(
    "fixture_name", ["diagnostic_cube", "diagnostic_cube_multiple_time_points"]
)
def test_calculate_unstandardised_anomalies_gridded_data(
    request, fixture_name, mean_cube
):
    """Test that the plugin calculates unstandardised anomalies correctly."""

    diagnostic_cube = request.getfixturevalue(fixture_name)
    plugin = CalculateClimateAnomalies()
    if fixture_name == "diagnostic_cube":
        expected_anomalies = np.array([7], dtype=np.float32).reshape(1, 1, 1)
    elif fixture_name == "diagnostic_cube_multiple_time_points":
        expected_anomalies = np.array([2, 0, -2], dtype=np.float32).reshape(3, 1, 1)
    result = plugin.process(diagnostic_cube, mean_cube)
    np.testing.assert_allclose(result.data, expected_anomalies, rtol=1e-5)
    assert result.name() == diagnostic_cube.name() + "_anomaly"
    assert result.units == "K"
    assert "reference_epoch" in [coord.name() for coord in result.coords()]
    assert result.coord("reference_epoch").points == mean_cube.coord("time").points
    assert np.array_equal(
        result.coord("reference_epoch").bounds, mean_cube.coord("time").bounds
    )
    assert result.coord("reference_epoch").units == mean_cube.coord("time").units
    assert (
        "(CellMethod(method='anomaly', coord_names=('reference_epoch',), intervals=(), comments=())"
        in str(result.cell_methods)
    )


@pytest.mark.parametrize(
    "fixture_name", ["diagnostic_cube", "diagnostic_cube_multiple_time_points"]
)
def test_calculate_standardised_anomalies_gridded_data(
    request, fixture_name, mean_cube, variance_cube
):
    """Test that the plugin returns a cube with expected data."""
    diagnostic_cube = request.getfixturevalue(fixture_name)
    plugin = CalculateClimateAnomalies()
    if fixture_name == "diagnostic_cube":
        expected_anomalies = np.array([3.5], dtype=np.float32).reshape(1, 1, 1)
    elif fixture_name == "diagnostic_cube_multiple_time_points":
        expected_anomalies = np.array([1.0, 0.0, -1.0], dtype=np.float32).reshape(
            3, 1, 1
        )
    result = plugin.process(diagnostic_cube, mean_cube, variance_cube)
    np.testing.assert_allclose(result.data, expected_anomalies, rtol=1e-5)
    assert result.long_name == diagnostic_cube.name() + "_standard_anomaly"
    assert result.units == "1"
    assert "reference_epoch" in [coord.name() for coord in result.coords()]
    assert result.coord("reference_epoch")
    assert result.coord("reference_epoch").points == mean_cube.coord("time").points
    assert np.array_equal(
        result.coord("reference_epoch").bounds, mean_cube.coord("time").bounds
    )
    assert result.coord("reference_epoch").units == mean_cube.coord("time").units
    assert (
        "(CellMethod(method='anomaly', coord_names=('reference_epoch',), intervals=(), comments=())"
        in str(result.cell_methods)
    )


def test_calculate_unstandardised_anomalies_site_data(site_cubes):
    """Test that the plugin calculates unstandardised anomalies correctly for site data."""
    plugin = CalculateClimateAnomalies()
    site_cube_diagnostic, site_cube_mean, _ = site_cubes
    result = plugin.process(site_cube_diagnostic, site_cube_mean)
    expected_anomalies = np.array([7.0], dtype=np.float32)
    np.testing.assert_allclose(result.data, expected_anomalies, rtol=1e-5)
    assert result.name() == site_cube_diagnostic.name() + "_anomaly"
    assert result.units == "K"
    assert "reference_epoch" in [coord.name() for coord in result.coords()]
    assert result.coord("reference_epoch").points == site_cube_mean.coord("time").points
    assert np.array_equal(
        result.coord("reference_epoch").bounds, site_cube_mean.coord("time").bounds
    )
    assert result.coord("reference_epoch").units == site_cube_mean.coord("time").units
    assert (
        "(CellMethod(method='anomaly', coord_names=('reference_epoch',), intervals=(), comments=())"
        in str(result.cell_methods)
    )


def test_calculate_standardised_anomalies_site_data(site_cubes):
    """Test that the plugin calculates standardised anomalies correctly for site data."""
    plugin = CalculateClimateAnomalies()
    site_cube_diagnostic, site_cube_mean, site_cube_variance = site_cubes
    result = plugin.process(site_cube_diagnostic, site_cube_mean, site_cube_variance)
    expected_anomalies = np.array([3.5], dtype=np.float32)
    np.testing.assert_allclose(result.data, expected_anomalies, rtol=1e-5)
    assert result.long_name == site_cube_diagnostic.name() + "_standard_anomaly"
    assert result.units == "1"
    assert "reference_epoch" in [coord.name() for coord in result.coords()]
    assert result.coord("reference_epoch").points == site_cube_mean.coord("time").points
    assert np.array_equal(
        result.coord("reference_epoch").bounds, site_cube_mean.coord("time").bounds
    )
    assert result.coord("reference_epoch").units == site_cube_mean.coord("time").units
    assert (
        "(CellMethod(method='anomaly', coord_names=('reference_epoch',), intervals=(), comments=())"
        in str(result.cell_methods)
    )


def test_ignore_temporal_mismatch(diagnostic_cube, mean_cube, variance_cube):
    """Test that the plugin handles requests to ignore temporal mismatch
    between diagnostic cube and mean/variance cube.
    """
    diagnostic_cube.coord("time").points = (
        diagnostic_cube.coord("time").points + 10 * SECONDS_IN_HOUR
    )  # Moves diagnostic bounds outside mean bounds

    plugin = CalculateClimateAnomalies(ignore_temporal_mismatch=True)
    result = plugin.process(diagnostic_cube, mean_cube, variance_cube)

    assert result.long_name == diagnostic_cube.name() + "_standard_anomaly"
    assert result.units == "1"
    assert result.coord("reference_epoch")
    assert result.coord("reference_epoch").points == mean_cube.coord("time").points
    assert np.array_equal(
        result.coord("reference_epoch").bounds, mean_cube.coord("time").bounds
    )
    assert result.coord("reference_epoch").units == mean_cube.coord("time").units
    assert (
        "(CellMethod(method='anomaly', coord_names=('reference_epoch',), intervals=(), comments=())"
        in str(result.cell_methods)
    )


## Testing the plugin's internal verification checks
@pytest.mark.parametrize("error_to_check", ["mean_check", "variance_check"])
def test_error_units_mismatch(
    diagnostic_cube, mean_cube, variance_cube, error_to_check
):
    """Test that the plugin raises a ValueError if the units of the diagnostic and another cube mismatch"""
    plugin = CalculateClimateAnomalies()
    if error_to_check == "mean_check":
        mean_cube.units = "C"  # The units should be K ordinarily
        with pytest.raises(
            ValueError,
            match="The mean cube must have the same units as the diagnostic cube.",
        ):
            plugin.verify_units_match(diagnostic_cube, mean_cube, variance_cube=None)
    else:
        variance_cube.units = "K"  # The units should be K2 ordinarily
        with pytest.raises(
            ValueError, match="The variance cube must be the diagnostic cube squared."
        ):
            plugin.verify_units_match(diagnostic_cube, mean_cube, variance_cube)


def test_error_spatial_coords_mismatch_gridded_data(
    diagnostic_cube, mean_cube, variance_cube
):
    """Test that the plugin raises a ValueError if the spatial coordinates of the diagnostic cube and another cube mismatch"""
    mean_cube.coord("latitude").points = mean_cube.coord("latitude").points + 20
    mean_cube.coord("longitude").points = mean_cube.coord("longitude").points + 20
    plugin = CalculateClimateAnomalies()
    with pytest.raises(ValueError, match="The spatial coordinates must match."):
        plugin.verify_spatial_coords_match(diagnostic_cube, mean_cube, variance_cube)


@pytest.mark.parametrize(
    "error_to_raise", ["more_than_one_spot_index", "mismatching_spot_index_points"]
)
def test_error_spatial_coords_mismatch_site_data(site_cubes, mean_cube, error_to_raise):
    site_cube_diagnostic, site_cube_mean, site_cube_variance = site_cubes
    """Test that the plugin raises a ValueError if the spatial coordinates of the mean and variance cubes have different bounds."""
    plugin = CalculateClimateAnomalies()
    if error_to_raise == "more_than_one_spot_index":
        # Uses a cube without the spot_index coordinate (gridded mean_cube) to trigger the error
        with pytest.raises(
            ValueError,
            match="The cubes must all have the same spatial coordinates. Some cubes contain spot_index coordinates and some do not.",
        ):
            plugin.verify_spatial_coords_match(
                site_cube_diagnostic, mean_cube, site_cube_variance
            )
    else:
        site_cube_diagnostic.coord("spot_index").points = (
            site_cube_diagnostic.coord("spot_index").points + 1
        )
        with pytest.raises(
            ValueError,
            match="Mismatching spot_index coordinates were found on the input cubes.",
        ):
            plugin.verify_spatial_coords_match(
                site_cube_diagnostic, site_cube_mean, site_cube_variance
            )


@pytest.mark.parametrize(
    "check", ["mean_to_variance_check", "diagnostic_to_others_check"]
)
def test_error_time_coords_mismatch(diagnostic_cube, mean_cube, variance_cube, check):
    plugin = CalculateClimateAnomalies(ignore_temporal_mismatch=False)
    if check == "mean_to_variance_check":
        mean_cube.coord("time").bounds = mean_cube.coord("time").bounds + (
            1 * SECONDS_IN_HOUR,
            1 * SECONDS_IN_HOUR,
        )  # Moves mean bounds outside variance bounds
        with pytest.raises(
            ValueError,
            match="The mean and variance cubes must have compatible bounds. The following bounds were found: ",
        ):
            plugin.verify_time_coords_match(diagnostic_cube, mean_cube, variance_cube)
    else:
        diagnostic_cube.coord("time").points = (
            diagnostic_cube.coord("time").points + 10 * SECONDS_IN_HOUR
        )  # Moves diagnostic bounds outside mean bounds
        with pytest.raises(
            ValueError,
            match="The diagnostic cube's time points must fall within the bounds of the mean cube. The following was found:",
        ):
            plugin.verify_time_coords_match(diagnostic_cube, mean_cube, variance_cube)
