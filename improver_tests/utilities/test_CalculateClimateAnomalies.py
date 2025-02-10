# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the CalculateClimateAnomalies plugin within mathematical_operations.py"""

from datetime import datetime, timedelta
from iris.cube import CubeList
import numpy as np
import pytest

from improver.synthetic_data.set_up_test_cubes import (
    set_up_spot_variable_cube,
    set_up_variable_cube,
)
from improver.utilities.mathematical_operations import (
    CalculateClimateAnomalies,
)
from improver.constants import SECONDS_IN_HOUR

@pytest.fixture
def diagnostic_cube():
    """Fixture for creating a diagnostic cube"""
    data = np.full((1, 1, 1), 305, dtype=np.float32) 
    time_bounds = (datetime(2024, 10, 16, 0, 0), datetime(2024, 10, 16, 1, 0))
    time = datetime(2024, 10, 16, 1, 0)
    return set_up_variable_cube(data=data, time_bounds=time_bounds, time=time)

@pytest.fixture
def diagnostic_cube_multiple_time_points():
    """Fixture for creating a diagnostic cube with multiple time points"""
    times = [
        datetime(2024, 10, 16, 0, 0),
        datetime(2024, 10, 16, 1, 0),
        datetime(2024, 10, 16, 2, 0),
    ]
    period = 1  # hour
    cubes = CubeList()
    for time, value in zip(times, [300, 298, 296]):
        data = np.full((1, 1), value, dtype=np.float32)
        time_bounds = (time - timedelta(hours=period), time)
        cubes.append(set_up_variable_cube(data=data, time=time, time_bounds=time_bounds))

    return cubes.merge_cube()

@pytest.fixture
def mean_cube():
    """Fixture for creating a mean cube."""
    data = np.mean([300, 298, 296], dtype=np.float32).reshape(1, 1, 1)
    time_bounds = (datetime(2024, 10, 16, 0, 0), datetime(2024, 10, 16, 3, 0))
    time = datetime(2024, 10, 16, 0, 0)
    return set_up_variable_cube(data=data, time=time, time_bounds=time_bounds)

@pytest.fixture
def variance_cube():
    """Fixture for creating a variance cube"""
    data = np.array(np.var([300, 298, 296]), dtype=np.float32).reshape(1, 1, 1)
    time_bounds = (datetime(2024, 10, 16, 0, 0), datetime(2024, 10, 16, 3, 0))
    time = datetime(2024, 10, 16, 0, 0)
    units = "K2"
    return set_up_variable_cube(data=data, time=time, time_bounds=time_bounds, units=units)


@pytest.fixture
def site_cubes():
    """Fixture for creating site cubes."""
    site_cube_diagnostic_data = np.array(
        [305], dtype=np.float32
    )
    site_cube_diagnostic = set_up_spot_variable_cube(site_cube_diagnostic_data, 
                                                     time_bounds=(datetime(2024, 10, 16, 0, 0), 
                                                                  datetime(2024, 10, 16, 1, 0)),
                                                     time=datetime(2024, 10, 16, 1, 0))
    
    site_cube_mean_data = np.array([np.mean(
        [300, 298, 296])], dtype=np.float32
    )
    site_cube_mean = set_up_spot_variable_cube(site_cube_mean_data,
                                               time_bounds=(datetime(2024, 10, 16, 0, 0),
                                                            datetime(2024, 10, 16, 3, 0)),
                                                time=datetime(2024, 10, 16, 0, 0))
    site_cube_variance_data = np.array([np.var(
        [300, 298, 296])], dtype=np.float32
    )
    site_cube_variance = set_up_spot_variable_cube(site_cube_variance_data, 
                                                   time_bounds=(datetime(2024, 10, 16, 0, 0),
                                                                datetime(2024, 10, 16, 3, 0)),
                                                    time=datetime(2024, 10, 16, 0, 0),
                                                   units="K2")
    return site_cube_diagnostic, site_cube_mean, site_cube_variance


## Testing the plugin's basic functionality
@pytest.mark.parametrize("fixture_name", ["diagnostic_cube", "diagnostic_cube_multiple_time_points"])
def test_calculate_unstandardised_anomalies_gridded_data(request, fixture_name, mean_cube):
    """Test that the plugin calculates unstandardised anomalies correctly."""

    diagnostic_cube = request.getfixturevalue(fixture_name)
    plugin = CalculateClimateAnomalies()
    if fixture_name == "diagnostic_cube":
        expected_anomalies = np.array([7], dtype=np.float32).reshape(1, 1, 1)
    elif fixture_name == "diagnostic_cube_multiple_time_points":
        expected_anomalies = np.array([2, 0, -2], dtype=np.float32).reshape(3, 1, 1)

    result = plugin.process(diagnostic_cube, mean_cube)
    np.testing.assert_allclose(result.data, expected_anomalies, rtol=1e-5)

@pytest.mark.parametrize("fixture_name", ["diagnostic_cube", "diagnostic_cube_multiple_time_points"])
def test_calculate_standardised_anomalies_gridded_data(request, fixture_name, mean_cube, variance_cube):
    """Test that the plugin returns a cube with expected data."""
    diagnostic_cube = request.getfixturevalue(fixture_name)
    plugin = CalculateClimateAnomalies()
    if fixture_name == "diagnostic_cube":
        expected_anomalies = np.array([4.286607], dtype=np.float32).reshape(1, 1, 1)
    elif fixture_name == "diagnostic_cube_multiple_time_points":
        expected_anomalies = np.array([1.224745, 0.0, -1.224745], dtype=np.float32).reshape(3, 1, 1)
    
    result = plugin.process(diagnostic_cube, mean_cube, variance_cube)
    np.testing.assert_allclose(result.data, expected_anomalies, rtol=1e-5)

def test_calculate_unstandardised_anomalies_site_data(site_cubes):
    """Test that the plugin calculates unstandardised anomalies correctly for site data."""
    plugin = CalculateClimateAnomalies()
    site_cube_diagnostic, site_cube_mean, _ = site_cubes
    result = plugin.process(site_cube_diagnostic, site_cube_mean)
    expected_anomalies = np.array(
        [7.0], dtype=np.float32
    )
    np.testing.assert_allclose(result.data, expected_anomalies, rtol=1e-5)

def test_calculate_standardised_anomalies_site_data(site_cubes):
    """Test that the plugin calculates standardised anomalies correctly for site data."""
    plugin = CalculateClimateAnomalies()
    result = plugin.process(*site_cubes)
    expected_anomalies = np.array(
        [4.286607049870562], dtype=np.float32
    )
    np.testing.assert_allclose(result.data, expected_anomalies, rtol=1e-5)

## Testing the plugin's internal verification checks
def test_verify_units_match(diagnostic_cube, mean_cube, variance_cube):
    """Test that the plugin verifies the units of the diagnostic and mean cubes match."""
    plugin = CalculateClimateAnomalies()
    plugin.verify_units_match(diagnostic_cube, mean_cube, variance_cube)
    assert True

@pytest.mark.parametrize("error_to_check", ["mean_check", "variance_check"])
def test_error_units_mismatch(diagnostic_cube, mean_cube, variance_cube, error_to_check):
    """Test that the plugin raises a ValueError if the units of the diagnostic and another cube mismatch"""
    plugin = CalculateClimateAnomalies()
    if error_to_check == "mean_check":
        mean_cube.units = "C" # The units should be K ordinarily
        with pytest.raises(ValueError, match="The mean cube must have the same units as the diagnostic cube."):
            plugin.verify_units_match(diagnostic_cube, mean_cube, variance_cube=None)
    else:
        variance_cube.units = "K" # The units should be K2 ordinarily
        with pytest.raises(ValueError, match="The variance cube must be the diagnostic cube squared."):
            plugin.verify_units_match(diagnostic_cube, mean_cube, variance_cube)

def test_verify_spatial_coords_match_gridded_data(diagnostic_cube, mean_cube, variance_cube):
    """Test that the plugin verifies the spatial coordinates of the diagnostic cube and another cube match."""
    plugin = CalculateClimateAnomalies()
    plugin.verify_spatial_coords_match(diagnostic_cube, mean_cube, variance_cube)

def test_verify_spatial_coords_match_site_data(site_cubes):
    """Test that the plugin verifies the spatial coordinates of the diagnostic cube and another cube match."""
    plugin = CalculateClimateAnomalies()
    plugin.verify_spatial_coords_match(*site_cubes)

def test_error_spatial_coords_mismatch_gridded_data(diagnostic_cube, mean_cube, variance_cube):
    """Test that the plugin raises a ValueError if the spatial coordinates of the diagnostic cube and another cube mismatch"""
    plugin = CalculateClimateAnomalies()
    with pytest.raises(ValueError, match = "The spatial coordinates must match."):
        mean_cube.coord("latitude").points = mean_cube.coord("latitude").points + 20
        mean_cube.coord("longitude").points = mean_cube.coord("longitude").points + 20
        plugin.verify_spatial_coords_match(diagnostic_cube, mean_cube, variance_cube)

@pytest.mark.parametrize("error_to_raise", ["more_than_one_spot_index", "mismatching_spot_index_points"])
def test_error_spatial_coords_mismatch_site_data(site_cubes, error_to_raise):
    site_cube_diagnostic, site_cube_mean, site_cube_variance = site_cubes
    """Test that the plugin raises a ValueError if the spatial coordinates of the mean and variance cubes have different bounds."""
    plugin = CalculateClimateAnomalies()
    if error_to_raise == "more_than_one_spot_index":
        # Remove the spot_index coordinate from one of the cubes to trigger the error
        site_cube_mean.remove_coord("spot_index")
        with pytest.raises(ValueError, match = "The cubes must all have the same spatial coordinates. Some cubes contain spot_index coordinates and some do not."):
            plugin.verify_spatial_coords_match(
                site_cube_diagnostic, site_cube_mean, site_cube_variance)
    else:
        site_cube_mean.coord("spot_index").points = site_cube_diagnostic.coord("spot_index").points + 1
        site_cube_variance.coord("spot_index").points = site_cube_diagnostic.coord("spot_index").points + 1
        with pytest.raises(ValueError, match = "Mismatching spot_index coordinates were found on the input cubes."):
            plugin.verify_spatial_coords_match(
                site_cube_diagnostic, site_cube_mean, site_cube_variance)
            

@pytest.mark.parametrize("check", ["ignore_temporal_mismatch_true_false", "ignore_temporal_mismatch_true"])
def test_verify_time_coords_match(diagnostic_cube, mean_cube, variance_cube, check):
    """Test that the verify_time_coords_match() function handles requests to ignore temporal mismatch between diagnostic cube and mean/variance cube."""
    if check == "ignore_temporal_mismatch_true_false":
        plugin = CalculateClimateAnomalies(ignore_temporal_mismatch=False)
        plugin.verify_time_coords_match(diagnostic_cube, mean_cube, variance_cube)
    else:
        plugin = CalculateClimateAnomalies(ignore_temporal_mismatch=True)
        plugin.verify_time_coords_match(diagnostic_cube, mean_cube, variance_cube)

@pytest.mark.parametrize("check", ["mean_to_variance_check", "diagnostic_to_others_check"])
def test_error_time_coords_mismatch(diagnostic_cube, mean_cube, variance_cube, check):
    if check == "mean_to_variance_check":
        plugin = CalculateClimateAnomalies(ignore_temporal_mismatch=False)
        mean_cube.coord("time").bounds = mean_cube.coord("time").bounds + 1 # Move mean bounds outside variance bounds
        with pytest.raises(ValueError, match="The mean and variance cubes must have compatible bounds. The following bounds were found:"):
            plugin.verify_time_coords_match(diagnostic_cube, mean_cube, variance_cube)
    else:
        plugin = CalculateClimateAnomalies(ignore_temporal_mismatch=False)
        diagnostic_cube.coord("time").points = diagnostic_cube.coord("time").points -2*SECONDS_IN_HOUR # Move diagnostic bounds outside mean bounds
        with pytest.raises(ValueError, match="The diagnostic cube's time points must fall within the bounds of the mean cube. The following was found:"):
            plugin.verify_time_coords_match(diagnostic_cube, mean_cube, variance_cube)

@pytest.mark.parametrize("check", ["unstandardised", "standardised"])
def test_metadata_of_cubes_correct(diagnostic_cube, mean_cube, variance_cube, check):
    """Test that the metadata on the output cube is as expected when no variance cube is provided"""
    plugin = CalculateClimateAnomalies()
    if check == "unstandardised":
        result = plugin.process(diagnostic_cube, mean_cube, variance_cube=None)
        assert result.name() == diagnostic_cube.name() + "_anomaly"
        assert result.units == "K"
    if check == "standardised":
        result = plugin.process(diagnostic_cube, mean_cube, variance_cube)
        assert result.long_name == diagnostic_cube.name() + "_standard_anomaly"
        assert result.units == "1"
    assert "reference_epoch" in [coord.name() for coord in result.coords()]
    assert "(CellMethod(method='anomaly', coord_names=('reference_epoch',), intervals=(), comments=())" in str(result.cell_methods)