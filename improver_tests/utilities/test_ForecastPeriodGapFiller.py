# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for ForecastPeriodGapFiller."""

import datetime

import numpy as np
import pytest
from iris.cube import CubeList

from improver.utilities.temporal_interpolation import ForecastPeriodGapFiller
from improver_tests.utilities.test_TemporalInterpolation import (
    diagnostic_cube,
    multi_time_cube,
    setup_google_film_mock,
)


def setup_cubes_with_gaps(hours, data_values=None, realizations=None, npoints=10):
    """Helper function to create cubelists for testing.

    Args:
        hours: List of hours (integers) for validity times.
        data_values: Optional list of data values for each time. If None,
            uses sequential values (1.0, 2.0, 3.0, ...).
        realizations: Optional list of realization indices.
        npoints: Spatial grid size (default 10x10).

    Returns:
        CubeList containing cubes for the specified times.
    """
    times = [datetime.datetime(2017, 11, 1, hour) for hour in hours]
    frt = datetime.datetime(2017, 11, 1, 0)

    if data_values is None:
        data_values = [float(i + 1) for i in range(len(hours))]

    cubelist = CubeList()
    for time, value in zip(times, data_values):
        data = np.ones((npoints, npoints), dtype=np.float32) * value
        cube = diagnostic_cube(time, frt, data, "latlon", realizations=realizations)
        cubelist.append(cube)

    return cubelist


def test_init_default_parameters():
    """Test ForecastPeriodGapFiller initializes with default parameters."""
    plugin = ForecastPeriodGapFiller(interval_in_minutes=60)

    assert plugin.interval_in_minutes == 60
    assert plugin.interpolation_method == "linear"
    assert plugin.cluster_sources_attribute is None
    assert plugin.interpolation_window_in_hours is None
    assert plugin.model_path is None
    assert plugin.scaling == "minmax"
    assert plugin.clipping_bounds == (0.0, 1.0)


def test_init_custom_parameters():
    """Test ForecastPeriodGapFiller initializes with custom parameters."""
    plugin = ForecastPeriodGapFiller(
        interval_in_minutes=120,
        interpolation_method="google_film",
        cluster_sources_attribute="cluster_sources",
        interpolation_window_in_hours=2,
        model_path="/mock/path",
        scaling="log10",
        clipping_bounds=(0.0, 5.0),
    )

    assert plugin.interval_in_minutes == 120
    assert plugin.interpolation_method == "google_film"
    assert plugin.cluster_sources_attribute == "cluster_sources"
    assert plugin.interpolation_window_in_hours == 2
    assert plugin.model_path == "/mock/path"
    assert plugin.scaling == "log10"
    assert plugin.clipping_bounds == (0.0, 5.0)


def test_process_no_gaps():
    """Test that process returns unchanged cubelist when no gaps exist."""
    times = [datetime.datetime(2017, 11, 1, hour) for hour in [3, 6, 9]]
    data = np.ones((10, 10), dtype=np.float32)
    cube = multi_time_cube(times, data, "latlon")
    cubelist = CubeList(cube.slices_over("time"))

    plugin = ForecastPeriodGapFiller(interval_in_minutes=180)
    result = plugin.process(cubelist)

    assert result.shape[0] == 3
    # Check that forecast periods match
    for orig, res in zip(cubelist, result.slices_over("time")):
        assert (
            orig.coord("forecast_period").points[0]
            == res.coord("forecast_period").points[0]
        )


def test_process_fills_single_gap():
    """Test that process fills a single missing period."""
    # Create cubes for T+3 and T+9, missing T+6
    cubelist = setup_cubes_with_gaps(hours=[3, 9], data_values=[1.0, 7.0])

    plugin = ForecastPeriodGapFiller(interval_in_minutes=180)
    result = plugin.process(cubelist)

    # Should now have 3 time points: T+3, T+6 (filled), T+9
    assert result.shape[0] == 3

    # Check forecast periods are correct
    result_periods = [
        int(round(cube.coord("forecast_period").points[0] / 3600))
        for cube in result.slices_over("time")
    ]
    assert result_periods == [3, 6, 9]

    # Check interpolated data is the midpoint (linear interpolation)
    assert np.allclose(list(result.slices_over("time"))[1].data, 4.0)


def test_process_fills_multiple_gaps():
    """Test that process fills multiple missing periods."""
    # Create cubes for T+3, T+9, and T+15, missing T+6 and T+12
    cubelist = setup_cubes_with_gaps(hours=[3, 9, 15], data_values=[1.0, 5.0, 9.0])

    plugin = ForecastPeriodGapFiller(interval_in_minutes=180)
    result = plugin.process(cubelist)

    # Should now have 5 time points: T+3, T+6, T+9, T+12, T+15
    assert result.shape[0] == 5

    # Check forecast periods are correct
    result_periods = [
        int(round(cube.coord("forecast_period").points[0] / 3600))
        for cube in result.slices_over("time")
    ]
    assert result_periods == [3, 6, 9, 12, 15]


def test_process_with_realizations():
    """Test that process works correctly with ensemble realizations."""
    times = [datetime.datetime(2017, 11, 1, 3), datetime.datetime(2017, 11, 1, 9)]
    npoints = 10
    data = np.stack(
        [
            np.ones((npoints, npoints), dtype=np.float32) * 2.0,
            np.ones((npoints, npoints), dtype=np.float32) * 8.0,
        ]
    )
    cube = multi_time_cube(times, data, "latlon", realizations=[0, 1, 2])
    cubelist = CubeList(cube.slices_over("time"))

    plugin = ForecastPeriodGapFiller(interval_in_minutes=180)
    result = plugin.process(cubelist)

    # Should have 3 time points with 3 realizations each
    assert result.shape[0] == 3
    cubes = list(result.slices_over("time"))
    assert all(cube.coords("realization") for cube in cubes)
    assert all(len(cube.coord("realization").points) == 3 for cube in cubes)


def test_process_unsorted_input():
    """Test that process handles unsorted input correctly."""
    # Create cubes in reverse order
    cubelist = setup_cubes_with_gaps(hours=[9, 3, 6], data_values=[1.0, 2.0, 3.0])

    plugin = ForecastPeriodGapFiller(interval_in_minutes=180)
    result = plugin.process(cubelist)

    # Result should be sorted by forecast period
    result_periods = [
        int(round(cube.coord("forecast_period").points[0] / 3600))
        for cube in result.slices_over("time")
    ]
    assert result_periods == [3, 6, 9]


def test_process_empty_cubelist_raises_error():
    """Test that process raises error for empty CubeList."""
    cubelist = CubeList()
    plugin = ForecastPeriodGapFiller(interval_in_minutes=60)

    with pytest.raises(ValueError, match="requires at least 2 cubes"):
        plugin.process(cubelist)


def test_process_single_cube_raises_error():
    """Test that process raises error for single cube."""
    cubelist = setup_cubes_with_gaps(hours=[3])

    plugin = ForecastPeriodGapFiller(interval_in_minutes=60)

    with pytest.raises(ValueError, match="requires at least 2 cubes"):
        plugin.process(cubelist)


def test_process_missing_forecast_period_raises_error():
    """Test that process raises error when cubes lack forecast_period coordinate."""
    times = [datetime.datetime(2017, 11, 1, hour) for hour in [3, 6]]
    data = np.ones((10, 10), dtype=np.float32)
    cube = multi_time_cube(times, data, "latlon")
    cubelist = CubeList(cube.slices_over("time"))

    # Remove forecast_period from first cube
    cubelist[0].remove_coord("forecast_period")

    plugin = ForecastPeriodGapFiller(interval_in_minutes=60)

    with pytest.raises(ValueError, match="must have a forecast_period coordinate"):
        plugin.process(cubelist)


def test_process_with_google_film_method(monkeypatch):
    """Test that process works with google_film interpolation method."""
    setup_google_film_mock(monkeypatch)

    times = [datetime.datetime(2017, 11, 1, 3), datetime.datetime(2017, 11, 1, 9)]
    npoints = 10
    data = np.stack(
        [
            np.ones((npoints, npoints), dtype=np.float32) * 1.0,
            np.ones((npoints, npoints), dtype=np.float32) * 7.0,
        ]
    )
    cube = multi_time_cube(times, data, "latlon", realizations=[0, 1])
    cubelist = CubeList(cube.slices_over("time"))

    plugin = ForecastPeriodGapFiller(
        interval_in_minutes=180,
        interpolation_method="google_film",
        model_path="/mock/path",
    )
    result = plugin.process(cubelist)

    # Should have 3 cubes including the interpolated one
    assert result.shape[0] == 3

    # Check forecast periods
    result_periods = [
        int(round(cube.coord("forecast_period").points[0] / 3600))
        for cube in result.slices_over("time")
    ]
    assert result_periods == [
        6,
        9,
        12,
    ]  # Times 3h and 9h with 6h interval results in forecast periods
    # of T+6, T+9(gap), T+12


def test_process_with_hourly_interval():
    """Test process with hourly forecast periods."""
    cubelist = setup_cubes_with_gaps(hours=[3, 6], data_values=[1.0, 4.0])

    plugin = ForecastPeriodGapFiller(interval_in_minutes=60)
    result = plugin.process(cubelist)

    # Should fill T+4 and T+5
    assert result.shape[0] == 4
    result_periods = [
        int(round(cube.coord("forecast_period").points[0] / 3600))
        for cube in result.slices_over("time")
    ]
    assert result_periods == [3, 4, 5, 6]


def test_process_maintains_metadata():
    """Test that process maintains cube metadata."""
    cubelist = setup_cubes_with_gaps(hours=[3, 9])

    # Add test attribute to all cubes
    for cube in cubelist:
        cube.attributes["test_attribute"] = "test_value"

    plugin = ForecastPeriodGapFiller(interval_in_minutes=180)
    result = plugin.process(cubelist)

    # Interpolated cube should have the same attribute
    assert "test_attribute" in list(result.slices_over("time"))[1].attributes
