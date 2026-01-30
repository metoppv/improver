# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for ForecastTrajectoryGapFiller."""

import copy
import datetime
import json

import numpy as np
import pytest
from iris.cube import CubeList

from improver.utilities.temporal_interpolation import ForecastTrajectoryGapFiller
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
    """Test ForecastTrajectoryGapFiller initializes with default parameters."""
    plugin = ForecastTrajectoryGapFiller(interval_in_minutes=60)

    assert plugin.interval_in_minutes == 60
    assert plugin.interpolation_method == "linear"
    assert plugin.cluster_sources_attribute is None
    assert plugin.interpolation_window_in_minutes is None
    assert plugin.model_path is None
    assert plugin.scaling == "minmax"
    assert plugin.clipping_bounds is None


def test_init_custom_parameters():
    """Test ForecastTrajectoryGapFiller initializes with custom parameters."""
    plugin = ForecastTrajectoryGapFiller(
        interval_in_minutes=120,
        interpolation_method="google_film",
        cluster_sources_attribute="cluster_sources",
        interpolation_window_in_minutes=120,
        model_path="/mock/path",
        scaling="log10",
        clipping_bounds=(0.0, 5.0),
    )

    assert plugin.interval_in_minutes == 120
    assert plugin.interpolation_method == "google_film"
    assert plugin.cluster_sources_attribute == "cluster_sources"
    assert plugin.interpolation_window_in_minutes == 120
    assert plugin.model_path == "/mock/path"
    assert plugin.scaling == "log10"
    assert plugin.clipping_bounds == (0.0, 5.0)


def test_process_no_gaps():
    """Test that process returns unchanged cubelist when no gaps exist."""
    times = [datetime.datetime(2017, 11, 1, hour) for hour in [3, 6, 9]]
    data = np.ones((10, 10), dtype=np.float32)
    cube = multi_time_cube(times, data, "latlon")
    cubelist = CubeList(cube.slices_over("time"))

    plugin = ForecastTrajectoryGapFiller(interval_in_minutes=180)
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

    plugin = ForecastTrajectoryGapFiller(interval_in_minutes=180)
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

    plugin = ForecastTrajectoryGapFiller(interval_in_minutes=180)
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

    plugin = ForecastTrajectoryGapFiller(interval_in_minutes=180)
    result = plugin.process(cubelist)

    # Should have 3 time points with 3 realizations each
    assert result.shape[0] == 3
    assert np.allclose(result[1].data, 5)
    cubes = list(result.slices_over("time"))
    assert all(cube.coords("realization") for cube in cubes)
    assert all(len(cube.coord("realization").points) == 3 for cube in cubes)


def test_process_unsorted_input():
    """Test that process handles unsorted input correctly."""
    # Create cubes in reverse order
    cubelist = setup_cubes_with_gaps(hours=[9, 3, 6], data_values=[1.0, 2.0, 3.0])

    plugin = ForecastTrajectoryGapFiller(interval_in_minutes=120)
    result = plugin.process(cubelist)

    # Result should be sorted by forecast period
    result_periods = [
        int(round(cube.coord("forecast_period").points[0] / 3600))
        for cube in result.slices_over("time")
    ]
    assert result_periods == [3, 5, 6, 7, 9]
    assert np.allclose(result[0].data, 2.0)  # T+3
    assert np.allclose(result[1].data, 2 + 2 / 3)  # T+5
    assert np.allclose(result[2].data, 3.0)  # T+6
    assert np.allclose(result[3].data, 2 + 1 / 3)  # T+7
    assert np.allclose(result[4].data, 1.0)  # T+9


@pytest.mark.parametrize(
    "input_case,call_method",
    [
        ("empty_cubelist", "validate"),
        ("single_cube", "validate"),
        ("single_cube_no_time", "process"),
    ],
)
def test_too_few_cubes_or_no_time_dimension_raises(input_case, call_method):
    """Test plugin raises ValueError for too few cubes or a single cube with no time/forecast_period dimension."""
    plugin = ForecastTrajectoryGapFiller(interval_in_minutes=60)

    if input_case == "empty_cubelist":
        input_data = CubeList()
    elif input_case == "single_cube":
        input_data = setup_cubes_with_gaps(hours=[3])
    elif input_case == "single_cube_no_time":
        cube = setup_cubes_with_gaps(hours=[3])[0]
        cube.remove_coord("time")
        cube.remove_coord("forecast_period")
        input_data = cube
    else:
        raise ValueError("Unknown input_case for test.")

    if call_method == "validate":
        with pytest.raises(ValueError, match="requires at least 2 cubes"):
            plugin._validate_input(input_data)
    elif call_method == "process":
        with pytest.raises(ValueError, match="requires at least 2 cubes"):
            plugin.process(input_data)
    else:
        raise ValueError("Unknown call_method for test.")


@pytest.mark.parametrize(
    "coord_to_remove",
    ["forecast_period", "time"],
)
def test_validate_input_missing_coord(coord_to_remove):
    """Test _validate_input raises an exception if a required coordinate is missing."""
    plugin = ForecastTrajectoryGapFiller(interval_in_minutes=60)
    cubes = setup_cubes_with_gaps(hours=[3, 6])
    cubes[0].remove_coord(coord_to_remove)
    cubelist = CubeList([cubes[0], cubes[1]])
    with pytest.raises(
        ValueError, match="All cubes must have forecast_period, time coordinates"
    ):
        plugin._validate_input(cubelist)


@pytest.mark.parametrize(
    "case",
    ["not_enough_forecast_periods", "not_enough_times"],
)
def test_validate_input_not_enough_uniqueness(case):
    """Test _validate_input raises if not enough unique forecast_periods or times."""
    plugin = ForecastTrajectoryGapFiller(interval_in_minutes=60)
    if case == "not_enough_forecast_periods":
        cubelist = setup_cubes_with_gaps(hours=[3, 3])
    elif case == "not_enough_times":
        cube = setup_cubes_with_gaps(hours=[3])[0]
        cubelist = CubeList([cube, copy.deepcopy(cube)])
    else:
        raise ValueError("Unknown case for test.")

    with pytest.raises(
        ValueError,
        match="requires cubes with multiple, different forecast_periods, times",
    ):
        plugin._validate_input(cubelist)


def test_validate_input_different_forecast_reference_time():
    """Test _validate_input raises an exception if forecast_reference_time differs."""
    plugin = ForecastTrajectoryGapFiller(interval_in_minutes=60)
    cube1 = setup_cubes_with_gaps(hours=[3])[0]
    cube2 = setup_cubes_with_gaps(hours=[6])[0]
    # Change the forecast_reference_time of cube2
    cube2.coord("forecast_reference_time").points = (
        cube2.coord("forecast_reference_time").points + 3600
    )
    cubelist = CubeList([cube1, cube2])
    with pytest.raises(
        ValueError,
        match="All cubes in cubelist must have the same forecast_reference_time",
    ):
        plugin._validate_input(cubelist)


@pytest.mark.parametrize(
    "parallel_backend,n_workers",
    [
        (None, None),
        ("loky", 2),
    ],
)
def test_process_with_google_film_method(monkeypatch, parallel_backend, n_workers):
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

    plugin = ForecastTrajectoryGapFiller(
        interval_in_minutes=180,
        interpolation_method="google_film",
        model_path="/mock/path",
        parallel_backend=parallel_backend,
        n_workers=n_workers,
    )
    assert plugin.parallel_backend == parallel_backend
    assert plugin.n_workers == n_workers
    result = plugin.process(cubelist)

    # Should have 3 cubes including the interpolated one
    assert result.shape[0] == 3

    # Check forecast periods
    result_periods = [
        int(round(cube.coord("forecast_period").points[0] / 3600))
        for cube in result.slices_over("time")
    ]
    # Times 3h and 9h with 6h interval results in forecast periods of
    # T+6, T+9(gap), T+12
    assert result_periods == [6, 9, 12]


def test_process_with_hourly_interval():
    """Test process with hourly forecast periods."""
    cubelist = setup_cubes_with_gaps(hours=[3, 6], data_values=[1.0, 4.0])

    plugin = ForecastTrajectoryGapFiller(interval_in_minutes=60)
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

    plugin = ForecastTrajectoryGapFiller(interval_in_minutes=180)
    result = plugin.process(cubelist)

    # Interpolated cube should have the same attribute
    assert result.attributes["test_attribute"] == "test_value"


def test_process_invalid_input_type_raises_typeerror():
    """Test that process raises TypeError for invalid input types."""
    plugin = ForecastTrajectoryGapFiller(interval_in_minutes=60)
    invalid_input = 123  # Not a Cube or CubeList

    with pytest.raises(TypeError, match="Expected Cube or CubeList, got <class 'int'>"):
        plugin.process(invalid_input)


def test_process_no_gaps_warns():
    """Test that a warning is raised when no gaps or regenerations are identified."""
    cubelist = setup_cubes_with_gaps(hours=[3, 6, 9])

    plugin = ForecastTrajectoryGapFiller(interval_in_minutes=180)
    with pytest.warns(UserWarning, match="No gaps or regenerations identified"):
        result = plugin.process(cubelist)
    assert result.shape[0] == 3


def test_process_raises_if_interval_in_minutes_not_set():
    """Test that process raises ValueError if interval_in_minutes is not set."""
    cubelist = setup_cubes_with_gaps(hours=[3, 6, 9])
    plugin = ForecastTrajectoryGapFiller(interval_in_minutes=None)
    with pytest.raises(
        ValueError,
        match="interval_in_minutes must be set to identify gaps in forecast period.",
    ):
        plugin.process(cubelist)


# fmt: off
@pytest.mark.parametrize(
    "cluster_sources,expected_exception,expected_message",
    [
        # Attribute missing
        (None, None, None),
        # Attribute as invalid JSON string
        ('{"bad_json": [}', ValueError, "Failed to parse cluster sources JSON"),
        # Attribute as non-dict JSON string
        ('["not", "a", "dict"]', ValueError, "Cluster sources attribute must be a dictionary"),
        # Attribute as dict, but sources not dict
        (json.dumps({"0": ["not_a_dict"]}), ValueError, "Sources for realization 0 must be a dictionary"),
        # Attribute as dict, but periods not list
        (json.dumps({"0": {"sourceA": "not_a_list"}}), ValueError, "Periods for source sourceA in realization 0 must be a list"),
        # Attribute as valid JSON string
        (json.dumps({"0": {"sourceA": [3, 6], "sourceB": [9]}}), None, None),
        # Attribute as valid dict
        ({"0": {"sourceA": [3, 6], "sourceB": [9]}}, None, None),
    ],
)
# fmt: on
def test_parse_cluster_sources(cluster_sources, expected_exception, expected_message):
    """Test _parse_cluster_sources via the process method with various
    cluster_sources attribute values."""
    # Setup cubes
    cubelist = setup_cubes_with_gaps(hours=[3, 6, 9])
    for cube in cubelist:
        if cluster_sources is not None:
            cube.attributes["cluster_sources"] = cluster_sources

    plugin = ForecastTrajectoryGapFiller(
        interval_in_minutes=60,
        cluster_sources_attribute="cluster_sources",
        interpolation_window_in_minutes=60,
    )

    if expected_exception:
        with pytest.raises(expected_exception, match=expected_message):
            plugin.process(cubelist)
    else:
        # Should run without error
        plugin.process(cubelist)


@pytest.mark.parametrize(
    "cluster_sources,expected_regenerated_periods",
    [
        # No transitions (single source)
        ({"0": {"sourceA": [3, 6, 9]}}, []),
        # One transition for realization 0: sourceA -> sourceB at period 6
        ({"0": {"sourceA": [3, 6], "sourceB": [9]}}, [6]),
        # Two transitions for realization 0: sourceA -> sourceB at 6, sourceB -> sourceC at 9
        ({"0": {"sourceA": [3, 6], "sourceB": [9], "sourceC": [12]}}, [6, 9]),
        # Multiple realizations, transitions for both
        (
            {
                "0": {"sourceA": [3, 6], "sourceB": [9]},
                "1": {"sourceA": [3], "sourceB": [6, 9]},
            },
            [6, 6],  # Both realizations have a transition at 6
        ),
    ],
)
def test_process_triggers_source_transitions(cluster_sources, expected_regenerated_periods):
    """Test that process triggers regeneration at source transitions."""
    # Setup cubes for periods 3, 6, 9, 12
    cubelist = setup_cubes_with_gaps(hours=[3, 6, 9, 12])
    for cube in cubelist:
        cube.attributes["cluster_sources"] = json.dumps(cluster_sources)

    plugin = ForecastTrajectoryGapFiller(
        interval_in_minutes=180,
        cluster_sources_attribute="cluster_sources",
        interpolation_window_in_minutes=180,
    )
    result = plugin.process(cubelist)

    # Extract forecast periods from result
    result_periods = [
        int(round(cube.coord("forecast_period").points[0] / 3600))
        for cube in result.slices_over("time")
    ]

    # Check that expected regenerated periods are present
    for period in expected_regenerated_periods:
        assert period in result_periods


@pytest.mark.parametrize(
    "input_type",
    [
        "single_cube",    # Single Cube with a time dimension
        "multiple_args",  # Multiple cubes as separate arguments
    ],
)
def test_process_various_input_forms(input_type):
    """Test process with different input forms produces correct forecast periods."""
    expected_periods = [3, 6, 9]
    cubes = setup_cubes_with_gaps(hours=expected_periods)
    plugin = ForecastTrajectoryGapFiller(interval_in_minutes=180)

    if input_type == "single_cube":
        input_data = cubes.merge_cube()
        result = plugin.process(input_data)
    elif input_type == "multiple_args":
        result = plugin.process(*cubes)
    else:
        raise ValueError("Unknown input_type for test.")

    assert result.shape[0] == len(expected_periods)
    result_periods = [
        int(round(c.coord("forecast_period").points[0] / 3600))
        for c in result.slices_over("time")
    ]
    assert result_periods == expected_periods
