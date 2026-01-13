# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for GoogleFilmInterpolation plugin."""

import datetime

import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import CubeList

from improver.utilities.temporal_interpolation import GoogleFilmInterpolation
from improver_tests.utilities.test_TemporalInterpolation import (
    diagnostic_cube,
    multi_time_cube,
    setup_google_film_mock,
)


@pytest.fixture
def google_film_sample_cubes():
    """Create sample cubes for GoogleFilmInterpolation testing."""
    times = [datetime.datetime(2017, 11, 1, hour) for hour in [3, 9]]
    npoints = 5
    data = np.stack(
        [
            np.ones((npoints, npoints), dtype=np.float32),
            np.ones((npoints, npoints), dtype=np.float32) * 7,
        ]
    )
    cube = multi_time_cube(times, data, "latlon")
    return cube[0], cube[1]


@pytest.fixture
def google_film_template_cube():
    """Create a template interpolated cube."""
    times = [datetime.datetime(2017, 11, 1, hour) for hour in [3, 6, 9]]
    npoints = 5
    data = np.ones((npoints, npoints), dtype=np.float32) * 4
    cube = multi_time_cube(times, data, "latlon")
    return cube[1]  # Return only the interpolated time at T+6


def test_google_film_init():
    """Test GoogleFilmInterpolation initialization."""
    model_path = "/path/to/model"
    scaling = "minmax"
    clipping_bounds = (0.0, 10.0)

    plugin = GoogleFilmInterpolation(
        model_path=model_path, scaling=scaling, clipping_bounds=clipping_bounds
    )

    assert plugin.model_path == model_path
    assert plugin.scaling == scaling
    assert plugin.clipping_bounds == clipping_bounds


def test_google_film_init_default_values():
    """Test GoogleFilmInterpolation initialization with default values."""
    model_path = "/my/model/is/here"
    plugin = GoogleFilmInterpolation(
        model_path=model_path, scaling="log10", clipping_bounds=(-10, 10)
    )

    assert plugin.model_path == model_path
    assert plugin.scaling == "log10"
    assert plugin.clipping_bounds == (-10, 10)


def test_google_film_init_clip_args():
    """Test GoogleFilmInterpolation initialization with clip_in_scaled_space and clip_to_physical_bounds."""
    plugin = GoogleFilmInterpolation(
        model_path="/mock/path",
        scaling="minmax",
        clipping_bounds=(0.0, 10.0),
        clip_in_scaled_space=False,
        clip_to_physical_bounds=True,
    )
    assert plugin.clip_in_scaled_space is False
    assert plugin.clip_to_physical_bounds is True


@pytest.mark.parametrize("scaling", ["log10", "minmax"])
def test_google_film_process_with_mock_model(
    google_film_sample_cubes,
    google_film_template_cube,
    monkeypatch,
    scaling,
):
    """Test the process method with a mock model using different scaling methods."""
    cube1, cube2 = google_film_sample_cubes

    # Use the shared setup_google_film_mock to patch GoogleFilmInterpolation
    setup_google_film_mock(monkeypatch)
    plugin = GoogleFilmInterpolation(model_path="/mock/path", scaling=scaling)
    result = plugin.process(cube1, cube2, google_film_template_cube)

    assert isinstance(result, CubeList)
    assert len(result) == 1  # One interpolated time

    # Check that output times match template times
    for result_cube, template_slice in zip(
        result, google_film_template_cube.slices_over("time")
    ):
        assert (
            result_cube.coord("time").points[0]
            == template_slice.coord("time").points[0]
        )
    if scaling == "log10":
        assert np.allclose(result[0].data, 3)
    else:
        assert np.allclose(result[0].data, 4)


def test_google_film_process_preserves_metadata(
    google_film_sample_cubes,
    google_film_template_cube,
    monkeypatch,
):
    """Test that process preserves metadata from template cube."""
    cube1, cube2 = google_film_sample_cubes

    setup_google_film_mock(monkeypatch)

    plugin = GoogleFilmInterpolation(model_path="/mock/path")
    result = plugin.process(cube1, cube2, google_film_template_cube)

    # Check that metadata matches template
    for result_cube, template_slice in zip(
        result, google_film_template_cube.slices_over("time")
    ):
        assert result_cube.metadata == template_slice.metadata
        # Check time coordinate matches
        assert (
            result_cube.coord("time").points[0]
            == template_slice.coord("time").points[0]
        )
        assert (
            result_cube.coord("forecast_period").points[0]
            == template_slice.coord("forecast_period").points[0]
        )


def test_google_film_process_with_clipping(
    google_film_sample_cubes,
    google_film_template_cube,
    monkeypatch,
):
    """Test that clipping bounds are applied correctly in scaled space."""
    cube1, cube2 = google_film_sample_cubes

    setup_google_film_mock(monkeypatch)

    # Clipping in scaled space - these bounds will be applied after scaling
    # but before reverse scaling
    clipping_bounds = (0.0, 0.5)
    plugin = GoogleFilmInterpolation(
        model_path="/mock/path", scaling="log10", clipping_bounds=clipping_bounds
    )
    result = plugin.process(cube1, cube2, google_film_template_cube)

    # After log10 scaling, interpolation, clipping to (0, 0.5), then reverse:
    # Values should be clipped to 10^0 = 1.0 and 10^0.5 = 3.16
    for result_cube in result:
        assert np.all(result_cube.data >= 1.0)  # 10^0
        assert np.all(result_cube.data <= 3.2)  # ~10^0.5


def test_google_film_process_time_fraction_calculation(
    google_film_sample_cubes, monkeypatch
):
    """Test that time fractions are calculated correctly for interpolation."""
    cube1, cube2 = google_film_sample_cubes

    # Create template with a single intermediate time (50% through)
    time = datetime.datetime(2017, 11, 1, 6)  # Midpoint between 3 and 9
    npoints = 5
    data = np.ones((npoints, npoints), dtype=np.float32) * 4
    template = diagnostic_cube(
        data=data,
        time=time,
        frt=datetime.datetime(2017, 11, 1, 3),
        spatial_grid="latlon",
    )

    captured_time_fractions = []

    # Patch with a capturing mock model
    captured_time_fractions = []

    class CapturingMockModel:
        def __call__(self, inputs):
            captured_time_fractions.append(inputs["time"][0])
            time_frac = inputs["time"][0]
            x0 = inputs["x0"][0]
            x1 = inputs["x1"][0]
            interpolated = x0 + time_frac * (x1 - x0)
            return {"image": np.expand_dims(interpolated, axis=0)}

    def mock_load_model(model_path):
        return CapturingMockModel()

    monkeypatch.setattr(
        "improver.utilities.temporal_interpolation.load_model", mock_load_model
    )

    plugin = GoogleFilmInterpolation(model_path="/mock/path")
    plugin.process(cube1, cube2, template)

    # Time should be 50% through (6 AM is midpoint between 3 AM and 9 AM)
    assert len(captured_time_fractions) == 1
    np.testing.assert_almost_equal(captured_time_fractions[0], 0.5, decimal=5)


@pytest.mark.parametrize(
    "max_batch,parallel_backend,n_workers",
    [
        (None, None, None),
        (3, None, None),
        (None, "loky", 2),
        (3, "loky", 2),
    ],
)
def test_google_film_process_multiple_times(
    google_film_sample_cubes, monkeypatch, max_batch, parallel_backend, n_workers
):
    """Test processing with multiple interpolation times, max_batch, and
    parallel options."""
    cube1, cube2 = google_film_sample_cubes

    # Create template with multiple times
    times = [datetime.datetime(2017, 11, 1, hour) for hour in [4, 5, 6, 7, 8]]
    npoints = 5
    data = np.ones((npoints, npoints), dtype=np.float32)
    template = multi_time_cube(times, data, "latlon")
    setup_google_film_mock(monkeypatch)
    plugin = GoogleFilmInterpolation(
        model_path="/mock/path",
        max_batch=max_batch,
        parallel_backend=parallel_backend,
        n_workers=n_workers,
    )
    result = plugin.process(cube1, cube2, template)

    assert isinstance(result, CubeList)
    assert len(result) == 5  # Five interpolated times

    # Check times are in correct order
    for i, result_cube in enumerate(result):
        expected_time = template[i].coord("time").points[0]
        assert result_cube.coord("time").points[0] == expected_time


@pytest.mark.parametrize(
    "max_batch,parallel_backend,n_workers",
    [
        (None, None, None),
        (3, None, None),
        (None, "loky", 2),
        (3, "loky", 2),
    ],
)
def test_google_film_process_multiple_times_and_realizations(
    google_film_sample_cubes, monkeypatch, max_batch, parallel_backend, n_workers
):
    """Test processing with multiple interpolation times, realizations, max_batch,
    and parallel options."""
    cube1, cube2 = google_film_sample_cubes
    nrealizations = 3
    npoints = cube1.shape[0]

    cube1s = CubeList([])
    cube2s = CubeList([])
    for nrealization in range(nrealizations):
        coord = DimCoord(nrealization, standard_name="realization")
        cube1_realization = cube1.copy()
        cube2_realization = cube2.copy()
        cube1_realization.add_aux_coord(coord)
        cube2_realization.add_aux_coord(coord)
        cube1s.append(cube1_realization)
        cube2s.append(cube2_realization)
    cube1 = cube1s.merge_cube()
    cube2 = cube2s.merge_cube()

    times = [datetime.datetime(2017, 11, 1, hour) for hour in [4, 5, 6, 7, 8]]
    data = np.ones((npoints, npoints), dtype=np.float32)
    template = multi_time_cube(
        times, data, "latlon", realizations=list(range(nrealizations))
    )

    setup_google_film_mock(monkeypatch)
    plugin = GoogleFilmInterpolation(
        model_path="/mock/path",
        max_batch=max_batch,
        parallel_backend=parallel_backend,
        n_workers=n_workers,
    )
    result = plugin.process(cube1, cube2, template)

    assert isinstance(result, CubeList)
    assert len(result) == 5  # Five interpolated times
    # Each result cube should have realization dimension of length 3
    for result_cube in result:
        assert result_cube.coord("realization").shape[0] == nrealizations
        assert result_cube.shape[0] == nrealizations
    # Check times are in correct order
    for i, result_cube in enumerate(result):
        expected_time = template[i].coord("time").points[0]
        assert result_cube.coord("time").points[0] == expected_time


@pytest.mark.parametrize(
    "clipping_bounds,scaling,expected_min,expected_max",
    [
        ((0.0, 0.5), "log10", 1.0, 3.2),  # 10^0 to 10^0.5
        ((0.0, 1.0), "minmax", 1.0, 7.0),  # Full range, no clipping
        ((-0.3, 0.5), "log10", 0.5, 3.2),  # 10^-0.3 to 10^0.5
    ],
)
def test_google_film_clipping_bounds_enforcement(
    google_film_sample_cubes,
    google_film_template_cube,
    monkeypatch,
    clipping_bounds,
    scaling,
    expected_min,
    expected_max,
):
    """Test that different clipping bounds are enforced correctly in scaled space."""
    cube1, cube2 = google_film_sample_cubes

    setup_google_film_mock(monkeypatch)

    plugin = GoogleFilmInterpolation(
        model_path="/mock/path", scaling=scaling, clipping_bounds=clipping_bounds
    )
    result = plugin.process(cube1, cube2, google_film_template_cube)

    for result_cube in result:
        assert result_cube.data.min() >= expected_min - 0.1  # Small tolerance
        assert result_cube.data.max() <= expected_max + 0.1  # Small tolerance


def test_google_film_process_preserves_cube_shape(
    google_film_sample_cubes,
    google_film_template_cube,
    monkeypatch,
):
    """Test that output cubes have the same spatial shape as inputs."""
    cube1, cube2 = google_film_sample_cubes

    setup_google_film_mock(monkeypatch)

    plugin = GoogleFilmInterpolation(model_path="/mock/path")
    result = plugin.process(cube1, cube2, google_film_template_cube)

    for result_cube in result:
        assert result_cube.shape == cube1.shape
        assert result_cube.shape == cube2.shape


def test_google_film_process_clip_in_scaled_space(
    google_film_sample_cubes,
    google_film_template_cube,
    monkeypatch,
):
    """Test that enabling clip_in_scaled_space enables clipping in scaled space."""
    cube1, cube2 = google_film_sample_cubes
    setup_google_film_mock(monkeypatch)
    # Set tight bounds and enable clip_in_scaled_space
    plugin = GoogleFilmInterpolation(
        model_path="/mock/path",
        scaling="minmax",
        clipping_bounds=(0.0, 0.1),
        clip_in_scaled_space=True,
        clip_to_physical_bounds=False,
    )
    result = plugin.process(cube1, cube2, google_film_template_cube)
    # When clipping is done in scaled space, the final data may not be within (0.0, 0.1)
    # after reverse scaling. However, all values should be within the physical bounds
    # (1.0 to 7.0)
    for result_cube in result:
        # The physical range is between the min and max of the input cubes (1.0 and 7.0)
        assert result_cube.data.min() >= 1.0 - 1e-6
        assert result_cube.data.max() <= 7.0 + 1e-6


def test_google_film_process_clip_to_physical_bounds(
    google_film_sample_cubes,
    google_film_template_cube,
    monkeypatch,
):
    """Test that enabling clip_to_physical_bounds applies clipping after reverse scaling."""
    cube1, cube2 = google_film_sample_cubes
    setup_google_film_mock(monkeypatch)
    # Set bounds that would be exceeded after reverse scaling
    plugin = GoogleFilmInterpolation(
        model_path="/mock/path",
        scaling="minmax",
        clipping_bounds=(0.0, 0.1),
        clip_in_scaled_space=False,
        clip_to_physical_bounds=True,
    )
    result = plugin.process(cube1, cube2, google_film_template_cube)
    # Data should be clipped to (0.0, 0.1) after reverse scaling
    for result_cube in result:
        assert result_cube.data.min() >= 0.0
        assert result_cube.data.max() <= 0.1
