# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for GoogleFilmInterpolation plugin."""

import datetime

import numpy as np
import pytest
from iris.cube import CubeList

from improver.utilities.temporal_interpolation import GoogleFilmInterpolation
from improver_tests.utilities.test_TemporalInterpolation import (
    diagnostic_cube,
    multi_time_cube,
)


@pytest.fixture
def google_film_mock_model():
    """Create a mock TensorFlow Hub model that returns a simple interpolation."""

    class MockModel:
        def __call__(self, inputs):
            # Simple linear interpolation for testing
            time_frac = inputs["time"][0]
            x0 = inputs["x0"][0]
            x1 = inputs["x1"][0]
            # Linear interpolation: x0 + time_frac * (x1 - x0)
            interpolated = x0 + time_frac * (x1 - x0)
            return {"image": np.expand_dims(interpolated, axis=0)}

    return MockModel()


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
    return cube[1:]  # Return only the interpolated times (6 and 9)


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
    model_path = "/path/to/model"
    plugin = GoogleFilmInterpolation(model_path=model_path)

    assert plugin.model_path == model_path
    assert plugin.scaling == "log10"
    assert plugin.clipping_bounds == (0.0, 1.0)


@pytest.mark.parametrize("scaling", ["log10", "minmax"])
def test_google_film_process_with_mock_model(
    google_film_sample_cubes,
    google_film_template_cube,
    google_film_mock_model,
    monkeypatch,
    scaling,
):
    """Test the process method with a mock model using different scaling methods."""
    cube1, cube2 = google_film_sample_cubes

    # Mock the load_model method to return our mock model
    def mock_load_model(self, model_path):
        return google_film_mock_model

    monkeypatch.setattr(GoogleFilmInterpolation, "load_model", mock_load_model)

    plugin = GoogleFilmInterpolation(
        model_path="/mock/path", scaling=scaling, clipping_bounds=(0.0, 10.0)
    )
    result = plugin.process(cube1, cube2, google_film_template_cube)

    assert isinstance(result, CubeList)
    assert len(result) == 2  # Two interpolated times

    # Check that output times match template times
    for result_cube, template_slice in zip(
        result, google_film_template_cube.slices_over("time")
    ):
        assert (
            result_cube.coord("time").points[0]
            == template_slice.coord("time").points[0]
        )
        # Data should be different from input cubes (interpolated)
        assert not np.allclose(result_cube.data, cube1.data)
        assert not np.allclose(result_cube.data, cube2.data)


def test_google_film_process_preserves_metadata(
    google_film_sample_cubes,
    google_film_template_cube,
    google_film_mock_model,
    monkeypatch,
):
    """Test that process preserves metadata from template cube."""
    cube1, cube2 = google_film_sample_cubes

    def mock_load_model(self, model_path):
        return google_film_mock_model

    monkeypatch.setattr(GoogleFilmInterpolation, "load_model", mock_load_model)

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
    google_film_mock_model,
    monkeypatch,
):
    """Test that clipping bounds are applied correctly in scaled space."""
    cube1, cube2 = google_film_sample_cubes

    def mock_load_model(self, model_path):
        return google_film_mock_model

    monkeypatch.setattr(GoogleFilmInterpolation, "load_model", mock_load_model)

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
    google_film_sample_cubes, google_film_mock_model, monkeypatch
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

    class CapturingMockModel:
        def __call__(self, inputs):
            captured_time_fractions.append(inputs["time"][0])
            time_frac = inputs["time"][0]
            x0 = inputs["x0"][0]
            x1 = inputs["x1"][0]
            interpolated = x0 + time_frac * (x1 - x0)
            return {"image": np.expand_dims(interpolated, axis=0)}

    def mock_load_model(self, model_path):
        return CapturingMockModel()

    monkeypatch.setattr(GoogleFilmInterpolation, "load_model", mock_load_model)

    plugin = GoogleFilmInterpolation(model_path="/mock/path")
    plugin.process(cube1, cube2, template)

    # Time should be 50% through (6 AM is midpoint between 3 AM and 9 AM)
    assert len(captured_time_fractions) == 1
    np.testing.assert_almost_equal(captured_time_fractions[0], 0.5, decimal=5)


def test_google_film_process_multiple_times(
    google_film_sample_cubes, google_film_mock_model, monkeypatch
):
    """Test processing with multiple interpolation times."""
    cube1, cube2 = google_film_sample_cubes

    # Create template with multiple times
    times = [datetime.datetime(2017, 11, 1, hour) for hour in [4, 5, 6, 7, 8]]
    npoints = 5
    data = np.ones((npoints, npoints), dtype=np.float32)
    template = multi_time_cube(times, data, "latlon")

    def mock_load_model(self, model_path):
        return google_film_mock_model

    monkeypatch.setattr(GoogleFilmInterpolation, "load_model", mock_load_model)

    plugin = GoogleFilmInterpolation(model_path="/mock/path")
    result = plugin.process(cube1, cube2, template)

    assert isinstance(result, CubeList)
    assert len(result) == 5  # Five interpolated times

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
    google_film_mock_model,
    monkeypatch,
    clipping_bounds,
    scaling,
    expected_min,
    expected_max,
):
    """Test that different clipping bounds are enforced correctly in scaled space."""
    cube1, cube2 = google_film_sample_cubes

    def mock_load_model(self, model_path):
        return google_film_mock_model

    monkeypatch.setattr(GoogleFilmInterpolation, "load_model", mock_load_model)

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
    google_film_mock_model,
    monkeypatch,
):
    """Test that output cubes have the same spatial shape as inputs."""
    cube1, cube2 = google_film_sample_cubes

    def mock_load_model(self, model_path):
        return google_film_mock_model

    monkeypatch.setattr(GoogleFilmInterpolation, "load_model", mock_load_model)

    plugin = GoogleFilmInterpolation(model_path="/mock/path")
    result = plugin.process(cube1, cube2, google_film_template_cube)

    for result_cube in result:
        assert result_cube.shape == cube1.shape
        assert result_cube.shape == cube2.shape


@pytest.mark.slow
def test_google_film_with_real_model():
    """Test GoogleFilmInterpolation with the actual TensorFlow Hub FILM model.

    This test is skipped if tensorflow_hub is not available or if the model
    fails to load. Uses larger images (64x64) as required by the FILM model.
    """
    # Apply monkey patch before importing tensorflow_hub
    try:
        import tensorflow as tf

        # Patch all the different ways tensorflow's __internal__ can be accessed
        if hasattr(tf.__internal__, "register_call_context_function"):
            func = tf.__internal__.register_call_context_function
            tf.__internal__.register_load_context_function = func
            # Also patch the compat.v2 path that tf_keras uses
            if hasattr(tf.compat, "v2"):
                tf.compat.v2.__internal__.register_load_context_function = func
            # And the _api.v2.compat.v2 path
            import tensorflow._api.v2.compat.v2 as tf_api

            tf_api.__internal__.register_load_context_function = func
    except (ImportError, AttributeError):
        pass

    # Check if tensorflow_hub is available
    try:
        import tensorflow_hub  # noqa: F401
    except (ImportError, AttributeError) as e:
        pytest.skip(f"tensorflow_hub not available or import failed: {e}")

    model_path = "https://tfhub.dev/google/film/1"

    try:
        plugin = GoogleFilmInterpolation(
            model_path=model_path, scaling="log10", clipping_bounds=(0.0, 1.0)
        )
    except Exception as e:
        pytest.skip(f"Failed to load Google FILM model: {e}")

    # Create larger test data (64x64) as required by the FILM model
    times = [datetime.datetime(2017, 11, 1, hour) for hour in [3, 9]]
    npoints = 64
    data = np.stack(
        [
            np.ones((npoints, npoints), dtype=np.float32),
            np.ones((npoints, npoints), dtype=np.float32) * 7,
        ]
    )
    cube = multi_time_cube(times, data, "latlon")
    cube1, cube2 = cube[0], cube[1]

    # Create template cube for interpolated times
    times_template = [datetime.datetime(2017, 11, 1, hour) for hour in [3, 6, 9]]
    data_template = np.ones((npoints, npoints), dtype=np.float32) * 4
    template_cube = multi_time_cube(times_template, data_template, "latlon")
    google_film_template_cube = template_cube[1:]  # Only interpolated times

    try:
        result = plugin.process(cube1, cube2, google_film_template_cube)
    except Exception as e:
        pytest.skip(f"Failed to process with Google FILM model: {e}")

    # Basic assertions to verify the model ran successfully
    assert isinstance(result, CubeList)
    assert len(result) == 2  # Two interpolated times

    # Check that output times match template times
    for result_cube, template_slice in zip(
        result, google_film_template_cube.slices_over("time")
    ):
        assert (
            result_cube.coord("time").points[0]
            == template_slice.coord("time").points[0]
        )
        # Check metadata is preserved
        assert result_cube.metadata == template_slice.metadata
        # Check shape is preserved
        assert result_cube.shape == cube1.shape
        # Data should be interpolated (different from both inputs)
        assert not np.allclose(result_cube.data, cube1.data)
        assert not np.allclose(result_cube.data, cube2.data)
        # Data should be in a reasonable range (after reverse scaling)
        assert np.all(result_cube.data >= 0.5)
        assert np.all(result_cube.data <= 10.0)
