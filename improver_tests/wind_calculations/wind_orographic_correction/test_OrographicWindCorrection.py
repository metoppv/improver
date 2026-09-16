# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the wind orographic correction module."""

import numpy as np
import pytest
from scipy.special import kv

import improver.wind_calculations.wind_orographic_correction as wind_orographic_correction
from improver.synthetic_data.set_up_test_cubes import (
    set_up_variable_cube,
)


def _make_xy_cube(data: np.ndarray, name: str, units: str):
    return set_up_variable_cube(
        data.astype(np.float32),
        name=name,
        units=units,
        spatial_grid="equalarea",
        domain_corner=(-1036000, -1158000),
        x_grid_spacing=2000,
        y_grid_spacing=2000,
    )


def _make_wind_profile_cube(
    heights: np.ndarray,
    values_at_heights: np.ndarray,
    shape: tuple[int, int] = (2, 2),
):
    data = np.stack(
        [
            np.full(shape, value, dtype=np.float32)
            for value in np.asarray(values_at_heights, dtype=np.float32)
        ],
        axis=0,
    )
    return set_up_variable_cube(
        data,
        name="wind_speed",
        units="m s-1",
        spatial_grid="equalarea",
        domain_corner=(-1036000, -1158000),
        x_grid_spacing=2000,
        y_grid_spacing=2000,
        vertical_levels=np.asarray(heights, dtype=np.float32).tolist(),
        height=True,
    )


def _make_ancillary_cubes(
    shape: tuple[int, int] = (2, 2),
    high_res_orog_value: float = 120.0,
    model_orog_value: float = 100.0,
    model_orog_stddev_value: float = 20.0,
    silhouette_roughness_value: float = 0.2,
):
    high_res_orog = _make_xy_cube(
        np.full(shape, high_res_orog_value, dtype=np.float32),
        "surface_altitude",
        "m",
    )
    model_orog = _make_xy_cube(
        np.full(shape, model_orog_value, dtype=np.float32),
        "surface_altitude",
        "m",
    )
    model_orog_stddev = _make_xy_cube(
        np.full(shape, model_orog_stddev_value, dtype=np.float32),
        "standard_deviation_of_height_in_grid_cell",
        "m",
    )
    silhouette_roughness = _make_xy_cube(
        np.full(shape, silhouette_roughness_value, dtype=np.float32),
        "silhouette_roughness",
        "1",
    )
    return high_res_orog, model_orog, model_orog_stddev, silhouette_roughness


def _make_plugin(
    shape: tuple[int, int] = (2, 2),
    high_res_orog_value: float = 120.0,
    model_orog_value: float = 100.0,
    model_orog_stddev_value: float = 20.0,
    silhouette_roughness_value: float = 0.2,
):
    high_res_orog, model_orog, model_orog_stddev, silhouette_roughness = (
        _make_ancillary_cubes(
            shape=shape,
            high_res_orog_value=high_res_orog_value,
            model_orog_value=model_orog_value,
            model_orog_stddev_value=model_orog_stddev_value,
            silhouette_roughness_value=silhouette_roughness_value,
        )
    )
    return wind_orographic_correction.OrographicWindCorrection(
        high_res_orog_cube=high_res_orog,
        model_orog_cube=model_orog,
        model_orog_stddev_cube=model_orog_stddev,
        model_silhouette_roughness_cube=silhouette_roughness,
    )


@pytest.fixture
def standard_heights() -> np.ndarray:
    return np.array([10.0, 20.0, 30.0], dtype=np.float32)


@pytest.fixture
def standard_profile_values() -> np.ndarray:
    return np.array([2.0, 4.0, 6.0], dtype=np.float32)


@pytest.fixture
def standard_wind_profile_cube(
    standard_heights: np.ndarray,
    standard_profile_values: np.ndarray,
):
    return _make_wind_profile_cube(
        heights=standard_heights,
        values_at_heights=standard_profile_values,
    )


@pytest.fixture
def flat_orography_plugin():
    return _make_plugin(
        shape=(2, 2),
        high_res_orog_value=100.0,
        model_orog_value=100.0,
    )


def test_plugin_initialises_with_ancillary_cubes():
    """Plugin construction should store the ancillary cubes it receives."""
    high_res_orog = _make_xy_cube(
        np.full((2, 2), 120.0, dtype=np.float32),
        "surface_altitude",
        "m",
    )
    model_orog = _make_xy_cube(
        np.full((2, 2), 100.0, dtype=np.float32),
        "surface_altitude",
        "m",
    )
    model_orog_stddev = _make_xy_cube(
        np.full((2, 2), 20.0, dtype=np.float32),
        "standard_deviation_of_height_in_grid_cell",
        "m",
    )
    silhouette_roughness = _make_xy_cube(
        np.full((2, 2), 0.2, dtype=np.float32),
        "silhouette_roughness",
        "1",
    )

    plugin = wind_orographic_correction.OrographicWindCorrection(
        high_res_orog_cube=high_res_orog,
        model_orog_cube=model_orog,
        model_orog_stddev_cube=model_orog_stddev,
        model_silhouette_roughness_cube=silhouette_roughness,
    )

    assert plugin.high_res_orog_cube is high_res_orog
    assert plugin.model_orog_cube is model_orog
    assert plugin.model_orog_stddev_cube is model_orog_stddev
    assert plugin.model_silhouette_roughness_cube is silhouette_roughness


def test_process_returns_corrected_cube_on_default_heights(
    standard_wind_profile_cube,
):
    """Process should preserve shape, metadata, and finite winds on defaults."""
    plugin = _make_plugin()
    wind = standard_wind_profile_cube

    result = plugin.process(wind)

    assert result.shape == wind.shape
    assert result.name() == wind.name()
    assert str(result.units) == str(wind.units)
    assert np.isfinite(result.data).all()
    assert np.all(result.data > 0.0)
    np.testing.assert_allclose(
        result.coord("height").points, wind.coord("height").points
    )


def test_process_returns_corrected_cube_on_explicit_heights(
    flat_orography_plugin,
    standard_wind_profile_cube,
):
    """Process should correct winds on caller-supplied target heights."""
    shape = (2, 2)
    plugin = flat_orography_plugin
    wind = standard_wind_profile_cube

    result = plugin.process(wind, target_height_levels=[15.0, 25.0])

    expected = np.stack(
        [
            np.full(shape, 3.0, dtype=np.float32),
            np.full(shape, 5.0, dtype=np.float32),
        ],
        axis=0,
    )
    np.testing.assert_allclose(result.data, expected)
    np.testing.assert_allclose(
        result.coord("height").points,
        np.array([15.0, 25.0], dtype=np.float32),
    )


def test_process_checks_all_required_horizontal_grids():
    """Process should reject ancillary cubes that do not share the wind grid."""
    plugin = _make_plugin(shape=(2, 2))
    wind = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
        shape=(3, 2),
    )

    with pytest.raises(ValueError, match="horizontal shape"):
        plugin.process(wind)


def test_process_converts_ancillary_orography_units_before_use():
    """Process should normalise orography-related ancillaries to metres in place."""
    shape = (2, 2)
    high_res_orog = _make_xy_cube(
        np.full(shape, 0.12, dtype=np.float32),
        "surface_altitude",
        "km",
    )
    model_orog = _make_xy_cube(
        np.full(shape, 0.10, dtype=np.float32),
        "surface_altitude",
        "km",
    )
    model_orog_stddev = _make_xy_cube(
        np.full(shape, 0.02, dtype=np.float32),
        "standard_deviation_of_height_in_grid_cell",
        "km",
    )
    silhouette_roughness = _make_xy_cube(
        np.full(shape, 0.2, dtype=np.float32),
        "silhouette_roughness",
        "1",
    )

    plugin = wind_orographic_correction.OrographicWindCorrection(
        high_res_orog_cube=high_res_orog,
        model_orog_cube=model_orog,
        model_orog_stddev_cube=model_orog_stddev,
        model_silhouette_roughness_cube=silhouette_roughness,
    )
    wind = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
    )

    assert str(plugin.high_res_orog_cube.units) == "km"
    assert str(plugin.model_orog_cube.units) == "km"
    assert str(plugin.model_orog_stddev_cube.units) == "km"

    plugin.process(wind)

    assert str(plugin.high_res_orog_cube.units) == "m"
    assert str(plugin.model_orog_cube.units) == "m"
    assert str(plugin.model_orog_stddev_cube.units) == "m"


def test_process_does_not_mutate_input_wind_data(standard_wind_profile_cube):
    """Process should leave input wind data and heights unchanged."""
    plugin = _make_plugin()
    wind = standard_wind_profile_cube
    data_before = np.ma.asarray(wind.data).copy()
    heights_before = wind.coord("height").points.copy()

    plugin.process(wind)

    np.testing.assert_allclose(np.ma.asarray(wind.data), data_before)
    np.testing.assert_allclose(wind.coord("height").points, heights_before)


def test_process_preserves_masked_target_winds_in_output(
    monkeypatch, flat_orography_plugin
):
    """Masked target winds should remain masked in the corrected output."""
    wind = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
    )

    masked_target_winds = np.ma.array(
        np.array(
            [
                [[3.0, 3.0], [3.0, 3.0]],
                [[5.0, 5.0], [5.0, 5.0]],
            ],
            dtype=np.float32,
        ),
        mask=np.array(
            [
                [[False, True], [False, False]],
                [[False, True], [False, False]],
            ]
        ),
    )

    monkeypatch.setattr(
        wind_orographic_correction,
        "get_target_wind_speeds",
        lambda *_args, **_kwargs: masked_target_winds,
    )
    monkeypatch.setattr(
        wind_orographic_correction,
        "calculate_speed_up_factor",
        lambda *args, **kwargs: np.ones((2, 2, 2), dtype=np.float32),
    )

    result = flat_orography_plugin.process(wind, target_height_levels=[15.0, 25.0])

    np.testing.assert_array_equal(
        np.ma.getmaskarray(result.data),
        np.ma.getmaskarray(masked_target_winds),
    )


def test_process_combines_target_winds_with_speed_up_factor(monkeypatch):
    """Process should multiply interpolated target winds by the speed-up factor."""
    shape = (2, 2)
    plugin = _make_plugin(
        shape=shape,
        high_res_orog_value=100.0,
        model_orog_value=100.0,
    )
    wind = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
    )

    mocked_speed_up = np.array(
        [
            [[2.0, 1.0], [0.5, 1.0]],
            [[1.5, 0.5], [1.0, 2.0]],
        ],
        dtype=np.float32,
    )

    def _fake_speed_up_factor(*_args, **_kwargs):
        return mocked_speed_up

    monkeypatch.setattr(
        wind_orographic_correction,
        "calculate_speed_up_factor",
        _fake_speed_up_factor,
    )

    result = plugin.process(wind, target_height_levels=[15.0, 25.0])

    target_winds = np.stack(
        [
            np.full(shape, 3.0, dtype=np.float32),
            np.full(shape, 5.0, dtype=np.float32),
        ],
        axis=0,
    )
    expected = target_winds * mocked_speed_up

    np.testing.assert_allclose(result.data, expected)


def test_get_target_height_levels_defaults_to_input_cube_heights():
    """Target heights should default to the wind cube's own height coordinate."""
    wind = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
    )

    result = wind_orographic_correction.get_target_height_levels(
        wind,
        target_height_levels=None,
    )

    np.testing.assert_allclose(
        result,
        np.array([10.0, 20.0, 30.0], dtype=np.float32),
    )


def test_get_target_height_levels_sorts_requested_values():
    """Target heights should be returned in ascending order."""
    wind = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
    )

    result = wind_orographic_correction.get_target_height_levels(
        wind,
        target_height_levels=[30.0, 10.0, 20.0],
    )

    np.testing.assert_allclose(
        result,
        np.array([10.0, 20.0, 30.0], dtype=np.float32),
    )


def test_get_target_height_levels_casts_to_float_array():
    """Target heights should be converted to a floating-point ndarray."""
    wind = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
    )

    result = wind_orographic_correction.get_target_height_levels(
        wind,
        target_height_levels=[30, 10, 20],
    )

    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.floating)


def test_prepare_target_wind_speeds_returns_mask_and_nan_values():
    """Preparation helper should split mask information from values."""
    target_winds = np.ma.array(
        [[[1.0, 2.0], [3.0, 4.0]]],
        mask=[[[False, True], [False, False]]],
        dtype=np.float32,
    )

    mask, values = wind_orographic_correction.prepare_target_wind_speeds(target_winds)

    np.testing.assert_array_equal(
        mask,
        np.array([[[False, True], [False, False]]]),
    )
    np.testing.assert_allclose(values[0, 0, 0], 1.0)
    assert np.isnan(values[0, 0, 1])


def test_prepare_target_wind_speeds_handles_plain_ndarrays():
    """Preparation helper should also accept unmasked arrays."""
    target_winds = np.array(
        [[[1.0, 2.0], [3.0, 4.0]]],
        dtype=np.float32,
    )

    mask, values = wind_orographic_correction.prepare_target_wind_speeds(target_winds)

    np.testing.assert_array_equal(mask, np.zeros_like(target_winds, dtype=bool))
    np.testing.assert_allclose(values, target_winds.astype(float))


def test_get_target_wind_speeds_returns_direct_copy_when_heights_match(monkeypatch):
    """Matching target heights should avoid interpolation and return a copy."""
    wind = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
    )

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError(
            "Spline fitting should not be called for matching heights."
        )

    monkeypatch.setattr(
        wind_orographic_correction,
        "fit_spline_wind_profile",
        _raise_if_called,
    )

    result = wind_orographic_correction.get_target_wind_speeds(
        wind,
        target_heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
    )

    np.testing.assert_allclose(result, wind.data)
    assert result is not wind.data

    original_value = result[0, 0, 0]
    wind.data[0, 0, 0] = 999.0
    np.testing.assert_allclose(result[0, 0, 0], original_value)


def test_get_target_wind_speeds_interpolates_when_heights_differ(
    standard_wind_profile_cube,
):
    """Different target heights should be produced by spline interpolation."""
    wind = standard_wind_profile_cube

    result = wind_orographic_correction.get_target_wind_speeds(
        wind,
        target_heights=np.array([15.0, 25.0], dtype=np.float32),
    )

    expected = np.stack(
        [
            np.full((2, 2), 3.0, dtype=np.float32),
            np.full((2, 2), 5.0, dtype=np.float32),
        ],
        axis=0,
    )
    np.testing.assert_allclose(result, expected)


def test_get_target_wind_speeds_reuses_supplied_spline_when_present(monkeypatch):
    """Provided spline objects should be reused rather than refit."""
    wind = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
    )

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("fit_spline_wind_profile should not be called.")

    monkeypatch.setattr(
        wind_orographic_correction,
        "fit_spline_wind_profile",
        _raise_if_called,
    )

    expected = np.array(
        [
            np.full((2, 2), 11.0, dtype=np.float32),
            np.full((2, 2), 13.0, dtype=np.float32),
        ],
        dtype=np.float32,
    )

    class _FakeSpline:
        def __call__(self, heights):
            np.testing.assert_allclose(
                heights, np.array([15.0, 25.0], dtype=np.float32)
            )
            return expected

    result = wind_orographic_correction.get_target_wind_speeds(
        wind,
        target_heights=np.array([15.0, 25.0], dtype=np.float32),
        spline=_FakeSpline(),
    )

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "target_heights",
    [
        np.array([15.0], dtype=np.float32),
        np.array([15.0, 25.0], dtype=np.float32),
    ],
)
def test_get_target_wind_speeds_rejects_single_level_interpolation(target_heights):
    """Single-level inputs should not be interpolated to different heights."""
    wind = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
    )[0]

    with pytest.raises(ValueError, match="Cannot interpolate wind_profile_cube"):
        wind_orographic_correction.get_target_wind_speeds(
            wind,
            target_heights=target_heights,
        )


def test_get_target_wind_speeds_rejects_height_coordinate_mismatch(monkeypatch):
    """Target wind data should agree with the height coordinate length."""
    wind = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
    )[0]

    monkeypatch.setattr(
        wind_orographic_correction,
        "get_height_levels_from_cube",
        lambda _cube: np.array([10.0, 20.0, 30.0], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="data first dimension must match"):
        wind_orographic_correction.get_target_wind_speeds(
            wind,
            target_heights=np.array([10.0, 20.0], dtype=np.float32),
        )


def test_create_corrected_wind_speed_cube_sets_target_heights():
    """Corrected output cube should carry the requested height points."""
    template = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0], dtype=np.float32),
    )
    corrected = np.array(
        [
            np.full((2, 2), 3.0, dtype=np.float32),
            np.full((2, 2), 5.0, dtype=np.float32),
        ],
        dtype=np.float32,
    )
    target_heights = np.array([15.0, 25.0], dtype=np.float32)

    result = wind_orographic_correction.create_corrected_wind_speed_cube(
        template,
        corrected,
        target_heights,
    )

    np.testing.assert_allclose(result.coord("height").points, target_heights)


def test_create_corrected_wind_speed_cube_preserves_metadata_template():
    """Corrected output should inherit metadata from the template cube."""
    template = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0], dtype=np.float32),
    )
    corrected = np.array(
        [
            np.full((2, 2), 3.0, dtype=np.float32),
            np.full((2, 2), 5.0, dtype=np.float32),
        ],
        dtype=np.float32,
    )

    result = wind_orographic_correction.create_corrected_wind_speed_cube(
        template,
        corrected,
        target_heights=np.array([15.0, 25.0], dtype=np.float32),
    )

    assert result.name() == template.name()
    assert str(result.units) == str(template.units)
    assert result.attributes == template.attributes
    assert result.cell_methods == template.cell_methods
    np.testing.assert_allclose(
        result.coord(axis="x").points,
        template.coord(axis="x").points,
    )
    np.testing.assert_allclose(
        result.coord(axis="y").points,
        template.coord(axis="y").points,
    )
    assert str(result.coord("height").units) == "m"
    assert result.coord("height").bounds is None


def test_create_corrected_wind_speed_cube_rejects_height_count_mismatch():
    """Cube creation should fail when data slices and target heights disagree."""
    template = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0], dtype=np.float32),
    )
    corrected = np.array(
        [
            np.full((2, 2), 3.0, dtype=np.float32),
            np.full((2, 2), 5.0, dtype=np.float32),
        ],
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="must match the number of target heights"):
        wind_orographic_correction.create_corrected_wind_speed_cube(
            template,
            corrected,
            target_heights=np.array([15.0], dtype=np.float32),
        )


def test_create_corrected_wind_speed_cube_handles_two_dimensional_templates():
    """2D template cubes should still produce a valid corrected output."""
    template_2d = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0], dtype=np.float32),
    )[0]
    corrected = np.array(
        [
            np.full((2, 2), 7.0, dtype=np.float32),
        ],
        dtype=np.float32,
    )

    result = wind_orographic_correction.create_corrected_wind_speed_cube(
        template_2d,
        corrected,
        target_heights=np.array([15.0], dtype=np.float32),
    )

    np.testing.assert_allclose(result.data, corrected[0])
    np.testing.assert_allclose(
        result.coord("height").points, np.array([15.0], dtype=np.float32)
    )


def test_get_height_levels_from_cube_converts_coordinate_units():
    """Height extraction should convert the coordinate to metres."""
    wind = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
    )
    wind.coord("height").convert_units("km")

    result = wind_orographic_correction.get_height_levels_from_cube(wind)

    np.testing.assert_allclose(result, np.array([10.0, 20.0, 30.0], dtype=np.float32))


def test_get_height_levels_from_cube_returns_1d_array():
    """Height extraction should always return a one-dimensional array."""
    wind_2d = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0], dtype=np.float32),
    )[0]

    result = wind_orographic_correction.get_height_levels_from_cube(wind_2d)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    np.testing.assert_allclose(result, np.array([10.0], dtype=np.float32))


def test_fit_spline_wind_profile_rejects_non_increasing_heights(monkeypatch):
    """Spline fitting should reject repeated or decreasing height levels."""
    wind = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
    )

    monkeypatch.setattr(
        wind_orographic_correction,
        "get_height_levels_from_cube",
        lambda _cube: np.array([10.0, 10.0, 30.0], dtype=float),
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        wind_orographic_correction.fit_spline_wind_profile(wind)


def test_fit_spline_wind_profile_rejects_non_finite_heights(monkeypatch):
    """Spline fitting should reject NaN or infinite height values."""
    wind = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
    )

    monkeypatch.setattr(
        wind_orographic_correction,
        "get_height_levels_from_cube",
        lambda _cube: np.array([10.0, np.nan, 30.0], dtype=float),
    )

    with pytest.raises(ValueError, match="must all be finite"):
        wind_orographic_correction.fit_spline_wind_profile(wind)


def test_fit_spline_wind_profile_rejects_shape_mismatch(monkeypatch):
    """Spline fitting should fail if data and coordinate lengths disagree."""
    wind = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
    )

    monkeypatch.setattr(
        wind_orographic_correction,
        "get_height_levels_from_cube",
        lambda _cube: np.array([10.0, 20.0], dtype=float),
    )

    with pytest.raises(ValueError, match="first dimension of wind_speeds"):
        wind_orographic_correction.fit_spline_wind_profile(wind)


def test_evaluate_spline_at_reference_heights_masks_out_of_range_points():
    """Out-of-range reference heights should be masked in the output."""
    wind = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
    )
    spline = wind_orographic_correction.fit_spline_wind_profile(wind)

    reference_height_cube = _make_xy_cube(
        np.array([[5.0, 15.0], [35.0, 20.0]], dtype=np.float32),
        "reference_height",
        "m",
    )
    reference_height_cube.data = np.ma.array(
        reference_height_cube.data,
        mask=np.array([[False, False], [False, True]]),
    )

    result = wind_orographic_correction.evaluate_spline_at_reference_heights(
        spline,
        reference_height_cube,
    )

    np.testing.assert_array_equal(
        np.ma.getmaskarray(result),
        np.array([[True, False], [True, True]]),
    )


def test_evaluate_spline_at_reference_heights_returns_exact_level_values():
    """Reference heights on fitted levels should return the exact fitted value."""
    wind = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
    )
    spline = wind_orographic_correction.fit_spline_wind_profile(wind)

    reference_height_cube = _make_xy_cube(
        np.full((2, 2), 20.0, dtype=np.float32),
        "reference_height",
        "m",
    )

    result = wind_orographic_correction.evaluate_spline_at_reference_heights(
        spline,
        reference_height_cube,
    )

    np.testing.assert_allclose(result.data, np.full((2, 2), 4.0, dtype=np.float32))
    np.testing.assert_array_equal(
        np.ma.getmaskarray(result), np.zeros((2, 2), dtype=bool)
    )


def test_evaluate_spline_at_reference_heights_handles_top_boundary():
    """The highest fitted height should map to the final spline interval."""
    wind = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
    )
    spline = wind_orographic_correction.fit_spline_wind_profile(wind)

    reference_height_cube = _make_xy_cube(
        np.full((2, 2), 30.0, dtype=np.float32),
        "reference_height",
        "m",
    )

    result = wind_orographic_correction.evaluate_spline_at_reference_heights(
        spline,
        reference_height_cube,
    )

    np.testing.assert_allclose(result.data, np.full((2, 2), 6.0, dtype=np.float32))
    np.testing.assert_array_equal(
        np.ma.getmaskarray(result), np.zeros((2, 2), dtype=bool)
    )


def _make_speed_up_inputs():
    return {
        "characteristic_wavenumber": np.full((2, 2), 0.01, dtype=float),
        "unresolved_orography_height": np.full((2, 2), 20.0, dtype=float),
        "target_heights": np.array([10.0, 100.0], dtype=float),
        "target_wind_speeds": np.full((2, 2, 2), 5.0, dtype=float),
        "reference_wind_speed": np.full((2, 2), 8.0, dtype=float),
        "roughness_length": np.full((2, 2), 0.1, dtype=float),
    }


def test_broadcast_2d_to_3d_adds_height_axis():
    """Broadcast helper should add a leading singleton height dimension."""
    array_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)

    result = wind_orographic_correction._broadcast_2d_to_3d(array_2d)

    assert result.shape == (1, 2, 2)
    np.testing.assert_allclose(result[0], array_2d)


def test_compute_inner_layer_response_uses_bessel_ratio():
    """Inner-layer response should be derived from the modified Bessel ratio."""
    roughness_scaled_wavenumber_3d = np.array([[[0.25, 0.5]]], dtype=float)
    target_heights_broadcast = np.array([[[10.0, 20.0]]], dtype=float)
    roughness_length_field = np.array([[[0.1, 0.2]]], dtype=float)

    result = wind_orographic_correction._compute_inner_layer_response(
        roughness_scaled_wavenumber_3d,
        target_heights_broadcast,
        roughness_length_field,
    )

    target_argument = (
        (1.0 + 1.0j)
        * np.sqrt(roughness_scaled_wavenumber_3d * target_heights_broadcast)
        / wind_orographic_correction.VON_KARMAN_CONSTANT
    )
    roughness_argument = (
        (1.0 + 1.0j)
        * np.sqrt(roughness_scaled_wavenumber_3d * roughness_length_field)
        / wind_orographic_correction.VON_KARMAN_CONSTANT
    )
    expected = np.real(1.0 - kv(0, target_argument) / kv(0, roughness_argument))

    np.testing.assert_allclose(result, expected)


def test_compute_inner_layer_response_handles_invalid_values():
    """Inner-layer response should tolerate divide-by-zero and invalid inputs."""
    roughness_scaled_wavenumber_3d = np.array([[[np.nan, 0.25]]], dtype=float)
    target_heights_broadcast = np.array([[[10.0, 10.0]]], dtype=float)
    roughness_length_field = np.array([[[0.0, 0.1]]], dtype=float)

    result = wind_orographic_correction._compute_inner_layer_response(
        roughness_scaled_wavenumber_3d,
        target_heights_broadcast,
        roughness_length_field,
    )

    assert result.shape == (1, 1, 2)
    assert np.isnan(result[0, 0, 0])
    assert np.isfinite(result[0, 0, 1])


@pytest.mark.parametrize(
    "target_heights,error_match",
    [
        (np.array([], dtype=float), "must not be empty"),
        (np.array([10.0, np.inf], dtype=float), "contains non-finite"),
        (np.array([10.0, np.nan], dtype=float), "contains non-finite"),
    ],
)
def test_calculate_speed_up_factor_validates_target_heights(
    target_heights,
    error_match,
):
    """Speed-up calculation should reject invalid target-height inputs."""
    inputs = _make_speed_up_inputs()

    with pytest.raises(ValueError, match=error_match):
        wind_orographic_correction.calculate_speed_up_factor(
            characteristic_wavenumber=inputs["characteristic_wavenumber"],
            unresolved_orography_height=inputs["unresolved_orography_height"],
            target_heights=target_heights,
            target_wind_speeds=inputs["target_wind_speeds"],
            reference_wind_speed=inputs["reference_wind_speed"],
            roughness_length=inputs["roughness_length"],
        )


def test_calculate_speed_up_factor_returns_neutral_for_invalid_inputs():
    """Invalid ancillary data should collapse to a neutral correction factor."""
    inputs = _make_speed_up_inputs()

    result = wind_orographic_correction.calculate_speed_up_factor(
        characteristic_wavenumber=np.full((2, 2), np.nan, dtype=float),
        unresolved_orography_height=inputs["unresolved_orography_height"],
        target_heights=inputs["target_heights"],
        target_wind_speeds=inputs["target_wind_speeds"],
        reference_wind_speed=inputs["reference_wind_speed"],
        roughness_length=inputs["roughness_length"],
    )

    np.testing.assert_allclose(result, np.ones((2, 2, 2), dtype=float))


def test_calculate_speed_up_factor_returns_unity_for_zero_unresolved_orography():
    """Zero unresolved terrain should produce a neutral correction factor."""
    inputs = _make_speed_up_inputs()

    result = wind_orographic_correction.calculate_speed_up_factor(
        characteristic_wavenumber=inputs["characteristic_wavenumber"],
        unresolved_orography_height=np.zeros((2, 2), dtype=float),
        target_heights=inputs["target_heights"],
        target_wind_speeds=inputs["target_wind_speeds"],
        reference_wind_speed=inputs["reference_wind_speed"],
        roughness_length=inputs["roughness_length"],
    )

    np.testing.assert_allclose(result, np.ones((2, 2, 2), dtype=float), atol=1e-8)


def test_calculate_speed_up_factor_matches_expected_numeric_solution():
    """Speed-up factor should satisfy independent scalar checkpoints."""
    characteristic_wavenumber = np.array([[0.01]], dtype=float)
    unresolved_orography_height = np.array([[20.0]], dtype=float)
    target_heights = np.array([10.0, 100.0], dtype=float)
    target_wind_speeds = np.array([[[5.0]], [[5.0]]], dtype=float)
    reference_wind_speed = np.array([[8.0]], dtype=float)
    roughness_length = np.array([[0.1]], dtype=float)

    result = wind_orographic_correction.calculate_speed_up_factor(
        characteristic_wavenumber=characteristic_wavenumber,
        unresolved_orography_height=unresolved_orography_height,
        target_heights=target_heights,
        target_wind_speeds=target_wind_speeds,
        reference_wind_speed=reference_wind_speed,
        roughness_length=roughness_length,
    )

    # Expected values were independently generated for this fixed 1x1 case.
    np.testing.assert_allclose(result[0, 0, 0], 1.2867160430532548, rtol=1e-8)
    np.testing.assert_allclose(result[1, 0, 0], 1.1161342251046130, rtol=1e-8)
    assert result[0, 0, 0] > result[1, 0, 0]


def test_calculate_speed_up_factor_clips_fractional_perturbation(monkeypatch):
    """Fractional perturbations should be clipped to the allowed range."""
    inputs = _make_speed_up_inputs()

    def _huge_response(roughness_scaled_wavenumber_3d, *_args):
        return np.full_like(roughness_scaled_wavenumber_3d, 1.0e6, dtype=float)

    monkeypatch.setattr(
        wind_orographic_correction,
        "_compute_inner_layer_response",
        _huge_response,
    )

    unresolved = np.array([[20.0, -20.0], [20.0, -20.0]], dtype=float)
    target_winds = np.full((2, 2, 2), 1.0, dtype=float)
    result = wind_orographic_correction.calculate_speed_up_factor(
        characteristic_wavenumber=inputs["characteristic_wavenumber"],
        unresolved_orography_height=unresolved,
        target_heights=inputs["target_heights"],
        target_wind_speeds=target_winds,
        reference_wind_speed=inputs["reference_wind_speed"],
        roughness_length=inputs["roughness_length"],
    )

    np.testing.assert_allclose(result[:, :, 0], np.full((2, 2), 2.0, dtype=float))
    np.testing.assert_allclose(result[:, :, 1], np.full((2, 2), 1.0, dtype=float))
    assert np.all(result >= 1.0)
    assert np.all(result <= 2.0)


def test_calculate_speed_up_factor_applies_vertical_decay(monkeypatch):
    """The correction should weaken with increasing height."""
    inputs = _make_speed_up_inputs()

    def _unit_response(roughness_scaled_wavenumber_3d, *_args):
        return np.ones_like(roughness_scaled_wavenumber_3d, dtype=float)

    monkeypatch.setattr(
        wind_orographic_correction,
        "_compute_inner_layer_response",
        _unit_response,
    )

    result = wind_orographic_correction.calculate_speed_up_factor(**inputs)

    assert np.all(result[0] > result[1])


def test_calculate_speed_up_factor_supports_realization_dimension(monkeypatch):
    """Speed-up calculation should support extra dimensions such as realization."""
    inputs = _make_speed_up_inputs()

    def _unit_response(roughness_scaled_wavenumber_3d, *_args):
        return np.ones_like(roughness_scaled_wavenumber_3d, dtype=float)

    monkeypatch.setattr(
        wind_orographic_correction,
        "_compute_inner_layer_response",
        _unit_response,
    )

    target_winds = np.stack(
        [
            np.full((2, 2, 2), 5.0, dtype=float),
            np.full((2, 2, 2), 6.0, dtype=float),
        ],
        axis=1,
    )

    result = wind_orographic_correction.calculate_speed_up_factor(
        characteristic_wavenumber=inputs["characteristic_wavenumber"],
        unresolved_orography_height=inputs["unresolved_orography_height"],
        target_heights=inputs["target_heights"],
        target_wind_speeds=target_winds,
        reference_wind_speed=inputs["reference_wind_speed"],
        roughness_length=inputs["roughness_length"],
    )

    assert result.shape == (2, 2, 2, 2)
    assert np.all(result >= 1.0)


def test_calculate_speed_up_factor_masks_non_positive_target_winds(monkeypatch):
    """Zero or negative target winds should not receive perturbations."""
    inputs = _make_speed_up_inputs()

    def _unit_response(roughness_scaled_wavenumber_3d, *_args):
        return np.ones_like(roughness_scaled_wavenumber_3d, dtype=float)

    monkeypatch.setattr(
        wind_orographic_correction,
        "_compute_inner_layer_response",
        _unit_response,
    )

    target_winds = np.array(
        [
            [[5.0, 0.0], [5.0, -1.0]],
            [[5.0, 0.0], [5.0, -1.0]],
        ],
        dtype=float,
    )
    result = wind_orographic_correction.calculate_speed_up_factor(
        characteristic_wavenumber=inputs["characteristic_wavenumber"],
        unresolved_orography_height=inputs["unresolved_orography_height"],
        target_heights=inputs["target_heights"],
        target_wind_speeds=target_winds,
        reference_wind_speed=inputs["reference_wind_speed"],
        roughness_length=inputs["roughness_length"],
    )

    np.testing.assert_allclose(result[:, 0, 1], np.array([1.0, 1.0], dtype=float))
    np.testing.assert_allclose(result[:, 1, 1], np.array([1.0, 1.0], dtype=float))


def test_calculate_speed_up_factor_handles_masked_target_winds(monkeypatch):
    """Masked target winds should produce a neutral correction at masked points."""
    inputs = _make_speed_up_inputs()

    def _unit_response(roughness_scaled_wavenumber_3d, *_args):
        return np.ones_like(roughness_scaled_wavenumber_3d, dtype=float)

    monkeypatch.setattr(
        wind_orographic_correction,
        "_compute_inner_layer_response",
        _unit_response,
    )

    target_winds = np.ma.array(
        np.full((2, 2, 2), 5.0, dtype=float),
        mask=np.array(
            [
                [[False, True], [False, False]],
                [[False, True], [False, False]],
            ]
        ),
    )

    result = wind_orographic_correction.calculate_speed_up_factor(
        characteristic_wavenumber=inputs["characteristic_wavenumber"],
        unresolved_orography_height=inputs["unresolved_orography_height"],
        target_heights=inputs["target_heights"],
        target_wind_speeds=target_winds,
        reference_wind_speed=inputs["reference_wind_speed"],
        roughness_length=inputs["roughness_length"],
    )

    np.testing.assert_allclose(result[:, 0, 1], np.array([1.0, 1.0], dtype=float))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda inputs: {
            **inputs,
            "characteristic_wavenumber": np.array(
                [[0.01, -0.01], [0.01, 0.01]], dtype=float
            ),
        },
        lambda inputs: {
            **inputs,
            "roughness_length": np.array([[0.1, np.nan], [0.1, 0.1]], dtype=float),
        },
        lambda inputs: {
            **inputs,
            "characteristic_wavenumber": np.array(
                [[0.01, np.nan], [0.01, 0.01]], dtype=float
            ),
        },
        lambda inputs: {
            **inputs,
            "unresolved_orography_height": np.array(
                [[20.0, np.nan], [20.0, 20.0]], dtype=float
            ),
        },
    ],
)
def test_calculate_speed_up_factor_sets_unity_for_invalid_ancillary_points(
    monkeypatch,
    mutator,
):
    """Invalid ancillary points should produce a neutral correction."""
    inputs = _make_speed_up_inputs()

    def _unit_response(roughness_scaled_wavenumber_3d, *_args):
        return np.ones_like(roughness_scaled_wavenumber_3d, dtype=float)

    monkeypatch.setattr(
        wind_orographic_correction,
        "_compute_inner_layer_response",
        _unit_response,
    )

    mutated = mutator(inputs)
    result = wind_orographic_correction.calculate_speed_up_factor(
        characteristic_wavenumber=mutated["characteristic_wavenumber"],
        unresolved_orography_height=mutated["unresolved_orography_height"],
        target_heights=mutated["target_heights"],
        target_wind_speeds=mutated["target_wind_speeds"],
        reference_wind_speed=mutated["reference_wind_speed"],
        roughness_length=mutated["roughness_length"],
    )

    np.testing.assert_allclose(result[:, 0, 1], np.array([1.0, 1.0], dtype=float))


def test_approximate_roughness_length_returns_search_bounds():
    """Initial roughness-length approximation should return a bracketing interval."""
    fit_heights = np.array([10.0, 20.0, 30.0], dtype=float)
    z0_true = 0.1
    fit_winds = np.stack(
        [
            np.full((2, 2), np.log((h + z0_true) / z0_true), dtype=float)
            for h in fit_heights
        ],
        axis=0,
    )
    valid = np.ones_like(fit_winds, dtype=bool)
    valid_count = np.sum(valid, axis=0)

    lower_z0, upper_z0 = wind_orographic_correction._approximate_roughness_length(
        fit_heights,
        fit_winds,
        valid,
        valid_count,
        min_roughness_length=1e-5,
        max_roughness_length=5.0,
    )

    assert np.all(lower_z0 > 0.0)
    assert np.all(upper_z0 > 0.0)
    assert np.all(lower_z0 < upper_z0)
    assert np.all(lower_z0 <= z0_true)
    assert np.all(upper_z0 >= z0_true)


def test_approximate_roughness_length_falls_back_to_global_bounds():
    """Bad initial fits should fall back to the configured z0 bounds."""
    fit_heights = np.array([10.0, 20.0, 30.0], dtype=float)
    fit_winds = np.stack(
        [
            np.full((2, 2), 5.0, dtype=float),
            np.full((2, 2), 5.0, dtype=float),
            np.full((2, 2), 5.0, dtype=float),
        ],
        axis=0,
    )
    valid = np.ones_like(fit_winds, dtype=bool)
    valid_count = np.sum(valid, axis=0)

    lower_z0, upper_z0 = wind_orographic_correction._approximate_roughness_length(
        fit_heights,
        fit_winds,
        valid,
        valid_count,
        min_roughness_length=1e-5,
        max_roughness_length=5.0,
    )

    np.testing.assert_allclose(lower_z0, np.full((2, 2), 1e-5, dtype=float))
    np.testing.assert_allclose(upper_z0, np.full((2, 2), 5.0, dtype=float))


def test_evaluate_log_profile_fit_returns_friction_velocity_and_error():
    """Log-profile evaluation should return the fitted friction velocity and error."""
    fit_heights = np.array([10.0, 20.0, 30.0], dtype=float)
    z0 = np.full((2, 2), 0.1, dtype=float)
    u_star = 0.3
    fit_winds = np.stack(
        [
            np.full((2, 2), (u_star / 0.4) * np.log((h + 0.1) / 0.1), dtype=float)
            for h in fit_heights
        ],
        axis=0,
    )
    valid = np.ones_like(fit_winds, dtype=bool)
    valid_count = np.sum(valid, axis=0)

    friction_velocity, squared_error = (
        wind_orographic_correction._evaluate_log_profile_fit(
            fit_heights,
            fit_winds,
            valid,
            valid_count,
            z0,
            von_karman_constant=0.4,
            min_friction_velocity=0.001,
            max_friction_velocity=5.0,
        )
    )

    np.testing.assert_allclose(friction_velocity, np.full((2, 2), u_star, dtype=float))
    np.testing.assert_allclose(squared_error, np.zeros((2, 2), dtype=float), atol=1e-10)


def test_evaluate_log_profile_fit_masks_insufficient_profiles():
    """Profiles with too few valid levels should get infinite error."""
    fit_heights = np.array([10.0, 20.0, 30.0], dtype=float)
    fit_winds = np.stack(
        [
            np.full((2, 2), 3.0, dtype=float),
            np.full((2, 2), 4.0, dtype=float),
            np.full((2, 2), 5.0, dtype=float),
        ],
        axis=0,
    )
    valid = np.ones_like(fit_winds, dtype=bool)
    valid[1:, 0, 1] = False
    valid_count = np.sum(valid, axis=0)

    _, squared_error = wind_orographic_correction._evaluate_log_profile_fit(
        fit_heights,
        fit_winds,
        valid,
        valid_count,
        roughness_length=np.full((2, 2), 0.1, dtype=float),
        von_karman_constant=0.4,
        min_friction_velocity=0.001,
        max_friction_velocity=5.0,
    )

    assert np.isinf(squared_error[0, 1])
    assert np.isfinite(squared_error[0, 0])


def test_refine_roughness_length_runs_golden_section_search(monkeypatch):
    """Roughness refinement should iterate in log-z0 space."""
    call_count = {"n": 0}

    def _fake_evaluate(
        fit_heights,
        fit_winds,
        valid,
        valid_count,
        roughness_length,
        von_karman_constant,
        min_friction_velocity,
        max_friction_velocity,
    ):
        call_count["n"] += 1
        friction_velocity = np.ones_like(roughness_length, dtype=float)
        squared_error = (roughness_length - 0.2) ** 2
        return friction_velocity, squared_error

    monkeypatch.setattr(
        wind_orographic_correction,
        "_evaluate_log_profile_fit",
        _fake_evaluate,
    )

    result = wind_orographic_correction._refine_roughness_length(
        fit_heights=np.array([10.0, 20.0], dtype=float),
        fit_winds=np.ones((2, 1, 1), dtype=float),
        valid=np.ones((2, 1, 1), dtype=bool),
        valid_count=np.full((1, 1), 2, dtype=int),
        lower_z0=np.full((1, 1), 0.01, dtype=float),
        upper_z0=np.full((1, 1), 1.0, dtype=float),
        refinement_iterations=6,
        von_karman_constant=0.4,
        min_friction_velocity=0.001,
        max_friction_velocity=5.0,
    )

    assert call_count["n"] == 12
    assert 0.01 <= result[0, 0] <= 1.0
    assert abs(result[0, 0] - 0.2) < 0.1


def test_refine_roughness_length_returns_positive_values():
    """Refined roughness lengths should remain strictly positive."""
    fit_heights = np.array([10.0, 20.0, 30.0], dtype=float)
    fit_winds = np.stack(
        [
            np.full((2, 2), 3.0, dtype=float),
            np.full((2, 2), 4.0, dtype=float),
            np.full((2, 2), 5.0, dtype=float),
        ],
        axis=0,
    )
    valid = np.ones_like(fit_winds, dtype=bool)
    valid_count = np.sum(valid, axis=0)

    result = wind_orographic_correction._refine_roughness_length(
        fit_heights,
        fit_winds,
        valid,
        valid_count,
        lower_z0=np.full((2, 2), 1e-4, dtype=float),
        upper_z0=np.full((2, 2), 1.0, dtype=float),
        refinement_iterations=5,
        von_karman_constant=0.4,
        min_friction_velocity=0.001,
        max_friction_velocity=5.0,
    )

    assert np.all(result > 0.0)


def test_fit_log_wind_profile_returns_nan_when_too_few_levels():
    """Log-profile fitting should return NaN when fewer than two levels are valid."""
    wind = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0], dtype=np.float32),
        values_at_heights=np.array([3.0, -1.0], dtype=np.float32),
    )

    result = wind_orographic_correction.fit_log_wind_profile(wind)

    assert np.isnan(result).all()


def test_fit_log_wind_profile_returns_finite_values_for_valid_profiles():
    """Valid wind profiles should produce finite roughness-length estimates."""
    z0_true = 0.2
    u_star = 0.35
    heights = np.array([10.0, 30.0, 100.0], dtype=np.float32)
    values = np.array(
        [
            (u_star / wind_orographic_correction.VON_KARMAN_CONSTANT)
            * np.log((height + z0_true) / z0_true)
            for height in heights
        ],
        dtype=np.float32,
    )
    wind = _make_wind_profile_cube(
        heights=heights,
        values_at_heights=values,
    )

    result = wind_orographic_correction.fit_log_wind_profile(wind)

    assert np.isfinite(result).all()
    assert np.all(result > 0.0)
    np.testing.assert_allclose(result, np.full((2, 2), z0_true, dtype=float), rtol=0.25)


def test_fit_log_wind_profile_applies_height_limits(monkeypatch):
    """Log-profile fitting should use only heights within the requested bounds."""
    captured = {"fit_heights": None}

    def _fake_approximate(
        fit_heights,
        fit_winds,
        valid,
        valid_count,
        min_roughness_length,
        max_roughness_length,
    ):
        captured["fit_heights"] = fit_heights.copy()
        return (
            np.full(valid_count.shape, min_roughness_length, dtype=float),
            np.full(valid_count.shape, max_roughness_length, dtype=float),
        )

    monkeypatch.setattr(
        wind_orographic_correction,
        "_approximate_roughness_length",
        _fake_approximate,
    )
    monkeypatch.setattr(
        wind_orographic_correction,
        "_refine_roughness_length",
        lambda *args, **kwargs: np.full((2, 2), 0.1, dtype=float),
    )
    monkeypatch.setattr(
        wind_orographic_correction,
        "_evaluate_log_profile_fit",
        lambda *args, **kwargs: (
            np.full((2, 2), 0.2, dtype=float),
            np.zeros((2, 2), dtype=float),
        ),
    )

    wind = _make_wind_profile_cube(
        heights=np.array([5.0, 50.0, 400.0], dtype=np.float32),
        values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
    )

    wind_orographic_correction.fit_log_wind_profile(
        wind,
        lower_height_limit=10.0,
        upper_height_limit=300.0,
    )

    np.testing.assert_allclose(captured["fit_heights"], np.array([50.0], dtype=float))


def test_fit_log_wind_profile_marks_invalid_result_points_nan(monkeypatch):
    """Invalid fitted points should be replaced with NaN in the final output."""
    monkeypatch.setattr(
        wind_orographic_correction,
        "_refine_roughness_length",
        lambda *args, **kwargs: np.array(
            [[np.nan, 0.1], [0.1, 0.1]],
            dtype=float,
        ),
    )
    monkeypatch.setattr(
        wind_orographic_correction,
        "_evaluate_log_profile_fit",
        lambda *args, **kwargs: (
            np.full((2, 2), 0.2, dtype=float),
            np.zeros((2, 2), dtype=float),
        ),
    )

    wind = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        values_at_heights=np.array([3.0, 4.0, 5.0], dtype=np.float32),
    )

    result = wind_orographic_correction.fit_log_wind_profile(wind)

    assert np.isnan(result[0, 0])
    assert np.isfinite(result[0, 1])


def test_calculate_unresolved_orography_height_converts_units():
    """Unresolved orography should be computed after unit conversion to metres."""
    high_res_orog = _make_xy_cube(
        np.full((2, 2), 0.12, dtype=np.float32),
        "surface_altitude",
        "km",
    )
    model_orog = _make_xy_cube(
        np.full((2, 2), 100.0, dtype=np.float32),
        "surface_altitude",
        "m",
    )

    result = wind_orographic_correction.calculate_unresolved_orography_height(
        high_res_orog,
        model_orog,
    )

    np.testing.assert_allclose(result.data, np.full((2, 2), 20.0, dtype=np.float32))


def test_calculate_unresolved_orography_height_sets_expected_name_and_units():
    """Unresolved orography output should have the documented metadata."""
    high_res_orog = _make_xy_cube(
        np.full((2, 2), 120.0, dtype=np.float32),
        "surface_altitude",
        "m",
    )
    model_orog = _make_xy_cube(
        np.full((2, 2), 100.0, dtype=np.float32),
        "surface_altitude",
        "m",
    )

    result = wind_orographic_correction.calculate_unresolved_orography_height(
        high_res_orog,
        model_orog,
    )

    assert result.name() == "unresolved_orography_height"
    assert str(result.units) == "m"


def test_calculate_reference_height_inverts_wavenumber():
    """Reference height should be the reciprocal of the wavenumber field."""
    wavenumber_cube = _make_xy_cube(
        np.array([[0.01, 0.02], [0.04, 0.05]], dtype=np.float32),
        "characteristic_unresolved_orography_wavenumber",
        "m-1",
    )

    result = wind_orographic_correction.calculate_reference_height(wavenumber_cube)

    expected = 1.0 / wavenumber_cube.data
    np.testing.assert_allclose(result.data, expected)


def test_calculate_reference_height_preserves_mask():
    """Reference height should preserve masking from the input cube."""
    wavenumber_cube = _make_xy_cube(
        np.array([[0.01, 0.02], [0.04, 0.05]], dtype=np.float32),
        "characteristic_unresolved_orography_wavenumber",
        "m-1",
    )
    wavenumber_cube.data = np.ma.array(
        wavenumber_cube.data,
        mask=np.array([[False, True], [False, False]]),
    )

    result = wind_orographic_correction.calculate_reference_height(wavenumber_cube)

    np.testing.assert_array_equal(
        np.ma.getmaskarray(result.data),
        np.array([[False, True], [False, False]]),
    )


def test_calculate_reference_height_sets_expected_metadata():
    """Reference height output should be renamed and assigned metres."""
    wavenumber_cube = _make_xy_cube(
        np.full((2, 2), 0.01, dtype=np.float32),
        "characteristic_unresolved_orography_wavenumber",
        "m-1",
    )

    result = wind_orographic_correction.calculate_reference_height(wavenumber_cube)

    assert result.name() == "unresolved_orography_reference_height"
    assert str(result.units) == "m"


def test_calculate_characteristic_wavenumber_applies_input_thresholds():
    """Characteristic wavenumber should only be calculated for valid inputs."""
    orog_stddev_cube = _make_xy_cube(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "standard_deviation_of_height_in_grid_cell",
        "m",
    )
    silhouette_roughness_cube = _make_xy_cube(
        np.array([[1.0, 1.0], [-0.1, 0.5]], dtype=np.float32),
        "silhouette_roughness",
        "1",
    )

    result = wind_orographic_correction.calculate_characteristic_wavenumber(
        orog_stddev_cube,
        silhouette_roughness_cube,
    )

    expected_mask = np.array([[True, False], [True, False]])
    np.testing.assert_array_equal(np.ma.getmaskarray(result.data), expected_mask)

    expected_valid_01 = np.pi * np.clip(
        1.0 / max(np.sqrt(2.0) * 2.0, 1.0),
        1.0 / 4000.0,
        1.0 / 500.0,
    )
    expected_valid_11 = np.pi * np.clip(
        0.5 / max(np.sqrt(2.0) * 4.0, 1.0),
        1.0 / 4000.0,
        1.0 / 500.0,
    )
    np.testing.assert_allclose(result.data[0, 1], expected_valid_01)
    np.testing.assert_allclose(result.data[1, 1], expected_valid_11)


def test_calculate_characteristic_wavenumber_clips_length_scale_bounds():
    """Characteristic wavenumber should respect the configured length-scale limits."""
    orog_stddev_cube = _make_xy_cube(
        np.full((2, 2), 10.0, dtype=np.float32),
        "standard_deviation_of_height_in_grid_cell",
        "m",
    )
    silhouette_roughness_cube = _make_xy_cube(
        np.array([[100.0, 0.0001], [1.0, 1.0]], dtype=np.float32),
        "silhouette_roughness",
        "1",
    )

    result = wind_orographic_correction.calculate_characteristic_wavenumber(
        orog_stddev_cube,
        silhouette_roughness_cube,
    )

    np.testing.assert_allclose(result.data[0, 0], np.pi / 500.0)
    np.testing.assert_allclose(result.data[0, 1], np.pi / 4000.0)


def test_calculate_characteristic_wavenumber_masks_invalid_points():
    """Invalid terrain points should remain masked in the output cube."""
    orog_stddev_cube = _make_xy_cube(
        np.array([[3.0, 3.0], [1.0, 3.0]], dtype=np.float32),
        "standard_deviation_of_height_in_grid_cell",
        "m",
    )
    silhouette_roughness_cube = _make_xy_cube(
        np.full((2, 2), 1.0, dtype=np.float32),
        "silhouette_roughness",
        "1",
    )
    orog_stddev_cube.data = np.ma.array(
        orog_stddev_cube.data,
        mask=np.array([[True, False], [False, False]]),
    )
    silhouette_roughness_cube.data = np.ma.array(
        silhouette_roughness_cube.data,
        mask=np.array([[False, True], [False, False]]),
    )

    result = wind_orographic_correction.calculate_characteristic_wavenumber(
        orog_stddev_cube,
        silhouette_roughness_cube,
    )

    np.testing.assert_array_equal(
        np.ma.getmaskarray(result.data),
        np.array([[True, True], [True, False]]),
    )


def test_check_same_grid_allows_single_cube():
    """A single cube should not trigger a grid comparison failure."""
    cube = _make_xy_cube(
        np.full((2, 2), 1.0, dtype=np.float32),
        "surface_altitude",
        "m",
    )

    wind_orographic_correction.check_same_grid(cube)


def test_check_same_grid_allows_matching_horizontal_shapes():
    """Matching trailing horizontal shapes should pass the grid check."""
    wind_cube = _make_wind_profile_cube(
        heights=np.array([10.0, 20.0], dtype=np.float32),
        values_at_heights=np.array([3.0, 4.0], dtype=np.float32),
        shape=(2, 2),
    )
    xy_cube = _make_xy_cube(
        np.full((2, 2), 50.0, dtype=np.float32),
        "surface_altitude",
        "m",
    )

    wind_orographic_correction.check_same_grid(wind_cube, xy_cube)


@pytest.mark.parametrize("bad_shape", [(3, 2), (2, 3), (1, 1)])
def test_check_same_grid_raises_for_mismatched_horizontal_shapes(bad_shape):
    """Mismatched trailing shapes should raise a clear ValueError."""
    reference_cube = _make_xy_cube(
        np.full((2, 2), 1.0, dtype=np.float32),
        "surface_altitude",
        "m",
    )
    bad_cube = _make_xy_cube(
        np.full(bad_shape, 1.0, dtype=np.float32),
        "surface_altitude",
        "m",
    )

    with pytest.raises(
        ValueError,
        match=rf"horizontal shape \({bad_shape[0]}, {bad_shape[1]}\), expected \(2, 2\)",
    ):
        wind_orographic_correction.check_same_grid(reference_cube, bad_cube)
