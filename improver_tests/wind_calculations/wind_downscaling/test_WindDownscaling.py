# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for wind downscaling behaviour and helper utilities."""

from unittest.mock import patch

import numpy as np
import pytest

from improver import BasePlugin
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)
from improver.wind_calculations.wind_downscaling import (
    WindDownscaling,
    calculate_characteristic_wavenumber,
    calculate_reference_height,
    calculate_speed_up_factor,
    calculate_unresolved_orography_height,
    check_same_grid,
    create_corrected_wind_speed_cube,
    crop_wind_profile_cube,
    evaluate_spline_at_reference_heights,
    fit_log_wind_profile,
    fit_spline_wind_profile,
    get_cubes_to_check,
    get_height_levels_from_cube,
    get_target_height_levels,
    get_target_wind_speeds,
    prepare_target_wind_speeds,
)


def _make_xy_cube(data: np.ndarray, name: str, units: str):
    """Build a simple 2D equal-area cube for ancillary inputs."""
    return set_up_variable_cube(
        data.astype(np.float32),
        name=name,
        units=units,
        spatial_grid="equalarea",
        domain_corner=(-1036000, -1158000),
        x_grid_spacing=2000,
        y_grid_spacing=2000,
    )


def _make_wind_cube(
    heights: np.ndarray,
    values_at_heights: np.ndarray,
    shape: tuple[int, int] = (2, 2),
):
    """Build a wind-speed cube on height levels with simple x/y fields."""
    base = _make_xy_cube(np.ones(shape, dtype=np.float32), "wind_speed", "m s-1")
    cube = add_coordinate(base, heights.tolist(), "height", "m")

    data_3d = np.stack(
        [
            np.full(shape, value, dtype=np.float32)
            for value in np.asarray(values_at_heights, dtype=np.float32)
        ],
        axis=0,
    )
    cube.data = data_3d
    return cube


def _make_plugin(landmask_value: float = 1.0):
    """Create a WindDownscaling plugin with matching ancillary cubes."""
    shape = (2, 2)
    high_res_orog = _make_xy_cube(
        np.full(shape, 120.0, dtype=np.float32),
        "surface_altitude",
        "m",
    )
    model_orog = _make_xy_cube(
        np.full(shape, 100.0, dtype=np.float32),
        "surface_altitude",
        "m",
    )
    model_orog_stddev = _make_xy_cube(
        np.full(shape, 20.0, dtype=np.float32),
        "standard_deviation_of_height_in_grid_cell",
        "m",
    )
    silhouette_roughness = _make_xy_cube(
        np.full(shape, 0.2, dtype=np.float32),
        "silhouette_roughness",
        "1",
    )
    landmask = _make_xy_cube(
        np.full(shape, landmask_value, dtype=np.float32),
        "land_binary_mask",
        "1",
    )

    return WindDownscaling(
        high_res_orog_cube=high_res_orog,
        model_orog_cube=model_orog,
        model_orog_stddev_cube=model_orog_stddev,
        model_silhouette_roughness_cube=silhouette_roughness,
        landmask_cube=landmask,
    )


class TestWindDownscaling:
    """End-to-end and helper tests for the WindDownscaling plugin."""

    def test_inherits_base_plugin(self):
        """Plugin should expose the standard IMPROVER plugin interface."""
        plugin = _make_plugin()
        assert isinstance(plugin, BasePlugin)

    def test_requires_target_wind_speed_cube_argument(self):
        """Process call should require an explicit target-wind cube."""
        plugin = _make_plugin()
        wind = _make_wind_cube(
            heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
        )

        with pytest.raises(TypeError, match="target_wind_speed_cube"):
            plugin(wind)

    def test_get_target_height_levels_returns_sorted_requested_values(self):
        """Requested target heights should be sorted before use."""
        wind = _make_wind_cube(
            heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
        )
        result = get_target_height_levels(
            wind,
            target_height_levels=[30.0, 10.0],
        )
        np.testing.assert_allclose(result, np.array([10.0, 30.0], dtype=np.float32))

    def test_get_target_height_levels_defaults_to_wind_cube_levels(self):
        """If no target is supplied, use heights from the wind cube."""
        wind = _make_wind_cube(
            heights=np.array([30.0, 10.0, 20.0], dtype=np.float32),
            values_at_heights=np.array([6.0, 2.0, 4.0], dtype=np.float32),
        )

        result = get_target_height_levels(
            wind,
            target_height_levels=None,
        )

        np.testing.assert_allclose(
            result, np.array([10.0, 20.0, 30.0], dtype=np.float32)
        )

    def test_get_target_height_levels_uses_target_cube_height_in_metres(self):
        """Use the target-cube height, converting to metres when needed."""
        wind = _make_wind_cube(
            heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
        )

        target_cube = wind[0].copy(data=np.full((2, 2), 8.0, dtype=np.float32))
        height_coord = target_cube.coord("height")
        height_coord.points = np.array([0.015], dtype=np.float32)
        height_coord.units = "km"
        height_coord.bounds = None

        result = get_target_height_levels(
            target_cube,
            target_height_levels=None,
        )

        np.testing.assert_allclose(result, np.array([15.0], dtype=np.float32))

    def test_crop_wind_profile_cube_uses_1300m_minimum_upper_bound(self):
        """Profile crop should keep levels up to 1300 m even for lower targets."""
        wind = _make_wind_cube(
            heights=np.array(
                [50.0, 150.0, 300.0, 500.0, 1000.0, 1250.0, 1500.0], dtype=np.float32
            ),
            values_at_heights=np.array(
                [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0], dtype=np.float32
            ),
        )

        cropped = crop_wind_profile_cube(
            wind,
            np.array([200.0], dtype=np.float32),
        )

        np.testing.assert_allclose(
            cropped.coord("height").points,
            np.array([50.0, 150.0, 300.0, 500.0, 1000.0, 1250.0], dtype=np.float32),
        )

    def test_crop_wind_profile_cube_uses_target_max_when_above_300m(self):
        """Profile crop should keep levels up to the maximum target height."""
        wind = _make_wind_cube(
            heights=np.array([50.0, 150.0, 300.0, 500.0, 700.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float32),
        )

        cropped = crop_wind_profile_cube(
            wind,
            np.array([600.0], dtype=np.float32),
        )

        np.testing.assert_allclose(
            cropped.coord("height").points,
            np.array([50.0, 150.0, 300.0, 500.0], dtype=np.float32),
        )

    def test_no_correction_over_sea_for_native_levels(self):
        """Sea points should keep the original winds on native levels."""
        plugin = _make_plugin(landmask_value=0.0)
        wind = _make_wind_cube(
            heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
        )

        result = plugin(wind, wind)
        np.testing.assert_allclose(result.data, wind.data)
        np.testing.assert_allclose(
            result.coord("height").points, wind.coord("height").points
        )

    def test_no_correction_over_sea_for_requested_levels(self):
        """At sea, requested-height output should match background interpolation."""
        plugin = _make_plugin(landmask_value=0.0)
        wind = _make_wind_cube(
            heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
        )

        result = plugin(wind, wind, target_height_levels=[10.0, 30.0])

        expected = np.stack(
            [
                np.full((2, 2), 2.0, dtype=np.float32),
                np.full((2, 2), 6.0, dtype=np.float32),
            ],
            axis=0,
        )
        np.testing.assert_allclose(result.data, expected)
        np.testing.assert_allclose(
            result.coord("height").points,
            np.array([10.0, 30.0], dtype=np.float32),
        )

    def test_target_wind_speed_cube_is_used_directly_over_sea(self):
        """When provided, use target-wind values directly at sea points."""
        plugin = _make_plugin(landmask_value=0.0)
        wind = _make_wind_cube(
            heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
        )

        target_cube = wind[0].copy(data=np.full((2, 2), 9.0, dtype=np.float32))
        height_coord = target_cube.coord("height")
        height_coord.convert_units("m")
        height_coord.points = np.array([15.0], dtype=np.float32)
        height_coord.bounds = None

        result = plugin(wind, target_cube)
        np.testing.assert_allclose(result.data, np.full((2, 2), 9.0, dtype=np.float32))
        np.testing.assert_allclose(
            result.coord("height").points,
            np.array([15.0], dtype=np.float32),
        )

    def test_raises_for_inconsistent_horizontal_grid(self):
        """Mismatched horizontal grids should raise a helpful error."""
        plugin = _make_plugin()
        wind = _make_wind_cube(
            heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
            shape=(3, 2),
        )

        msg = "horizontal shape"
        with pytest.raises(ValueError, match=msg):
            plugin(wind, wind)

    def test_create_corrected_wind_speed_cube_raises_for_height_length_mismatch(self):
        """Reject output data if height count and data slices do not match."""
        template = _make_wind_cube(
            heights=np.array([10.0, 20.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0], dtype=np.float32),
        )
        corrected = np.ones((2, 2, 2), dtype=np.float32)

        msg = "must match the number of target heights"
        with pytest.raises(ValueError, match=msg):
            create_corrected_wind_speed_cube(
                template,
                corrected,
                target_heights=np.array([10.0], dtype=np.float32),
            )

    def test_create_corrected_wind_speed_cube_uses_float32_output(self):
        """Ensure output data and height points comply with mandatory dtype."""
        template = _make_wind_cube(
            heights=np.array([10.0, 20.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0], dtype=np.float32),
        )
        corrected = np.array(
            [
                np.full((2, 2), 3.0, dtype=np.float64),
                np.full((2, 2), 5.0, dtype=np.float64),
            ]
        )

        result = create_corrected_wind_speed_cube(
            template,
            corrected,
            target_heights=np.array([10.0, 20.0], dtype=np.float64),
        )

        assert result.dtype == np.dtype(np.float32)
        assert result.coord("height").dtype == np.dtype(np.float32)
        assert result.coord("height").points.dtype == np.dtype(np.float32)

    def test_get_cubes_to_check_contains_required_cubes(self):
        """Build the required set of cubes in the expected order."""
        plugin = _make_plugin()
        wind = _make_wind_cube(
            heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
        )

        cubes = get_cubes_to_check(
            wind,
            wind,
            plugin.high_res_orog_cube,
            plugin.model_orog_cube,
            plugin.model_orog_stddev_cube,
            plugin.model_silhouette_roughness_cube,
            plugin.landmask_cube,
        )

        expected = [
            wind,
            wind,
            plugin.high_res_orog_cube,
            plugin.model_orog_cube,
            plugin.landmask_cube,
            plugin.model_orog_stddev_cube,
            plugin.model_silhouette_roughness_cube,
        ]
        assert cubes == expected

    def test_get_cubes_to_check_places_target_cube_second(self):
        """Ensure the target cube is retained in the expected position."""
        plugin = _make_plugin()
        wind = _make_wind_cube(
            heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
        )
        target_cube = wind[0].copy(data=np.full((2, 2), 7.0, dtype=np.float32))

        cubes = get_cubes_to_check(
            wind,
            target_cube,
            plugin.high_res_orog_cube,
            plugin.model_orog_cube,
            plugin.model_orog_stddev_cube,
            plugin.model_silhouette_roughness_cube,
            plugin.landmask_cube,
        )

        assert len(cubes) == 7
        assert cubes[1] is target_cube

    def test_get_target_wind_speeds_from_target_cube_returns_independent_copy(self):
        """Target-wind data should be copied into a new array with height axis."""
        wind = _make_wind_cube(
            heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
        )
        target_data = np.ma.array(
            [[3.0, 4.0], [5.0, 6.0]],
            mask=[[False, True], [False, False]],
            dtype=np.float32,
        )
        target_cube = wind[0].copy(data=target_data)

        result = get_target_wind_speeds(
            target_wind_speed_cube=target_cube,
            target_heights=np.array([10.0], dtype=np.float32),
        )

        assert result.shape == (1, 2, 2)
        assert np.ma.isMaskedArray(result)
        np.testing.assert_allclose(result[0, 0, 0], 3.0)
        assert np.ma.getmaskarray(result)[0, 0, 1]

        target_cube.data[0, 0] = 999.0
        np.testing.assert_allclose(result[0, 0, 0], 3.0)

    def test_get_target_wind_speeds_interpolates_from_target_cube(self):
        """Requested heights should be interpolated from target-cube levels."""
        wind = _make_wind_cube(
            heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
        )
        base = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        gradient = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        wind.data = np.stack(
            [
                base + gradient * 10.0,
                base + gradient * 20.0,
                base + gradient * 30.0,
            ],
            axis=0,
        )

        result = get_target_wind_speeds(
            target_wind_speed_cube=wind,
            target_heights=np.array([15.0, 25.0], dtype=np.float32),
        )

        expected = np.stack(
            [
                base + gradient * 15.0,
                base + gradient * 25.0,
            ],
            axis=0,
        )
        np.testing.assert_allclose(result, expected)

    def test_get_target_wind_speeds_uses_copy_for_close_height_match(self):
        """Near-equal target heights should use direct-copy path, not interpolation."""
        wind = _make_wind_cube(
            heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
        )

        with patch(
            "improver.wind_calculations.wind_downscaling.fit_spline_wind_profile",
            side_effect=AssertionError("Interpolation path should not be used."),
        ):
            result = get_target_wind_speeds(
                target_wind_speed_cube=wind,
                target_heights=np.array([10.0 + 1e-7, 20.0 - 1e-7, 30.0], dtype=float),
            )

        np.testing.assert_allclose(result, wind.data)

    def test_get_target_wind_speeds_raises_for_single_level_mismatch(self):
        """Single-level target cubes cannot be interpolated to new heights."""
        wind_profile = _make_wind_cube(
            heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
        )
        single_level_target = wind_profile[0].copy(
            data=np.full((2, 2), 2.0, dtype=np.float32)
        )

        msg = "Cannot interpolate target_wind_speed_cube"
        with pytest.raises(ValueError, match=msg):
            get_target_wind_speeds(
                target_wind_speed_cube=single_level_target,
                target_heights=np.array([20.0], dtype=np.float32),
            )

    def test_get_target_wind_speeds_raises_for_height_data_mismatch(self):
        """Reject cubes whose height coordinate does not match first data axis."""
        wind = _make_wind_cube(
            heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
        )
        target_cube = wind[0].copy(data=np.full((2, 2), 7.0, dtype=np.float32))

        msg = "data first dimension must match"
        with patch(
            "improver.wind_calculations.wind_downscaling.get_height_levels_from_cube",
            return_value=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        ):
            with pytest.raises(ValueError, match=msg):
                get_target_wind_speeds(
                    target_wind_speed_cube=target_cube,
                    target_heights=np.array([10.0, 20.0], dtype=np.float32),
                )

    def test_create_corrected_wind_speed_cube_sets_values_and_heights(self):
        """Output cube should keep corrected values and requested heights."""
        template = _make_wind_cube(
            heights=np.array([10.0, 20.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0], dtype=np.float32),
        )
        corrected = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=np.float32,
        )
        target_heights = np.array([15.0, 35.0], dtype=np.float32)

        result = create_corrected_wind_speed_cube(template, corrected, target_heights)

        np.testing.assert_allclose(result.data, corrected)
        np.testing.assert_allclose(result.coord("height").points, target_heights)

    def test_calculate_speed_up_factor_clips_and_forces_sea_points(self):
        """Clip land corrections and force sea points back to unity."""
        characteristic_wavenumber = np.array(
            [[0.2, 0.2], [0.2, 0.2]],
            dtype=np.float32,
        )
        unresolved_orography_height = np.array(
            [[1500.0, 1500.0], [0.0, 1500.0]],
            dtype=np.float32,
        )
        target_heights = np.array([10.0], dtype=np.float32)
        target_wind_speeds = np.array(
            [[[1.0, 1.0], [1.0, 1.0]]],
            dtype=np.float32,
        )
        reference_wind_speed = np.array(
            [[80.0, 80.0], [80.0, 80.0]],
            dtype=np.float32,
        )
        roughness_length = np.array(
            [[0.1, 0.1], [0.1, 0.1]],
            dtype=np.float32,
        )
        land_mask = np.array(
            [[1.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        )

        speed_up = calculate_speed_up_factor(
            characteristic_wavenumber,
            unresolved_orography_height,
            target_heights,
            target_wind_speeds,
            reference_wind_speed,
            roughness_length,
            land_mask=land_mask,
        )

        assert speed_up.shape == (1, 2, 2)
        np.testing.assert_allclose(speed_up[0, :, 1], np.ones(2, dtype=np.float32))
        np.testing.assert_allclose(speed_up[0, 1, 0], 1.0)
        assert speed_up[0, 0, 0] > 1.0
        assert speed_up[0, 0, 0] <= 2.0

    def test_calculate_speed_up_factor_returns_unity_for_masked_or_zero_winds(self):
        """Masked or non-positive winds should not receive perturbations."""
        characteristic_wavenumber = np.array([[0.2, 0.2]], dtype=np.float32)
        unresolved_orography_height = np.array([[500.0, 500.0]], dtype=np.float32)
        target_heights = np.array([10.0], dtype=np.float32)
        target_wind_speeds = np.ma.array(
            [[[0.0, 4.0]]],
            mask=[[[False, True]]],
            dtype=np.float32,
        )
        reference_wind_speed = np.array([[50.0, 50.0]], dtype=np.float32)
        roughness_length = np.array([[0.1, 0.1]], dtype=np.float32)

        speed_up = calculate_speed_up_factor(
            characteristic_wavenumber,
            unresolved_orography_height,
            target_heights,
            target_wind_speeds,
            reference_wind_speed,
            roughness_length,
        )

        np.testing.assert_allclose(speed_up, np.ones((1, 1, 2), dtype=np.float32))

    def test_calculate_speed_up_factor_clips_strong_negative_to_zero(self):
        """Very negative perturbations should clip to a zero speed-up."""
        characteristic_wavenumber = np.array([[0.2]], dtype=np.float32)
        unresolved_orography_height = np.array([[-3000.0]], dtype=np.float32)
        target_heights = np.array([10.0], dtype=np.float32)
        target_wind_speeds = np.array([[[1.0]]], dtype=np.float32)
        reference_wind_speed = np.array([[80.0]], dtype=np.float32)
        roughness_length = np.array([[0.1]], dtype=np.float32)

        speed_up = calculate_speed_up_factor(
            characteristic_wavenumber,
            unresolved_orography_height,
            target_heights,
            target_wind_speeds,
            reference_wind_speed,
            roughness_length,
            land_mask=np.array([[1.0]], dtype=np.float32),
        )

        np.testing.assert_allclose(speed_up, np.zeros((1, 1, 1), dtype=np.float32))

    def test_check_same_grid_error_contains_cube_name_and_shapes(self):
        """Grid-shape errors should name the cube and both shapes."""
        reference = _make_xy_cube(
            np.ones((2, 2), dtype=np.float32), "wind_speed", "m s-1"
        )
        mismatch = _make_xy_cube(
            np.ones((3, 2), dtype=np.float32), "surface_altitude", "m"
        )

        with pytest.raises(ValueError, match="surface_altitude") as exc_info:
            check_same_grid(reference, mismatch)

        message = str(exc_info.value)
        assert "(3, 2)" in message
        assert "(2, 2)" in message

    def test_get_height_levels_from_cube_converts_units_and_handles_scalar(self):
        """Height extraction should convert to metres and return a 1D array."""
        wind = _make_wind_cube(
            heights=np.array([10.0, 20.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0], dtype=np.float32),
        )

        target = wind[0].copy(data=np.full((2, 2), 6.0, dtype=np.float32))
        height_coord = target.coord("height")
        height_coord.points = np.array([0.025], dtype=np.float32)
        height_coord.units = "km"
        height_coord.bounds = None

        levels = get_height_levels_from_cube(target)

        assert levels.shape == (1,)
        np.testing.assert_allclose(levels, np.array([25.0], dtype=np.float32))

    def test_evaluate_spline_at_reference_heights_masks_out_of_range_points(self):
        """Spline evaluation should mask points outside valid height range."""
        wind = _make_wind_cube(
            heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
        )
        spline = fit_spline_wind_profile(wind)

        reference_cube = _make_xy_cube(
            np.array([[5.0, 15.0], [30.0, 35.0]], dtype=np.float32),
            "reference_height",
            "m",
        )

        result = evaluate_spline_at_reference_heights(spline, reference_cube)

        expected_mask = np.array([[True, False], [False, True]])
        np.testing.assert_array_equal(np.ma.getmaskarray(result), expected_mask)
        np.testing.assert_allclose(result[0, 1], 3.0)
        np.testing.assert_allclose(result[1, 0], 6.0)

    def test_evaluate_spline_at_reference_heights_exact_boundary_value(self):
        """A reference height on an input level should return the exact value."""
        wind = _make_wind_cube(
            heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
        )
        spline = fit_spline_wind_profile(wind)
        reference_cube = _make_xy_cube(
            np.array([[20.0, 20.0], [20.0, 20.0]], dtype=np.float32),
            "reference_height",
            "m",
        )

        result = evaluate_spline_at_reference_heights(spline, reference_cube)

        np.testing.assert_allclose(result.data, np.full((2, 2), 4.0, dtype=np.float32))
        assert not np.any(np.ma.getmaskarray(result))

    def test_process_returns_background_when_unresolved_orography_is_zero(self):
        """If unresolved orography is zero, land correction should be neutral."""
        shape = (2, 2)
        high_res_orog = _make_xy_cube(
            np.full(shape, 100.0, dtype=np.float32),
            "surface_altitude",
            "m",
        )
        model_orog = _make_xy_cube(
            np.full(shape, 100.0, dtype=np.float32),
            "surface_altitude",
            "m",
        )
        model_orog_stddev = _make_xy_cube(
            np.full(shape, 20.0, dtype=np.float32),
            "standard_deviation_of_height_in_grid_cell",
            "m",
        )
        silhouette_roughness = _make_xy_cube(
            np.full(shape, 0.2, dtype=np.float32),
            "silhouette_roughness",
            "1",
        )
        landmask = _make_xy_cube(
            np.ones(shape, dtype=np.float32),
            "land_binary_mask",
            "1",
        )

        plugin = WindDownscaling(
            high_res_orog_cube=high_res_orog,
            model_orog_cube=model_orog,
            model_orog_stddev_cube=model_orog_stddev,
            model_silhouette_roughness_cube=silhouette_roughness,
            landmask_cube=landmask,
        )

        wind = _make_wind_cube(
            heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
        )
        requested_heights = [15.0, 25.0]

        result = plugin(wind, wind, target_height_levels=requested_heights)

        expected = np.stack(
            [
                np.full((2, 2), 3.0, dtype=np.float32),
                np.full((2, 2), 5.0, dtype=np.float32),
            ],
            axis=0,
        )
        np.testing.assert_allclose(result.data, expected)
        np.testing.assert_allclose(
            result.coord("height").points,
            np.array(requested_heights, dtype=np.float32),
        )

    def test_calculate_speed_up_factor_exact_clip_endpoints(self):
        """Large perturbations should clip cleanly to 2 and 0 endpoints."""
        characteristic_wavenumber = np.array([[0.2, 0.2]], dtype=np.float32)
        unresolved_orography_height = np.array([[4000.0, -4000.0]], dtype=np.float32)
        target_heights = np.array([10.0], dtype=np.float32)
        target_wind_speeds = np.array([[[1.0, 1.0]]], dtype=np.float32)
        reference_wind_speed = np.array([[80.0, 80.0]], dtype=np.float32)
        roughness_length = np.array([[0.1, 0.1]], dtype=np.float32)

        speed_up = calculate_speed_up_factor(
            characteristic_wavenumber,
            unresolved_orography_height,
            target_heights,
            target_wind_speeds,
            reference_wind_speed,
            roughness_length,
            land_mask=np.array([[1.0, 1.0]], dtype=np.float32),
        )

        np.testing.assert_allclose(speed_up, np.array([[[2.0, 0.0]]], dtype=np.float32))

    def test_calculate_speed_up_factor_sets_unity_for_invalid_characteristic_wavenumber(
        self,
    ):
        """NaN wavenumber values should force unity correction at that point."""
        characteristic_wavenumber = np.array([[np.nan, 0.2]], dtype=np.float32)
        unresolved_orography_height = np.array([[100.0, 100.0]], dtype=np.float32)
        target_heights = np.array([10.0], dtype=np.float32)
        target_wind_speeds = np.array([[[5.0, 5.0]]], dtype=np.float32)
        reference_wind_speed = np.array([[20.0, 20.0]], dtype=np.float32)
        roughness_length = np.array([[0.1, 0.1]], dtype=np.float32)

        speed_up = calculate_speed_up_factor(
            characteristic_wavenumber,
            unresolved_orography_height,
            target_heights,
            target_wind_speeds,
            reference_wind_speed,
            roughness_length,
        )

        np.testing.assert_allclose(speed_up[0, 0, 0], 1.0)
        assert speed_up[0, 0, 1] != 1.0

    def test_process_mixed_surface_branches_with_target_wind_cube(self):
        """One run should hit sea forcing and opposite-sign land responses."""
        shape = (2, 2)
        high_res_orog = _make_xy_cube(
            np.array([[120.0, 120.0], [100.0, 80.0]], dtype=np.float32),
            "surface_altitude",
            "m",
        )
        model_orog = _make_xy_cube(
            np.full(shape, 100.0, dtype=np.float32),
            "surface_altitude",
            "m",
        )
        model_orog_stddev = _make_xy_cube(
            np.full(shape, 20.0, dtype=np.float32),
            "standard_deviation_of_height_in_grid_cell",
            "m",
        )
        silhouette_roughness = _make_xy_cube(
            np.full(shape, 0.2, dtype=np.float32),
            "silhouette_roughness",
            "1",
        )
        landmask = _make_xy_cube(
            np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            "land_binary_mask",
            "1",
        )

        plugin = WindDownscaling(
            high_res_orog_cube=high_res_orog,
            model_orog_cube=model_orog,
            model_orog_stddev_cube=model_orog_stddev,
            model_silhouette_roughness_cube=silhouette_roughness,
            landmask_cube=landmask,
        )

        wind = _make_wind_cube(
            heights=np.array([100.0, 200.0, 300.0], dtype=np.float32),
            values_at_heights=np.array([20.0, 40.0, 60.0], dtype=np.float32),
        )
        target_cube = wind[0].copy(data=np.full((2, 2), 10.0, dtype=np.float32))
        target_cube.coord("height").points = np.array([150.0], dtype=np.float32)
        target_cube.coord("height").bounds = None

        result = plugin(wind, target_cube)

        # Sea point stays unchanged.
        np.testing.assert_allclose(result.data[0, 1], 10.0)
        # Zero unresolved orography point stays unchanged.
        np.testing.assert_allclose(result.data[1, 0], 10.0)
        # Positive and negative unresolved-terrain responses are opposite in sign.
        assert result.data[0, 0] > 10.0
        assert result.data[1, 1] < 10.0

    def test_fit_spline_wind_profile_raises_for_non_increasing_heights(self):
        """Spline fitting should reject repeated or decreasing height levels."""
        wind = _make_wind_cube(
            heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
        )

        with patch(
            "improver.wind_calculations.wind_downscaling.get_height_levels_from_cube",
            return_value=np.array([10.0, 10.0, 30.0], dtype=np.float32),
        ):
            with pytest.raises(ValueError, match="strictly increasing"):
                fit_spline_wind_profile(wind)

    def test_fit_spline_wind_profile_raises_for_non_finite_heights(self):
        """Spline fitting should reject NaN or inf height levels."""
        wind = _make_wind_cube(
            heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
        )

        with patch(
            "improver.wind_calculations.wind_downscaling.get_height_levels_from_cube",
            return_value=np.array([10.0, np.nan, 30.0], dtype=np.float32),
        ):
            with pytest.raises(ValueError, match="all be finite"):
                fit_spline_wind_profile(wind)

    def test_fit_spline_wind_profile_raises_for_height_data_size_mismatch(self):
        """Spline fitting should fail if coordinate levels and data axis disagree."""
        wind = _make_wind_cube(
            heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            values_at_heights=np.array([2.0, 4.0, 6.0], dtype=np.float32),
        )

        with patch(
            "improver.wind_calculations.wind_downscaling.get_height_levels_from_cube",
            return_value=np.array([10.0, 20.0], dtype=np.float32),
        ):
            with pytest.raises(ValueError, match="first dimension"):
                fit_spline_wind_profile(wind)

    def test_fit_log_wind_profile_returns_nan_with_fewer_than_two_valid_levels(self):
        """Log-profile fit should return NaN where fewer than two valid points exist."""
        wind = _make_wind_cube(
            heights=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            values_at_heights=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            shape=(1, 1),
        )
        wind.data = np.array([[[np.nan]], [[2.0]], [[np.nan]]], dtype=np.float32)

        roughness = fit_log_wind_profile(wind)

        assert roughness.shape == (1, 1)
        assert np.isnan(roughness[0, 0])

    def test_calculate_characteristic_wavenumber_applies_thresholds_and_landmask(self):
        """Wavenumber should only be computed at valid land points."""
        sigma_cube = _make_xy_cube(
            np.array([[10.0, 1.0], [10.0, 10.0]], dtype=np.float32),
            "standard_deviation_of_height_in_grid_cell",
            "m",
        )
        silhouette_cube = _make_xy_cube(
            np.array([[2.0, 2.0], [-0.5, 2.0]], dtype=np.float32),
            "silhouette_roughness",
            "1",
        )
        landmask_cube = _make_xy_cube(
            np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            "land_binary_mask",
            "1",
        )

        result = calculate_characteristic_wavenumber(
            sigma_cube,
            silhouette_cube,
            landmask_cube,
        )

        mask = np.ma.getmaskarray(result.data)
        np.testing.assert_array_equal(mask, np.array([[False, True], [True, True]]))
        np.testing.assert_allclose(result.data[0, 0], np.pi / 500.0)

    def test_calculate_characteristic_wavenumber_clips_to_max_length_scale(self):
        """Very smooth terrain should clip to the longest permitted scale."""
        sigma_cube = _make_xy_cube(
            np.full((1, 1), 500.0, dtype=np.float32),
            "standard_deviation_of_height_in_grid_cell",
            "m",
        )
        silhouette_cube = _make_xy_cube(
            np.full((1, 1), 0.01, dtype=np.float32),
            "silhouette_roughness",
            "1",
        )
        landmask_cube = _make_xy_cube(
            np.full((1, 1), 1.0, dtype=np.float32),
            "land_binary_mask",
            "1",
        )

        result = calculate_characteristic_wavenumber(
            sigma_cube,
            silhouette_cube,
            landmask_cube,
        )

        np.testing.assert_allclose(result.data[0, 0], np.pi / 4000.0)

    def test_calculate_unresolved_orography_height_converts_units_and_sets_metadata(
        self,
    ):
        """Unresolved orography helper should convert units and set output metadata."""
        high_res = _make_xy_cube(
            np.full((2, 2), 0.12, dtype=np.float32),
            "surface_altitude",
            "km",
        )
        model = _make_xy_cube(
            np.full((2, 2), 100.0, dtype=np.float32),
            "surface_altitude",
            "m",
        )

        result = calculate_unresolved_orography_height(high_res, model)

        assert result.name() == "unresolved_orography_height"
        assert str(result.units) == "m"
        np.testing.assert_allclose(result.data, np.full((2, 2), 20.0, dtype=np.float32))

    def test_calculate_reference_height_preserves_mask_and_sets_metadata(self):
        """Reference-height helper should preserve mask and update metadata."""
        wavenumber = _make_xy_cube(
            np.array([[0.1, 0.2]], dtype=np.float32),
            "characteristic_unresolved_orography_wavenumber",
            "m-1",
        )
        wavenumber.data = np.ma.array(
            wavenumber.data,
            mask=np.array([[False, True]]),
        )

        result = calculate_reference_height(wavenumber)

        assert result.name() == "unresolved_orography_reference_height"
        assert str(result.units) == "m"
        np.testing.assert_allclose(result.data[0, 0], 10.0)
        assert np.ma.getmaskarray(result.data)[0, 1]

    def test_check_same_grid_allows_single_cube(self):
        """A single cube should be a no-op for grid consistency checks."""
        cube = _make_xy_cube(np.ones((2, 2), dtype=np.float32), "wind_speed", "m s-1")

        check_same_grid(cube)

    def test_prepare_target_wind_speeds_returns_mask_and_nan_filled_values(self):
        """Preparation helper should return explicit mask and NaN-filled values."""
        target_winds = np.ma.array(
            [[[1.0, 2.0], [3.0, 4.0]]],
            mask=[[[False, True], [False, False]]],
            dtype=np.float32,
        )

        mask, values = prepare_target_wind_speeds(target_winds)

        np.testing.assert_array_equal(mask, np.array([[[False, True], [False, False]]]))
        np.testing.assert_allclose(values[0, 0, 0], 1.0)
        assert np.isnan(values[0, 0, 1])

    def test_fit_log_wind_profile_returns_finite_for_valid_profiles(self):
        """Valid wind profiles should produce finite roughness estimates."""
        wind = _make_wind_cube(
            heights=np.array([10.0, 30.0, 80.0, 150.0], dtype=np.float32),
            values_at_heights=np.array([4.0, 6.0, 8.0, 10.0], dtype=np.float32),
            shape=(2, 2),
        )

        roughness = fit_log_wind_profile(wind)

        assert roughness.shape == (2, 2)
        assert np.all(np.isfinite(roughness))
        assert np.all(roughness > 0.0)
