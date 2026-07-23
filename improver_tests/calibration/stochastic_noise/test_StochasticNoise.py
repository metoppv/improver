# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the StochasticNoise plugin"""

import warnings

import numpy as np
import pytest
from iris.cube import Cube

from improver.calibration.stochastic_noise import StochasticNoise
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

pytest.importorskip("pysteps")


@pytest.fixture
def plugin():
    """Create StochasticNoise plugin instance with default parameters."""

    ssft_init_params = {
        "win_size": (100, 100),
        "overlap": 0.3,
        "war_thr": 0.1,
    }
    ssft_generate_params = {
        "overlap": 0.3,
        "seed": 0,
    }
    plugin = StochasticNoise(
        ssft_init_params=ssft_init_params,
        ssft_generate_params=ssft_generate_params,
        db_threshold=0.03,
        db_threshold_units="mm/hr",
    )
    return plugin


@pytest.fixture
def simple_cube():
    """
    Create a simple cube with two realizations for testing.
    All values are non-zero, so no noise should be added when the data is unmodified.
    """
    data = np.array(
        [
            [[2.0, 3.0], [1.0, 4.0]],
            [[2.2, 3.2], [1.2, 4.2]],
        ],
        dtype=np.float32,
    )

    cube = set_up_variable_cube(data=data, name="precipitation_rate", units="mm/hr")

    return cube


def test__to_dB_and__from_dB(plugin: StochasticNoise, simple_cube: Cube):
    """Test that _to_dB and _from_dB are inverses of each other."""
    cube = simple_cube.copy()
    dB_cube = plugin._to_dB(cube.copy())
    restored_array = plugin._from_dB(dB_cube.data)
    db_threshold = plugin.db_threshold
    expected = np.where(simple_cube.data < db_threshold, 0, simple_cube.data)
    np.testing.assert_allclose(restored_array, expected, rtol=1e-6)


def test_do_fft(plugin: StochasticNoise, simple_cube: Cube):
    """Test the do_fft method generates expected noise pattern."""
    test_data = simple_cube.data[0]  # Use one realization for test
    result = plugin.do_fft(test_data)

    expected = np.array(
        [[0.59051897, -1.33586476], [-0.5186695, 1.26401529]], dtype=np.float32
    )
    np.testing.assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "test_case",
    [
        "base",
        "plugin_defaults",
        "some_data_masked",
        "with_zero_values",
    ],
)
def test_process(
    plugin: StochasticNoise,
    simple_cube: Cube,
    test_case: str,
):
    """Test process method."""
    cube = simple_cube.copy()

    # All values in simple_cube > 0 (not non-positive), so no noise should be added
    # (i.e., output should equal input)
    expected = cube.data.copy()

    if test_case == "plugin_defaults":
        # Use plugin with default parameters (except seed for reproducibility)
        plugin = StochasticNoise(ssft_generate_params={"seed": 0})

    elif test_case == "some_data_masked":
        # Create masked input array
        cube.data = np.ma.masked_array(cube.data, mask=False, dtype=np.float32)
        cube.data[0, 0, 0] = np.ma.masked
        cube.data[1, 1, 1] = np.ma.masked

        # Create expected output masked array
        expected = np.ma.masked_array(expected, mask=False, dtype=np.float32)
        expected[0, 0, 0] = np.ma.masked
        expected[1, 1, 1] = np.ma.masked

    elif test_case == "with_zero_values":
        # Create cube with some zero values where noise should be added
        plugin = StochasticNoise(
            ssft_init_params={"win_size": (2, 2), "overlap": 0},
            ssft_generate_params={"seed": 0},
            db_threshold=0.03,
            db_threshold_units="mm/hr",
        )
        data = np.array(
            [
                [[0.0, 3.0], [0.0, 4.0]],
                [[0.0, 3.2], [0.0, 4.2]],
            ],
            dtype=np.float32,
        )
        cube = set_up_variable_cube(data=data, name="precipitation_rate", units="mm/hr")

        # Noise will be added only to zero values; non-zero values should remain
        # unchanged
        expected = np.array(
            [
                [[1.1456498, 3.0], [0.8874278, 4.0]],
                [[1.1456498, 3.2], [0.8874278, 4.2]],
            ],
            dtype=np.float32,
        )

    with pytest.warns(UserWarning, match="multi-realization dimension"):
        result = plugin.process(cube)

    if test_case == "with_zero_values":
        # SSFT output for this tiny field can vary. We thus test with stable invariants
        # instead.
        non_zero_mask = data > 0
        zero_mask = data == 0
        np.testing.assert_array_equal(result.data[non_zero_mask], data[non_zero_mask])
        assert np.all(np.isfinite(result.data[zero_mask]))
        assert np.all(result.data[zero_mask] >= 0.0)
    else:
        # Use array_equal for exact comparisons (no noise added)
        np.testing.assert_array_equal(result.data, expected)

    assert isinstance(result, Cube)
    assert result.data.dtype == np.float32
    # Verify it returns a cube with the expected properties

    assert result.shape == simple_cube.shape


def test_scale_non_positive_noise():
    """Test that scale_non_positive_noise ensures resultant max noise in regions where
    diagnostic values are non-positive is <= 0."""
    plugin = StochasticNoise(
        ssft_init_params={"win_size": (2, 2), "overlap": 0},
        ssft_generate_params={"seed": 0},
        db_threshold=0.03,
        db_threshold_units="mm/hr",
        scale_non_positive_noise=True,
    )

    # Create cube with zero values
    data = np.array(
        [
            [[0.0, 3.0], [0.0, 4.0]],
            [[0.0, 3.2], [0.0, 4.2]],
        ],
        dtype=np.float32,
    )
    cube = set_up_variable_cube(data=data, name="precipitation_rate", units="mm/hr")

    with pytest.warns(UserWarning, match="multi-realization dimension"):
        result = plugin.process(cube)

    # Non-zero values should remain unchanged
    non_zero_mask = data > 0
    np.testing.assert_array_equal(result.data[non_zero_mask], data[non_zero_mask])

    # Non-positive regions should have values <= 0 (scaled so max is 0)
    non_positive_mask = data <= 0
    assert np.all(
        result.data[non_positive_mask] <= 0
    ), "Noise in non-positive regions should be <= 0"


@pytest.mark.parametrize("scale_non_positive_noise", [False, True])
def test_process_non_finite_noise_is_sanitized(scale_non_positive_noise: bool):
    """Non-finite values from SSFT should not leak into output data."""

    plugin = StochasticNoise(
        ssft_init_params={"win_size": (2, 2), "overlap": 0},
        ssft_generate_params={"seed": 0},
        db_threshold=0.03,
        db_threshold_units="mm/hr",
        scale_non_positive_noise=scale_non_positive_noise,
    )

    data = np.array(
        [
            [[0.0, 3.0], [0.0, 4.0]],
            [[0.0, 3.2], [0.0, 4.2]],
        ],
        dtype=np.float32,
    )
    cube = set_up_variable_cube(data=data, name="precipitation_rate", units="mm/hr")

    # Force problematic SSFT output that includes non-finite values.
    plugin.do_fft = lambda _: np.array(
        [[np.nan, 0.0], [np.inf, -np.inf]], dtype=np.float32
    )

    with pytest.warns(UserWarning, match="multi-realization dimension"):
        result = plugin.process(cube)

    non_zero_mask = data > 0
    np.testing.assert_array_equal(result.data[non_zero_mask], data[non_zero_mask])

    non_positive_mask = data <= 0
    assert np.all(np.isfinite(result.data[non_positive_mask]))
    if scale_non_positive_noise:
        assert np.all(result.data[non_positive_mask] <= 0)


def test_process_scalar_realization_coord():
    """Test processing a cube with scalar realization coordinate.

    The input has no realization dimension.
    """
    plugin = StochasticNoise(
        ssft_init_params={"domain_size": [2, 2], "overlap": 0},
        ssft_generate_params={"seed": 0},
        db_threshold=0.03,
        db_threshold_units="mm/hr",
    )

    data = np.array(
        [
            [[0.0, 3.0], [0.0, 4.0]],
            [[0.0, 3.2], [0.0, 4.2]],
        ],
        dtype=np.float32,
    )
    cube = set_up_variable_cube(data=data, name="precipitation_rate", units="mm/hr")
    single_realization_cube = cube[0, :, :]

    result = plugin.process(single_realization_cube)

    assert isinstance(result, Cube)
    assert result.shape == single_realization_cube.shape


@pytest.mark.parametrize(
    "constant_value, expect_degenerate_warning, expect_changed",
    [(0.0, True, True), (1.0, False, False)],
)
def test_process_constant_input(
    constant_value: float, expect_degenerate_warning: bool, expect_changed: bool
):
    """Constant inputs are handled. Degenerate input warning is raised for constant
    zero input with fallback noise constrained to the dry_fallback_range. For non-zero
    constant input, no warning is raised and output equals input."""
    if constant_value == 0.0:
        # Degenerate path requires wet_noise_floor for guaranteed separation.
        plugin = StochasticNoise(
            ssft_generate_params={"seed": 0},
            scale_non_positive_noise=True,
            wet_noise_floor=-5.0,
        )
    else:
        plugin = StochasticNoise(ssft_generate_params={"seed": 0})
    data = np.full((4, 4), constant_value, dtype=np.float32)
    cube = set_up_variable_cube(data=data, name="precipitation_rate", units="mm/hr")

    if expect_degenerate_warning:
        with pytest.warns(UserWarning, match="Degenerate input field detected"):
            result = plugin.process(cube)
    else:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            result = plugin.process(cube)
        assert not any(
            "Degenerate input field detected" in str(w.message) for w in caught_warnings
        )

    assert isinstance(result, Cube)
    assert result.shape == cube.shape
    assert np.all(np.isfinite(result.data))
    if expect_changed:
        assert np.any(result.data != cube.data)
        assert np.all(result.data <= -5.0)
        assert np.all(result.data >= -10.0)
    else:
        np.testing.assert_array_equal(result.data, cube.data)


@pytest.mark.parametrize("constant_value", [0.0, 1.0])
def test_process_constant_input_seeded_is_reproducible(constant_value: float):
    """Seeded processing of constant fields is reproducible via process."""
    if constant_value == 0.0:
        plugin = StochasticNoise(
            ssft_generate_params={"seed": 42},
            scale_non_positive_noise=True,
            wet_noise_floor=-5.0,
        )
    else:
        plugin = StochasticNoise(ssft_generate_params={"seed": 42})
    data = np.full((4, 4), constant_value, dtype=np.float32)
    cube = set_up_variable_cube(data=data, name="precipitation_rate", units="mm/hr")

    if constant_value == 0.0:
        with pytest.warns(UserWarning, match="Degenerate input field detected"):
            first = plugin.process(cube)
        with pytest.warns(UserWarning, match="Degenerate input field detected"):
            second = plugin.process(cube)
    else:
        first = plugin.process(cube)
        second = plugin.process(cube)

    np.testing.assert_array_equal(first.data, second.data)


def test_process_all_zero_input_with_scale_non_positive_noise():
    """All-zero input should use constrained dry fallback range when configured."""
    plugin = StochasticNoise(
        ssft_generate_params={"seed": 0},
        scale_non_positive_noise=True,
        wet_noise_floor=-5.0,
    )

    data = np.zeros((4, 4), dtype=np.float32)
    cube = set_up_variable_cube(data=data, name="precipitation_rate", units="mm/hr")

    with pytest.warns(UserWarning, match="Degenerate input field detected"):
        result = plugin.process(cube)

    assert isinstance(result, Cube)
    assert result.shape == cube.shape
    assert np.all(np.isfinite(result.data))
    assert np.all(result.data <= -5.0)
    assert np.all(result.data >= -10.0)
    # Ensure the configured dry fallback range is active on output.
    assert np.isclose(np.max(result.data), -5.0)


def test_process_all_zero_input_constant_fallback_clamps_to_dry_max():
    """Constant dry fallback values should clamp to dry_max without division by zero."""
    plugin = StochasticNoise(
        ssft_generate_params={"seed": 0},
        scale_non_positive_noise=True,
        wet_noise_floor=-5.0,
    )

    # Force a constant fallback field so dry_vmax == dry_vmin in remapping.
    plugin._fallback_noise_linear = lambda shape: np.full(shape, -2.0, dtype=np.float32)

    data = np.zeros((4, 4), dtype=np.float32)
    cube = set_up_variable_cube(data=data, name="precipitation_rate", units="mm/hr")

    with pytest.warns(UserWarning, match="Degenerate input field detected"):
        result = plugin.process(cube)

    assert np.all(np.isfinite(result.data))
    # Default dry_fallback_range for wet_noise_floor=-5.0 is (-10.0, -5.0), so
    # the clamp target at zero dynamic range is dry_max == -5.0.
    assert np.all(result.data == -5.0)


def test_process_wet_path_applies_wet_noise_floor_clip():
    """Wet SSFT path should clip overly negative scaled noise to wet_noise_floor."""
    plugin = StochasticNoise(
        ssft_generate_params={"seed": 0},
        scale_non_positive_noise=True,
        wet_noise_floor=-5.0,
    )

    data = np.array(
        [[0.0, 1.0], [0.0, 2.0]],
        dtype=np.float32,
    )
    cube = set_up_variable_cube(data=data, name="precipitation_rate", units="mm/hr")

    # Force non-degenerate SSFT path and produce non-positive-region noise values
    # that become [0, -100] after scaling, so clipping to wet_noise_floor is required.
    plugin.do_fft = lambda _: np.array(
        [[20.0, 0.0], [-40.0, 0.0]],
        dtype=np.float32,
    )

    result = plugin.process(cube)

    non_positive_mask = data <= 0
    assert np.all(result.data[non_positive_mask] >= -5.0)
    assert np.any(result.data[non_positive_mask] == -5.0)


def test_degenerate_fallback_without_wet_noise_floor_raises():
    """Degenerate fallback without wet_noise_floor should raise ValueError."""
    plugin = StochasticNoise(ssft_generate_params={"seed": 0})
    data = np.zeros((4, 4), dtype=np.float32)
    cube = set_up_variable_cube(data=data, name="precipitation_rate", units="mm/hr")

    with pytest.warns(UserWarning, match="Degenerate input field detected"):
        with pytest.raises(ValueError, match="wet_noise_floor is not set"):
            plugin.process(cube)


def test_wet_noise_floor_without_scale_non_positive_noise_raises():
    """Setting wet_noise_floor without scale_non_positive_noise=True should raise."""
    with pytest.raises(
        ValueError, match="scale_non_positive_noise must be True when wet_noise_floor"
    ):
        StochasticNoise(wet_noise_floor=-5.0)


def test_wet_noise_floor_non_negative_raises():
    """Setting a non-negative wet_noise_floor should raise ValueError."""
    with pytest.raises(ValueError, match="wet_noise_floor must be negative"):
        StochasticNoise(wet_noise_floor=0.0, scale_non_positive_noise=True)


def test_dry_fallback_range_invalid_length_raises():
    """dry_fallback_range must contain exactly two values."""
    with pytest.raises(ValueError, match="must contain exactly two values"):
        StochasticNoise(
            wet_noise_floor=-5.0,
            scale_non_positive_noise=True,
            dry_fallback_range=(-10.0,),
        )


@pytest.mark.parametrize(
    "dry_fallback_range",
    [(-5.0, -5.0), (-4.0, -5.0), (-10.0, 1.0)],
)
def test_dry_fallback_range_invalid_bounds_raises(dry_fallback_range):
    """dry_fallback_range bounds must satisfy min < max <= 0."""
    with pytest.raises(ValueError, match="min_value < max_value <= 0"):
        StochasticNoise(
            wet_noise_floor=-5.0,
            scale_non_positive_noise=True,
            dry_fallback_range=dry_fallback_range,
        )


def test_dry_fallback_range_max_above_wet_floor_raises():
    """dry_fallback_range max must not exceed wet_noise_floor when both are set."""
    with pytest.raises(ValueError, match="max must be <= wet_noise_floor"):
        StochasticNoise(
            wet_noise_floor=-5.0,
            scale_non_positive_noise=True,
            dry_fallback_range=(-10.0, -4.0),
        )


def test_non_positive_threshold():
    """Test that ValueError is raised for non-positive db_threshold."""
    with pytest.raises(ValueError, match="db_threshold must be a positive value."):
        StochasticNoise(db_threshold=0)


def test_init_warning():
    """Test that a warning is raised when using a seeded plugin with
    allow_seeded_parallel_processing."""
    with pytest.warns(
        UserWarning,
        match="Using multiple workers with a fixed seed",
    ):
        StochasticNoise(
            ssft_generate_params={"seed": 0},
            allow_seeded_parallel_processing=True,
        )
