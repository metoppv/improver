# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the StochasticNoise plugin"""

import numpy as np
import pytest
from iris.cube import Cube

from improver.precipitation.stochastic_noise import StochasticNoise
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


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


@pytest.mark.parametrize("test_case", ["same_units", "different_units"])
def test__convert_threshold_units(
    plugin: StochasticNoise, simple_cube: Cube, test_case: str
):
    """Test that db_threshold is converted to the correct units."""
    if test_case == "same_units":
        expected_threshold = 0.03

    elif test_case == "different_units":
        simple_cube.units = "mm/s"
        # 0.03 mm/hr = 0.03/3600 mm/s = 8.333e-6 mm/s
        expected_threshold = 8.333e-06

    result = plugin._convert_threshold_units(simple_cube)
    assert np.isclose(result, expected_threshold)


def test__to_dB_and__from_dB(plugin: StochasticNoise, simple_cube: Cube):
    """Test that _to_dB and _from_dB are inverses of each other."""
    cube = simple_cube.copy()
    dB_cube = plugin._to_dB(cube.copy())
    restored_cube = plugin._from_dB(dB_cube.data, simple_cube)
    db_threshold = plugin.db_threshold
    expected = np.where(simple_cube.data < db_threshold, 0, simple_cube.data)
    np.testing.assert_allclose(restored_cube.data, expected, rtol=1e-6)


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
def test_stochastic_noise_to_input_cube(
    plugin: StochasticNoise,
    simple_cube: Cube,
    test_case: str,
):
    """Test stochastic_noise_to_input_cube method."""
    cube = simple_cube.copy()

    # All values in simple_cube > 0 (not dry), so no noise should be added (i.e., output
    # should equal input)
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

        # Noise will be added only to zero values; non-zero values should remain
        # unchanged
        expected = [
            [[1.1456498, 3.0], [0.8874278, 4.0]],
            [[1.1456498, 3.2], [0.8874278, 4.2]],
        ]

    result = plugin.stochastic_noise_to_input_cube(cube)

    if test_case == "with_zero_values":
        # Use allclose for floating point comparisons
        np.testing.assert_allclose(result.data, expected, rtol=1e-6)
    else:
        # Use array_equal for exact comparisons (no noise added)
        np.testing.assert_array_equal(result.data, expected)
    assert result.data.dtype == np.float32


def test_process(plugin: StochasticNoise, simple_cube: Cube):
    """Test the process method (public API wrapper)."""
    result = plugin.process(simple_cube)

    # Verify it returns a cube with the expected properties
    assert isinstance(result, Cube)
    assert result.shape == simple_cube.shape
    assert result.data.dtype == np.float32

    # All values in simple_cube are non-zero, so output should equal input
    np.testing.assert_array_equal(result.data, simple_cube.data)


def test_scale_dry_noise():
    """Test that scale_dry_noise ensures max noise in dry regions is <= 0."""
    plugin = StochasticNoise(
        ssft_init_params={"domain_size": [2, 2], "overlap": 0},
        ssft_generate_params={"seed": 0},
        db_threshold=0.03,
        db_threshold_units="mm/hr",
        scale_dry_noise=True,
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

    result = plugin.stochastic_noise_to_input_cube(cube)

    # Non-zero values should remain unchanged
    non_zero_mask = data > 0
    np.testing.assert_array_equal(result.data[non_zero_mask], data[non_zero_mask])

    # Dry regions should have values <= 0 (scaled so max is 0)
    dry_mask = data <= 0
    assert np.all(result.data[dry_mask] <= 0), "Noise in dry regions should be <= 0"


def test_non_positive_threshold():
    """Test that ValueError is raised for non-positive db_threshold."""
    with pytest.raises(ValueError, match="db_threshold must be a positive value"):
        StochasticNoise(db_threshold=0)
