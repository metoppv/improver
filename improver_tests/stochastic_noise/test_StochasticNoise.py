# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the StochasticNoise plugin"""

import numpy as np
import pytest
from iris.cube import Cube

from improver.stochastic_noise import StochasticNoise
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
        expected = np.array(
            [
                [[1.1456498, 3.0], [0.8874278, 4.0]],
                [[1.1456498, 3.2], [0.8874278, 4.2]],
            ],
            dtype=np.float32,
        )

    result = plugin.process(cube)

    if test_case == "with_zero_values":
        # Use allclose for floating point comparisons
        np.testing.assert_allclose(result.data, expected, rtol=1e-6)
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
        ssft_init_params={"domain_size": [2, 2], "overlap": 0},
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

    result = plugin.process(cube)

    # Non-zero values should remain unchanged
    non_zero_mask = data > 0
    np.testing.assert_array_equal(result.data[non_zero_mask], data[non_zero_mask])

    # Non-positive regions should have values <= 0 (scaled so max is 0)
    non_positive_mask = data <= 0
    assert np.all(
        result.data[non_positive_mask] <= 0
    ), "Noise in non-positive regions should be <= 0"


def test_non_positive_threshold():
    """Test that ValueError is raised for non-positive db_threshold."""
    with pytest.raises(ValueError, match="db_threshold must be a positive value."):
        StochasticNoise(db_threshold=0)
