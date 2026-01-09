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
        threshold=0.03,
        threshold_units="mm/hr",
    )
    return plugin


@pytest.fixture
def simple_cube():
    """
    Create a simple cube with two realizations for testing, where values to the bottom
    left are lower than all others, but bottom right are the highest. If spatial
    structure is maintained, the noise added to the bottom left should be lower
    than that added to the bottom right.
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


@pytest.mark.parametrize(
    "test_case", ["same_units", "different_units", "incompatible_units"]
)
def test__convert_threshold_units(
    plugin: StochasticNoise, simple_cube: Cube, test_case: str
):
    """Test that threshold is converted to the correct units."""
    if test_case == "same_units":
        expected_threshold = 0.03

    elif test_case == "different_units":
        simple_cube.units = "mm/s"
        # 0.03 mm/hr = 0.03/3600 mm/s = 8.333e-6 mm/s
        expected_threshold = 8.333e-06

    elif test_case == "incompatible_units":
        simple_cube.units = "K"
        with pytest.raises(ValueError, match="Cannot convert"):
            plugin._convert_threshold_units(simple_cube)
        return

    result = plugin._convert_threshold_units(simple_cube)
    assert np.isclose(result, expected_threshold)


def test__to_dB_and__from_dB(plugin: StochasticNoise, simple_cube: Cube):
    """Test that _to_dB and _from_dB are inverses of each other."""
    cube = simple_cube.copy()
    dB_cube = plugin._to_dB(cube.copy())
    restored_cube = plugin._from_dB(dB_cube.copy())
    threshold = plugin.threshold
    expected = np.where(simple_cube.data < threshold, 0, simple_cube.data)
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
        "fully_masked_array",
        "unmodified_below_threshold",
    ],
)
def test_stochastic_noise_to_dependence_template(
    plugin: StochasticNoise,
    simple_cube: Cube,
    test_case: str,
):
    """Test stochastic_noise_to_dependence_template method."""

    base_expected = np.array(
        [
            [[3.5848932, 4.995262], [2.2589254, 6.5118866]],
            [[3.859587, 5.289296], [2.5182567, 6.830268]],
        ],
        dtype=np.float32,
    )

    cube = simple_cube.copy()
    expected = base_expected.copy()

    if test_case == "plugin_defaults":
        plugin = StochasticNoise()  # Use default parameters

    elif test_case in ["some_data_masked", "fully_masked_array"]:
        cube.data = np.ma.masked_array(cube.data, mask=False, dtype=np.float32)

        if test_case == "some_data_masked":
            cube.data[0, 0, 0] = np.ma.masked
            cube.data[1, 1, 1] = np.ma.masked

            expected = np.ma.masked_array(base_expected, mask=False, dtype=np.float32)
            expected[0, 0, 0] = np.ma.masked
            expected[1, 1, 1] = np.ma.masked

        elif test_case == "fully_masked_array":
            cube.data[:, :, :] = np.ma.masked
            expected = cube.data

        cube.data = np.ma.filled(cube.data, 0).astype(np.float32)
        expected = np.ma.filled(expected, 0).astype(np.float32)

    elif test_case == "unmodified_below_threshold":
        cube.data[0, 0, 0] = 0.01  # Below threshold of 0.03
        cube.data[1, 0, 0] = 0.02  # Below threshold of 0.03

        expected = base_expected.copy()
        expected[0, 0, 0] = 0.01
        expected[1, 0, 0] = 0.02

    result = plugin.stochastic_noise_to_dependence_template(cube)

    # Use looser tolerance for stochastic/FFT-based calculations
    # which can vary slightly across platforms and library versions
    np.testing.assert_allclose(result.data, expected, rtol=1e-4, atol=1e-6)
    assert result.data.dtype == np.float32


def test_process(plugin: StochasticNoise, simple_cube: Cube):
    """Test the process method (public API wrapper)."""
    result = plugin.process(simple_cube)

    # Verify it returns a cube with the expected properties
    assert isinstance(result, Cube)
    assert result.shape == simple_cube.shape
    assert result.data.dtype == np.float32

    # Verify noise was added (output differs from input)
    assert not np.allclose(result.data, simple_cube.data)
