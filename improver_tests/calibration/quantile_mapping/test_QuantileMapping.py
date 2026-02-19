# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

import numpy as np
import pytest
from iris.cube import Cube

from improver.calibration.quantile_mapping import (
    QuantileMapping,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


@pytest.fixture
def uniform_reference_array():
    """Fixture for creating a simple reference array."""
    return np.array([10, 20, 30, 40, 50])


@pytest.fixture
def uniform_forecast_array():
    """Fixture for creating a simple forecast array"""
    return np.array([5, 15, 25, 35, 45])


@pytest.fixture
def complex_reference_array():
    """Non-uniform spacing with outliers for edge case testing."""
    return np.array([5, 10, 15, 50, 100])


@pytest.fixture
def complex_forecast_array():
    """Non-uniform forecast with different distribution characteristics."""
    return np.array([8, 12, 20, 45, 90])


@pytest.fixture
def duplicate_zeroes_reference_array():
    """Fixture for creating a reference array with duplicate zeroes."""
    return np.array([0, 0, 0, 0, 0, 0, 0, 10, 20, 40, 50])


@pytest.fixture
def duplicate_zeroes_forecast_array():
    """Fixture for creating a forecast array with duplicate zeroes.
    Contains 8 zeroes (7 in the reference array).
    """
    return np.array([0, 0, 0, 0, 0, 0, 0, 0, 10, 20, 30])


def test___init__():
    """Test that QuantileMapping can be instantiated with default and custom parameters."""
    # Test default parameters
    plugin_default = QuantileMapping()
    assert plugin_default.preservation_threshold is None
    assert plugin_default.method == "step"

    # Test custom parameters
    plugin_custom = QuantileMapping(preservation_threshold=0.5, method="continuous")
    assert plugin_custom.preservation_threshold == 0.5
    assert plugin_custom.method == "continuous"

    # Test invalid method raises error
    with pytest.raises(ValueError, match="Unsupported method"):
        QuantileMapping(method="unsupported_method")


@pytest.mark.parametrize(
    ["method", "num_points", "expected_quantiles"],
    [
        ("step", 5, np.array([0.2, 0.4, 0.6, 0.8, 1.0])),
        ("continuous", 5, np.array([0.1, 0.3, 0.5, 0.7, 0.9])),
        ("step", 3, np.array([1 / 3, 2 / 3, 1.0])),
        ("continuous", 3, np.array([1 / 6, 0.5, 5 / 6])),
        ("step", 1, np.array([1.0])),
        ("continuous", 1, np.array([0.5])),
    ],
)
def test__plotting_positions(method, num_points, expected_quantiles):
    """Test _plotting_positions returns correct quantiles for both methods.

    Tests:
    - Standard plotting positions (i/n) for 'step' method
    - Midpoint plotting positions ((i-0.5)/n) for 'continuous' method
    - Edge cases with small sample sizes (n=1, n=3)
    """
    plugin = QuantileMapping(method=method)
    result = plugin._plotting_positions(num_points)
    np.testing.assert_allclose(result, expected_quantiles, rtol=1e-10)


@pytest.mark.parametrize(
    ["method", "input_data", "expected_sorted", "expected_quantiles"],
    [
        # Uniform data - 'step' method
        (
            "step",
            np.array([10, 20, 30, 40, 50]),
            np.array([10, 20, 30, 40, 50]),
            np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
        ),
        # Uniform data - 'continuous' method
        (
            "continuous",
            np.array([10, 20, 30, 40, 50]),
            np.array([10, 20, 30, 40, 50]),
            np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
        ),
        # Unsorted data - should be sorted
        (
            "step",
            np.array([50, 10, 30, 20, 40]),
            np.array([10, 20, 30, 40, 50]),
            np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
        ),
        # Data with duplicates - 'step' method
        (
            "step",
            np.array([10, 10, 20, 30, 30]),
            np.array([10, 10, 20, 30, 30]),
            np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
        ),
        # Data with duplicates - 'continuous' method
        (
            "continuous",
            np.array([10, 10, 20, 30, 30]),
            np.array([10, 10, 20, 30, 30]),
            np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
        ),
        # Single value
        ("step", np.array([42.0]), np.array([42.0]), np.array([1.0])),
        ("continuous", np.array([42.0]), np.array([42.0]), np.array([0.5])),
    ],
)
def test__build_empirical_cdf(method, input_data, expected_sorted, expected_quantiles):
    """Test _build_empirical_cdf for various data patterns.

    Tests:
    - Sorted and unsorted data
    - Data with duplicate values
    - Single-value arrays
    - Both 'step' and 'continuous' methods
    """
    plugin = QuantileMapping(method=method)
    sorted_values, quantiles = plugin._build_empirical_cdf(input_data)

    np.testing.assert_array_equal(sorted_values, expected_sorted)
    np.testing.assert_allclose(quantiles, expected_quantiles, rtol=1e-10)


@pytest.mark.parametrize(
    ["method", "forecast", "expected_quantiles"],
    [
        ("step", np.array([10, 20, 30, 40, 50]), np.array([0.2, 0.4, 0.6, 0.8, 1.0])),
        (
            "continuous",
            np.array([10, 20, 30, 40, 50]),
            np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
        ),
    ],
)
def test__forecast_to_quantiles_methods(method, forecast, expected_quantiles):
    """Test that _forecast_to_quantiles correctly assigns quantiles for both methods."""
    result = QuantileMapping(method=method)._forecast_to_quantiles(forecast)
    np.testing.assert_allclose(result, expected_quantiles, rtol=1e-10)


@pytest.mark.parametrize(
    ["method", "quantiles", "expected"],
    [
        ("step", np.array([0.1, 0.5, 0.9]), np.array([10, 30, 40])),
        ("continuous", np.array([0.1, 0.5, 0.9]), np.array([10, 30, 50])),
    ],
)
def test__inverted_cdf_methods(uniform_reference_array, method, quantiles, expected):
    """Test _inverted_cdf returns correct values for both methods."""
    plugin = QuantileMapping(method=method)
    result = plugin._inverted_cdf(uniform_reference_array, quantiles)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize(
    ["method", "expected"],
    [
        ("step", np.array([10, 20, 30, 40, 50])),
        ("continuous", np.array([10, 20, 30, 40, 50])),
    ],
)
def test__map_quantiles(
    method,
    expected,
    uniform_reference_array,
    uniform_forecast_array,
):
    """Test _map_quantiles returns the correct mapped values."""
    result = QuantileMapping(method=method)._map_quantiles(
        uniform_reference_array,
        uniform_forecast_array,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ["method", "reference_array", "forecast_array", "expected"],
    [
        (  # Single values, step method
            "step",
            np.array([40]),
            np.array([100]),
            np.array([40]),
        ),
        (  # Single values, continuous method
            "continuous",
            np.array([40]),
            np.array([100]),
            np.array([40]),
        ),
        (  # All identical values, step
            "step",
            np.array([5, 5, 5]),
            np.array([10, 10, 10]),
            np.array([5, 5, 5]),
        ),
        (  # All identical values, continuous
            "continuous",
            np.array([5, 5, 5]),
            np.array([10, 10, 10]),
            np.array([5, 5, 5]),
        ),
        (  # Different sized arrays, step method
            "step",
            np.array([20, 25, 30, 35, 40]),
            np.array([0, 5, 10, 15]),
            np.array([25, 30, 35, 40]),
        ),
        (  # Different sized arrays, continuous method
            "continuous",
            np.array([20, 25, 30, 35, 40]),
            np.array([0, 5, 10, 15]),
            np.array([20.625, 26.875, 33.125, 39.375]),
        ),
        (  # Duplicate zeroes in reference, step method
            "step",
            np.array([0, 0, 0, 0, 0, 0, 0, 10, 20, 40, 50]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 10, 20, 30]),
            np.array([10, 10, 10, 10, 10, 10, 10, 10, 20, 40, 50]),
        ),
        (  # Duplicate zeroes in reference, continuous method
            "continuous",
            np.array([0, 0, 0, 0, 0, 0, 0, 10, 20, 40, 50]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 10, 20, 30]),
            np.array([0, 0, 0, 0, 0, 0, 0, 10, 20, 40, 50]),
        ),
    ],
)
def test__map_quantiles_edge_cases(method, reference_array, forecast_array, expected):
    """Test _map_quantiles handles edge cases correctly."""
    plugin = QuantileMapping(method=method)
    result = plugin._map_quantiles(reference_array, forecast_array)
    np.testing.assert_array_equal(result, expected)


@pytest.fixture
def reference_cube():
    """Fixture for creating a reference precipitation rate (mm/h) cube."""
    data = np.array(
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            [
                [0.7, 1.8, 2.8],
                [3.8, 4.9, 5.8],
                [6.8, 7.7, 8.7],
            ],
        ],
        dtype=np.float32,
    )

    return set_up_variable_cube(data, name="lwe_precipitation_rate", units="mm h-1")


@pytest.fixture
def forecast_cube():
    """Fixture for creating a forecast precipitation rate (mm/h) cube."""
    data = np.array(
        [
            [
                [0.6, 1.7, 2.7],
                [3.7, 4.8, 5.7],
                [6.7, 7.6, 8.6],
            ],
            [
                [0.5, 1.6, 2.6],
                [3.6, 4.7, 5.6],
                [6.6, 7.5, 8.5],
            ],
        ],
        dtype=np.float32,
    )
    return set_up_variable_cube(data, name="lwe_precipitation_rate", units="mm h-1")


@pytest.mark.parametrize(
    "test_case",
    [
        "same_units",
        "different_units",
        "incompatible_units",
    ],
)
def test__convert_reference_cube_to_forecast_units(
    reference_cube,
    forecast_cube,
    test_case,
):
    """Test handling of cubes with same, different, and incompatible units."""
    plugin = QuantileMapping()

    if test_case == "same_units":
        # Both cubes already in mm h-1, should work normally
        result = plugin.process(reference_cube, forecast_cube)
        assert result.units == forecast_cube.units

    elif test_case == "different_units":
        # Convert forecast to different (but compatible) units
        forecast_cube_copy = forecast_cube.copy()
        forecast_cube_copy.convert_units("m s-1")
        result = plugin.process(reference_cube, forecast_cube_copy)
        # Result should be in forecast units (m s-1)
        assert result.units == forecast_cube_copy.units

    elif test_case == "incompatible_units":
        # Set incompatible units and expect error
        forecast_cube_copy = forecast_cube.copy()
        forecast_cube_copy.units = "Celsius"
        with pytest.raises(ValueError, match="Cannot convert cube with units"):
            plugin.process(reference_cube, forecast_cube_copy)


@pytest.mark.parametrize(
    ["preservation_threshold", "expected"],
    [
        (
            None,
            np.array(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                    [[0.7, 1.8, 2.8], [3.8, 4.9, 5.8], [6.8, 7.7, 8.7]],
                ],
                dtype=np.float32,
            ),
        ),
        (
            0.51,
            np.array(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                    [[0.5, 1.8, 2.8], [3.8, 4.9, 5.8], [6.8, 7.7, 8.7]],
                ],
                dtype=np.float32,
            ),
        ),
    ],
)
def test_quantile_mapping_process_thresholding(
    reference_cube, forecast_cube, preservation_threshold, expected
):
    """Test quantile mapping with and without a preservation threshold."""
    plugin = QuantileMapping(preservation_threshold=preservation_threshold)
    result = plugin.process(reference_cube, forecast_cube)

    assert isinstance(result, Cube)
    assert result.shape == forecast_cube.shape
    assert result.data.dtype == np.float32
    assert not np.ma.is_masked(result.data)
    np.testing.assert_array_equal(result.data, expected)


@pytest.mark.parametrize(
    "test_case",
    [
        "one_input_masked",
        "both_inputs_masked",
    ],
)
def test_masked_input(reference_cube, forecast_cube, test_case):
    """Test behaviour when one or both inputs have masked values.
    In both cases, the mask should be a union of cube masks."""

    # Make copies to avoid fixture mutation
    reference_cube = reference_cube.copy()
    forecast_cube = forecast_cube.copy()

    # Mask reference at position [0, 0, 0]
    reference_cube.data = np.ma.masked_array(
        reference_cube.data, mask=np.zeros_like(reference_cube.data, dtype=bool)
    )
    reference_cube.data[0, 0, 0] = np.ma.masked

    if test_case == "one_input_masked":
        expected_mask_count = 1

    elif test_case == "both_inputs_masked":
        # Also mask forecast at position [0, 0, 1]
        forecast_cube.data = np.ma.masked_array(
            forecast_cube.data, mask=np.zeros_like(forecast_cube.data, dtype=bool)
        )
        forecast_cube.data[0, 0, 1] = np.ma.masked
        expected_mask_count = 2

    plugin = QuantileMapping()
    result = plugin.process(reference_cube, forecast_cube)

    # Check that result is masked
    assert np.ma.is_masked(result.data)
    # Check mask count matches expected (union of input masks)
    assert expected_mask_count == np.ma.count_masked(result.data)
    # Check that the correct positions are masked
    if test_case == "one_input_masked":
        assert result.data.mask[0, 0, 0]
        assert not result.data.mask[0, 0, 1]
    elif test_case == "both_inputs_masked":
        assert result.data.mask[0, 0, 0]
        assert result.data.mask[0, 0, 1]


def test_metadata_preservation(reference_cube, forecast_cube):
    """Test that metadata from forecast cube is preserved."""
    plugin = QuantileMapping()
    reference_cube.long_name = "kittens"
    result = plugin.process(reference_cube, forecast_cube)

    # Check key metadata is preserved
    assert result.long_name == forecast_cube.long_name
