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
def simple_reference_array():
    """Fixture for creating a simple reference array."""
    return np.array([10, 20, 30, 40, 50])


@pytest.fixture
def simple_forecast_array():
    """Fixture for creating a simple forecast array"""
    return np.array([5, 15, 25, 35, 45])


def test__build_empirical_cdf(simple_reference_array):
    """Test _build_empirical_cdf returns the correct empirical CDF."""
    sorted_values, quantiles = QuantileMapping()._build_empirical_cdf(
        simple_reference_array
    )

    np.testing.assert_array_equal(sorted_values, np.array([10, 20, 30, 40, 50]))
    np.testing.assert_array_equal(quantiles, np.array([0.2, 0.4, 0.6, 0.8, 1.0]))


def test__inverted_cdf(simple_reference_array):
    """Test _inverted_cdf returns the correct values. Values output should be the
    same as values input in this case."""
    _, quantiles = QuantileMapping()._build_empirical_cdf(simple_reference_array)
    result = QuantileMapping()._inverted_cdf(simple_reference_array, quantiles)
    np.testing.assert_array_equal(result, np.array([10, 20, 30, 40, 50]))


def test__map_quantiles(
    simple_reference_array,
    simple_forecast_array,
):
    expected = np.array([10, 20, 30, 40, 50])
    result = QuantileMapping()._map_quantiles(
        simple_reference_array,
        simple_forecast_array,
    )
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
                [
                    6.8,
                    7.7,
                    8.7,
                ],
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


@pytest.fixture
def expected_result_no_threshold():
    """Expected result for quantile mapping without a preservation threshold."""
    return np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[0.7, 1.8, 2.8], [3.8, 4.9, 5.8], [6.8, 7.7, 8.7]],
        ],
        dtype=np.float32,
    )


@pytest.mark.parametrize(
    "test_case",
    [
        "same_units",
        "different_units",
        "incompatible_units",
    ],
)
def test__convert_reference_cube_to_forecast(
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


def test_quantile_mapping_process_no_threshold(
    reference_cube, forecast_cube, expected_result_no_threshold
):
    """Test quantile mapping with no preservation threshold."""
    plugin = QuantileMapping()
    result = plugin.process(reference_cube, forecast_cube)

    assert isinstance(result, Cube)
    assert result.shape == forecast_cube.shape
    assert result.data.dtype == np.float32
    assert not np.ma.is_masked(result.data)
    np.testing.assert_array_equal(result.data, expected_result_no_threshold)


def test_quantile_mapping_process_with_threshold(reference_cube, forecast_cube):
    """Test quantile mapping with preservation threshold.
    Index [1,0,0] should remain 0.5, despite the reference normally transforming
    it to the reference value of 0.7.
    """
    plugin = QuantileMapping(preservation_threshold=0.51)
    result = plugin.process(reference_cube, forecast_cube)

    expected_result = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[0.5, 1.8, 2.8], [3.8, 4.9, 5.8], [6.8, 7.7, 8.7]],
        ],
        dtype=np.float32,
    )

    assert isinstance(result, Cube)
    assert result.shape == forecast_cube.shape
    assert result.data.dtype == np.float32
    assert result.data.mask is not False
    np.testing.assert_array_equal(result.data, expected_result)


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
