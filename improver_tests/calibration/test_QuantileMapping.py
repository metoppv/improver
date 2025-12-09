# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

import numpy as np
import pytest
from iris.cube import Cube

from improver.calibration.quantile_mapping import (
    QuantileMapping,
    _build_empirical_cdf,
    _interpolated_inverted_cdf,
    _inverted_cdf,
    quantile_mapping,
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


@pytest.fixture
def simple_new_values_to_map_array():
    """Fixture for creating a simple alternative forecast array to correct using mapping
    from a reference array and forecast array.
    """
    return np.array([7.5, 17.5, 27.5, 37.5, 47.5])


def test__build_empirical_cdf(simple_reference_array):
    """Test _build_empirical_cdf returns the correct empirical CDF."""
    sorted_values, quantiles = _build_empirical_cdf(simple_reference_array)

    np.testing.assert_array_equal(sorted_values, np.array([10, 20, 30, 40, 50]))
    np.testing.assert_array_equal(quantiles, np.array([0.2, 0.4, 0.6, 0.8, 1.0]))


def test__inverted_cdf(simple_reference_array):
    """Test _inverted_cdf returns the correct values. Values output should be the
    same as values input in this case."""
    _, quantiles = _build_empirical_cdf(simple_reference_array)
    result = _inverted_cdf(simple_reference_array, quantiles)
    np.testing.assert_array_equal(result, np.array([10, 20, 30, 40, 50]))


def test__interpolated_inverted_cdf(simple_reference_array):
    """Test _interpolated_inverted_cdf returns correct interpolated values."""
    # Test with quantiles that fall between the reference data points
    target_quantiles = np.array([0.3, 0.5, 0.7, 0.9])
    result = _interpolated_inverted_cdf(simple_reference_array, target_quantiles)
    # At quartile 0.3: interpolate between 0.2 (10) and 0.4 (20) -> 15
    # At quartile 0.5: interpolate between 0.4 (20) and 0.6 (30) -> 25
    # At quartile 0.7: interpolate between 0.6 (30) and 0.8 (40) -> 35
    # At quartile 0.9: interpolate between 0.8 (40) and 1.0 (50) -> 45
    expected = np.array([15, 25, 35, 45])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "use_new_values, mapping_method, expected",
    [
        (False, "floor", np.array([10, 20, 30, 40, 50])),
        (False, "interp", np.array([10, 20, 30, 40, 50])),
        (True, "floor", np.array([20, 20, 30, 40, 50])),
        (True, "interp", np.array([12.5, 22.5, 32.5, 42.5, 50])),
    ],
    ids=[
        "same_values_to_map_floor",
        "same_values_to_map_interp",
        "different_values_to_map_floor",
        "different_values_to_map_interp",
    ],
)
def test_quantile_mapping(
    simple_reference_array,
    simple_forecast_array,
    simple_new_values_to_map_array,
    use_new_values,
    mapping_method,
    expected,
):
    values_to_map = (
        simple_new_values_to_map_array if use_new_values else simple_forecast_array
    )
    result = quantile_mapping(
        simple_reference_array,
        simple_forecast_array,
        values_to_map,
        mapping_method=mapping_method,
    )
    np.testing.assert_array_equal(result, expected)


def test_invalid_mapping_method_raises_error(
    simple_reference_array, simple_forecast_array
):
    """Test that invalid mapping_method raises ValueError."""
    with pytest.raises(ValueError, match="Unknown mapping method"):
        quantile_mapping(
            simple_reference_array, simple_forecast_array, mapping_method="kitten"
        )


@pytest.fixture
def reference_cube():
    """Fixture for creating a reference precipitation rate (mm/s) cube."""
    data = np.array(
        [
            [
                [2.63564289e-07, 8.47503543e-08, 3.35276127e-08],
                [4.65661287e-08, 2.14204192e-08, 1.67638063e-08],
                [8.38190317e-09, 1.21071935e-08, 2.23517418e-08],
            ],
            [
                [5.58793545e-09, 3.81842256e-08, 2.03959644e-07],
                [2.51457095e-08, 6.61239028e-08, 1.89989805e-07],
                [5.49480319e-08, 9.40635800e-08, 1.64844096e-07],
            ],
        ],
        dtype=np.float32,
    )

    return set_up_variable_cube(data, units="mm h-1")


@pytest.fixture
def forecast_cube():
    """Fixture for creating a forecast precipitation rate (mm/s) cube."""
    data = np.array(
        [
            [
                [4.7218055e-07, 9.1269612e-07, 1.3476238e-06],
                [8.7451190e-07, 1.4798716e-06, 1.9185245e-06],
                [9.0710819e-07, 1.3411045e-06, 1.6242266e-06],
            ],
            [
                [3.4458935e-08, 1.3038516e-08, 3.7252903e-09],
                [5.7742000e-08, 2.1420419e-08, 2.7939677e-09],
                [1.1455268e-07, 4.0046871e-08, 6.5192580e-09],
            ],
        ],
        dtype=np.float32,
    )
    return set_up_variable_cube(data, units="mm h-1")


@pytest.fixture
def custom_values_to_map_cube():
    """Fixture for creating custom values to map cube (different from forecast cube)."""
    data = np.array(
        [
            [[1e-7, 2e-7, 3e-7], [4e-7, 5e-7, 6e-7], [7e-7, 8e-7, 9e-7]],
            [[1e-8, 2e-8, 3e-8], [4e-8, 5e-8, 6e-8], [7e-8, 8e-8, 9e-8]],
        ],
        dtype=np.float32,
    )
    return set_up_variable_cube(data, units="mm h-1")


@pytest.fixture
def expected_result_floor_no_threshold():
    """Expected result for quantile mapping with floor mapping_method, no threshold."""
    return np.array(
        [
            [
                [4.65661287e-08, 8.47503543e-08, 1.64844096e-07],
                [5.49480319e-08, 1.89989805e-07, 2.63564289e-07],
                [6.61239028e-08, 9.40635800e-08, 2.03959644e-07],
            ],
            [
                [2.23517418e-08, 1.67638063e-08, 8.38190317e-09],
                [3.35276127e-08, 2.14204192e-08, 5.58793545e-09],
                [3.81842256e-08, 2.51457095e-08, 1.21071935e-08],
            ],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def expected_result_floor_with_threshold():
    """Expected result for quantile mapping with floor mapping_method and preservation threshold."""
    return np.array(
        [
            [
                [4.65661287e-08, 8.47503543e-08, 1.64844096e-07],
                [5.49480319e-08, 1.89989805e-07, 2.63564289e-07],
                [6.61239028e-08, 9.40635800e-08, 2.03959644e-07],
            ],
            [
                [2.23517418e-08, 1.67638063e-08, 3.7252903e-09],
                [3.35276127e-08, 2.14204192e-08, 2.7939677e-09],
                [3.81842256e-08, 2.51457095e-08, 6.5192580e-09],
            ],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def expected_result_interp_no_threshold():
    """Expected result for quantile mapping with interp mapping_method, no threshold."""
    return np.array(
        [
            [
                [4.65661287e-08, 8.47503543e-08, 1.64844096e-07],
                [5.49480319e-08, 1.89989805e-07, 2.63564289e-07],
                [6.61239028e-08, 9.40635800e-08, 2.03959644e-07],
            ],
            [
                [2.23517418e-08, 1.67638063e-08, 8.38190317e-09],
                [3.35276127e-08, 2.14204192e-08, 5.58793545e-09],
                [3.81842256e-08, 2.51457095e-08, 1.21071935e-08],
            ],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def expected_result_interp_with_threshold():
    """Expected result for quantile mapping with interp mapping_method and preservation threshold."""
    return np.array(
        [
            [
                [4.6566129e-08, 8.4750354e-08, 1.6484410e-07],
                [5.4948032e-08, 1.8998981e-07, 2.6356429e-07],
                [6.6123903e-08, 9.4063580e-08, 2.0395964e-07],
            ],
            [
                [2.2351742e-08, 1.6763806e-08, 3.7252903e-09],
                [3.3527613e-08, 2.1420419e-08, 2.7939677e-09],
                [3.8184226e-08, 2.5145710e-08, 6.5192580e-09],
            ],
        ],
        dtype=np.float32,
    )


def test_quantile_mapping_process_floor_no_threshold(
    reference_cube, forecast_cube, expected_result_floor_no_threshold
):
    """Test quantile mapping with floor method and no threshold."""
    plugin = QuantileMapping()
    result = plugin.process(reference_cube, forecast_cube, mapping_method="floor")

    assert isinstance(result, Cube)
    assert result.shape == forecast_cube.shape
    assert result.data.dtype == np.float32
    np.testing.assert_array_equal(result.data, expected_result_floor_no_threshold)


def test_quantile_mapping_process_floor_with_threshold(
    reference_cube, forecast_cube, expected_result_floor_with_threshold
):
    """Test quantile mapping with floor method and preservation threshold."""
    plugin = QuantileMapping(preservation_threshold=8.333333e-09)
    result = plugin.process(reference_cube, forecast_cube, mapping_method="floor")

    assert isinstance(result, Cube)
    assert result.shape == forecast_cube.shape
    assert result.data.dtype == np.float32
    np.testing.assert_array_equal(result.data, expected_result_floor_with_threshold)


def test_quantile_mapping_process_interp_no_threshold(
    reference_cube, forecast_cube, expected_result_interp_no_threshold
):
    """Test quantile mapping with interp method and no threshold."""
    plugin = QuantileMapping()
    result = plugin.process(reference_cube, forecast_cube, mapping_method="interp")

    assert isinstance(result, Cube)
    assert result.shape == forecast_cube.shape
    assert result.data.dtype == np.float32
    np.testing.assert_array_equal(result.data, expected_result_interp_no_threshold)


def test_quantile_mapping_process_interp_with_threshold(
    reference_cube, forecast_cube, expected_result_interp_with_threshold
):
    """Test quantile mapping with interp method and preservation threshold."""
    plugin = QuantileMapping(preservation_threshold=8.333333e-09)
    result = plugin.process(reference_cube, forecast_cube, mapping_method="interp")

    assert isinstance(result, Cube)
    assert result.shape == forecast_cube.shape
    assert result.data.dtype == np.float32
    np.testing.assert_array_equal(result.data, expected_result_interp_with_threshold)


def test_quantile_mapping_process_custom_values_to_map(
    reference_cube, forecast_cube, custom_values_to_map_cube
):
    """Test quantile mapping with custom forecast_to_calibrate cube."""
    plugin = QuantileMapping()

    result_custom = plugin.process(
        reference_cube,
        forecast_cube,
        forecast_to_calibrate=custom_values_to_map_cube,
        mapping_method="interp",
    )
    result_default = plugin.process(
        reference_cube, forecast_cube, mapping_method="interp"
    )

    # Results should be different since we're mapping different values
    assert not np.array_equal(result_custom.data, result_default.data)
    assert result_custom.shape == custom_values_to_map_cube.shape
    assert result_custom.data.dtype == np.float32


def test_mask_preservation(reference_cube, forecast_cube):
    """Test that masks are preserved in output."""
    # Mask some values
    forecast_cube.data = np.ma.masked_where(
        forecast_cube.data <= 2.7939677e-09, forecast_cube.data
    )
    reference_cube.data = np.ma.masked_where(
        reference_cube.data <= 2.7939677e-09, reference_cube.data
    )

    plugin = QuantileMapping()
    result = plugin.process(reference_cube, forecast_cube)

    # Check output is masked
    assert np.ma.is_masked(result.data)
    # Check mask count matches forecast
    assert np.ma.count_masked(result.data) == np.ma.count_masked(forecast_cube.data)


def test_non_masked_input_produces_non_masked_output(reference_cube, forecast_cube):
    """Test that non-masked inputs produce non-masked outputs."""
    plugin = QuantileMapping()
    result = plugin.process(reference_cube, forecast_cube)

    assert not np.ma.is_masked(result.data)


def test_unit_conversion(reference_cube, forecast_cube):
    """Test that unit conversion is handled correctly."""
    # Convert reference cube to different units
    reference_cube.convert_units("m h-1")

    plugin = QuantileMapping()
    result = plugin.process(reference_cube, forecast_cube)

    # Result should have forecast units
    assert result.units == forecast_cube.units


def test_incompatible_units_raises_error(reference_cube, forecast_cube):
    """Test that incompatible units raise ValueError."""
    # Change reference cube to incompatible units
    reference_cube.units = "K"  # Temperature instead of precipitation

    plugin = QuantileMapping()

    with pytest.raises(ValueError, match="Cannot convert cube with units"):
        plugin.process(reference_cube, forecast_cube)


def test_threshold_preserves_small_values(reference_cube, forecast_cube):
    """Test that values below threshold are not modified."""
    threshold = 1e-7
    plugin = QuantileMapping(preservation_threshold=threshold)
    result = plugin.process(reference_cube, forecast_cube)

    # Values below threshold should match the original forecast
    below_threshold_mask = forecast_cube.data < threshold
    np.testing.assert_array_equal(
        result.data[below_threshold_mask], forecast_cube.data[below_threshold_mask]
    )
