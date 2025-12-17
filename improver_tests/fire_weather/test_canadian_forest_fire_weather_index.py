# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the CanadianForestFireWeatherIndex plugin."""

import warnings

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.fire_weather.canadian_forest_fire_weather_index import (
    CanadianForestFireWeatherIndex,
)
from improver_tests.fire_weather import make_cube, make_input_cubes


def input_cubes(
    isi_val: float = 10.0,
    bui_val: float = 50.0,
    shape: tuple[int, int] = (5, 5),
    isi_units: str = "1",
    bui_units: str = "1",
) -> list[Cube]:
    """Create a list of dummy input cubes for FWI tests, with configurable units.

    ISI cube has time coordinates; BUI cube does not (following the pattern).

    Args:
        isi_val (float): ISI value for all grid points.
        bui_val (float): BUI value for all grid points.
        shape (tuple[int, int]): Shape of the grid for each cube.
        isi_units (str): Units for ISI cube.
        bui_units (str): Units for BUI cube.

    Returns:
        list[Cube]: List of Iris Cubes for ISI and BUI.
    """
    return make_input_cubes(
        [
            ("initial_spread_index", isi_val, isi_units, True),
            ("build_up_index", bui_val, bui_units, False),
        ],
        shape=shape,
    )


@pytest.mark.parametrize(
    "bui_val, expected_dmf",
    [
        # Case 0: BUI = 0
        (0.0, 2.0),
        # Case 1: BUI = 20 (equation 28a)
        (20.0, 9.065),
        # Case 2: BUI = 50 (equation 28a)
        (50.0, 16.827),
        # Case 3: BUI = 80 boundary (equation 28a)
        (80.0, 23.686),
        # Case 4: BUI = 100 (equation 28b)
        (100.0, 27.861),
        # Case 5: BUI = 200 (equation 28b)
        (200.0, 38.326),
    ],
)
def test__calculate_extrapolated_duff_moisture_function(
    bui_val: float,
    expected_dmf: float,
) -> None:
    """Test calculation of extrapolated DMF from BUI.

    Args:
        bui_val (float): BUI value to test.
        expected_dmf (float): Expected extrapolated DMF value.
    """
    cubes = input_cubes(isi_val=10.0, bui_val=bui_val)
    plugin = CanadianForestFireWeatherIndex()
    plugin.load_input_cubes(CubeList(cubes))
    dmf = plugin._calculate_extrapolated_duff_moisture_function()

    assert np.allclose(dmf, expected_dmf, rtol=1e-2, atol=0.1)


def test__calculate_extrapolated_duff_moisture_function_no_negative() -> None:
    """Test that extrapolated DMF calculation never produces negative values."""
    bui_values = np.array([0.0, 10.0, 50.0, 80.0, 100.0, 150.0, 250.0])

    for bui in bui_values:
        cubes = input_cubes(isi_val=10.0, bui_val=bui)
        plugin = CanadianForestFireWeatherIndex()
        plugin.load_input_cubes(CubeList(cubes))
        dmf = plugin._calculate_extrapolated_duff_moisture_function()
        assert np.all(dmf >= 0.0), f"Negative DMF for BUI={bui}"


@pytest.mark.parametrize(
    "isi_val, bui_val, expected_fwi",
    [
        # Case 0: Both zero
        (0.0, 0.0, 0.0),
        # Case 1: BUI = 0 only
        (10.0, 0.0, 3.492),
        # Case 2: ISI = 0 only
        (0.0, 50.0, 0.0),
        # Case 3: Low values, BUI <= 80 (equation 28a), B <= 1 (equation 30b)
        (1.0, 20.0, 0.906),
        # Case 4: Mid values, BUI <= 80 (equation 28a), B > 1 (equation 30a)
        (10.0, 50.0, 22.241),
        # Case 5: BUI = 80 boundary (equation 28a)
        (15.0, 80.0, 37.003),
        # Case 6: High BUI > 80 (equation 28b)
        (20.0, 100.0, 49.368),
        # Case 7: Very high values
        (50.0, 200.0, 103.265),
    ],
)
def test__calculate_fwi(
    isi_val: float,
    bui_val: float,
    expected_fwi: float,
) -> None:
    """Test calculation of FWI from ISI and BUI.

    Args:
        isi_val (float): ISI value to test.
        bui_val (float): BUI value to test.
        expected_fwi (float): Expected FWI value.
    """
    cubes = input_cubes(isi_val=isi_val, bui_val=bui_val)
    plugin = CanadianForestFireWeatherIndex()
    plugin.load_input_cubes(CubeList(cubes))
    extrapolated_DMF = plugin._calculate_extrapolated_duff_moisture_function()
    fwi = plugin._calculate_fwi(extrapolated_DMF)

    assert np.allclose(fwi, expected_fwi, rtol=1e-2, atol=0.2)


def test__calculate_fwi_no_negative_values() -> None:
    """Test that FWI calculation never produces negative values."""
    # Test a range of ISI and BUI values
    isi_values = np.array([0.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    bui_values = np.array([0.0, 10.0, 50.0, 80.0, 150.0, 250.0])

    for isi in isi_values:
        for bui in bui_values:
            cubes = input_cubes(isi_val=isi, bui_val=bui)
            plugin = CanadianForestFireWeatherIndex()
            plugin.load_input_cubes(CubeList(cubes))
            extrapolated_DMF = plugin._calculate_extrapolated_duff_moisture_function()
            fwi = plugin._calculate_fwi(extrapolated_DMF)
            assert np.all(fwi >= 0.0), f"Negative FWI for ISI={isi}, BUI={bui}"


def test__calculate_fwi_spatially_varying() -> None:
    """Test FWI calculation with spatially varying ISI and BUI.

    Verifies vectorized FWI calculation with varying values across the grid.
    """
    isi_data = np.array([[5.0, 10.0, 20.0], [8.0, 15.0, 30.0], [12.0, 25.0, 50.0]])
    bui_data = np.array([[20.0, 40.0, 60.0], [30.0, 70.0, 90.0], [50.0, 100.0, 150.0]])

    cubes = [
        make_cube(isi_data, "initial_spread_index", "1", add_time_coord=True),
        make_cube(bui_data, "build_up_index", "1"),
    ]

    plugin = CanadianForestFireWeatherIndex()
    plugin.load_input_cubes(CubeList(cubes))
    extrapolated_DMF = plugin._calculate_extrapolated_duff_moisture_function()
    fwi = plugin._calculate_fwi(extrapolated_DMF)

    # Verify shape and all values are non-negative
    assert fwi.shape == (3, 3)
    assert np.all(fwi >= 0.0)

    # Verify unique values (no broadcast errors)
    assert len(np.unique(fwi)) > 1

    # Check specific position - low ISI and BUI should give low FWI
    assert fwi[0, 0] < 10.0

    # Check specific position - high ISI and BUI should give high FWI
    assert fwi[2, 2] > 50.0


@pytest.mark.parametrize(
    "isi_val, bui_val, expected_fwi",
    [
        # Case 0: Both zero
        (0.0, 0.0, 0.0),
        # Case 1: Low values
        (1.0, 20.0, 0.906),
        # Case 2: Mid values
        (10.0, 50.0, 22.241),
        # Case 3: High values
        (20.0, 100.0, 49.368),
    ],
)
def test_process(
    isi_val: float,
    bui_val: float,
    expected_fwi: float,
) -> None:
    """Integration test for the complete FWI calculation process.

    Verifies end-to-end FWI calculation with various input conditions.

    Args:
        isi_val (float): ISI value to test.
        bui_val (float): BUI value to test.
        expected_fwi (float): Expected FWI output value.
    """
    cubes = input_cubes(isi_val=isi_val, bui_val=bui_val)
    result = CanadianForestFireWeatherIndex().process(CubeList(cubes))

    assert isinstance(result, Cube)
    assert result.shape == (5, 5)
    assert result.long_name == "canadian_forest_fire_weather_index"
    assert result.units == "1"
    assert np.allclose(result.data, expected_fwi, rtol=1e-2, atol=0.2)
    assert result.dtype == np.float32


def test_process_spatially_varying() -> None:
    """Integration test with spatially varying input data.

    Verifies vectorized FWI calculation with varying values across the grid.
    """
    isi_data = np.array([[5.0, 10.0, 20.0], [8.0, 15.0, 30.0], [12.0, 25.0, 50.0]])
    bui_data = np.array([[20.0, 40.0, 60.0], [30.0, 70.0, 90.0], [50.0, 100.0, 150.0]])

    cubes = [
        make_cube(isi_data, "initial_spread_index", "1", add_time_coord=True),
        make_cube(bui_data, "build_up_index", "1"),
    ]

    result = CanadianForestFireWeatherIndex().process(CubeList(cubes))

    # Verify shape, type, and all values are non-negative
    assert result.data.shape == (3, 3)
    assert result.data.dtype == np.float32
    assert np.all(result.data >= 0.0)

    # Verify increasing ISI and BUI both increase FWI (generally)
    # Compare bottom-right vs top-left
    assert result.data[2, 2] > result.data[0, 0]

    # Verify unique values (no broadcast errors)
    assert len(np.unique(result.data)) > 1

    # Check that different environmental conditions produce different outputs
    assert not np.allclose(result.data[0, 0], result.data[2, 2], atol=1.0)


def test_process_isi_zero() -> None:
    """Test that when ISI=0, FWI equals 0."""
    isi_values = np.zeros((3, 3))
    bui_values = np.array(
        [[10.0, 50.0, 100.0], [20.0, 75.0, 150.0], [30.0, 90.0, 200.0]]
    )

    cubes = [
        make_cube(isi_values, "initial_spread_index", "1", add_time_coord=True),
        make_cube(bui_values, "build_up_index", "1"),
    ]

    result = CanadianForestFireWeatherIndex().process(CubeList(cubes))

    # When ISI=0, FWI should be 0
    assert np.allclose(result.data, 0.0, atol=1e-6)


def test_process_bui_zero() -> None:
    """Test that when BUI=0, FWI is calculated with minimum extrapolated DMF (2.0)."""
    isi_values = np.array([[5.0, 10.0, 20.0], [8.0, 15.0, 30.0], [12.0, 25.0, 50.0]])
    bui_values = np.zeros((3, 3))

    cubes = [
        make_cube(isi_values, "initial_spread_index", "1", add_time_coord=True),
        make_cube(bui_values, "build_up_index", "1"),
    ]

    result = CanadianForestFireWeatherIndex().process(CubeList(cubes))

    # All values should be positive and vary with ISI
    assert np.all(result.data > 0.0)
    assert len(np.unique(result.data)) > 1  # Different ISI values give different FWI
    assert np.isclose(result.data[0, 1], 3.492, rtol=1e-2)


def test_process_both_zero() -> None:
    """Test that when both ISI and BUI are zero, FWI equals 0."""
    cubes = input_cubes(isi_val=0.0, bui_val=0.0)
    result = CanadianForestFireWeatherIndex().process(CubeList(cubes))

    assert isinstance(result, Cube)
    assert result.long_name == "canadian_forest_fire_weather_index"
    assert result.units == "1"
    assert np.allclose(result.data, 0.0, atol=1e-6)
    assert result.dtype == np.float32


@pytest.mark.parametrize(
    "isi_val, bui_val, expected_error",
    [
        (-5.0, 50.0, "initial_spread_index contains values below valid minimum"),
        (150.0, 50.0, "initial_spread_index contains values above valid maximum"),
        (10.0, -5.0, "build_up_index contains values below valid minimum"),
        (10.0, 600.0, "build_up_index contains values above valid maximum"),
    ],
)
def test_invalid_input_ranges_raise_errors(
    isi_val: float, bui_val: float, expected_error: str
) -> None:
    """Test that invalid input values raise appropriate ValueError.

    Verifies that the base class validation catches physically meaningless
    or out-of-range input values and raises descriptive errors.

    Args:
        isi_val (float): ISI value for all grid points.
        bui_val (float): BUI value for all grid points.
        expected_error (str): Expected error message substring.
    """
    cubes = input_cubes(isi_val, bui_val)
    plugin = CanadianForestFireWeatherIndex()

    with pytest.raises(ValueError, match=expected_error):
        plugin.load_input_cubes(CubeList(cubes))


@pytest.mark.parametrize(
    "invalid_input_type,expected_error",
    [
        ("initial_spread_index_nan", "initial_spread_index contains NaN"),
        ("initial_spread_index_inf", "initial_spread_index contains infinite"),
        ("build_up_index_nan", "build_up_index contains NaN"),
        ("build_up_index_inf", "build_up_index contains infinite"),
    ],
)
def test_nan_and_inf_values_raise_errors(
    invalid_input_type: str, expected_error: str
) -> None:
    """Test that NaN and Inf values in inputs raise appropriate ValueError.

    Verifies that the validation catches non-finite values (NaN, Inf) in input data.

    Args:
        invalid_input_type (str): Which input to make invalid and how.
        expected_error (str): Expected error message substring.
    """
    # Start with valid values
    isi_val, bui_val = 10.0, 50.0

    # Replace the appropriate value with NaN or Inf
    if invalid_input_type == "initial_spread_index_nan":
        isi_val = np.nan
    elif invalid_input_type == "initial_spread_index_inf":
        isi_val = np.inf
    elif invalid_input_type == "build_up_index_nan":
        bui_val = np.nan
    elif invalid_input_type == "build_up_index_inf":
        bui_val = -np.inf

    cubes = input_cubes(isi_val, bui_val)
    plugin = CanadianForestFireWeatherIndex()

    with pytest.raises(ValueError, match=expected_error):
        plugin.load_input_cubes(CubeList(cubes))


def test_output_validation_no_warning_for_valid_output() -> None:
    """Test that valid output values do not trigger warnings.

    Uses moderate inputs to verify that as long as the output
    stays within the expected range (0-100 for FWI), no warning is issued.
    This demonstrates that the FWI calculation naturally constrains outputs
    to valid ranges with typical inputs.
    """
    # Use moderate inputs that produce valid FWI
    # Moderate ISI and BUI produce moderate FWI within valid range
    cubes = [
        make_cube(np.array([[10.0]]), "initial_spread_index", "1", add_time_coord=True),
        make_cube(np.array([[50.0]]), "build_up_index", "1"),
    ]
    plugin = CanadianForestFireWeatherIndex()

    # Process should complete without warnings since output stays in valid range
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        result = plugin.process(CubeList(cubes))

    # No warnings should have been raised (if any were, they'd be errors)
    assert isinstance(result, Cube)
    # Verify output is within expected range (0-100 for FWI)
    assert np.all(result.data >= 0.0)
    assert np.all(result.data <= 100.0)
