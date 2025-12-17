# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import warnings

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.fire_weather.build_up_index import BuildUpIndex
from improver_tests.fire_weather import make_cube, make_input_cubes


def input_cubes(
    dmc_val: float = 10.0,
    dc_val: float = 15.0,
    shape: tuple[int, int] = (5, 5),
    dmc_units: str = "1",
    dc_units: str = "1",
) -> list[Cube]:
    """Create a list of dummy input cubes for BUI tests, with configurable units.

    DMC cube has time coordinates; DC cube does not (following the pattern).

    Args:
        dmc_val (float): DMC value for all grid points.
        dc_val (float): DC value for all grid points.
        shape (tuple[int, int]): Shape of the grid for each cube.
        dmc_units (str): Units for DMC cube.
        dc_units (str): Units for DC cube.

    Returns:
        list[Cube]: List of Iris Cubes for DMC and DC.
    """
    return make_input_cubes(
        [
            ("duff_moisture_code", dmc_val, dmc_units, True),
            ("drought_code", dc_val, dc_units, False),
        ],
        shape=shape,
    )


@pytest.mark.parametrize(
    "dmc_val, dc_val, expected_bui",
    [
        # Case 0: Both zero (special case)
        (0.0, 0.0, 0.0),
        # Case 1: DC only (DMC=0) -> BUI should be 0
        (0.0, 50.0, 0.0),
        # Case 2: Small DMC, large DC (DMC <= 0.4*DC)
        (5.0, 30.0, 7.06),
        # Case 3: DMC=10, DC=30 (DMC <= 0.4*DC)
        (10.0, 30.0, 10.91),
        # Case 4: DMC > 0.4*DC case
        (20.0, 30.0, 19.76),
        # Case 5: Equal DMC and DC (DMC > 0.4*DC)
        (50.0, 50.0, 49.41),
        # Case 6: High DC (DMC <= 0.4*DC)
        (30.0, 200.0, 43.64),
        # Case 7: High DMC and DC (DMC > 0.4*DC)
        (45.9, 123.9, 47.67),
        # Case 8: High values with DMC > 0.4*DC
        (100.0, 150.0, 99.46),
    ],
)
def test__calculate(
    dmc_val: float,
    dc_val: float,
    expected_bui: float,
) -> None:
    """Test calculation of BUI from DMC and DC.

    Verifies BUI calculation from DMC and DC values.

    Args:
        dmc_val (float): DMC value to test.
        dc_val (float): DC value to test.
        expected_bui (float): Expected BUI value.
    """
    cubes = input_cubes(dmc_val=dmc_val, dc_val=dc_val)
    plugin = BuildUpIndex()
    plugin.load_input_cubes(CubeList(cubes))
    bui = plugin._calculate()

    assert np.allclose(bui, expected_bui, rtol=1e-2, atol=0.1)


def test__calculate_no_negative_values() -> None:
    """Test that BUI calculation never produces negative values."""
    # Test a range of DMC and DC values
    dmc_values = np.array([0.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    dc_values = np.array([0.0, 10.0, 30.0, 100.0, 300.0, 500.0])

    for dmc in dmc_values:
        for dc in dc_values:
            cubes = input_cubes(dmc_val=dmc, dc_val=dc)
            plugin = BuildUpIndex()
            plugin.load_input_cubes(CubeList(cubes))
            bui = plugin._calculate()
            assert np.all(bui >= 0.0), f"Negative BUI for DMC={dmc}, DC={dc}"


def test__calculate_spatially_varying() -> None:
    """Test BUI calculation with spatially varying DMC and DC.

    Verifies vectorized BUI calculation with varying values across the grid.
    """
    dmc_data = np.array([[5.0, 10.0, 20.0], [15.0, 25.0, 35.0], [30.0, 45.0, 60.0]])
    dc_data = np.array([[10.0, 20.0, 40.0], [30.0, 50.0, 70.0], [60.0, 90.0, 120.0]])

    cubes = [
        make_cube(dmc_data, "duff_moisture_code", "1", add_time_coord=True),
        make_cube(dc_data, "drought_code", "1"),
    ]

    plugin = BuildUpIndex()
    plugin.load_input_cubes(CubeList(cubes))
    bui = plugin._calculate()

    # Verify shape and all values are non-negative
    assert bui.shape == (3, 3)
    assert np.all(bui >= 0.0)

    # Verify unique values (no broadcast errors)
    assert len(np.unique(bui)) > 1

    # Check specific positions using the equations
    # Position [0,0]: DMC=5, DC=10, use eq 27b
    assert np.allclose(bui[0, 0], 4.90, rtol=0.02)

    # Position [2,2]: DMC=60, DC=120, use eq 27b
    assert np.allclose(bui[2, 2], 59.84, rtol=0.02)


@pytest.mark.parametrize(
    "dmc_val, dc_val, expected_bui",
    [
        # Case 0: Typical mid-range values (DMC <= 0.4*DC, use eq 27a)
        (10.0, 30.0, 10.91),
        # Case 1: DMC > 0.4*DC case (use eq 27b)
        (20.0, 30.0, 19.76),
        # Case 2: High DC (DMC <= 0.4*DC, use eq 27a)
        (30.0, 200.0, 43.64),
        # Case 3: High values with DMC > 0.4*DC (use eq 27b)
        (100.0, 150.0, 99.46),
    ],
)
def test_process(
    dmc_val: float,
    dc_val: float,
    expected_bui: float,
) -> None:
    """Integration test for the complete BUI calculation process.

    Verifies end-to-end BUI calculation with various input conditions. Tests both
    equation branches. Edge cases (DMC=0, DC=0, both zero) are covered by dedicated tests.

    Args:
        dmc_val (float): DMC value to test.
        dc_val (float): DC value to test.
        expected_bui (float): Expected BUI output value.
    """
    cubes = input_cubes(dmc_val=dmc_val, dc_val=dc_val)
    result = BuildUpIndex().process(CubeList(cubes))

    assert isinstance(result, Cube)
    assert result.shape == (5, 5)
    assert result.long_name == "build_up_index"
    assert result.units == "1"
    assert np.allclose(result.data, expected_bui, rtol=1e-2, atol=0.1)
    assert result.dtype == np.float32


def test_process_spatially_varying() -> None:
    """Integration test with spatially varying input data.

    Verifies vectorized BUI calculation with varying values across the grid.
    """
    dmc_data = np.array([[5.0, 15.0, 30.0], [10.0, 25.0, 45.0], [20.0, 35.0, 60.0]])
    dc_data = np.array([[10.0, 30.0, 60.0], [20.0, 50.0, 90.0], [40.0, 70.0, 120.0]])

    cubes = [
        make_cube(dmc_data, "duff_moisture_code", "1", add_time_coord=True),
        make_cube(dc_data, "drought_code", "1"),
    ]

    result = BuildUpIndex().process(CubeList(cubes))

    # Verify shape, type, and all values are non-negative
    assert result.data.shape == (3, 3)
    assert result.data.dtype == np.float32
    assert np.all(result.data >= 0.0)

    # Verify increasing DMC and DC both increase BUI (generally)
    # Compare bottom-right vs top-left
    assert result.data[2, 2] > result.data[0, 0]

    # Verify unique values (no broadcast errors)
    assert len(np.unique(result.data)) > 1

    # Check that different environmental conditions produce different outputs
    assert not np.allclose(result.data[0, 0], result.data[2, 2], atol=1.0)


def test_process_dmc_only() -> None:
    """Test that when DC=0, BUI calculation gives result close to DMC."""
    dmc_values = np.array([[10.0, 20.0, 30.0], [15.0, 25.0, 35.0], [5.0, 40.0, 50.0]])
    dc_values = np.zeros((3, 3))

    # Calculate expected BUI for each DMC value when DC=0
    expected_bui = dmc_values - (0.92 + (0.0114 * dmc_values) ** 1.7)

    cubes = [
        make_cube(dmc_values, "duff_moisture_code", "1", add_time_coord=True),
        make_cube(dc_values, "drought_code", "1"),
    ]

    result = BuildUpIndex().process(CubeList(cubes))

    # When DC=0, BUI = DMC - [0.92 + (0.0114*DMC)^1.7]
    assert np.allclose(result.data, expected_bui, rtol=1e-3)


def test_process_dc_only() -> None:
    """Test that when DMC=0, BUI equals 0."""
    dmc_values = np.zeros((3, 3))
    dc_values = np.array(
        [[10.0, 50.0, 100.0], [20.0, 75.0, 150.0], [30.0, 90.0, 200.0]]
    )

    cubes = [
        make_cube(dmc_values, "duff_moisture_code", "1", add_time_coord=True),
        make_cube(dc_values, "drought_code", "1"),
    ]

    result = BuildUpIndex().process(CubeList(cubes))

    # When DMC=0, BUI should be 0 (no fuel moisture contribution)
    assert np.allclose(result.data, 0.0, atol=1e-6)


def test_process_both_zero() -> None:
    """Test that when both DMC and DC are zero, BUI equals 0."""
    cubes = input_cubes(dmc_val=0.0, dc_val=0.0)
    result = BuildUpIndex().process(CubeList(cubes))

    assert isinstance(result, Cube)
    assert result.long_name == "build_up_index"
    assert result.units == "1"
    assert np.allclose(result.data, 0.0, atol=1e-6)
    assert result.dtype == np.float32


def test_input_attribute_mapping() -> None:
    """Test that input cubes are mapped to correct attribute names.

    Verifies that the base class correctly maps cube standard names to
    internal attribute names (duff_moisture_code -> input_dmc,
    drought_code -> input_dc).
    """
    cubes = input_cubes(dmc_val=10.0, dc_val=30.0)
    plugin = BuildUpIndex()
    plugin.load_input_cubes(CubeList(cubes))

    # Check that attributes exist and have correct values
    assert hasattr(plugin, "input_dmc")
    assert hasattr(plugin, "input_dc")
    assert np.allclose(plugin.input_dmc.data, 10.0)
    assert np.allclose(plugin.input_dc.data, 30.0)


@pytest.mark.parametrize(
    "dmc_val, dc_val, expected_error",
    [
        (-5.0, 15.0, "input_dmc contains values below valid minimum"),
        (10.0, -5.0, "input_dc contains values below valid minimum"),
    ],
)
def test_invalid_input_ranges_raise_errors(
    dmc_val: float, dc_val: float, expected_error: str
) -> None:
    """Test that invalid input values raise appropriate ValueError.

    Verifies that the base class validation catches physically meaningless
    or out-of-range input values and raises descriptive errors.

    Args:
        dmc_val (float): DMC value for all grid points.
        dc_val (float): DC value for all grid points.
        expected_error (str): Expected error message substring.
    """
    cubes = input_cubes(dmc_val, dc_val)
    plugin = BuildUpIndex()

    with pytest.raises(ValueError, match=expected_error):
        plugin.load_input_cubes(CubeList(cubes))


@pytest.mark.parametrize(
    "invalid_input_type,expected_error",
    [
        ("input_dmc_nan", "input_dmc contains NaN"),
        ("input_dmc_inf", "input_dmc contains infinite"),
        ("input_dc_nan", "input_dc contains NaN"),
        ("input_dc_inf", "input_dc contains infinite"),
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
    dmc_val, dc_val = 10.0, 30.0

    # Replace the appropriate value with NaN or Inf
    if invalid_input_type == "input_dmc_nan":
        dmc_val = np.nan
    elif invalid_input_type == "input_dmc_inf":
        dmc_val = np.inf
    elif invalid_input_type == "input_dc_nan":
        dc_val = np.nan
    elif invalid_input_type == "input_dc_inf":
        dc_val = -np.inf

    cubes = input_cubes(dmc_val, dc_val)
    plugin = BuildUpIndex()

    with pytest.raises(ValueError, match=expected_error):
        plugin.load_input_cubes(CubeList(cubes))


def test_output_validation_no_warning_for_valid_output() -> None:
    """Test that valid output values do not trigger warnings.

    Uses moderate inputs to verify that as long as the output
    stays within the expected range (0-500 for BUI), no warning is issued.
    This demonstrates that the BUI calculation naturally constrains outputs
    to valid ranges with typical inputs.
    """
    # Use moderate inputs that produce valid BUI
    # Moderate DMC and DC produce moderate BUI within valid range
    cubes = [
        make_cube(np.array([[45.9]]), "duff_moisture_code", "1", add_time_coord=True),
        make_cube(np.array([[123.9]]), "drought_code", "1"),
    ]
    plugin = BuildUpIndex()

    # Process should complete without warnings since output stays in valid range
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        result = plugin.process(CubeList(cubes))

    # No warnings should have been raised (if any were, they'd be errors)
    assert isinstance(result, Cube)
    # Verify output is within expected range (0-500 for BUI)
    assert np.all(result.data >= 0.0)
    assert np.all(result.data <= 500.0)
