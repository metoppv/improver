# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import warnings

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.fire_weather.initial_spread_index import InitialSpreadIndex
from improver_tests.fire_weather import make_cube, make_input_cubes


def input_cubes(
    wind_val: float = 10.0,
    ffmc_val: float = 85.0,
    shape: tuple[int, int] = (5, 5),
    wind_units: str = "km/h",
    ffmc_units: str = "1",
) -> list[Cube]:
    """Create a list of dummy input cubes for ISI tests, with configurable units.

    All cubes have forecast_reference_time. FFMC cube also has time coordinates with bounds.

    Args:
        wind_val (float): Wind speed value for all grid points.
        ffmc_val (float): FFMC value for all grid points.
        shape (tuple[int, int]): Shape of the grid for each cube.
        wind_units (str): Units for wind speed cube.
        ffmc_units (str): Units for FFMC cube.

    Returns:
        list[Cube]: List of Iris Cubes for wind speed and FFMC.
    """
    return make_input_cubes(
        [
            ("wind_speed", wind_val, wind_units, False),
            ("fine_fuel_moisture_content", ffmc_val, ffmc_units, True),
        ],
        shape=shape,
    )


def test_input_attribute_mapping() -> None:
    """Test that INPUT_ATTRIBUTE_MAPPINGS correctly disambiguates input FFMC."""
    cubes = input_cubes()
    plugin = InitialSpreadIndex()
    plugin.load_input_cubes(CubeList(cubes))

    # Check that the mapping was applied correctly
    assert hasattr(plugin, "input_ffmc")
    assert isinstance(plugin.input_ffmc, Cube)
    assert plugin.input_ffmc.long_name == "fine_fuel_moisture_content"
    assert np.allclose(plugin.input_ffmc.data, 85.0)


@pytest.mark.parametrize(
    "ffmc_val, expected_fm",
    [
        # Case 0: Typical FFMC value
        (85.0, 16.2990),
        # Case 1: Very low FFMC (wet conditions)
        (30.0, 116.7732),
        # Case 2: Very high FFMC (dry conditions)
        (95.0, 5.7165),
        # Case 3: Maximum FFMC
        (101.0, 0.0),
        # Case 4: Zero FFMC
        (0.0, 249.8689),
        # Case 5: Mid-range FFMC
        (70.0, 35.2371),
        # Case 6: Another typical value
        (80.0, 22.1591),
    ],
)
def test__calculate_fine_fuel_moisture(
    ffmc_val: float,
    expected_fm: float,
) -> None:
    """Test calculation of fine fuel moisture from FFMC.

    Args:
        ffmc_val (float): FFMC value to test.
        expected_fm (float): Expected fine fuel moisture content.
    """
    cubes = input_cubes(wind_val=10.0, ffmc_val=ffmc_val)
    plugin = InitialSpreadIndex()
    plugin.load_input_cubes(CubeList(cubes))
    plugin._calculate_fine_fuel_moisture()
    assert np.allclose(plugin.moisture_content, expected_fm, rtol=1e-4)


@pytest.mark.parametrize(
    "wind_val, expected_wf",
    [
        # Case 0: Zero wind
        (0.0, 1.0),
        # Case 1: Typical wind speed
        (10.0, 1.6552),
        # Case 2: High wind speed
        (30.0, 4.5344),
        # Case 3: Very high wind speed
        (50.0, 12.4224),
        # Case 4: Low wind speed
        (5.0, 1.2865),
        # Case 5: Another typical wind speed
        (15.0, 2.1294),
        # Case 6: Moderate wind speed
        (20.0, 2.7396),
    ],
)
def test__calculate_wind_function(
    wind_val: float,
    expected_wf: float,
) -> None:
    """Test calculation of wind function.

    Args:
        wind_val (float): Wind speed value to test.
        expected_wf (float): Expected wind function value.
    """
    cubes = input_cubes(wind_val=wind_val, ffmc_val=85.0)
    plugin = InitialSpreadIndex()
    plugin.load_input_cubes(CubeList(cubes))
    wind_function = plugin._calculate_wind_function()
    assert np.allclose(wind_function, expected_wf, rtol=1e-4)


@pytest.mark.parametrize(
    "ffmc_val, expected_isi",
    [
        # Case 0: Typical FFMC value (ISI with zero wind)
        (85.0, 2.1073),
        # Case 1: Low FFMC (wet conditions, low spread)
        (30.0, 0.0034466),
        # Case 2: High FFMC (dry conditions, high spread)
        (95.0, 8.6572),
        # Case 3: Zero FFMC
        (0.0, 0.0),
        # Case 4: Maximum FFMC
        (101.0, 19.1152),
        # Case 5: Mid-range FFMC
        (70.0, 0.6256),
    ],
)
def test__calculate_spread_factor(
    ffmc_val: float,
    expected_isi: float,
) -> None:
    """Test calculation of spread factor component of ISI.

    Verifies spread factor calculation from FFMC with zero wind.

    Args:
        ffmc_val (float): FFMC value to test.
        expected_isi (float): Expected ISI value with zero wind.
    """
    cubes = input_cubes(wind_val=0.0, ffmc_val=ffmc_val)
    plugin = InitialSpreadIndex()
    plugin.load_input_cubes(CubeList(cubes))
    plugin._calculate_fine_fuel_moisture()
    wind_function = plugin._calculate_wind_function()
    spread_factor = plugin._calculate_spread_factor()
    isi = plugin._calculate_isi(spread_factor, wind_function)
    # With zero wind, wind_function=1.0, ISI = 0.208 * SF * 1.0
    assert np.allclose(isi, expected_isi, rtol=1e-4)


@pytest.mark.parametrize(
    "wind_val, ffmc_val, expected_isi",
    [
        # Case 0: Typical values
        (10.0, 85.0, 3.4879),
        # Case 1: Zero wind
        (0.0, 85.0, 2.1073),
        # Case 2: High wind, high FFMC
        (30.0, 95.0, 39.2554),
        # Case 3: Low wind, low FFMC
        (5.0, 60.0, 0.5266),
        # Case 4: Moderate wind, moderate FFMC
        (15.0, 75.0, 1.6311),
    ],
)
def test__calculate_isi(
    wind_val: float,
    ffmc_val: float,
    expected_isi: float,
) -> None:
    """Test calculation of ISI from fine fuel moisture and wind.

    Args:
        wind_val (float): Wind speed value to test.
        ffmc_val (float): FFMC value to test.
        expected_isi (float): Expected ISI value.
    """
    cubes = input_cubes(wind_val=wind_val, ffmc_val=ffmc_val)
    plugin = InitialSpreadIndex()
    plugin.load_input_cubes(CubeList(cubes))
    plugin._calculate_fine_fuel_moisture()
    wind_function = plugin._calculate_wind_function()
    spread_factor = plugin._calculate_spread_factor()
    isi = plugin._calculate_isi(spread_factor, wind_function)
    assert np.allclose(isi, expected_isi, rtol=1e-4)


@pytest.mark.parametrize(
    "wind_val, ffmc_val, expected_isi",
    [
        # Case 0: Typical mid-range values
        (10.0, 85.0, 3.4879),
        # Case 1: Zero wind, typical FFMC
        (0.0, 85.0, 2.1073),
        # Case 2: High wind, high FFMC (extreme fire spread conditions)
        (30.0, 95.0, 39.2554),
        # Case 3: Low wind, low FFMC (low fire spread conditions)
        (5.0, 60.0, 0.5266),
        # Case 4: Moderate conditions
        (15.0, 75.0, 1.6311),
        # Case 5: Very high wind, moderate FFMC
        (50.0, 80.0, 14.1266),
        # Case 6: Calm conditions
        (2.0, 70.0, 0.6919),
        # Case 7: Maximum FFMC, moderate wind
        (20.0, 101.0, 52.3674),
    ],
)
def test_process(
    wind_val: float,
    ffmc_val: float,
    expected_isi: float,
) -> None:
    """Integration test for the complete ISI calculation process.

    Verifies end-to-end ISI calculation with various environmental conditions.

    Args:
        wind_val (float): Wind speed value to test.
        ffmc_val (float): FFMC value to test.
        expected_isi (float): Expected ISI output value.
    """
    cubes = input_cubes(wind_val=wind_val, ffmc_val=ffmc_val)
    result = InitialSpreadIndex().process(CubeList(cubes))

    assert isinstance(result, Cube)
    assert result.shape == (5, 5)
    assert result.long_name == "initial_spread_index"
    assert result.units == "1"
    assert np.allclose(result.data, expected_isi, rtol=1e-4)
    assert result.dtype == np.float32


def test_process_spatially_varying() -> None:
    """Integration test with spatially varying input data.

    Verifies vectorized ISI implementation with varying values across the grid.
    """
    wind_data = np.array([[5.0, 10.0, 15.0], [10.0, 15.0, 20.0], [15.0, 20.0, 25.0]])
    ffmc_data = np.array([[70.0, 80.0, 85.0], [75.0, 85.0, 90.0], [80.0, 88.0, 92.0]])

    cubes = [
        make_cube(wind_data, "wind_speed", "km/h"),
        make_cube(ffmc_data, "fine_fuel_moisture_content", "1", add_time_coord=True),
    ]

    result = InitialSpreadIndex().process(CubeList(cubes))

    # Verify shape, type, and all values are positive
    assert result.data.shape == (3, 3)
    assert result.data.dtype == np.float32
    assert np.all(result.data >= 0.0)

    # Verify increasing wind and FFMC both increase ISI
    # Compare bottom-right (high wind, high FFMC) vs top-left (low wind, low FFMC)
    assert result.data[2, 2] > result.data[0, 0]

    # Compare along diagonal: increasing wind and FFMC should increase ISI
    assert result.data[1, 1] > result.data[0, 0]
    assert result.data[2, 2] > result.data[1, 1]

    # Verify unique values (no broadcast errors)
    assert len(np.unique(result.data)) > 1

    # Check that different environmental conditions produce different outputs
    assert not np.allclose(result.data[0, 0], result.data[2, 2], atol=0.1)


def test_process_with_varying_wind() -> None:
    """Test ISI increases with increasing wind speed at constant FFMC.

    Verifies wind speed relationship in ISI calculation.
    """
    wind_data = np.array([[0.0, 10.0, 20.0], [5.0, 15.0, 25.0], [10.0, 20.0, 30.0]])
    ffmc_data = np.full((3, 3), 85.0)

    cubes = [
        make_cube(wind_data, "wind_speed", "km/h"),
        make_cube(ffmc_data, "fine_fuel_moisture_content", "1", add_time_coord=True),
    ]

    result = InitialSpreadIndex().process(CubeList(cubes))

    # ISI should increase with wind at constant FFMC
    # Check each row
    for row in range(3):
        assert result.data[row, 1] > result.data[row, 0]
        assert result.data[row, 2] > result.data[row, 1]


def test_process_with_varying_ffmc() -> None:
    """Test ISI increases with increasing FFMC at constant wind speed.

    Verifies FFMC relationship in ISI calculation.
    """
    wind_data = np.full((3, 3), 10.0)
    ffmc_data = np.array([[60.0, 70.0, 80.0], [65.0, 75.0, 85.0], [70.0, 80.0, 90.0]])

    cubes = [
        make_cube(wind_data, "wind_speed", "km/h"),
        make_cube(ffmc_data, "fine_fuel_moisture_content", "1", add_time_coord=True),
    ]

    result = InitialSpreadIndex().process(CubeList(cubes))

    # ISI should increase with FFMC at constant wind
    # Check each row
    for row in range(3):
        assert result.data[row, 1] > result.data[row, 0]
        assert result.data[row, 2] > result.data[row, 1]


@pytest.mark.parametrize(
    "wind_val, ffmc_val, expected_error",
    [
        (-5.0, 85.0, "wind_speed contains values below valid minimum"),
        (10.0, 120.0, "input_ffmc contains values above valid maximum"),
        (10.0, -5.0, "input_ffmc contains values below valid minimum"),
    ],
)
def test_invalid_input_ranges_raise_errors(
    wind_val: float, ffmc_val: float, expected_error: str
) -> None:
    """Test that invalid input values raise appropriate ValueError.

    Verifies that the base class validation catches physically meaningless
    or out-of-range input values and raises descriptive errors.

    Args:
        wind_val (float): Wind speed value for all grid points.
        ffmc_val (float): FFMC value for all grid points.
        expected_error (str): Expected error message substring.
    """
    cubes = [
        make_cube(np.array([[wind_val]]), "wind_speed", "km/h"),
        make_cube(
            np.array([[ffmc_val]]),
            "fine_fuel_moisture_content",
            "1",
            add_time_coord=True,
        ),
    ]
    plugin = InitialSpreadIndex()

    with pytest.raises(ValueError, match=expected_error):
        plugin.load_input_cubes(CubeList(cubes))


@pytest.mark.parametrize(
    "invalid_input_type,expected_error",
    [
        ("wind_speed_nan", "wind_speed contains NaN"),
        ("wind_speed_inf", "wind_speed contains infinite"),
        ("input_ffmc_nan", "input_ffmc contains NaN"),
        ("input_ffmc_inf", "input_ffmc contains infinite"),
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
    wind_val, ffmc_val = 10.0, 85.0

    # Replace the appropriate value with NaN or Inf
    if invalid_input_type == "wind_speed_nan":
        wind_val = np.nan
    elif invalid_input_type == "wind_speed_inf":
        wind_val = np.inf
    elif invalid_input_type == "input_ffmc_nan":
        ffmc_val = np.nan
    elif invalid_input_type == "input_ffmc_inf":
        ffmc_val = -np.inf

    cubes = [
        make_cube(np.array([[wind_val]]), "wind_speed", "km/h"),
        make_cube(
            np.array([[ffmc_val]]),
            "fine_fuel_moisture_content",
            "1",
            add_time_coord=True,
        ),
    ]
    plugin = InitialSpreadIndex()

    with pytest.raises(ValueError, match=expected_error):
        plugin.load_input_cubes(CubeList(cubes))


def test_output_validation_no_warning_for_valid_output() -> None:
    """Test that valid output values do not trigger warnings.

    Uses moderate inputs to verify that as long as the output
    stays within the expected range (0-100 for ISI), no warning is issued.
    This demonstrates that the ISI calculation naturally constrains outputs
    to valid ranges with typical inputs.
    """
    # Use moderate inputs that produce valid ISI
    # Moderate wind and high FFMC produce moderate ISI within valid range
    cubes = [
        make_cube(np.array([[15.0]]), "wind_speed", "km/h"),
        make_cube(
            np.array([[85.0]]), "fine_fuel_moisture_content", "1", add_time_coord=True
        ),
    ]
    plugin = InitialSpreadIndex()

    # Process should complete without warnings since output stays in valid range
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        result = plugin.process(CubeList(cubes))

    # No warnings should have been raised (if any were, they'd be errors)
    assert isinstance(result, Cube)
    # Verify output is within expected range (0-100 for ISI)
    assert np.all(result.data >= 0.0)
    assert np.all(result.data <= 100.0)
