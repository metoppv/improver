import numpy as np

# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import pytest
from iris.cube import Cube, CubeList

from improver.fire_weather.fine_fuel_moisture_content import FineFuelMoistureContent


def make_cube(data: np.ndarray, name: str, units: str) -> Cube:
    """Create a dummy Iris Cube with specified data, name, and units.

    Args:
        data (np.ndarray): The data array for the cube.
        name (str): The long name for the cube.
        units (str): The units for the cube.

    Returns:
        Cube: The constructed Iris Cube with the given properties.
    """
    arr = np.array(data, dtype=np.float64)
    cube = Cube(arr, long_name=name)
    cube.units = units
    return cube


def input_cubes(
    temp_val: float = 20.0,
    precip_val: float = 1.0,
    rh_val: float = 50.0,
    wind_val: float = 10.0,
    ffmc_val: float = 85.0,
    shape: tuple[int, int] = (5, 5),
    temp_units: str = "degC",
    precip_units: str = "mm",
    rh_units: str = "1",
    wind_units: str = "km/h",
    ffmc_units: str = "1",
) -> list[Cube]:
    """Create a list of dummy input cubes for FFMC tests, with configurable units.

    Args:
        temp_val (float): Temperature value for all grid points.
        precip_val (float): Precipitation value for all grid points.
        rh_val (float): Relative humidity value for all grid points.
        wind_val (float): Wind speed value for all grid points.
        ffmc_val (float): FFMC value for all grid points.
        shape (tuple[int, int]): Shape of the grid for each cube.
        temp_units (str): Units for temperature cube.
        precip_units (str): Units for precipitation cube.
        rh_units (str): Units for relative humidity cube.
        wind_units (str): Units for wind speed cube.
        ffmc_units (str): Units for FFMC cube.

    Returns:
        list[Cube]: List of Iris Cubes for temperature, precipitation, relative humidity, wind speed, and FFMC.
    """
    temp = make_cube(np.full(shape, temp_val), "air_temperature", temp_units)
    precip = make_cube(
        np.full(shape, precip_val),
        "lwe_thickness_of_precipitation_amount",
        precip_units,
    )
    rh = make_cube(np.full(shape, rh_val), "relative_humidity", rh_units)
    wind = make_cube(np.full(shape, wind_val), "wind_speed", wind_units)
    ffmc = make_cube(np.full(shape, ffmc_val), "fine_fuel_moisture_content", ffmc_units)
    return [temp, precip, rh, wind, ffmc]


@pytest.mark.parametrize(
    "temp_val, precip_val, rh_val, wind_val, ffmc_val",
    [
        # Case 0: Typical mid-range values
        (20.0, 1.0, 50.0, 10.0, 85.0),
        # Case 1: All zeros (edge case)
        (0.0, 0.0, 0.0, 0.0, 0.0),
        # Case 2: All maximums/extremes
        (100.0, 100.0, 100.0, 100.0, 100.0),
        # Case 3: Low temperature, low precip, low RH, low wind, low FFMC
        (-10.0, 0.5, 10.0, 2.0, 60.0),
        # Case 4: High temp, high precip, high RH, high wind, high FFMC
        (30.0, 10.0, 90.0, 20.0, 120.0),
    ],
)
def test_load_input_cubes(
    temp_val: float,
    precip_val: float,
    rh_val: float,
    wind_val: float,
    ffmc_val: float,
) -> None:
    """Test FineFuelMoistureContent.load_input_cubes with various input conditions.

    Args:
        temp_val (float): Temperature value for all grid points.
        precip_val (float): Precipitation value for all grid points.
        rh_val (float): Relative humidity value for all grid points.
        wind_val (float): Wind speed value for all grid points.
        ffmc_val (float): FFMC value for all grid points.

    Raises:
        AssertionError: If the loaded cubes do not match expected shapes and types.
    """
    cubes = input_cubes(temp_val, precip_val, rh_val, wind_val, ffmc_val)
    plugin = FineFuelMoistureContent()
    plugin.load_input_cubes(CubeList(cubes))

    attributes = [
        plugin.temperature,
        plugin.precipitation,
        plugin.relative_humidity,
        plugin.wind_speed,
        plugin.input_ffmc,
    ]
    input_values = [temp_val, precip_val, rh_val, wind_val, ffmc_val]

    for attr, val in zip(attributes, input_values):
        assert isinstance(attr, np.ndarray)
        assert attr.shape == (5, 5)
        assert np.allclose(attr, val)


@pytest.mark.parametrize(
    "param, input_val, input_unit, expected_val",
    [
        # 0: Temperature: Kelvin -> degC
        ("temperature", 293.15, "K", 20.0),
        # 1: Precipitation: m -> mm
        ("precipitation", 0.001, "m", 1.0),
        # 2: Relative humidity: percentage -> fraction
        ("relative_humidity", 10.0, "%", 0.1),
        # 3: Wind speed: m/s -> km/h
        ("wind_speed", 2.7777777777777777, "m/s", 10.0),
        # 4: Input FFMC: percentage -> fraction
        ("input_ffmc", 85.0, "%", 0.85),
    ],
)
def test_load_input_cubes_unit_conversion(
    param: str,
    input_val: float,
    input_unit: str,
    expected_val: float,
) -> None:
    """
    Test that load_input_cubes correctly converts a single alternative unit for each input cube.

    Args:
        param (str): Name of the parameter to test (e.g., 'temperature', 'precipitation', etc.).
        input_val (float): Value to use for the tested parameter.
        input_unit (str): Unit to use for the tested parameter.
        expected_val (float): Expected value after conversion.

    Raises:
        AssertionError: If the converted value does not match the expected value.
    """

    # Override the value and unit for the parameter being tested
    if param == "temperature":
        cubes = input_cubes(temp_val=input_val, temp_units=input_unit)
    elif param == "precipitation":
        cubes = input_cubes(precip_val=input_val, precip_units=input_unit)
    elif param == "relative_humidity":
        cubes = input_cubes(rh_val=input_val, rh_units=input_unit)
    elif param == "wind_speed":
        cubes = input_cubes(wind_val=input_val, wind_units=input_unit)
    elif param == "input_ffmc":
        cubes = input_cubes(ffmc_val=input_val, ffmc_units=input_unit)

    plugin = FineFuelMoistureContent()
    plugin.load_input_cubes(CubeList(cubes))
    # Check only the parameter being tested
    result = getattr(plugin, param)
    assert np.allclose(result, expected_val)


@pytest.mark.parametrize(
    "temp_val, precip_val, rh_val, wind_val, ffmc_val",
    [
        # Case 0: Typical mid-range values
        (20.0, 1.0, 50.0, 10.0, 85.0),
        # Case 1: All zeros (edge case)
        (0.0, 0.0, 0.0, 0.0, 0.0),
        # Case 2: All maximums/extremes
        (100.0, 100.0, 100.0, 100.0, 100.0),
        # Case 3: Low temperature, low precip, low RH, low wind, low FFMC
        (-10.0, 0.5, 10.0, 2.0, 60.0),
        # Case 4: High temp, high precip, high RH, high wind, high FFMC
        (30.0, 10.0, 90.0, 20.0, 120.0),
    ],
)
def test__calculate_moisture_content(
    temp_val: float,
    precip_val: float,
    rh_val: float,
    wind_val: float,
    ffmc_val: float,
) -> None:
    """Test _calculate_moisture_content for various input scenarios.

    Args:
        temp_val (float): Temperature value for all grid points.
        precip_val (float): Precipitation value for all grid points.
        rh_val (float): Relative humidity value for all grid points.
        wind_val (float): Wind speed value for all grid points.
        ffmc_val (float): FFMC value for all grid points.

    Raises:
        AssertionError: If the calculated moisture content does not match expectations.
    """
    cubes = input_cubes(temp_val, precip_val, rh_val, wind_val, ffmc_val)
    plugin = FineFuelMoistureContent()
    plugin.load_input_cubes(CubeList(cubes))
    plugin._calculate_moisture_content()

    # Check that both initial and regular moisture content are set
    assert hasattr(plugin, "initial_moisture_content")
    assert hasattr(plugin, "moisture_content")

    # Check moisture_content shape and type
    assert plugin.moisture_content.shape == cubes[0].data.shape
    assert isinstance(plugin.moisture_content, np.ndarray)

    # Check initial moisture content calculation
    expected_mc = 147.2 * (101.0 - ffmc_val) / (59.5 + ffmc_val)
    assert np.allclose(plugin.moisture_content, expected_mc)


@pytest.mark.parametrize(
    "precip_val, initial_mc_val, expected_mc",
    [
        # Case 0: precip is zero, (no adjustment)
        (0.0, 100.0, 100.0),
        # Case 1: precip below threshold, (no adjustment)
        (0.1, 100.0, 100.0),
        # Case 2: precip on threshold limit, (no adjustment)
        (0.5, 100.0, 100.0),
        # Case 3: precip below threshold, moisture_content > 150 (no adjustment)
        (0.3, 200.0, 200.0),
        # Case 4: precip below threshold, moisture_content > 250 (no adjustment)
        (0.3, 260.0, 260.0),
        # Case 5: precip > 0.5, moisture_content <= 150 (adjustment1)
        (1.0, 100.0, 110.9584),
        # Case 6: precip > 0.5, moisture_content = 150 (adjustment1)
        (1.0, 150.0, 157.8952),
        # Case 7: precip > 0.5, moisture_content > 150 (adjustment1 + adjustment2)
        (1.0, 200.0, 205.6425),
        # Case 8: precip > 0.5, moisture_content > 250 (cap at 250)
        (10.0, 260.0, 250.0),
    ],
)
def test__perform_rainfall_adjustment(
    precip_val: float,
    initial_mc_val: float,
    expected_mc: float,
) -> None:
    """
    Test _perform_rainfall_adjustment for all logical branches: no adjustment, adjustment1, adjustment2, cap at 250.

    Args:
        precip_val (float): Precipitation value for all grid points.
        initial_mc_val (float): Initial moisture content value for all grid points.
        expected_mc (float): Expected moisture content after adjustment.

    Raises:
        AssertionError: If the moisture content adjustment does not match expectations.
    """
    cubes = input_cubes(
        precip_val=precip_val,
    )
    plugin = FineFuelMoistureContent()
    plugin.load_input_cubes(CubeList(cubes))
    # Overwrite moisture_content and initial_moisture_content for explicit test control
    plugin.moisture_content = np.full(plugin.precipitation.shape, initial_mc_val)
    plugin.initial_moisture_content = np.full(
        plugin.precipitation.shape, initial_mc_val
    )
    plugin._perform_rainfall_adjustment()
    adjusted_mc = plugin.moisture_content
    # Check that all points are modified by the correct amount
    assert np.allclose(adjusted_mc, expected_mc, atol=0.01)


@pytest.mark.parametrize(
    "temp_val, rh_val, expected_E_d",
    [
        # Case 0: Typical mid-range values
        (20.0, 50.0, 13.69),
        # Case 1: All zeros (edge case)
        (0.0, 0.0, 0.0005),
        # Case 2: High temp, high RH
        (30.0, 90.0, 22.44),
        # Case 3: Low temp, low RH
        (-10.0, 10.0, 8.33),
    ],
)
def test_calculate_drying_phase(
    temp_val: float,
    rh_val: float,
    expected_E_d: float,
) -> None:
    """
    Test _calculate_drying_phase for given temperature and relative humidity, comparing to expected E_d value.

    Args:
        temp_val (float): Temperature value for all grid points.
        rh_val (float): Relative humidity value for all grid points.
        expected_E_d (float): Expected drying phase value.

    Raises:
        AssertionError: If the drying phase calculation does not match expectations.
    """
    cubes = input_cubes(temp_val=temp_val, rh_val=rh_val)
    plugin = FineFuelMoistureContent()
    plugin.load_input_cubes(CubeList(cubes))
    E_d = plugin._calculate_drying_phase()
    # Check output type and shape
    assert isinstance(E_d, np.ndarray)
    assert E_d.shape == cubes[0].data.shape
    # Check that drying phase matches expected value
    assert np.allclose(E_d, expected_E_d, atol=0.01)


@pytest.mark.parametrize(
    "moisture_content, relative_humidity, wind_speed, temperature, E_d, expected_output",
    [
        # Case 0: Some points above, some below E_d
        (
            np.array([10, 20, 10, 20, 10]),
            50,
            10,
            20,
            np.array([15, 15, 15, 15, 15]),
            np.array([13.80, 16.21, 13.80, 16.21, 13.80]),
        ),
        # Case 1: All points below E_d (mask all False)
        (
            np.array([5, 5, 5, 5, 5]),
            50,
            10,
            20,
            np.array([10, 10, 10, 10, 10]),
            np.array([8.80, 8.80, 8.80, 8.80, 8.80]),
        ),
        # Case 2: All points above E_d (mask all True)
        (
            np.array([20, 20, 20, 20, 20]),
            50,
            10,
            20,
            np.array([10, 10, 10, 10, 10]),
            np.array([12.41, 12.41, 12.41, 12.41, 12.41]),
        ),
        # Case 3: Mixed values, different RH and wind
        (
            np.array([10, 30, 50, 70, 90]),
            80,
            5,
            15,
            np.array([20, 40, 60, 80, 100]),
            np.array([14.56, 34.56, 54.56, 74.56, 94.56]),
        ),
        # Case 4: Edge case, moisture_content == E_d (mask all False)
        (
            np.array([10, 20, 30, 40, 50]),
            60,
            8,
            25,
            np.array([10, 20, 30, 40, 50]),
            np.array([10.01, 20.01, 30.01, 40.01, 50.01]),
        ),
    ],
)
def test_calculate_moisture_content_through_drying_rate(
    moisture_content: np.ndarray,
    relative_humidity: float,
    wind_speed: float,
    temperature: float,
    E_d: np.ndarray,
    expected_output: np.ndarray,
) -> None:
    """
    Test _calculate_moisture_content_through_drying_rate for given relative humidity, wind speed, temperature, moisture content, and E_d.
    Compares the output mask and moisture content to expected values.

    Args:
        moisture_content (np.ndarray): Moisture content values for all grid points.
        relative_humidity (float): Relative humidity value for all grid points.
        wind_speed (float): Wind speed value for all grid points.
        temperature (float): Temperature value for all grid points.
        E_d (np.ndarray): Drying phase values for all grid points.
        expected_output (np.ndarray): Expected output moisture content values.

    Raises:
        AssertionError: If the drying rate calculation does not match expectations.
    """
    plugin = FineFuelMoistureContent()
    plugin.initial_moisture_content = moisture_content.copy()
    plugin.moisture_content = moisture_content.copy()
    plugin.relative_humidity = np.full(moisture_content.shape, relative_humidity)
    plugin.wind_speed = np.full(moisture_content.shape, wind_speed)
    plugin.temperature = np.full(moisture_content.shape, temperature)

    expected_mask = moisture_content > E_d

    mask, new_mc = plugin._calculate_moisture_content_through_drying_rate(E_d)

    assert np.all(mask == expected_mask)
    assert np.allclose(new_mc, expected_output, atol=0.01)


@pytest.mark.parametrize(
    "temp_val, rh_val, expected_E_w",
    [
        # Case 0: Typical mid-range values
        (20.0, 50.0, 12.0222),
        # Case 1: All zeros (edge case)
        (0.0, 0.0, 0.0004540),
        # Case 2: High temp, high RH
        (30.0, 90.0, 20.3803),
        # Case 3: Low temp, low RH
        (-10.0, 10.0, 7.3261),
    ],
)
def test_calculate_wetting_phase(
    temp_val: float,
    rh_val: float,
    expected_E_w: float,
) -> None:
    """
    Test _calculate_wetting_phase for given temperature and relative humidity, comparing to expected E_w value.

    Args:
        temp_val (float): Temperature value for all grid points.
        rh_val (float): Relative humidity value for all grid points.
        expected_E_w (float): Expected wetting phase value.

    Raises:
        AssertionError: If the wetting phase calculation does not match expectations.
    """
    cubes = input_cubes(temp_val=temp_val, rh_val=rh_val)
    plugin = FineFuelMoistureContent()
    plugin.load_input_cubes(CubeList(cubes))
    E_w = plugin._calculate_wetting_phase()
    with open("debug_E_w.txt", "a") as f:
        f.write(f"E_w: {E_w[0,0]}\n")
    # Check output type and shape
    assert isinstance(E_w, np.ndarray)
    assert E_w.shape == cubes[0].data.shape
    # Check that wetting phase matches expected value
    assert np.allclose(E_w, expected_E_w, atol=0.01)


@pytest.mark.parametrize(
    "moisture_content, relative_humidity, wind_speed, temperature, E_w, expected_output",
    [
        # Case 0: Some points below, some above E_w
        (
            np.array([10, 20, 10, 20, 10]),
            50,
            10,
            20,
            np.array([15, 15, 15, 15, 15]),
            np.array([13.79, 16.21, 13.79, 16.21, 13.79]),
        ),
        # Case 1: All points above E_w (mask all False)
        (
            np.array([20, 20, 20, 20, 20]),
            50,
            10,
            20,
            np.array([10, 10, 10, 10, 10]),
            np.array([12.41, 12.41, 12.41, 12.41, 12.41]),
        ),
        # Case 2: All points below E_w (mask all True)
        (
            np.array([5, 5, 5, 5, 5]),
            50,
            10,
            20,
            np.array([10, 10, 10, 10, 10]),
            np.array([8.79, 8.79, 8.79, 8.79, 8.79]),
        ),
        # Case 3: Mixed values, different RH and wind
        (
            np.array([10, 30, 50, 70, 90]),
            80,
            5,
            15,
            np.array([20, 40, 60, 80, 100]),
            np.array([17.21, 37.21, 57.21, 77.21, 97.21]),
        ),
        # Case 4: Edge case, moisture_content == E_w (mask all False)
        (
            np.array([10, 20, 30, 40, 50]),
            60,
            8,
            25,
            np.array([10, 20, 30, 40, 50]),
            np.array([10, 20, 30, 40, 50]),
        ),
    ],
)
def test_calculate_moisture_content_through_wetting_equilibrium(
    moisture_content: np.ndarray,
    relative_humidity: float,
    wind_speed: float,
    temperature: float,
    E_w: np.ndarray,
    expected_output: np.ndarray,
) -> None:
    """
    Test _calculate_moisture_content_through_wetting_equilibrium for given relative humidity, wind speed, temperature, moisture content, and E_w.
    Compares the output mask and moisture content to expected values.

    Args:
        moisture_content (np.ndarray): Moisture content values for all grid points.
        relative_humidity (float): Relative humidity value for all grid points.
        wind_speed (float): Wind speed value for all grid points.
        temperature (float): Temperature value for all grid points.
        E_w (np.ndarray): Wetting phase values for all grid points.
        expected_output (np.ndarray): Expected output moisture content values.

    Raises:
        AssertionError: If the wetting equilibrium calculation does not match expectations.
    """
    plugin = FineFuelMoistureContent()
    plugin.initial_moisture_content = moisture_content.copy()
    plugin.moisture_content = moisture_content.copy()
    plugin.relative_humidity = np.full(moisture_content.shape, relative_humidity)
    plugin.wind_speed = np.full(moisture_content.shape, wind_speed)
    plugin.temperature = np.full(moisture_content.shape, temperature)

    expected_mask = moisture_content < E_w

    mask, new_mc = plugin._calculate_moisture_content_through_wetting_equilibrium(E_w)

    assert np.all(mask == expected_mask)
    assert np.allclose(new_mc, expected_output, atol=0.01)


@pytest.mark.parametrize(
    "moisture_content, E_d, E_w, expected_output",
    [
        # Case 0: All values between E_d and E_w
        (
            np.array([10, 20, 30, 40, 50]),
            np.array([60, 60, 60, 60, 60]),
            np.array([0, 0, 0, 0, 0]),
            np.array([90.84, 81.85, 73.87, 66.75, 60.34]),
        ),
        # Case 1: All values below E_w (should use initial moisture_content)
        (
            np.array([5, 5, 5, 5, 5]),
            np.array([60, 60, 60, 60, 60]),
            np.array([10, 10, 10, 10, 10]),
            np.array([95.78, 95.78, 95.78, 95.78, 95.78]),
        ),
        # Case 2: All values above E_d (should use initial moisture_content)
        (
            np.array([70, 70, 70, 70, 70]),
            np.array([60, 60, 60, 60, 60]),
            np.array([0, 0, 0, 0, 0]),
            np.array([49.31, 49.31, 49.31, 49.31, 49.31]),
        ),
        # Case 3: Mixed values
        (
            np.array([10, 70, 30, 80, 50]),
            np.array([60, 60, 60, 60, 60]),
            np.array([0, 0, 0, 0, 0]),
            np.array([90.84, 49.31, 73.87, 44.52, 60.34]),
        ),
    ],
)
def test_calculate_ffmc_from_moisture_content(
    moisture_content: np.ndarray,
    E_d: np.ndarray,
    E_w: np.ndarray,
    expected_output: np.ndarray,
) -> None:
    """
    Test _calculate_ffmc_from_moisture_content for given arrays of moisture_content, E_d, and E_w.
    Checks that the output matches the FFMC equation and has the correct structure.

    Args:
        moisture_content (np.ndarray): Moisture content values for all grid points.
        E_d (np.ndarray): Drying phase values for all grid points.
        E_w (np.ndarray): Wetting phase values for all grid points.
        expected_output (np.ndarray): Expected FFMC output values.

    Raises:
        AssertionError: If the FFMC calculation does not match expectations.
    """
    plugin = FineFuelMoistureContent()
    plugin.initial_moisture_content = moisture_content.copy()
    plugin.moisture_content = moisture_content.copy()
    ffmc = plugin._calculate_ffmc_from_moisture_content(E_d, E_w)
    with open("debug_ffmc.txt", "a") as f:
        f.write(f"FFMC: {ffmc}\n")
    # Check output type and shape
    assert isinstance(ffmc, np.ndarray)
    assert ffmc.shape == moisture_content.shape
    # Check that FFMC matches expected output
    assert np.allclose(ffmc, expected_output, atol=0.01)


@pytest.mark.parametrize(
    "temp_val, precip_val, rh_val, wind_val, ffmc_val, expected_output",
    [
        # Case 0: Typical mid-range values
        (20.0, 1.0, 50.0, 10.0, 85.0, 85.0),
        # Case 1: All zeros (edge case)
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.01964),
        # Case 2: High temp, high precip, high RH, high wind, high FFMC
        (30.0, 10.0, 90.0, 20.0, 120.0, 120.06),
        # Case 3: Low temp, low precip, low RH, low wind, low FFMC
        (-10.0, 0.5, 10.0, 2.0, 60.0, 60.04),
        # Case 4: Precipitation just below threshold (should not adjust)
        (20.0, 0.4, 50.0, 10.0, 85.0, 85.05),
    ],
)
def test_process(
    temp_val: float,
    precip_val: float,
    rh_val: float,
    wind_val: float,
    ffmc_val: float,
    expected_output: float,
) -> None:
    """
    Test process for various input scenarios, providing explicit expected FFMC output values.

    Args:
        temp_val (float): Temperature value for all grid points.
        precip_val (float): Precipitation value for all grid points.
        rh_val (float): Relative humidity value for all grid points.
        wind_val (float): Wind speed value for all grid points.
        ffmc_val (float): FFMC value for all grid points.
        expected_output (float): Expected FFMC output value for all grid points.

    Raises:
        AssertionError: If the process output does not match expectations.
    """
    cubes = input_cubes(temp_val, precip_val, rh_val, wind_val, ffmc_val)
    plugin = FineFuelMoistureContent()
    result = plugin.process(*cubes)
    # Check output type and shape
    assert hasattr(result, "data")
    assert result.data.shape == cubes[0].data.shape
    # Check that FFMC matches expected output within tolerance
    data = np.array(result.data)
    with open("debug_process_ffmc.txt", "a") as f:
        f.write(f"Process FFMC: {data[0]}\n")
    assert np.allclose(data, expected_output, atol=0.05)
