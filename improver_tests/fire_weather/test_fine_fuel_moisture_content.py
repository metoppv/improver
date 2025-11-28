# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

from datetime import datetime

import numpy as np
import pytest
from cf_units import Unit
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList

from improver.fire_weather.fine_fuel_moisture_content import FineFuelMoistureContent


def make_cube(
    data: np.ndarray,
    name: str,
    units: str,
    add_time_coord: bool = False,
) -> Cube:
    """Create a dummy Iris Cube with specified data, name, units, and optional
    time coordinates.

    All cubes include a forecast_reference_time coordinate by default.

    Args:
        data (np.ndarray): The data array for the cube.
        name (str): The long name for the cube.
        units (str): The units for the cube.
        add_time_coord (bool): Whether to add a time coordinate with bounds.

    Returns:
        Cube: The constructed Iris Cube with the given properties.
    """
    arr = np.array(data, dtype=np.float64)
    cube = Cube(arr, long_name=name)
    cube.units = units

    # Always add forecast_reference_time
    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"

    # Default forecast reference time: 2025-10-20 00:00:00
    frt = datetime(2025, 10, 20, 0, 0)
    frt_coord = AuxCoord(
        np.array([frt.timestamp() / 3600], dtype=np.float64),
        standard_name="forecast_reference_time",
        units=Unit(time_origin, calendar=calendar),
    )
    cube.add_aux_coord(frt_coord)

    # Optionally add time coordinate with bounds
    if add_time_coord:
        # Default valid time: 2025-10-20 12:00:00 with 12-hour bounds
        valid_time = datetime(2025, 10, 20, 12, 0)
        time_bounds = np.array(
            [
                [
                    (valid_time.timestamp() - 43200) / 3600,  # 12 hours earlier
                    valid_time.timestamp() / 3600,
                ]
            ],
            dtype=np.float64,
        )
        time_coord = AuxCoord(
            np.array([valid_time.timestamp() / 3600], dtype=np.float64),
            standard_name="time",
            bounds=time_bounds,
            units=Unit(time_origin, calendar=calendar),
        )
        cube.add_aux_coord(time_coord)

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

    All cubes have forecast_reference_time. Precipitation and FFMC cubes also have
    time coordinates with bounds.

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
    # Precipitation cube needs time coordinates for _make_ffmc_cube
    precip = make_cube(
        np.full(shape, precip_val),
        "lwe_thickness_of_precipitation_amount",
        precip_units,
        add_time_coord=True,
    )
    rh = make_cube(np.full(shape, rh_val), "relative_humidity", rh_units)
    wind = make_cube(np.full(shape, wind_val), "wind_speed", wind_units)
    # FFMC cube needs time coordinates for _make_ffmc_cube to copy metadata
    ffmc = make_cube(
        np.full(shape, ffmc_val),
        "fine_fuel_moisture_content",
        ffmc_units,
        add_time_coord=True,
    )
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
        assert isinstance(attr, Cube)
        assert attr.data.shape == (5, 5)
        assert np.allclose(attr.data, val)


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
    assert np.allclose(result.data, expected_val)


@pytest.mark.parametrize(
    "num_cubes, should_raise, expected_message",
    [
        # Case 0: Correct number of cubes (5)
        (5, False, None),
        # Case 1: Too few cubes (4 instead of 5)
        (4, True, "Expected 5 cubes, found 4"),
        # Case 2: No cubes (0 instead of 5)
        (0, True, "Expected 5 cubes, found 0"),
        # Case 3: Too many cubes (6 instead of 5)
        (6, True, "Expected 5 cubes, found 6"),
    ],
)
def test_load_input_cubes_wrong_number_raises_error(
    num_cubes: int,
    should_raise: bool,
    expected_message: str,
) -> None:
    """Test that load_input_cubes raises ValueError when given wrong number of cubes.

    Args:
        num_cubes (int): Number of cubes to provide to load_input_cubes.
        should_raise (bool): Whether a ValueError should be raised.
        expected_message (str): Expected error message (or None if no error expected).

    Raises:
        AssertionError: If ValueError behavior does not match expectations.
    """
    # Create a list with the specified number of cubes
    cubes = input_cubes()
    if num_cubes < len(cubes):
        cubes = cubes[:num_cubes]
    elif num_cubes > len(cubes):
        # Add extra dummy cube(s) to test "too many cubes" case
        for _ in range(num_cubes - len(cubes)):
            cubes.append(make_cube(np.full((5, 5), 0.0), "extra_cube", "1"))

    plugin = FineFuelMoistureContent()

    if should_raise:
        with pytest.raises(ValueError, match=expected_message):
            plugin.load_input_cubes(CubeList(cubes))
    else:
        # Should not raise - verify it loads successfully
        plugin.load_input_cubes(CubeList(cubes))
        assert isinstance(plugin.temperature, Cube)


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
    """Test _perform_rainfall_adjustment for various rainfall and moisture scenarios.

    Tests include: no adjustment (precip <= 0.5), adjustment1 only (mc <= 150),
    adjustment1 + adjustment2 (mc > 150), and capping at 250.

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
    plugin.moisture_content = np.full(plugin.precipitation.data.shape, initial_mc_val)
    plugin.initial_moisture_content = np.full(
        plugin.precipitation.data.shape, initial_mc_val
    )
    plugin._perform_rainfall_adjustment()
    adjusted_mc = plugin.moisture_content
    # Check that all points are modified by the correct amount
    assert np.allclose(adjusted_mc, expected_mc, atol=0.01)


def test__perform_rainfall_adjustment_spatially_varying() -> None:
    """Test rainfall adjustment with spatially varying data (vectorization check)."""
    shape = (4, 4)
    # Produce a checkerboard precipitation pattern (5mm and 0mm alternating)
    precip_data = np.zeros(shape)
    precip_data[::2, ::2] = precip_data[1::2, 1::2] = 5.0

    # Varying initial moisture content
    mc_data = np.array(
        [
            [50.0, 100.0, 150.0, 200.0],
            [75.0, 125.0, 175.0, 225.0],
            [60.0, 110.0, 160.0, 210.0],
            [80.0, 130.0, 180.0, 230.0],
        ]
    )

    cubes = [
        make_cube(np.full(shape, 20.0), "air_temperature", "degC"),
        make_cube(
            precip_data,
            "lwe_thickness_of_precipitation_amount",
            "mm",
            add_time_coord=True,
        ),
        make_cube(np.full(shape, 50.0), "relative_humidity", "1"),
        make_cube(np.full(shape, 10.0), "wind_speed", "km/h"),
        make_cube(
            np.full(shape, 85.0), "fine_fuel_moisture_content", "1", add_time_coord=True
        ),
    ]

    plugin = FineFuelMoistureContent()
    plugin.load_input_cubes(CubeList(cubes))
    plugin.moisture_content = mc_data.copy()
    plugin.initial_moisture_content = mc_data.copy()
    plugin._perform_rainfall_adjustment()

    # No-rain cells unchanged, rain cells increased
    assert np.allclose(plugin.moisture_content[0, 1], 100.0)
    assert np.allclose(plugin.moisture_content[0, 3], 200.0)
    assert np.all(plugin.moisture_content[::2, ::2] >= mc_data[::2, ::2])
    assert np.all(plugin.moisture_content[1::2, 1::2] >= mc_data[1::2, 1::2])
    # Verify unique values (no broadcast errors)
    assert len(np.unique(plugin.moisture_content)) > 1


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
def test__calculate_EMC_for_drying_phase(
    temp_val: float,
    rh_val: float,
    expected_E_d: float,
) -> None:
    """
    Test _calculate_EMC_for_drying_phase for given temperature and relative humidity,
    comparing to expected E_d value.

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
    E_d = plugin._calculate_EMC_for_drying_phase()
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
def test__calculate_moisture_content_through_drying_rate(
    moisture_content: np.ndarray,
    relative_humidity: float,
    wind_speed: float,
    temperature: float,
    E_d: np.ndarray,
    expected_output: np.ndarray,
) -> None:
    """Test _calculate_moisture_content_through_drying_rate with various moisture scenarios.

    Tests the calculated moisture content values after applying drying rate equations.

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
    plugin.relative_humidity = make_cube(
        np.full(moisture_content.shape, relative_humidity), "relative_humidity", "1"
    )
    plugin.wind_speed = make_cube(
        np.full(moisture_content.shape, wind_speed), "wind_speed", "km/h"
    )
    plugin.temperature = make_cube(
        np.full(moisture_content.shape, temperature), "air_temperature", "degC"
    )

    new_mc = plugin._calculate_moisture_content_through_drying_rate(E_d)

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
def test__calculate_EMC_for_wetting_phase(
    temp_val: float,
    rh_val: float,
    expected_E_w: float,
) -> None:
    """
    Test _calculate_EMC_for_wetting_phase for given temperature and relative humidity, comparing to expected E_w value.

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
    E_w = plugin._calculate_EMC_for_wetting_phase()
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
def test__calculate_moisture_content_through_wetting_equilibrium(
    moisture_content: np.ndarray,
    relative_humidity: float,
    wind_speed: float,
    temperature: float,
    E_w: np.ndarray,
    expected_output: np.ndarray,
) -> None:
    """
    Test _calculate_moisture_content_through_wetting_equilibrium for given relative humidity, wind speed, temperature, moisture content, and E_w.
    Compares the calculated moisture content to expected values.

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
    plugin.relative_humidity = make_cube(
        np.full(moisture_content.shape, relative_humidity), "relative_humidity", "1"
    )
    plugin.wind_speed = make_cube(
        np.full(moisture_content.shape, wind_speed), "wind_speed", "km/h"
    )
    plugin.temperature = make_cube(
        np.full(moisture_content.shape, temperature), "air_temperature", "degC"
    )

    new_mc = plugin._calculate_moisture_content_through_wetting_equilibrium(E_w)

    assert np.allclose(new_mc, expected_output, atol=0.01)


@pytest.mark.parametrize(
    "moisture_content, expected_output",
    [
        # Case 0: Low moisture content values
        (
            np.array([10, 20, 30, 40, 50]),
            np.array([90.84, 81.85, 73.87, 66.75, 60.34]),
        ),
        # Case 1: Very low moisture content
        (
            np.array([5, 5, 5, 5, 5]),
            np.array([95.78, 95.78, 95.78, 95.78, 95.78]),
        ),
        # Case 2: High moisture content
        (
            np.array([70, 70, 70, 70, 70]),
            np.array([49.31, 49.31, 49.31, 49.31, 49.31]),
        ),
        # Case 3: Mixed moisture content values
        (
            np.array([10, 70, 30, 80, 50]),
            np.array([90.84, 49.31, 73.87, 44.52, 60.34]),
        ),
        # Case 4: Near maximum moisture content (250)
        (
            np.array([250, 250, 250, 250, 250]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        ),
    ],
)
def test__calculate_ffmc_from_moisture_content(
    moisture_content: np.ndarray,
    expected_output: np.ndarray,
) -> None:
    """Test _calculate_ffmc_from_moisture_content with various moisture scenarios.

    Tests the FFMC calculation from moisture content using Van Wagner & Pickett Equation 10.

    Args:
        moisture_content (np.ndarray): Moisture content values for all grid points.
        expected_output (np.ndarray): Expected FFMC output values.

    Raises:
        AssertionError: If the FFMC calculation does not match expectations.
    """
    plugin = FineFuelMoistureContent()
    plugin.moisture_content = moisture_content.copy()
    ffmc = plugin._calculate_ffmc_from_moisture_content()
    # Check output type and shape
    assert isinstance(ffmc, np.ndarray)
    assert ffmc.shape == moisture_content.shape
    # Check that FFMC matches expected output
    assert np.allclose(ffmc, expected_output, atol=0.01)


@pytest.mark.parametrize(
    "ffmc_value, shape",
    [
        # Case 0: Typical mid-range FFMC value with standard grid
        (87.5, (5, 5)),
        # Case 1: Low FFMC value with different grid size
        (60.0, (3, 4)),
        # Case 2: High FFMC value with larger grid
        (95.0, (10, 10)),
        # Case 3: Zero FFMC (edge case) with small grid
        (0.0, (2, 2)),
        # Case 4: Maximum typical FFMC value
        (101.0, (5, 5)),
    ],
)
def test__make_ffmc_cube(
    ffmc_value: float,
    shape: tuple[int, int],
) -> None:
    """
    Test _make_ffmc_cube to ensure it creates an Iris Cube with correct properties
    for various FFMC values and grid shapes.

    Args:
        ffmc_value (float): FFMC data value to use for all grid points.
        shape (tuple[int, int]): Shape of the grid.

    Raises:
        AssertionError: If the created cube does not have expected properties.
    """
    # Create input cubes with specified shape
    cubes = input_cubes(shape=shape)

    # Initialize the plugin and load cubes
    plugin = FineFuelMoistureContent()
    plugin.load_input_cubes(CubeList(cubes))

    # Create test FFMC data
    ffmc_data = np.full(shape, ffmc_value, dtype=np.float64)

    # Call the method under test
    result_cube = plugin._make_ffmc_cube(ffmc_data)

    # Check that result is an Iris Cube with correct type and shape
    assert isinstance(result_cube, Cube)
    assert result_cube.data.dtype == np.float32
    assert result_cube.data.shape == shape
    assert np.allclose(result_cube.data, ffmc_value, atol=0.001)

    # Check that the cube has the correct name and units
    assert result_cube.long_name == "fine_fuel_moisture_content"
    assert result_cube.units == "1"

    # Check that forecast_reference_time is copied from precipitation cube
    result_frt = result_cube.coord("forecast_reference_time")
    expected_frt = plugin.precipitation.coord("forecast_reference_time")
    assert result_frt.points[0] == expected_frt.points[0]
    assert result_frt.units == expected_frt.units

    # Check that time coordinate is copied from precipitation cube
    result_time = result_cube.coord("time")
    expected_time = plugin.precipitation.coord("time")
    assert result_time.points[0] == expected_time.points[0]
    assert result_time.units == expected_time.units

    # Check that time coordinate has no bounds (removed by _make_ffmc_cube)
    assert result_time.bounds is None


@pytest.mark.parametrize(
    "temp_val, precip_val, rh_val, wind_val, ffmc_val, expected_output",
    [
        # Case 0: Typical mid-range values
        (20.0, 1.0, 50.0, 10.0, 85.0, 83.85),
        # Case 1: All zeros (edge case)
        (0.0, 0.0, 0.0, 0.0, 0.0, 22.30),
        # Case 2: High temp, no precip, low RH, high wind (produces high output FFMC)
        (35.0, 0.0, 15.0, 25.0, 90.0, 96.75),
        # Case 3: Low temp, high precip, high RH, low wind (produces low output FFMC)
        (10.0, 15.0, 95.0, 5.0, 85.0, 20.70),
        # Case 4: Precipitation just below threshold (should not adjust)
        (20.0, 0.4, 50.0, 10.0, 85.0, 86.82),
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
    """Integration test for the complete FFMC calculation process.

    Tests end-to-end functionality with various environmental conditions and
    verifies the final FFMC output matches expected values.

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
    result = plugin.process(CubeList(cubes))
    # Check output type and shape
    assert hasattr(result, "data")
    assert result.data.shape == cubes[0].data.shape
    # Check that FFMC matches expected output within tolerance
    data = np.array(result.data)
    assert np.allclose(data, expected_output, atol=0.05)


def test_process_spatially_varying() -> None:
    """Integration test with spatially varying data (vectorization check)."""
    temp_data = np.array([[10.0, 15.0, 20.0], [15.0, 20.0, 25.0], [20.0, 25.0, 30.0]])
    precip_data = np.array([[0.0, 1.0, 5.0], [0.0, 0.0, 10.0], [0.0, 0.0, 0.0]])
    rh_data = np.array([[40.0, 50.0, 60.0], [50.0, 60.0, 70.0], [60.0, 70.0, 80.0]])
    wind_data = np.array([[5.0, 10.0, 15.0], [10.0, 15.0, 20.0], [15.0, 20.0, 25.0]])
    ffmc_data = np.array([[70.0, 80.0, 85.0], [75.0, 85.0, 90.0], [80.0, 88.0, 92.0]])

    cubes = [
        make_cube(temp_data, "air_temperature", "degC"),
        make_cube(
            precip_data,
            "lwe_thickness_of_precipitation_amount",
            "mm",
            add_time_coord=True,
        ),
        make_cube(rh_data, "relative_humidity", "1"),
        make_cube(wind_data, "wind_speed", "km/h"),
        make_cube(ffmc_data, "fine_fuel_moisture_content", "1", add_time_coord=True),
    ]

    result = FineFuelMoistureContent().process(CubeList(cubes))

    # Verify shape, type, and all values in valid range (0-101)
    assert (
        result.data.shape == (3, 3)
        and result.data.dtype == np.float32
        and np.all(result.data >= 0.0)
        and np.all(result.data <= 101.0)
    )
    # Hot/dry/no-rain increases FFMC; heavy rain decreases; unique values (no broadcast errors)
    assert result.data[2, 0] > ffmc_data[2, 0]
    assert result.data[0, 2] < ffmc_data[0, 2]
    assert len(np.unique(result.data)) > 1
    # Check that different environmental conditions produce different outputs
    assert not np.allclose(result.data[0, 0], result.data[2, 2], atol=0.1)
