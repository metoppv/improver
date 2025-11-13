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
def test_calculate_drying_phase(
    temp_val: float,
    precip_val: float,
    rh_val: float,
    wind_val: float,
    ffmc_val: float,
) -> None:
    """Test _calculate_drying_phase for various input scenarios.

    Args:
        temp_val (float): Temperature value for all grid points.
        precip_val (float): Precipitation value for all grid points.
        rh_val (float): Relative humidity value for all grid points.
        wind_val (float): Wind speed value for all grid points.
        ffmc_val (float): FFMC value for all grid points.

    Raises:
        AssertionError: If the drying phase calculation does not match expectations.
    """
    cubes = input_cubes(temp_val, precip_val, rh_val, wind_val, ffmc_val)
    plugin = FineFuelMoistureContent()
    plugin.load_input_cubes(CubeList(cubes))
    E_d = plugin._calculate_drying_phase()
    # Check output type and shape
    assert isinstance(E_d, np.ndarray)
    assert E_d.shape == cubes[0].data.shape
    # Check that drying phase is non-negative
    assert np.all(E_d >= 0)


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
def test_calculate_moisture_content_through_drying_rate(
    temp_val: float,
    precip_val: float,
    rh_val: float,
    wind_val: float,
    ffmc_val: float,
) -> None:
    """Test _calculate_moisture_content_through_drying_rate for various input scenarios.

    Args:
        temp_val (float): Temperature value for all grid points.
        precip_val (float): Precipitation value for all grid points.
        rh_val (float): Relative humidity value for all grid points.
        wind_val (float): Wind speed value for all grid points.
        ffmc_val (float): FFMC value for all grid points.

    Raises:
        AssertionError: If the drying rate calculation does not match expectations.
    """
    cubes = input_cubes(temp_val, precip_val, rh_val, wind_val, ffmc_val)
    plugin = FineFuelMoistureContent()
    plugin.load_input_cubes(CubeList(cubes))
    E_d = plugin._calculate_drying_phase()
    plugin._calculate_moisture_content()
    mask, new_mc = plugin._calculate_moisture_content_through_drying_rate(E_d)
    # Check output types and shapes
    assert isinstance(mask, np.ndarray)
    assert isinstance(new_mc, np.ndarray)
    assert mask.shape == cubes[0].data.shape
    assert new_mc.shape == cubes[0].data.shape
    # Check that new moisture content is non-negative
    assert np.all(new_mc >= 0)


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
def test_calculate_wetting_phase(
    temp_val: float,
    precip_val: float,
    rh_val: float,
    wind_val: float,
    ffmc_val: float,
) -> None:
    """Test _calculate_wetting_phase for various input scenarios.

    Args:
        temp_val (float): Temperature value for all grid points.
        precip_val (float): Precipitation value for all grid points.
        rh_val (float): Relative humidity value for all grid points.
        wind_val (float): Wind speed value for all grid points.
        ffmc_val (float): FFMC value for all grid points.

    Raises:
        AssertionError: If the wetting phase calculation does not match expectations.
    """
    cubes = input_cubes(temp_val, precip_val, rh_val, wind_val, ffmc_val)
    plugin = FineFuelMoistureContent()
    plugin.load_input_cubes(CubeList(cubes))
    E_w = plugin._calculate_wetting_phase()
    # Check output type and shape
    assert isinstance(E_w, np.ndarray)
    assert E_w.shape == cubes[0].data.shape
    # Check that wetting phase is non-negative
    assert np.all(E_w >= 0)


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
def test_calculate_moisture_content_through_wetting_equilibrium(
    temp_val: float,
    precip_val: float,
    rh_val: float,
    wind_val: float,
    ffmc_val: float,
) -> None:
    """Test _calculate_moisture_content_through_wetting_equilibrium for various input scenarios.

    Args:
        temp_val (float): Temperature value for all grid points.
        precip_val (float): Precipitation value for all grid points.
        rh_val (float): Relative humidity value for all grid points.
        wind_val (float): Wind speed value for all grid points.
        ffmc_val (float): FFMC value for all grid points.

    Raises:
        AssertionError: If the wetting equilibrium calculation does not match expectations.
    """
    cubes = input_cubes(temp_val, precip_val, rh_val, wind_val, ffmc_val)
    plugin = FineFuelMoistureContent()
    plugin.load_input_cubes(CubeList(cubes))
    plugin._calculate_moisture_content()
    E_w = plugin._calculate_wetting_phase()
    mask, new_mc = plugin._calculate_moisture_content_through_wetting_equilibrium(E_w)
    # Check output types and shapes
    assert isinstance(mask, np.ndarray)
    assert isinstance(new_mc, np.ndarray)
    assert mask.shape == cubes[0].data.shape
    assert new_mc.shape == cubes[0].data.shape
    # Check that new moisture content is non-negative
    assert np.all(new_mc >= 0)


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
def test_calculate_ffmc_from_moisture_content(
    temp_val: float,
    precip_val: float,
    rh_val: float,
    wind_val: float,
    ffmc_val: float,
) -> None:
    """Test _calculate_ffmc_from_moisture_content for various input scenarios.

    Args:
        temp_val (float): Temperature value for all grid points.
        precip_val (float): Precipitation value for all grid points.
        rh_val (float): Relative humidity value for all grid points.
        wind_val (float): Wind speed value for all grid points.
        ffmc_val (float): FFMC value for all grid points.

    Raises:
        AssertionError: If the FFMC calculation does not match expectations.
    """
    cubes = input_cubes(temp_val, precip_val, rh_val, wind_val, ffmc_val)
    plugin = FineFuelMoistureContent()
    plugin.load_input_cubes(CubeList(cubes))
    plugin._calculate_moisture_content()
    E_d = plugin._calculate_drying_phase()
    E_w = plugin._calculate_wetting_phase()
    ffmc = plugin._calculate_ffmc_from_moisture_content(E_d, E_w)
    # Check output type and shape
    assert isinstance(ffmc, np.ndarray)
    assert ffmc.shape == cubes[0].data.shape
    # Check that FFMC is within expected bounds
    assert np.all((ffmc >= 0) & (ffmc <= 150))


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
def test_process(
    temp_val: float,
    precip_val: float,
    rh_val: float,
    wind_val: float,
    ffmc_val: float,
) -> None:
    """Test process for various input scenarios.

    Args:
        temp_val (float): Temperature value for all grid points.
        precip_val (float): Precipitation value for all grid points.
        rh_val (float): Relative humidity value for all grid points.
        wind_val (float): Wind speed value for all grid points.
        ffmc_val (float): FFMC value for all grid points.

    Raises:
        AssertionError: If the process output does not match expectations.
    """
    cubes = input_cubes(temp_val, precip_val, rh_val, wind_val, ffmc_val)
    plugin = FineFuelMoistureContent()
    result = plugin.process(*cubes)
    # Check output type and shape
    assert hasattr(result, "data")
    assert result.data.shape == cubes[0].data.shape
    # Check that FFMC is within expected bounds
    data = np.array(result.data)
    assert np.all((data >= 0) & (data <= 150))
