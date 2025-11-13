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
def test_load_input_cubes_cases(
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


#! add a test here for checking the unit conversions of load_input_cubes


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
def test_calculate_moisture_content(
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
    # Check moisture_content shape and type
    assert plugin.moisture_content.shape == cubes[0].data.shape
    assert isinstance(plugin.moisture_content, np.ndarray)
    # Check initial moisture content calculation
    expected_mc = 147.2 * (101.0 - ffmc_val) / (59.5 + ffmc_val)
    assert np.allclose(plugin.moisture_content, expected_mc)


@pytest.mark.parametrize(
    "temp_val, precip_val, rh_val, wind_val, ffmc_val",
    [
        # Case 0: No rainfall, FFMC mid-range (no adjustment expected)
        (20.0, 0.0, 50.0, 10.0, 85.0),
        # Case 1: Rainfall above threshold, FFMC mid-range (adjustment expected)
        (20.0, 1.0, 50.0, 10.0, 85.0),
        # Case 2: Rainfall at threshold, FFMC mid-range (no adjustment expected)
        (20.0, 0.5, 50.0, 10.0, 85.0),
        # Case 3: Heavy rainfall, low FFMC (large adjustment expected)
        (20.0, 10.0, 50.0, 10.0, 40.0),
        # Case 4: No rainfall, high FFMC (no adjustment expected)
        (20.0, 0.0, 50.0, 10.0, 120.0),
    ],
)
def test_perform_rainfall_adjustment(
    temp_val: float,
    precip_val: float,
    rh_val: float,
    wind_val: float,
    ffmc_val: float,
) -> None:
    """Test _perform_rainfall_adjustment for various rainfall and FFMC scenarios.

    Args:
        temp_val (float): Temperature value for all grid points.
        precip_val (float): Precipitation value for all grid points.
        rh_val (float): Relative humidity value for all grid points.
        wind_val (float): Wind speed value for all grid points.
        ffmc_val (float): FFMC value for all grid points.

    Raises:
        AssertionError: If the moisture content adjustment does not match expectations.
    """
    shape = (5, 5)
    cubes = input_cubes(temp_val, precip_val, rh_val, wind_val, ffmc_val, shape)
    plugin = FineFuelMoistureContent()
    plugin.load_input_cubes(CubeList(cubes))
    plugin._calculate_moisture_content()
    initial_mc = plugin.moisture_content.copy()
    plugin._perform_rainfall_adjustment()
    adjusted_mc = plugin.moisture_content
    if precip_val <= 0.5:
        # No adjustment expected: moisture content unchanged
        assert np.allclose(adjusted_mc, initial_mc)
    else:
        # Adjustment expected: moisture content should increase
        assert np.all(adjusted_mc >= initial_mc)
        # For heavy rainfall, check for significant increase
        if precip_val > 5.0:
            assert np.all(adjusted_mc - initial_mc > 1.0)


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
