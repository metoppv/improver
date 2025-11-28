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

from improver.fire_weather.initial_spread_index import InitialSpreadIndex


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
                    (valid_time.timestamp() - 43200) / 3600,
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
    wind = make_cube(np.full(shape, wind_val), "wind_speed", wind_units)
    # FFMC cube needs time coordinates for _make_isi_cube to copy metadata
    ffmc = make_cube(
        np.full(shape, ffmc_val),
        "fine_fuel_moisture_content",
        ffmc_units,
        add_time_coord=True,
    )
    return [wind, ffmc]


@pytest.mark.parametrize(
    "wind_val, ffmc_val",
    [
        # Case 0: Typical mid-range values
        (10.0, 85.0),
        # Case 1: Low values
        (0.0, 0.0),
        # Case 2: High values
        (100.0, 101.0),
        # Case 3: Low wind, high FFMC
        (2.0, 95.0),
        # Case 4: High wind, low FFMC
        (50.0, 60.0),
    ],
)
def test_load_input_cubes(
    wind_val: float,
    ffmc_val: float,
) -> None:
    """Test InitialSpreadIndex.load_input_cubes with various input conditions.

    Args:
        wind_val (float): Wind speed value for all grid points.
        ffmc_val (float): FFMC value for all grid points.

    Raises:
        AssertionError: If the loaded cubes do not match expected shapes and types.
    """
    cubes = input_cubes(wind_val, ffmc_val)
    plugin = InitialSpreadIndex()
    plugin.load_input_cubes(CubeList(cubes))

    attributes = [
        plugin.wind_speed,
        plugin.input_ffmc,
    ]
    input_values = [wind_val, ffmc_val]

    for attr, val in zip(attributes, input_values):
        assert isinstance(attr, Cube)
        assert attr.shape == (5, 5)
        assert np.allclose(attr.data, val)


@pytest.mark.parametrize(
    "param, input_val, input_unit, expected_val",
    [
        # 0: Wind speed: m/s -> km/h
        ("wind_speed", 2.7778, "m/s", 10.0),
        # 1: FFMC: percentage -> fraction
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
        param (str): Name of the parameter to test (e.g., 'wind_speed', 'input_ffmc').
        input_val (float): Value to use for the tested parameter.
        input_unit (str): Unit to use for the tested parameter.
        expected_val (float): Expected value after conversion.

    Raises:
        AssertionError: If the converted value does not match the expected value.
    """

    # Override the value and unit for the parameter being tested
    if param == "wind_speed":
        cubes = input_cubes(wind_val=input_val, wind_units=input_unit)
    elif param == "input_ffmc":
        cubes = input_cubes(ffmc_val=input_val, ffmc_units=input_unit)

    plugin = InitialSpreadIndex()
    plugin.load_input_cubes(CubeList(cubes))
    # Check only the parameter being tested
    result = getattr(plugin, param)
    assert np.allclose(result.data, expected_val)


@pytest.mark.parametrize(
    "num_cubes, should_raise, expected_message",
    [
        # Case 0: Correct number of cubes (2)
        (2, False, None),
        # Case 1: Too few cubes (1 instead of 2)
        (1, True, "Expected 2 cubes, found 1"),
        # Case 2: No cubes (0 instead of 2)
        (0, True, "Expected 2 cubes, found 0"),
        # Case 3: Too many cubes (3 instead of 2)
        (3, True, "Expected 2 cubes, found 3"),
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
        # Add dummy cubes
        for _ in range(num_cubes - len(cubes)):
            cubes.append(make_cube(np.full((5, 5), 0.0), "dummy", "1"))

    plugin = InitialSpreadIndex()

    if should_raise:
        with pytest.raises(ValueError, match=expected_message):
            plugin.load_input_cubes(CubeList(cubes))
    else:
        plugin.load_input_cubes(CubeList(cubes))


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

    Raises:
        AssertionError: If the calculated fine fuel moisture does not match expected value.
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

    Raises:
        AssertionError: If the calculated wind function does not match expected value.
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

    Args:
        ffmc_val (float): FFMC value to test.
        expected_isi (float): Expected ISI value with zero wind.

    Raises:
        AssertionError: If the calculated ISI does not match expected value.
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

    Raises:
        AssertionError: If the calculated ISI does not match expected value.
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
    "isi_value, shape",
    [
        # Case 0: Typical ISI value with standard grid
        (13.67, (5, 5)),
        # Case 1: Low ISI value with different grid size
        (2.5, (3, 4)),
        # Case 2: High ISI value with larger grid
        (79.15, (10, 10)),
        # Case 3: Zero ISI with small grid
        (0.0, (2, 2)),
        # Case 4: Another typical ISI value
        (8.26, (5, 5)),
    ],
)
def test__make_isi_cube(
    isi_value: float,
    shape: tuple[int, int],
) -> None:
    """Test creation of ISI cube from ISI data.

    Args:
        isi_value (float): ISI value to use.
        shape (tuple[int, int]): Shape of the grid.

    Raises:
        AssertionError: If the created cube does not have expected properties.
    """
    cubes = input_cubes(wind_val=10.0, ffmc_val=85.0, shape=shape)
    plugin = InitialSpreadIndex()
    plugin.load_input_cubes(CubeList(cubes))

    isi_data = np.full(shape, isi_value)
    isi_cube = plugin._make_isi_cube(isi_data)

    assert isinstance(isi_cube, Cube)
    assert isi_cube.shape == shape
    assert isi_cube.long_name == "initial_spread_index"
    assert isi_cube.units == "1"
    assert np.allclose(isi_cube.data, isi_value)
    assert isi_cube.dtype == np.float32
    assert isi_cube.coord("forecast_reference_time")
    assert isi_cube.coord("time")


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
    """Integration test for process method with various input conditions.

    Args:
        wind_val (float): Wind speed value to test.
        ffmc_val (float): FFMC value to test.
        expected_isi (float): Expected ISI output value.

    Raises:
        AssertionError: If the calculated ISI does not match expected value.
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
    """Integration test with spatially varying data (vectorization check)."""
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
    """Test that ISI increases with increasing wind speed at constant FFMC."""
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
    """Test that ISI increases with increasing FFMC at constant wind speed."""
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
