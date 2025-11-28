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

from improver.fire_weather.drought_code import DroughtCode


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
    dc_val: float = 15.0,
    shape: tuple[int, int] = (5, 5),
    temp_units: str = "degC",
    precip_units: str = "mm",
    dc_units: str = "1",
) -> list[Cube]:
    """Create a list of dummy input cubes for DC tests, with configurable units.

    All cubes have forecast_reference_time. Precipitation and DC cubes also have
    time coordinates with bounds.

    Args:
        temp_val (float): Temperature value for all grid points.
        precip_val (float): Precipitation value for all grid points.
        dc_val (float): DC value for all grid points.
        shape (tuple[int, int]): Shape of the grid for each cube.
        temp_units (str): Units for temperature cube.
        precip_units (str): Units for precipitation cube.
        dc_units (str): Units for DC cube.

    Returns:
        list[Cube]: List of Iris Cubes for temperature, precipitation, and DC.
    """
    temp = make_cube(np.full(shape, temp_val), "air_temperature", temp_units)
    # Precipitation cube needs time coordinates for _make_dc_cube
    precip = make_cube(
        np.full(shape, precip_val),
        "lwe_thickness_of_precipitation_amount",
        precip_units,
        add_time_coord=True,
    )
    # DC cube needs time coordinates for _make_dc_cube to copy metadata
    dc = make_cube(
        np.full(shape, dc_val),
        "drought_code",
        dc_units,
        add_time_coord=True,
    )
    return [temp, precip, dc]


@pytest.mark.parametrize(
    "temp_val, precip_val, dc_val",
    [
        # Case 0: Typical mid-range values
        (20.0, 1.0, 15.0),
        # Case 1: All zeros (edge case)
        (0.0, 0.0, 0.0),
        # Case 2: All maximums/extremes
        (100.0, 100.0, 1000.0),
        # Case 3: Low temperature, low precip, low DC
        (-10.0, 0.5, 5.0),
        # Case 4: High temp, high precip, high DC
        (30.0, 10.0, 500.0),
    ],
)
def test_load_input_cubes(
    temp_val: float,
    precip_val: float,
    dc_val: float,
) -> None:
    """Test DroughtCode.load_input_cubes with various input conditions.

    Args:
        temp_val (float): Temperature value for all grid points.
        precip_val (float): Precipitation value for all grid points.
        dc_val (float): DC value for all grid points.

    Raises:
        AssertionError: If the loaded cubes do not match expected shapes and types.
    """
    cubes = input_cubes(temp_val, precip_val, dc_val)
    plugin = DroughtCode()
    plugin.load_input_cubes(CubeList(cubes), month=7)

    attributes = [
        plugin.temperature,
        plugin.precipitation,
        plugin.input_dc,
    ]
    input_values = [temp_val, precip_val, dc_val]

    for attr, val in zip(attributes, input_values):
        assert isinstance(attr, Cube)
        assert attr.data.shape == (5, 5)
        assert np.allclose(attr.data, val)

    # Check that month is set correctly
    assert plugin.month == 7


@pytest.mark.parametrize(
    "param, input_val, input_unit, expected_val",
    [
        # Case 0: Temperature: Kelvin -> degC
        ("temperature", 293.15, "K", 20.0),
        # Case 1: Precipitation: m -> mm
        ("precipitation", 0.001, "m", 1.0),
        # Case 2: Input DC: no conversion needed (dimensionless)
        ("input_dc", 15.0, "1", 15.0),
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
    elif param == "input_dc":
        cubes = input_cubes(dc_val=input_val, dc_units=input_unit)

    plugin = DroughtCode()
    plugin.load_input_cubes(CubeList(cubes), month=7)
    # Check only the parameter being tested
    result = getattr(plugin, param)
    assert np.allclose(result.data, expected_val)


@pytest.mark.parametrize(
    "num_cubes, should_raise, expected_message",
    [
        # Case 0: Correct number of cubes (3)
        (3, False, None),
        # Case 1: Too few cubes (2 instead of 3)
        (2, True, "Expected 3 cubes, found 2"),
        # Case 2: No cubes (0 instead of 3)
        (0, True, "Expected 3 cubes, found 0"),
        # Case 3: Too many cubes (4 instead of 3)
        (4, True, "Expected 3 cubes, found 4"),
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

    plugin = DroughtCode()

    if should_raise:
        with pytest.raises(ValueError, match=expected_message):
            plugin.load_input_cubes(CubeList(cubes), month=7)
    else:
        # Should not raise - verify it loads successfully
        plugin.load_input_cubes(CubeList(cubes), month=7)
        assert isinstance(plugin.temperature, Cube)


@pytest.mark.parametrize(
    "month, should_raise, expected_message",
    [
        # Valid months
        (1, False, None),
        (6, False, None),
        (12, False, None),
        # Invalid months
        (0, True, "Month must be between 1 and 12, got 0"),
        (13, True, "Month must be between 1 and 12, got 13"),
        (-1, True, "Month must be between 1 and 12, got -1"),
    ],
)
def test_load_input_cubes_month_validation(
    month: int,
    should_raise: bool,
    expected_message: str,
) -> None:
    """Test that load_input_cubes validates month parameter correctly.

    Args:
        month (int): Month value to test.
        should_raise (bool): Whether a ValueError should be raised.
        expected_message (str): Expected error message (or None if no error expected).

    Raises:
        AssertionError: If month validation does not match expectations.
    """
    cubes = input_cubes()
    plugin = DroughtCode()

    if should_raise:
        with pytest.raises(ValueError, match=expected_message):
            plugin.load_input_cubes(CubeList(cubes), month=month)
    else:
        # Should not raise - verify it loads successfully
        plugin.load_input_cubes(CubeList(cubes), month=month)
        assert plugin.month == month


@pytest.mark.parametrize(
    "precip_val, prev_dc, expected_dc",
    [
        # Case 0: No rain, DC unchanged
        (0.0, 20.0, 20.0),
        # Case 1: Rain below threshold (2.8 mm), DC unchanged
        (1.0, 20.0, 20.0),
        # Case 2: Rain on threshold limit, DC unchanged
        (2.8, 20.0, 20.0),
        # Case 3: Rain just above threshold, DC decreases slightly
        (3.0, 20.0, 17.48),
        # Case 4: Moderate rain with low previous DC
        (5.0, 20.0, 14.08),
        # Case 5: Heavy rain with low previous DC
        (10.0, 20.0, 5.71),
        # Case 6: Heavy rain with high previous DC
        (10.0, 200.0, 177.81),
        # Case 7: Moderate rain with moderate DC
        (5.0, 100.0, 92.79),
        # Case 8: Very heavy rain with moderate DC
        (25.0, 100.0, 53.56),
        # Case 9: Rain with very low DC near zero
        (10.0, 5.0, 0.0),
        # Case 10: Rain with very high DC
        (5.0, 500.0, 480.69),
    ],
)
def test__perform_rainfall_adjustment(
    precip_val: float,
    prev_dc: float,
    expected_dc: float,
) -> None:
    """Test _perform_rainfall_adjustment for various rainfall and DC scenarios.

    Tests include: no adjustment (precip <= 2.8), and various rainfall amounts
    with different previous DC values.

    Args:
        precip_val (float): Precipitation value for all grid points.
        prev_dc (float): Previous DC value for all grid points.
        expected_dc (float): Expected DC after adjustment.

    Raises:
        AssertionError: If the DC adjustment does not match expectations.
    """
    cubes = input_cubes(precip_val=precip_val, dc_val=prev_dc)
    plugin = DroughtCode()
    plugin.load_input_cubes(CubeList(cubes), month=7)
    # previous_dc is set in load_input_cubes, overwriting for explicit test control
    plugin.previous_dc = np.full(plugin.precipitation.data.shape, prev_dc)
    plugin._perform_rainfall_adjustment()
    adjusted_dc = plugin.previous_dc
    # Check that all points are modified by the correct amount
    assert np.allclose(adjusted_dc, expected_dc, atol=0.05)


def test__perform_rainfall_adjustment_spatially_varying() -> None:
    """Test rainfall adjustment with spatially varying data (vectorization check)."""
    shape = (4, 4)
    # Produce a checkerboard precipitation pattern (10mm and 0mm alternating)
    precip_data = np.zeros(shape)
    precip_data[::2, ::2] = precip_data[1::2, 1::2] = 10.0

    dc_data = np.array(
        [
            [10.0, 50.0, 100.0, 200.0],
            [20.0, 75.0, 150.0, 300.0],
            [30.0, 100.0, 175.0, 400.0],
            [15.0, 60.0, 125.0, 250.0],
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
        make_cube(dc_data, "drought_code", "1", add_time_coord=True),
    ]

    plugin = DroughtCode()
    plugin.load_input_cubes(CubeList(cubes), month=7)
    plugin.previous_dc = dc_data.copy()
    plugin._perform_rainfall_adjustment()

    # No-rain cells unchanged, rain cells decreased
    assert np.allclose(plugin.previous_dc[0, 1], 50.0)
    assert np.allclose(plugin.previous_dc[0, 3], dc_data[0, 3])
    assert np.all(plugin.previous_dc[::2, ::2] <= dc_data[::2, ::2])
    assert np.all(plugin.previous_dc[1::2, 1::2] <= dc_data[1::2, 1::2])


@pytest.mark.parametrize(
    "temp_val, month, expected_pe",
    [
        # Case 0: Typical mid-range values for July
        (20.0, 7, 14.61),
        # Case 1: Cold January
        (0.0, 1, -0.59),
        # Case 2: Hot June
        (30.0, 6, 17.61),
        # Case 3: Very cold December
        (-10.0, 12, -1.60),
        # Case 4: Spring conditions (April)
        (15.0, 4, 7.31),
        # Case 5: Temperature at lower bound (-2.8°C)
        (-2.8, 7, 6.40),
        # Case 6: Temperature just below lower bound (should be clipped to -2.8°C)
        (-5.0, 7, 6.40),
        # Case 7: Cold winter month (February)
        (-5.0, 2, -1.60),
        # Case 8: Warm summer month (August)
        (25.0, 8, 15.01),
        # Case 9: Autumn conditions (September)
        (12.0, 9, 7.73),
        # Case 10: Cool October
        (8.0, 10, 4.29),
    ],
)
def test__calculate_potential_evapotranspiration(
    temp_val: float,
    month: int,
    expected_pe: float,
) -> None:
    """
    Test _calculate_potential_evapotranspiration for various temperature and month combinations.

    Args:
        temp_val (float): Temperature value for all grid points.
        month (int): Month of the year (1-12).
        expected_pe (float): Expected potential evapotranspiration value.

    Raises:
        AssertionError: If the PE calculation does not match expectations.
    """
    cubes = input_cubes(temp_val=temp_val)
    plugin = DroughtCode()
    plugin.load_input_cubes(CubeList(cubes), month=month)
    pe = plugin._calculate_potential_evapotranspiration()
    # Check output type and shape
    assert isinstance(pe, np.ndarray)
    assert pe.shape == cubes[0].data.shape
    # Check that potential evapotranspiration matches expected value
    assert np.allclose(pe, expected_pe, atol=0.01)


def test__calculate_potential_evapotranspiration_spatially_varying() -> None:
    """Test potential evapotranspiration with spatially varying temperature (vectorization check)."""
    temp_data = np.array([[-10.0, -5.0, 0.0], [5.0, 10.0, 15.0], [20.0, 25.0, 30.0]])

    cubes = [
        make_cube(temp_data, "air_temperature", "degC"),
        make_cube(
            np.zeros((3, 3)),
            "lwe_thickness_of_precipitation_amount",
            "mm",
            add_time_coord=True,
        ),
        make_cube(np.full((3, 3), 50.0), "drought_code", "1", add_time_coord=True),
    ]

    plugin = DroughtCode()
    plugin.load_input_cubes(CubeList(cubes), month=7)
    pot_evapotrans = plugin._calculate_potential_evapotranspiration()

    # Temperature effect: warmer produces higher Potential Evapotranspiration
    assert pot_evapotrans[2, 2] > pot_evapotrans[1, 1] > pot_evapotrans[0, 2]
    # Below temp bound gets clamped
    assert pot_evapotrans[0, 0] == pot_evapotrans[0, 1]
    # All values should be non-negative or reasonable for winter months
    assert isinstance(pot_evapotrans, np.ndarray)
    assert pot_evapotrans.shape == (3, 3)


@pytest.mark.parametrize(
    "prev_dc, potential_evapotranspiration, expected_dc",
    [
        # Case 0: Typical values
        (50.0, 10.0, 55.0),
        # Case 1: Zero previous DC
        (0.0, 5.0, 2.5),
        # Case 2: No evapotranspiration
        (100.0, 0.0, 100.0),
        # Case 3: High values
        (500.0, 20.0, 510.0),
        # Case 4: Negative PE (winter conditions)
        (50.0, -2.0, 49.0),
        # Case 5: Would be negative without lower bound
        (0.0, -5.0, 0.0),
    ],
)
def test__calculate_dc(
    prev_dc: float,
    potential_evapotranspiration: float,
    expected_dc: float,
) -> None:
    """Test _calculate_dc for various previous DC and potential evapotranspiration values.

    Args:
        prev_dc (float): Previous DC value.
        potential_evapotranspiration (float): Potential evapotranspiration value.
        expected_dc (float): Expected DC output value.

    Raises:
        AssertionError: If the DC calculation does not match expectations.
    """
    plugin = DroughtCode()
    plugin.previous_dc = np.array([prev_dc])
    dc = plugin._calculate_dc(np.array([potential_evapotranspiration]))
    # Check output type and shape
    assert isinstance(dc, np.ndarray)
    assert dc.shape == (1,)
    # Check that DC matches expected output
    assert np.allclose(dc, expected_dc, atol=0.01)


@pytest.mark.parametrize(
    "dc_value, shape",
    [
        # Case 0: Typical mid-range DC value with standard grid
        (50.0, (5, 5)),
        # Case 1: Low DC value with different grid size
        (0.0, (3, 4)),
        # Case 2: High DC value with larger grid
        (500.0, (10, 10)),
        # Case 3: Very high DC (edge case) with small grid
        (1000.0, (2, 2)),
        # Case 4: Standard DC value
        (15.0, (5, 5)),
    ],
)
def test__make_dc_cube(
    dc_value: float,
    shape: tuple[int, int],
) -> None:
    """
    Test _make_dc_cube to ensure it creates an Iris Cube with correct properties
    for various DC values and grid shapes.

    Args:
        dc_value (float): DC data value to use for all grid points.
        shape (tuple[int, int]): Shape of the grid.

    Raises:
        AssertionError: If the created cube does not have expected properties.
    """
    # Create input cubes with specified shape
    cubes = input_cubes(shape=shape)

    # Initialize the plugin and load cubes
    plugin = DroughtCode()
    plugin.load_input_cubes(CubeList(cubes), month=7)

    # Create test DC data
    dc_data = np.full(shape, dc_value, dtype=np.float64)

    # Call the method under test
    result_cube = plugin._make_dc_cube(dc_data)

    # Check that result is an Iris Cube with correct type and shape
    assert isinstance(result_cube, Cube)
    assert result_cube.data.dtype == np.float32
    assert result_cube.data.shape == shape
    assert np.allclose(result_cube.data, dc_value, atol=0.001)

    # Check that the cube has the correct name and units
    assert result_cube.long_name == "drought_code"
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

    # Check that time coordinate has no bounds (removed by _make_dc_cube)
    assert result_time.bounds is None


@pytest.mark.parametrize(
    "temp_val, precip_val, dc_val, month, expected_output",
    [
        # Case 0: Typical mid-range values for July
        (20.0, 1.0, 15.0, 7, 22.31),
        # Case 1: All zeros (edge case) for January
        (0.0, 0.0, 0.0, 1, 0.00),
        # Case 2: High temp, no precip, high DC (produces higher output DC) for June
        (35.0, 0.0, 200.0, 6, 209.70),
        # Case 3: Low temp, high precip, moderate DC (produces lower output DC) for August
        (10.0, 15.0, 100.0, 8, 77.50),
        # Case 4: Precipitation just below threshold (should not adjust) for May
        (20.0, 2.0, 100.0, 5, 106.00),
        # Case 5: Winter conditions with negative PE (December)
        (-5.0, 0.0, 50.0, 12, 49.20),
        # Case 6: Spring warming with moderate DC (April)
        (15.0, 0.0, 80.0, 4, 83.65),
        # Case 7: Heavy rain reduction (September)
        (18.0, 20.0, 150.0, 9, 113.28),
    ],
)
def test_process(
    temp_val: float,
    precip_val: float,
    dc_val: float,
    month: int,
    expected_output: float,
) -> None:
    """Integration test for the complete DC calculation process.

    Tests end-to-end functionality with various environmental conditions and
    verifies the final DC output matches expected values.

    Args:
        temp_val (float): Temperature value for all grid points.
        precip_val (float): Precipitation value for all grid points.
        dc_val (float): DC value for all grid points.
        month (int): Month of the year (1-12).
        expected_output (float): Expected DC output value for all grid points.

    Raises:
        AssertionError: If the process output does not match expectations.
    """
    cubes = input_cubes(temp_val, precip_val, dc_val)
    plugin = DroughtCode()
    result = plugin.process(CubeList(cubes), month=month)

    # Check output type and shape
    assert hasattr(result, "data")
    assert result.data.shape == cubes[0].data.shape

    # Check that DC matches expected output within tolerance
    data = np.array(result.data)
    assert np.allclose(data, expected_output, atol=0.05)


def test_process_default_month() -> None:
    """Test that process method works with default month parameter."""
    cubes = input_cubes()
    plugin = DroughtCode()

    # Should not raise - uses current month by default
    result = plugin.process(CubeList(cubes))

    # Check that the month is set to current month
    from datetime import datetime

    current_month = datetime.now().month
    assert plugin.month == current_month

    # Check that result is valid
    assert hasattr(result, "data")
    assert result.data.shape == cubes[0].data.shape
    assert isinstance(result.data[0][0], (float, np.floating))


def test_process_spatially_varying() -> None:
    """Integration test with spatially varying data (vectorization check)."""
    temp_data = np.array([[5.0, 10.0, 15.0], [10.0, 15.0, 20.0], [15.0, 20.0, 25.0]])
    precip_data = np.array([[0.0, 3.0, 10.0], [0.0, 0.0, 15.0], [0.0, 0.0, 0.0]])
    dc_data = np.array(
        [[10.0, 50.0, 100.0], [20.0, 150.0, 200.0], [50.0, 100.0, 300.0]]
    )

    cubes = [
        make_cube(temp_data, "air_temperature", "degC"),
        make_cube(
            precip_data,
            "lwe_thickness_of_precipitation_amount",
            "mm",
            add_time_coord=True,
        ),
        make_cube(dc_data, "drought_code", "1", add_time_coord=True),
    ]

    result = DroughtCode().process(CubeList(cubes), month=7)

    # Verify shape, type, and all non-negative
    assert (
        result.data.shape == (3, 3)
        and result.data.dtype == np.float32
        and np.all(result.data >= 0.0)
    )
    # Hot/dry/no-rain increases DC; heavy rain decreases; unique values (no broadcast errors)
    assert result.data[2, 0] > dc_data[2, 0]
    assert result.data[0, 2] < dc_data[0, 2]
    assert len(np.unique(result.data)) > 1


def test_day_length_factors_table() -> None:
    """Test that DC_DAY_LENGTH_FACTORS match the expected values from Van Wagner and Pickett Table 2."""
    expected_factors = [
        0.0,  # Placeholder
        -1.6,  # January
        -1.6,  # February
        -1.6,  # March
        0.9,  # April
        3.8,  # May
        5.8,  # June
        6.4,  # July
        5.0,  # August
        2.4,  # September
        0.4,  # October
        -1.6,  # November
        -1.6,  # December
    ]

    assert DroughtCode.DC_DAY_LENGTH_FACTORS == expected_factors
