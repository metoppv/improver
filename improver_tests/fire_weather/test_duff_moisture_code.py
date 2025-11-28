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

from improver.fire_weather.duff_moisture_code import DuffMoistureCode


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
    dmc_val: float = 6.0,
    shape: tuple[int, int] = (5, 5),
    temp_units: str = "degC",
    precip_units: str = "mm",
    rh_units: str = "1",
    dmc_units: str = "1",
) -> list[Cube]:
    """Create a list of dummy input cubes for DMC tests, with configurable units.

    All cubes have forecast_reference_time. Precipitation and DMC cubes also have
    time coordinates with bounds.

    Args:
        temp_val (float): Temperature value for all grid points.
        precip_val (float): Precipitation value for all grid points.
        rh_val (float): Relative humidity value for all grid points.
        dmc_val (float): DMC value for all grid points.
        shape (tuple[int, int]): Shape of the grid for each cube.
        temp_units (str): Units for temperature cube.
        precip_units (str): Units for precipitation cube.
        rh_units (str): Units for relative humidity cube.
        dmc_units (str): Units for DMC cube.

    Returns:
        list[Cube]: List of Iris Cubes for temperature, precipitation, relative humidity, and DMC.
    """
    temp = make_cube(np.full(shape, temp_val), "air_temperature", temp_units)
    # Precipitation cube needs time coordinates for _make_dmc_cube
    precip = make_cube(
        np.full(shape, precip_val),
        "lwe_thickness_of_precipitation_amount",
        precip_units,
        add_time_coord=True,
    )
    rh = make_cube(np.full(shape, rh_val), "relative_humidity", rh_units)
    # DMC cube needs time coordinates for _make_dmc_cube to copy metadata
    dmc = make_cube(
        np.full(shape, dmc_val),
        "duff_moisture_code",
        dmc_units,
        add_time_coord=True,
    )
    return [temp, precip, rh, dmc]


@pytest.mark.parametrize(
    "temp_val, precip_val, rh_val, dmc_val",
    [
        # Case 0: Typical mid-range values
        (20.0, 1.0, 50.0, 6.0),
        # Case 1: All zeros (edge case)
        (0.0, 0.0, 0.0, 0.0),
        # Case 2: All maximums/extremes
        (100.0, 100.0, 100.0, 100.0),
        # Case 3: Low temperature, low precip, low relative humidity, low DMC
        (-10.0, 0.5, 10.0, 2.0),
        # Case 4: High temp, high precip, high relative humidity, high DMC
        (30.0, 10.0, 90.0, 120.0),
    ],
)
def test_load_input_cubes(
    temp_val: float,
    precip_val: float,
    rh_val: float,
    dmc_val: float,
) -> None:
    """Test DuffMoistureCode.load_input_cubes with various input conditions.

    Args:
        temp_val (float): Temperature value for all grid points.
        precip_val (float): Precipitation value for all grid points.
        rh_val (float): Relative humidity value for all grid points.
        dmc_val (float): DMC value for all grid points.

    Raises:
        AssertionError: If the loaded cubes do not match expected shapes and types.
    """
    cubes = input_cubes(temp_val, precip_val, rh_val, dmc_val)
    plugin = DuffMoistureCode()
    plugin.load_input_cubes(CubeList(cubes), month=7)

    attributes = [
        plugin.temperature,
        plugin.precipitation,
        plugin.relative_humidity,
        plugin.input_dmc,
    ]
    input_values = [temp_val, precip_val, rh_val, dmc_val]

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
        # Case 2: Relative humidity: percentage -> fraction
        ("relative_humidity", 10.0, "%", 0.1),
        # Case 3: Input DMC: no conversion needed (dimensionless)
        ("input_dmc", 6.0, "1", 6.0),
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
    elif param == "input_dmc":
        cubes = input_cubes(dmc_val=input_val, dmc_units=input_unit)

    plugin = DuffMoistureCode()
    plugin.load_input_cubes(CubeList(cubes), month=7)
    # Check only the parameter being tested
    result = getattr(plugin, param)
    assert np.allclose(result.data, expected_val)


@pytest.mark.parametrize(
    "num_cubes, should_raise, expected_message",
    [
        # Case 0: Correct number of cubes (4)
        (4, False, None),
        # Case 1: Too few cubes (3 instead of 4)
        (3, True, "Expected 4 cubes, found 3"),
        # Case 2: No cubes (0 instead of 4)
        (0, True, "Expected 4 cubes, found 0"),
        # Case 3: Too many cubes (5 instead of 4)
        (5, True, "Expected 4 cubes, found 5"),
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

    plugin = DuffMoistureCode()

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
    plugin = DuffMoistureCode()

    if should_raise:
        with pytest.raises(ValueError, match=expected_message):
            plugin.load_input_cubes(CubeList(cubes), month=month)
    else:
        # Should not raise - verify it loads successfully
        plugin.load_input_cubes(CubeList(cubes), month=month)
        assert plugin.month == month


@pytest.mark.parametrize(
    "precip_val, prev_dmc, expected_dmc",
    [
        # Case 0: No rain, DMC unchanged
        (0.0, 10.0, 10.0),
        # Case 1: Rain below threshold (1.5 mm), DMC unchanged
        (1.0, 10.0, 10.0),
        # Case 2: Rain on threshold limit, DMC unchanged
        (1.5, 10.0, 10.0),
        # Case 3: Rain above threshold, DMC decreases
        (2.0, 10.0, 8.32),
        # Case 4: Heavy rain with low previous DMC
        (10.0, 10.0, 4.71),
        # Case 5: Heavy rain with high previous DMC
        (10.0, 50.0, 25.70),
        # Case 6: Moderate rain with moderate DMC
        (5.0, 30.0, 19.17),
        # Case 7: Rain with DMC <= 33 (tests Equation 13a)
        (5.0, 20.0, 12.50),
        # Case 8: Rain with 33 < DMC <= 65 (tests Equation 13b)
        (5.0, 45.0, 29.63),
        # Case 9: Rain with DMC > 65 (tests Equation 13c)
        (5.0, 80.0, 51.77),
        # Case 10: Rain with very low DMC near log domain edge
        (10.0, 2.0, 0.36),
    ],
)
def test__perform_rainfall_adjustment(
    precip_val: float,
    prev_dmc: float,
    expected_dmc: float,
) -> None:
    """Test _perform_rainfall_adjustment for various rainfall and DMC scenarios.

    Tests include: no adjustment (precip <= 1.5), and various rainfall amounts
    with different previous DMC values.

    Args:
        precip_val (float): Precipitation value for all grid points.
        prev_dmc (float): Previous DMC value for all grid points.
        expected_dmc (float): Expected DMC after adjustment.

    Raises:
        AssertionError: If the DMC adjustment does not match expectations.
    """
    cubes = input_cubes(precip_val=precip_val, dmc_val=prev_dmc)
    plugin = DuffMoistureCode()
    plugin.load_input_cubes(CubeList(cubes), month=7)
    # previous_dmc is set in load_input_cubes, overwriting for explicit test control
    plugin.previous_dmc = np.full(plugin.precipitation.data.shape, prev_dmc)
    plugin._perform_rainfall_adjustment()
    adjusted_dmc = plugin.previous_dmc
    # Check that all points are modified by the correct amount
    assert np.allclose(adjusted_dmc, expected_dmc, atol=0.05)


def test__perform_rainfall_adjustment_spatially_varying() -> None:
    """Test rainfall adjustment with spatially varying data (vectorization check)."""
    shape = (4, 4)
    # Produce a checkerboard precipitation pattern (5mm and 0mm alternating)
    precip_data = np.zeros(shape)
    precip_data[::2, ::2] = precip_data[1::2, 1::2] = 5.0

    dmc_data = np.array(
        [
            [10.0, 25.0, 40.0, 70.0],
            [15.0, 30.0, 50.0, 80.0],
            [20.0, 35.0, 60.0, 90.0],
            [12.0, 28.0, 45.0, 75.0],
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
        make_cube(dmc_data, "duff_moisture_code", "1", add_time_coord=True),
    ]

    plugin = DuffMoistureCode()
    plugin.load_input_cubes(CubeList(cubes), month=7)
    plugin.previous_dmc = dmc_data.copy()
    plugin._perform_rainfall_adjustment()

    # No-rain cells unchanged, rain cells decreased
    assert np.allclose(plugin.previous_dmc[0, 1], 25.0) and np.allclose(
        plugin.previous_dmc[0, 3], dmc_data[0, 3]
    )
    assert np.all(plugin.previous_dmc[::2, ::2] <= dmc_data[::2, ::2])
    assert np.all(plugin.previous_dmc[1::2, 1::2] <= dmc_data[1::2, 1::2])


@pytest.mark.parametrize(
    "temp_val, rh_val, month, expected_rate",
    [
        # Case 0: Typical mid-range values for July
        (20.0, 50.0, 7, 2.478),
        # Case 1: Cold, dry January
        (0.0, 0.0, 1, 0.135),
        # Case 2: Hot, humid June
        (30.0, 90.0, 6, 0.819),
        # Case 3: Very cold December
        (-10.0, 10.0, 12, 0.000),
        # Case 4: Spring conditions (April)
        (15.0, 60.0, 4, 1.561),
        # Case 5: Temperature at lower bound (-1.1°C)
        (-1.1, 50.0, 7, 0.000),
        # Case 6: Temperature just below lower bound (should be clipped to -1.1°C)
        (-5.0, 50.0, 7, 0.000),
    ],
)
def test__calculate_drying_rate(
    temp_val: float,
    rh_val: float,
    month: int,
    expected_rate: float,
) -> None:
    """
    Test _calculate_drying_rate for various temperature, relative humidity, and month combinations.

    Args:
        temp_val (float): Temperature value for all grid points.
        rh_val (float): Relative humidity value for all grid points.
        month (int): Month of the year (1-12).
        expected_rate (float): Expected drying rate value.

    Raises:
        AssertionError: If the drying rate calculation does not match expectations.
    """
    cubes = input_cubes(temp_val=temp_val, rh_val=rh_val)
    plugin = DuffMoistureCode()
    plugin.load_input_cubes(CubeList(cubes), month=month)
    rate = plugin._calculate_drying_rate()
    # Check output type and shape
    assert isinstance(rate, np.ndarray)
    assert rate.shape == cubes[0].data.shape
    # Check that drying rate matches expected value
    assert np.allclose(rate, expected_rate, atol=0.01)


def test__calculate_drying_rate_spatially_varying() -> None:
    """Test drying rate with spatially varying temperature/relative humidity (vectorization check)."""
    temp_data = np.array([[-5.0, 0.0, 10.0], [15.0, 20.0, 25.0], [30.0, 35.0, 40.0]])
    rh_data = np.array([[20.0, 30.0, 40.0], [50.0, 60.0, 70.0], [80.0, 90.0, 95.0]])

    cubes = [
        make_cube(temp_data, "air_temperature", "degC"),
        make_cube(
            np.zeros((3, 3)),
            "lwe_thickness_of_precipitation_amount",
            "mm",
            add_time_coord=True,
        ),
        make_cube(rh_data, "relative_humidity", "1"),
        make_cube(
            np.full((3, 3), 10.0), "duff_moisture_code", "1", add_time_coord=True
        ),
    ]

    plugin = DuffMoistureCode()
    plugin.load_input_cubes(CubeList(cubes), month=7)
    rate = plugin._calculate_drying_rate()

    # relative humidity dominates: low relative humidity beats high temp
    assert rate[2, 0] > rate[2, 1] and rate[2, 0] > rate[2, 2]
    # Temperature effect: warmer produces higher rate (same relative humidity column)
    assert rate[1, 2] > rate[0, 2]
    # Below temp bound gives zero, mid conditions beat extreme humid
    assert rate[0, 0] == 0.0 and rate[1, 1] > rate[2, 2] and np.all(rate >= 0.0)


@pytest.mark.parametrize(
    "prev_dmc, drying_rate, expected_dmc",
    [
        # Case 0: Typical values
        (10.0, 0.3, 10.3),
        # Case 1: Zero previous DMC
        (0.0, 0.1, 0.1),
        # Case 2: No drying
        (50.0, 0.0, 50.0),
        # Case 3: High values
        (100.0, 1.0, 101.0),
        # Case 4: Would be negative without lower bound
        (0.0, -0.5, 0.0),
    ],
)
def test__calculate_dmc(
    prev_dmc: float,
    drying_rate: float,
    expected_dmc: float,
) -> None:
    """Test _calculate_dmc for various previous DMC and drying rate values.

    Args:
        prev_dmc (float): Previous DMC value.
        drying_rate (float): Drying rate value.
        expected_dmc (float): Expected DMC output value.

    Raises:
        AssertionError: If the DMC calculation does not match expectations.
    """
    plugin = DuffMoistureCode()
    plugin.previous_dmc = np.array([prev_dmc])
    dmc = plugin._calculate_dmc(np.array([drying_rate]))
    # Check output type and shape
    assert isinstance(dmc, np.ndarray)
    assert dmc.shape == (1,)
    # Check that DMC matches expected output
    assert np.allclose(dmc, expected_dmc, atol=0.01)


@pytest.mark.parametrize(
    "dmc_value, shape",
    [
        # Case 0: Typical mid-range DMC value with standard grid
        (10.0, (5, 5)),
        # Case 1: Low DMC value with different grid size
        (0.0, (3, 4)),
        # Case 2: High DMC value with larger grid
        (50.0, (10, 10)),
        # Case 3: Very high DMC (edge case) with small grid
        (200.0, (2, 2)),
        # Case 4: Standard DMC value
        (6.0, (5, 5)),
    ],
)
def test__make_dmc_cube(
    dmc_value: float,
    shape: tuple[int, int],
) -> None:
    """
    Test _make_dmc_cube to ensure it creates an Iris Cube with correct properties
    for various DMC values and grid shapes.

    Args:
        dmc_value (float): DMC data value to use for all grid points.
        shape (tuple[int, int]): Shape of the grid.

    Raises:
        AssertionError: If the created cube does not have expected properties.
    """
    # Create input cubes with specified shape
    cubes = input_cubes(shape=shape)

    # Initialize the plugin and load cubes
    plugin = DuffMoistureCode()
    plugin.load_input_cubes(CubeList(cubes), month=7)

    # Create test DMC data
    dmc_data = np.full(shape, dmc_value, dtype=np.float64)

    # Call the method under test
    result_cube = plugin._make_dmc_cube(dmc_data)

    # Check that result is an Iris Cube with correct type and shape
    assert isinstance(result_cube, Cube)
    assert result_cube.data.dtype == np.float32
    assert result_cube.data.shape == shape
    assert np.allclose(result_cube.data, dmc_value, atol=0.001)

    # Check that the cube has the correct name and units
    assert result_cube.long_name == "duff_moisture_code"
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

    # Check that time coordinate has no bounds (removed by _make_dmc_cube)
    assert result_time.bounds is None


@pytest.mark.parametrize(
    "temp_val, precip_val, rh_val, dmc_val, month, expected_output",
    [
        # Case 0: Typical mid-range values
        (20.0, 1.0, 50.0, 6.0, 7, 8.48),
        # Case 1: All zeros (edge case)
        (0.0, 0.0, 0.0, 0.0, 1, 0.14),
        # Case 2: High temp, no precip, low relative humidity, high DMC (produces high output DMC)
        (35.0, 0.0, 15.0, 90.0, 6, 98.08),
        # Case 3: Low temp, high precip, high relative humidity (produces low output DMC)
        (10.0, 15.0, 95.0, 85.0, 8, 40.77),
        # Case 4: Precipitation just below threshold (should not adjust)
        (20.0, 0.4, 50.0, 85.0, 5, 87.78),
    ],
)
def test_process(
    temp_val: float,
    precip_val: float,
    rh_val: float,
    dmc_val: float,
    month: int,
    expected_output: float,
) -> None:
    """Integration test for the complete DMC calculation process.

    Tests end-to-end functionality with various environmental conditions and
    verifies the final DMC output matches expected values.

    Args:
        temp_val (float): Temperature value for all grid points.
        precip_val (float): Precipitation value for all grid points.
        rh_val (float): Relative humidity value for all grid points.
        dmc_val (float): DMC value for all grid points.
        month (int): Month of the year (1-12).
        expected_output (float): Expected DMC output value for all grid points.

    Raises:
        AssertionError: If the process output does not match expectations.
    """
    cubes = input_cubes(temp_val, precip_val, rh_val, dmc_val)
    plugin = DuffMoistureCode()
    result = plugin.process(CubeList(cubes), month=month)

    # Check output type and shape
    assert hasattr(result, "data")
    assert result.data.shape == cubes[0].data.shape

    # Check that DMC matches expected output within tolerance
    data = np.array(result.data)
    assert np.allclose(data, expected_output, atol=0.05)


def test_process_default_month() -> None:
    """Test that process method works with default month parameter."""
    cubes = input_cubes()
    plugin = DuffMoistureCode()

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
    temp_data = np.array([[10.0, 15.0, 20.0], [15.0, 20.0, 25.0], [20.0, 25.0, 30.0]])
    precip_data = np.array([[0.0, 2.0, 5.0], [0.0, 0.0, 10.0], [0.0, 0.0, 0.0]])
    rh_data = np.array([[40.0, 50.0, 60.0], [50.0, 60.0, 70.0], [60.0, 70.0, 80.0]])
    dmc_data = np.array([[5.0, 15.0, 30.0], [10.0, 50.0, 70.0], [20.0, 40.0, 90.0]])

    cubes = [
        make_cube(temp_data, "air_temperature", "degC"),
        make_cube(
            precip_data,
            "lwe_thickness_of_precipitation_amount",
            "mm",
            add_time_coord=True,
        ),
        make_cube(rh_data, "relative_humidity", "1"),
        make_cube(dmc_data, "duff_moisture_code", "1", add_time_coord=True),
    ]

    result = DuffMoistureCode().process(CubeList(cubes), month=7)

    # Verify shape, type, and all non-negative
    assert (
        result.data.shape == (3, 3)
        and result.data.dtype == np.float32
        and np.all(result.data >= 0.0)
    )
    # Hot/dry/no-rain increases DMC; heavy rain decreases; unique values (no broadcast errors)
    assert (
        result.data[2, 0] > dmc_data[2, 0] and result.data[0, 2] <= dmc_data[0, 2] + 2.0
    )
    assert len(np.unique(result.data)) > 1


def test_day_length_factors_table() -> None:
    """Test that DMC_DAY_LENGTH_FACTORS match the expected values from Van Wagner and Pickett Table 1."""
    expected_factors = [
        0.0,  # Placeholder for index 0
        6.5,  # January
        7.5,  # February
        9.0,  # March
        12.8,  # April
        13.9,  # May
        13.9,  # June
        12.4,  # July
        10.9,  # August
        9.4,  # September
        8.0,  # October
        7.0,  # November
        6.0,  # December
    ]

    assert DuffMoistureCode.DMC_DAY_LENGTH_FACTORS == expected_factors
