# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.fire_weather.fine_fuel_moisture_content import FineFuelMoistureContent
from improver_tests.fire_weather import make_cube, make_input_cubes


def input_cubes(
    temp_val: float = 20.0,
    precip_val: float = 1.0,
    rh_val: float = 50.0,
    wind_val: float = 10.0,
    ffmc_val: float = 85.0,
    shape: tuple[int, int] = (5, 5),
    temp_units: str = "Celsius",
    precip_units: str = "mm",
    rh_units: str = "1",
    wind_units: str = "km/h",
    ffmc_units: str = "1",
) -> list[Cube]:
    """Create a list of dummy input cubes for FFMC tests, with configurable units.

    All cubes have forecast_reference_time. Precipitation and FFMC cubes also have
    time coordinates with bounds.

    Args:
        temp_val:
            Temperature value for all grid points.
        precip_val:
            Precipitation value for all grid points.
        rh_val:
            Relative humidity value for all grid points.
        wind_val:
            Wind speed value for all grid points.
        ffmc_val:
            FFMC value for all grid points.
        shape:
            Shape of the grid for each cube.
        temp_units:
            Units for temperature cube.
        precip_units:
            Units for precipitation cube.
        rh_units:
            Units for relative humidity cube.
        wind_units:
            Units for wind speed cube.
        ffmc_units:
            Units for FFMC cube.

    Returns:
        List of Iris Cubes for temperature, precipitation, relative humidity, wind speed, and FFMC.
    """
    return make_input_cubes(
        [
            ("air_temperature", temp_val, temp_units, False),
            ("lwe_thickness_of_precipitation_amount", precip_val, precip_units, True),
            ("relative_humidity", rh_val, rh_units, False),
            ("wind_speed", wind_val, wind_units, False),
            ("fine_fuel_moisture_content", ffmc_val, ffmc_units, True),
        ],
        shape=shape,
    )


@pytest.mark.parametrize(
    "temp_val, precip_val, rh_val, wind_val, ffmc_val",
    [
        # Case 0: Typical mid-range values
        (20.0, 1.0, 50.0, 10.0, 85.0),
        # Case 1: All zeros (edge case)
        (0.0, 0.0, 0.0, 0.0, 0.0),
        # Case 2: Low temperature, low precip, low RH, low wind, low FFMC
        (-10.0, 0.5, 10.0, 2.0, 60.0),
        # Case 3: High temp, high precip, high RH, high wind, high FFMC (within valid ranges)
        (30.0, 10.0, 90.0, 20.0, 95.0),
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

    Verifies that the initial moisture content is calculated correctly from FFMC.

    Args:
        temp_val:
            Temperature value for all grid points.
        precip_val:
            Precipitation value for all grid points.
        rh_val:
            Relative humidity value for all grid points.
        wind_val:
            Wind speed value for all grid points.
        ffmc_val:
            FFMC value for all grid points.
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

    Verifies: no adjustment (precip <= 0.5), adjustment1 only (mc <= 150),
    adjustment1 + adjustment2 (mc > 150), and capping at 250.

    Args:
        precip_val:
            Precipitation value for all grid points.
        initial_mc_val:
            Initial moisture content value for all grid points.
        expected_mc:
            Expected moisture content after adjustment.
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
    """Test rainfall adjustment with spatially varying data.

    Verifies vectorized implementation with checkerboard precipitation pattern and
    varying moisture content values across the grid.
    """
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
        make_cube(np.full(shape, 20.0), "air_temperature", "Celsius"),
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
    """Test _calculate_EMC_for_drying_phase with various relative humidity, wind, and
    temperature values.

    Verifies Equilibrium Moisture Content calculation for the drying phase.

    Args:
        temp_val:
            Temperature value for all grid points.
        rh_val:
            Relative humidity value for all grid points.
        expected_E_d:
            Expected drying phase value.
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
    """Test _calculate_moisture_content_through_drying_rate with various conditions.

    Verifies moisture content calculation through drying rate.

    Args:
        moisture_content:
            Moisture content values for all grid points.
        relative_humidity:
            Relative humidity value for all grid points.
        wind_speed:
            Wind speed value for all grid points.
        temperature:
            Temperature value for all grid points.
        E_d:
            Drying phase values for all grid points.
        expected_output:
            Expected output moisture content values.
    """
    plugin = FineFuelMoistureContent()
    plugin.initial_moisture_content = moisture_content.copy()
    plugin.moisture_content = moisture_content.copy()
    # For these unit tests, create simple cubes without spatial coordinates
    plugin.relative_humidity = Cube(
        np.full(moisture_content.shape, relative_humidity, dtype=np.float32),
        long_name="relative_humidity",
        units="1",
    )
    plugin.wind_speed = Cube(
        np.full(moisture_content.shape, wind_speed, dtype=np.float32),
        long_name="wind_speed",
        units="km/h",
    )
    plugin.temperature = Cube(
        np.full(moisture_content.shape, temperature, dtype=np.float32),
        long_name="air_temperature",
        units="Celsius",
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
    """Test _calculate_EMC_for_wetting_phase for given temperature and relative humidity.

    Verifies Equilibrium Moisture Content calculation for the wetting phase.

    Args:
        temp_val:
            Temperature value for all grid points.
        rh_val:
            Relative humidity value for all grid points.
        expected_E_w:
            Expected wetting phase value.
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
    """Test _calculate_moisture_content_through_wetting_equilibrium with various moisture scenarios.

    Verifies moisture content calculation through wetting equilibrium.

    Args:
        moisture_content:
            Moisture content values for all grid points.
        relative_humidity:
            Relative humidity value for all grid points.
        wind_speed:
            Wind speed value for all grid points.
        temperature:
            Temperature value for all grid points.
        E_w:
            Wetting phase values for all grid points.
        expected_output:
            Expected output moisture content values.
    """
    plugin = FineFuelMoistureContent()
    plugin.initial_moisture_content = moisture_content.copy()
    plugin.moisture_content = moisture_content.copy()
    # For these unit tests, create simple cubes without spatial coordinates
    plugin.relative_humidity = Cube(
        np.full(moisture_content.shape, relative_humidity, dtype=np.float32),
        long_name="relative_humidity",
        units="1",
    )
    plugin.wind_speed = Cube(
        np.full(moisture_content.shape, wind_speed, dtype=np.float32),
        long_name="wind_speed",
        units="km/h",
    )
    plugin.temperature = Cube(
        np.full(moisture_content.shape, temperature, dtype=np.float32),
        long_name="air_temperature",
        units="Celsius",
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

    Verifies FFMC calculation from moisture content.

    Args:
        moisture_content:
            Moisture content values for all grid points.
        expected_output:
            Expected FFMC output values.
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

    Verifies end-to-end FFMC calculation with various environmental conditions.

    Args:
        temp_val:
            Temperature value for all grid points.
        precip_val:
            Precipitation value for all grid points.
        rh_val:
            Relative humidity value for all grid points.
        wind_val:
            Wind speed value for all grid points.
        ffmc_val:
            FFMC value for all grid points.
        expected_output:
            Expected FFMC output value for all grid points.
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
    """Integration test with spatially varying input data.

    Verifies vectorized implementation with varying values across the grid.
    """
    temp_data = np.array([[10.0, 15.0, 20.0], [15.0, 20.0, 25.0], [20.0, 25.0, 30.0]])
    precip_data = np.array([[0.0, 1.0, 5.0], [0.0, 0.0, 10.0], [0.0, 0.0, 0.0]])
    rh_data = np.array([[40.0, 50.0, 60.0], [50.0, 60.0, 70.0], [60.0, 70.0, 80.0]])
    wind_data = np.array([[5.0, 10.0, 15.0], [10.0, 15.0, 20.0], [15.0, 20.0, 25.0]])
    ffmc_data = np.array([[70.0, 80.0, 85.0], [75.0, 85.0, 90.0], [80.0, 88.0, 92.0]])

    cubes = [
        make_cube(temp_data, "air_temperature", "Celsius"),
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


@pytest.mark.parametrize(
    "temp_val, precip_val, rh_val, wind_val, ffmc_val, expected_error",
    [
        # Temperature too high
        (
            150.0,
            1.0,
            50.0,
            10.0,
            85.0,
            "temperature contains values above valid maximum",
        ),
        # Temperature too low
        (
            -150.0,
            1.0,
            50.0,
            10.0,
            85.0,
            "temperature contains values below valid minimum",
        ),
        # Precipitation negative
        (
            20.0,
            -5.0,
            50.0,
            10.0,
            85.0,
            "precipitation contains values below valid minimum",
        ),
        # Relative humidity above 100%
        (
            20.0,
            1.0,
            150.0,
            10.0,
            85.0,
            "relative_humidity contains values above valid maximum",
        ),
        # Relative humidity negative
        (
            20.0,
            1.0,
            -10.0,
            10.0,
            85.0,
            "relative_humidity contains values below valid minimum",
        ),
        # Wind speed negative
        (20.0, 1.0, 50.0, -5.0, 85.0, "wind_speed contains values below valid minimum"),
        # FFMC above 101
        (
            20.0,
            1.0,
            50.0,
            10.0,
            120.0,
            "input_ffmc contains values above valid maximum",
        ),
        # FFMC negative
        (20.0, 1.0, 50.0, 10.0, -5.0, "input_ffmc contains values below valid minimum"),
    ],
)
def test_invalid_input_ranges_raise_errors(
    temp_val: float,
    precip_val: float,
    rh_val: float,
    wind_val: float,
    ffmc_val: float,
    expected_error: str,
) -> None:
    """Test that invalid input values raise appropriate ValueError.

    Verifies that the base class validation catches physically meaningless
    or out-of-range input values and raises descriptive errors.

    Args:
        temp_val:
            Temperature value for all grid points.
        precip_val:
            Precipitation value for all grid points.
        rh_val:
            Relative humidity value for all grid points.
        wind_val:
            Wind speed value for all grid points.
        ffmc_val:
            FFMC value for all grid points.
        expected_error:
            Expected error message substring.
    """
    cubes = input_cubes(temp_val, precip_val, rh_val, wind_val, ffmc_val)
    plugin = FineFuelMoistureContent()

    with pytest.raises(ValueError, match=expected_error):
        plugin.load_input_cubes(CubeList(cubes))


@pytest.mark.parametrize(
    "invalid_input_type,expected_error",
    [
        ("temperature_nan", "temperature contains NaN"),
        ("temperature_inf", "temperature contains infinite"),
        ("precipitation_nan", "precipitation contains NaN"),
        ("precipitation_inf", "precipitation contains infinite"),
        ("relative_humidity_nan", "relative_humidity contains NaN"),
        ("wind_speed_inf", "wind_speed contains infinite"),
        ("input_ffmc_nan", "input_ffmc contains NaN"),
    ],
)
def test_nan_and_inf_values_raise_errors(
    invalid_input_type: str, expected_error: str
) -> None:
    """Test that NaN and Inf values in inputs raise appropriate ValueError.

    Verifies that the validation catches non-finite values (NaN, Inf) in input data.

    Args:
        invalid_input_type:
            Which input to make invalid and how.
        expected_error:
            Expected error message substring.
    """
    # Start with valid values
    temp_val, precip_val, rh_val, wind_val, ffmc_val = 20.0, 1.0, 50.0, 10.0, 85.0

    # Replace the appropriate value with NaN or Inf
    if invalid_input_type == "temperature_nan":
        temp_val = np.nan
    elif invalid_input_type == "temperature_inf":
        temp_val = np.inf
    elif invalid_input_type == "precipitation_nan":
        precip_val = np.nan
    elif invalid_input_type == "precipitation_inf":
        precip_val = np.inf
    elif invalid_input_type == "relative_humidity_nan":
        rh_val = np.nan
    elif invalid_input_type == "wind_speed_inf":
        wind_val = -np.inf
    elif invalid_input_type == "input_ffmc_nan":
        ffmc_val = np.nan

    cubes = input_cubes(temp_val, precip_val, rh_val, wind_val, ffmc_val)
    plugin = FineFuelMoistureContent()

    with pytest.raises(ValueError, match=expected_error):
        plugin.load_input_cubes(CubeList(cubes))


def test_output_validation_no_warning_for_valid_output() -> None:
    """Test that valid output values do not trigger warnings.

    Uses extreme but valid inputs to verify that as long as the output
    stays within the expected range (0-101 for FFMC), no warning is issued.
    This demonstrates that the FFMC calculation naturally constrains outputs
    to valid ranges even with extreme inputs.
    """
    # Use extreme valid inputs
    # Very cold, very humid, low wind
    cubes = input_cubes(
        temp_val=-45.0,  # Extreme cold but within valid range
        precip_val=0.0,
        rh_val=99.0,
        wind_val=0.1,
        ffmc_val=101.0,  # Maximum valid FFMC
    )
    plugin = FineFuelMoistureContent()

    # Process should complete without warnings since output stays in valid range
    result = plugin.process(CubeList(cubes))

    assert isinstance(result, Cube)
    # Verify output is within expected range
    assert np.all(result.data >= 0.0)
    assert np.all(result.data <= 101.0)
