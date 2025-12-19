# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import warnings

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.fire_weather.drought_code import DroughtCode
from improver_tests.fire_weather import make_cube, make_input_cubes


def input_cubes(
    temp_val: float = 20.0,
    precip_val: float = 1.0,
    dc_val: float = 15.0,
    shape: tuple[int, int] = (5, 5),
    temp_units: str = "Celsius",
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
    return make_input_cubes(
        [
            ("air_temperature", temp_val, temp_units, False),
            ("lwe_thickness_of_precipitation_amount", precip_val, precip_units, True),
            ("drought_code", dc_val, dc_units, True),
        ],
        shape=shape,
    )


def test_input_attribute_mapping() -> None:
    """Test that INPUT_ATTRIBUTE_MAPPINGS correctly stored inputted `drought_code`
    cubes as `input_dc`, while not changing the cube's metadata.
    """
    cubes = input_cubes()
    plugin = DroughtCode()
    plugin.load_input_cubes(CubeList(cubes), month=7)

    # Check that the mapping was applied correctly
    assert hasattr(plugin, "input_dc")
    assert isinstance(plugin.input_dc, Cube)
    assert plugin.input_dc.long_name == "drought_code"
    assert np.allclose(plugin.input_dc.data, 15.0)


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

    Tests include no adjustment (precip <= 2.8) and various rainfall amounts with
    different previous DC values.

    Args:
        precip_val (float): Precipitation value for all grid points.
        prev_dc (float): Previous DC value for all grid points.
        expected_dc (float): Expected DC after adjustment.
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
    """Test rainfall adjustment with spatially varying input data.

    Verifies vectorized DC rainfall adjustment with varying values across the grid.
    """
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
        make_cube(np.full(shape, 20.0), "air_temperature", "Celsius"),
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
    """Test _calculate_potential_evapotranspiration with various temperature and month combinations.

    Verifies potential evapotranspiration calculation for DC.

    Args:
        temp_val (float): Temperature value for all grid points.
        month (int): Month of the year (1-12).
        expected_pe (float): Expected potential evapotranspiration value.
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
    """Test potential evapotranspiration with spatially varying temperature.

    Verifies vectorized potential evapotranspiration calculation with varying values across the grid.
    """
    temp_data = np.array([[-10.0, -5.0, 0.0], [5.0, 10.0, 15.0], [20.0, 25.0, 30.0]])

    cubes = [
        make_cube(temp_data, "air_temperature", "Celsius"),
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

    Verifies DC calculation from previous DC and potential evapotranspiration.

    Args:
        prev_dc (float): Previous DC value.
        potential_evapotranspiration (float): Potential evapotranspiration value.
        expected_dc (float): Expected DC output value.
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

    Verifies end-to-end DC calculation with various environmental conditions.

    Args:
        temp_val (float): Temperature value for all grid points.
        precip_val (float): Precipitation value for all grid points.
        dc_val (float): DC value for all grid points.
        month (int): Month of the year (1-12).
        expected_output (float): Expected DC output value for all grid points.
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


def test_process_spatially_varying() -> None:
    """Integration test with spatially varying input data.

    Verifies vectorized DC implementation with varying values across the grid.
    """
    temp_data = np.array([[5.0, 10.0, 15.0], [10.0, 15.0, 20.0], [15.0, 20.0, 25.0]])
    precip_data = np.array([[0.0, 3.0, 10.0], [0.0, 0.0, 15.0], [0.0, 0.0, 0.0]])
    dc_data = np.array(
        [[10.0, 50.0, 100.0], [20.0, 150.0, 200.0], [50.0, 100.0, 300.0]]
    )

    cubes = [
        make_cube(temp_data, "air_temperature", "Celsius"),
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


def test_dc_day_length_factors_table() -> None:
    """Test that DC_DAY_LENGTH_FACTORS match the expected values from lookup table."""
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


@pytest.mark.parametrize(
    "temp_val, precip_val, dc_val, expected_error",
    [
        # Temperature too high
        (150.0, 1.0, 15.0, "temperature contains values above valid maximum"),
        # Temperature too low
        (-150.0, 1.0, 15.0, "temperature contains values below valid minimum"),
        # Precipitation negative
        (20.0, -5.0, 15.0, "precipitation contains values below valid minimum"),
        # DC negative
        (20.0, 1.0, -5.0, "input_dc contains values below valid minimum"),
    ],
)
def test_invalid_input_ranges_raise_errors(
    temp_val: float,
    precip_val: float,
    dc_val: float,
    expected_error: str,
) -> None:
    """Test that invalid input values raise appropriate ValueError.

    Verifies that the base class validation catches physically meaningless
    or out-of-range input values and raises descriptive errors.

    Args:
        temp_val (float): Temperature value for all grid points.
        precip_val (float): Precipitation value for all grid points.
        dc_val (float): DC value for all grid points.
        expected_error (str): Expected error message substring.
    """
    cubes = input_cubes(temp_val, precip_val, dc_val)
    plugin = DroughtCode()

    with pytest.raises(ValueError, match=expected_error):
        plugin.load_input_cubes(CubeList(cubes), month=7)


TEMP_VAL, PRECIP_VAL, DC_VAL = 20.0, 1.0, 15.0


@pytest.mark.parametrize(
    "temp_val, precip_val, dc_val, expected_error",
    [
        (np.nan, PRECIP_VAL, DC_VAL, "temperature contains NaN"),
        (np.inf, PRECIP_VAL, DC_VAL, "temperature contains infinite"),
        (TEMP_VAL, np.nan, DC_VAL, "precipitation contains NaN"),
        (TEMP_VAL, np.inf, DC_VAL, "precipitation contains infinite"),
        (TEMP_VAL, PRECIP_VAL, np.nan, "input_dc contains NaN"),
        (TEMP_VAL, PRECIP_VAL, np.inf, "input_dc contains infinite"),
    ],
)
def test_nan_and_inf_values_raise_errors(
    temp_val: float,
    precip_val: float,
    dc_val: float,
    expected_error: str,
) -> None:
    """Test that NaN and infinite input values raise appropriate ValueError.
    Verifies that the base class validation catches NaN and infinite input values
    and raises descriptive errors.

    Args:
        temp_val (float): Temperature value for all grid points.
        precip_val (float): Precipitation value for all grid points.
        dc_val (float): DC value for all grid points.
        expected_error (str): Expected error message substring.
    """
    cubes = input_cubes(temp_val, precip_val, dc_val)
    plugin = DroughtCode()

    with pytest.raises(ValueError, match=expected_error):
        plugin.load_input_cubes(CubeList(cubes), month=7)


def test_output_validation_no_warning_for_valid_output() -> None:
    """Test that valid output values do not trigger warnings.

    Uses valid inputs to verify that as long as the output
    stays within the expected range (0-1000 for DC), no warning is issued.
    """
    # Use normal valid inputs
    cubes = input_cubes(temp_val=20.0, precip_val=0.0, dc_val=50.0)
    plugin = DroughtCode()

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        result = plugin.process(CubeList(cubes), month=7)

    assert isinstance(result, Cube)
    # Verify output is within expected range
    assert np.all(result.data >= 0.0)
    assert np.all(result.data <= 1000.0)
