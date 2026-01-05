# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from improver.fire_weather.duff_moisture_code import DuffMoistureCode
from improver_tests.fire_weather import make_cube, make_input_cubes


def input_cubes(
    temp_val: float = 20.0,
    precip_val: float = 1.0,
    rh_val: float = 50.0,
    dmc_val: float = 6.0,
    shape: tuple[int, int] = (5, 5),
    temp_units: str = "Celsius",
    precip_units: str = "mm",
    rh_units: str = "1",
    dmc_units: str = "1",
) -> list[Cube]:
    """Create a list of dummy input cubes for DMC tests, with configurable units.

    All cubes have forecast_reference_time. Precipitation and DMC cubes also have
    time coordinates with bounds.

    Args:
        temp_val:
            Temperature value for all grid points.
        precip_val:
            Precipitation value for all grid points.
        rh_val:
            Relative humidity value for all grid points.
        dmc_val:
            DMC value for all grid points.
        shape:
            Shape of the grid for each cube.
        temp_units:
            Units for temperature cube.
        precip_units:
            Units for precipitation cube.
        rh_units:
            Units for relative humidity cube.
        dmc_units:
            Units for DMC cube.

    Returns:
        List of Iris Cubes for temperature, precipitation, relative humidity, and DMC.
    """
    return make_input_cubes(
        [
            ("air_temperature", temp_val, temp_units, False),
            ("lwe_thickness_of_precipitation_amount", precip_val, precip_units, True),
            ("relative_humidity", rh_val, rh_units, False),
            ("duff_moisture_code", dmc_val, dmc_units, True),
        ],
        shape=shape,
    )


def test_input_attribute_mapping() -> None:
    """Test that INPUT_ATTRIBUTE_MAPPINGS correctly disambiguates input DMC."""
    cubes = input_cubes()
    plugin = DuffMoistureCode()
    plugin.load_input_cubes(CubeList(cubes), month=7)

    # Check that the mapping was applied correctly
    assert hasattr(plugin, "input_dmc")
    assert isinstance(plugin.input_dmc, Cube)
    assert plugin.input_dmc.long_name == "duff_moisture_code"
    assert np.allclose(plugin.input_dmc.data, 6.0)


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

    Tests include no adjustment (precip <= 1.5) and various rainfall amounts with
    different previous DMC values.

    Args:
        precip_val:
            Precipitation value for all grid points.
        prev_dmc:
            Previous DMC value for all grid points.
        expected_dmc:
            Expected DMC after adjustment.
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
    """Test rainfall adjustment with spatially varying input data.

    Verifies vectorized DMC rainfall adjustment with varying values across the grid.
    """
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
        make_cube(np.full(shape, 20.0), "air_temperature", "Celsius"),
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
    """Test _calculate_drying_rate with various temperature, relative humidity,
    and month combinations.

    Verifies drying rate calculation for DMC.

    Args:
        temp_val:
            Temperature value for all grid points.
        rh_val:
            Relative humidity value for all grid points.
        month:
            Month of the year (1-12).
        expected_rate:
            Expected drying rate value.
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
    """Test drying rate with spatially varying temperature and relative humidity.

    Verifies vectorized drying rate calculation with varying values across the grid.
    """
    temp_data = np.array([[-5.0, 0.0, 10.0], [15.0, 20.0, 25.0], [30.0, 35.0, 40.0]])
    rh_data = np.array([[20.0, 30.0, 40.0], [50.0, 60.0, 70.0], [80.0, 90.0, 95.0]])

    cubes = [
        make_cube(temp_data, "air_temperature", "Celsius"),
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

    Verifies DMC calculation from previous DMC and drying rate.

    Args:
        prev_dmc:
            Previous DMC value.
        drying_rate:
            Drying rate value.
        expected_dmc:
            Expected DMC output value.
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

    Verifies end-to-end DMC calculation with various environmental conditions.

    Args:
        temp_val:
            Temperature value for all grid points.
        precip_val:
            Precipitation value for all grid points.
        rh_val:
            Relative humidity value for all grid points.
        dmc_val:
            DMC value for all grid points.
        month:
            Month of the year (1-12).
        expected_output:
            Expected DMC output value for all grid points.
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


def test_process_spatially_varying() -> None:
    """Integration test with spatially varying input data.

    Verifies vectorized DMC implementation with varying values across the grid.
    """
    temp_data = np.array([[10.0, 15.0, 20.0], [15.0, 20.0, 25.0], [20.0, 25.0, 30.0]])
    precip_data = np.array([[0.0, 2.0, 5.0], [0.0, 0.0, 10.0], [0.0, 0.0, 0.0]])
    rh_data = np.array([[40.0, 50.0, 60.0], [50.0, 60.0, 70.0], [60.0, 70.0, 80.0]])
    dmc_data = np.array([[5.0, 15.0, 30.0], [10.0, 50.0, 70.0], [20.0, 40.0, 90.0]])

    cubes = [
        make_cube(temp_data, "air_temperature", "Celsius"),
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


def test_dmc_day_length_factors_table() -> None:
    """Test that DMC_DAY_LENGTH_FACTORS match the expected values from lookup table."""
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


@pytest.mark.parametrize(
    "temp_val, precip_val, rh_val, dmc_val, expected_error",
    [
        # Temperature too high
        (150.0, 1.0, 50.0, 6.0, "temperature contains values above valid maximum"),
        # Temperature too low
        (-150.0, 1.0, 50.0, 6.0, "temperature contains values below valid minimum"),
        # Precipitation negative
        (20.0, -5.0, 50.0, 6.0, "precipitation contains values below valid minimum"),
        # Relative humidity above 100%
        (
            20.0,
            1.0,
            150.0,
            6.0,
            "relative_humidity contains values above valid maximum",
        ),
        # Relative humidity negative
        (
            20.0,
            1.0,
            -10.0,
            6.0,
            "relative_humidity contains values below valid minimum",
        ),
        # DMC negative
        (20.0, 1.0, 50.0, -5.0, "input_dmc contains values below valid minimum"),
    ],
)
def test_invalid_input_ranges_raise_errors(
    temp_val: float,
    precip_val: float,
    rh_val: float,
    dmc_val: float,
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
        dmc_val:
            DMC value for all grid points.
        expected_error:
            Expected error message substring.
    """
    cubes = input_cubes(temp_val, precip_val, rh_val, dmc_val)
    plugin = DuffMoistureCode()

    with pytest.raises(ValueError, match=expected_error):
        plugin.load_input_cubes(CubeList(cubes), month=7)


@pytest.mark.parametrize(
    "invalid_input_type,expected_error",
    [
        ("temperature_nan", "temperature contains NaN"),
        ("temperature_inf", "temperature contains infinite"),
        ("precipitation_nan", "precipitation contains NaN"),
        ("precipitation_inf", "precipitation contains infinite"),
        ("relative_humidity_nan", "relative_humidity contains NaN"),
        ("input_dmc_nan", "input_dmc contains NaN"),
        ("input_dmc_inf", "input_dmc contains infinite"),
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
    temp_val, precip_val, rh_val, dmc_val = 20.0, 1.0, 50.0, 6.0

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
    elif invalid_input_type == "input_dmc_nan":
        dmc_val = np.nan
    elif invalid_input_type == "input_dmc_inf":
        dmc_val = np.inf

    cubes = input_cubes(temp_val, precip_val, rh_val, dmc_val)
    plugin = DuffMoistureCode()

    with pytest.raises(ValueError, match=expected_error):
        plugin.load_input_cubes(CubeList(cubes), month=7)


def test_output_validation_no_warning_for_valid_output() -> None:
    """Test that valid output values do not trigger warnings.

    Uses valid inputs to verify that as long as the output
    stays within the expected range (0-400 for DMC), no warning is issued.
    """
    # Use normal valid inputs
    cubes = input_cubes(temp_val=20.0, precip_val=0.0, rh_val=50.0, dmc_val=50.0)
    plugin = DuffMoistureCode()

    # Process should complete without warnings since output stays in valid range
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        result = plugin.process(CubeList(cubes), month=7)

    assert isinstance(result, Cube)
    # Verify output is within expected range
    assert np.all(result.data >= 0.0)
    assert np.all(result.data <= 400.0)
