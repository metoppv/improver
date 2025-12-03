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

from improver.fire_weather import FireWeatherIndexBase


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


# Concrete implementation of FireWeatherIndexBase for testing
class ConcreteFireWeatherIndex(FireWeatherIndexBase):
    """Concrete implementation of FireWeatherIndexBase for testing purposes."""

    INPUT_CUBE_NAMES = ["air_temperature", "relative_humidity"]
    OUTPUT_CUBE_NAME = "test_index"
    REQUIRES_MONTH = False

    def _calculate(self) -> np.ndarray:
        """Simple test calculation: sum temperature and relative humidity."""
        return self.temperature.data + self.relative_humidity.data


class ConcreteFireWeatherIndexWithMonth(FireWeatherIndexBase):
    """Concrete implementation that requires a month parameter."""

    INPUT_CUBE_NAMES = ["air_temperature", "lwe_thickness_of_precipitation_amount"]
    OUTPUT_CUBE_NAME = "test_index_with_month"
    REQUIRES_MONTH = True

    def _calculate(self) -> np.ndarray:
        """Simple test calculation that uses month."""
        return self.temperature.data * self.month


class ConcreteFireWeatherIndexWithPrecipitation(FireWeatherIndexBase):
    """Concrete implementation with precipitation (for time coord testing)."""

    INPUT_CUBE_NAMES = [
        "air_temperature",
        "lwe_thickness_of_precipitation_amount",
        "relative_humidity",
    ]
    OUTPUT_CUBE_NAME = "test_index_with_precip"
    REQUIRES_MONTH = False

    def _calculate(self) -> np.ndarray:
        """Simple test calculation."""
        return (
            self.temperature.data
            + self.precipitation.data
            + self.relative_humidity.data
        )


class ConcreteFireWeatherIndexWithMappings(FireWeatherIndexBase):
    """Concrete implementation with INPUT_ATTRIBUTE_MAPPINGS for disambiguation."""

    INPUT_CUBE_NAMES = ["air_temperature", "test_index"]
    OUTPUT_CUBE_NAME = "test_index"
    REQUIRES_MONTH = False
    INPUT_ATTRIBUTE_MAPPINGS = {"test_index": "input_test_index"}

    def _calculate(self) -> np.ndarray:
        """Simple test calculation."""
        return self.temperature.data + self.input_test_index.data


def input_cubes_basic(
    temp_val: float = 20.0,
    rh_val: float = 50.0,
    shape: tuple[int, int] = (5, 5),
) -> list[Cube]:
    """Create basic input cubes for testing.

    Args:
        temp_val (float): Temperature value for all grid points.
        rh_val (float): Relative humidity value for all grid points.
        shape (tuple[int, int]): Shape of the grid for each cube.

    Returns:
        list[Cube]: List of Iris Cubes for temperature and relative humidity.
    """
    temp = make_cube(np.full(shape, temp_val), "air_temperature", "Celsius")
    rh = make_cube(np.full(shape, rh_val), "relative_humidity", "1")
    return [temp, rh]


def input_cubes_with_precip(
    temp_val: float = 20.0,
    precip_val: float = 1.0,
    rh_val: float = 50.0,
    shape: tuple[int, int] = (5, 5),
) -> list[Cube]:
    """Create input cubes including precipitation with time coordinates.

    Args:
        temp_val (float): Temperature value for all grid points.
        precip_val (float): Precipitation value for all grid points.
        rh_val (float): Relative humidity value for all grid points.
        shape (tuple[int, int]): Shape of the grid for each cube.

    Returns:
        list[Cube]: List of Iris Cubes for temperature, precipitation, and RH.
    """
    temp = make_cube(np.full(shape, temp_val), "air_temperature", "Celsius")
    precip = make_cube(
        np.full(shape, precip_val),
        "lwe_thickness_of_precipitation_amount",
        "mm",
        add_time_coord=True,
    )
    rh = make_cube(np.full(shape, rh_val), "relative_humidity", "1")
    return [temp, precip, rh]


@pytest.mark.parametrize(
    "temp_val, rh_val",
    [
        # Case 0: Typical mid-range values
        (20.0, 50.0),
        # Case 1: Zero values
        (0.0, 0.0),
        # Case 2: Maximum values
        (100.0, 100.0),
        # Case 3: Negative temperature
        (-10.0, 30.0),
        # Case 4: High values
        (40.0, 95.0),
    ],
)
def test_load_input_cubes_basic(temp_val: float, rh_val: float) -> None:
    """Test load_input_cubes with basic two-cube setup.

    Args:
        temp_val (float): Temperature value for all grid points.
        rh_val (float): Relative humidity value for all grid points.

    Raises:
        AssertionError: If the loaded cubes do not match expected properties.
    """
    cubes = input_cubes_basic(temp_val, rh_val)
    plugin = ConcreteFireWeatherIndex()
    plugin.load_input_cubes(CubeList(cubes))

    # Check attributes exist and have correct type
    assert isinstance(plugin.temperature, Cube)
    assert isinstance(plugin.relative_humidity, Cube)

    # Check shapes
    assert plugin.temperature.data.shape == (5, 5)
    assert plugin.relative_humidity.data.shape == (5, 5)

    # Check values
    assert np.allclose(plugin.temperature.data, temp_val)
    assert np.allclose(plugin.relative_humidity.data, rh_val)


@pytest.mark.parametrize(
    "param, input_val, input_unit, expected_val",
    [
        # Case 0: Temperature: Kelvin -> Celsius
        ("temperature", 293.15, "K", 20.0),
        # Case 1: Temperature: Fahrenheit -> Celsius
        ("temperature", 68.0, "fahrenheit", 20.0),
        # Case 2: Relative humidity: percent -> dimensionless
        ("relative_humidity", 50.0, "%", 0.5),
        # Case 3: Relative humidity: already dimensionless
        ("relative_humidity", 0.75, "1", 0.75),
    ],
)
def test_load_input_cubes_unit_conversion(
    param: str, input_val: float, input_unit: str, expected_val: float
) -> None:
    """Test that load_input_cubes correctly converts units.

    Args:
        param (str): Name of the parameter to test.
        input_val (float): Value to use for the tested parameter.
        input_unit (str): Unit to use for the tested parameter.
        expected_val (float): Expected value after conversion.

    Raises:
        AssertionError: If the converted value does not match expectations.
    """
    # Create cubes with custom units
    if param == "temperature":
        temp = make_cube(np.full((5, 5), input_val), "air_temperature", input_unit)
        rh = make_cube(np.full((5, 5), 50.0), "relative_humidity", "1")
        cubes = [temp, rh]
    else:  # relative_humidity
        temp = make_cube(np.full((5, 5), 20.0), "air_temperature", "Celsius")
        rh = make_cube(np.full((5, 5), input_val), "relative_humidity", input_unit)
        cubes = [temp, rh]

    plugin = ConcreteFireWeatherIndex()
    plugin.load_input_cubes(CubeList(cubes))

    # Check the converted value
    result = getattr(plugin, param)
    assert np.allclose(result.data, expected_val, atol=0.01)


@pytest.mark.parametrize(
    "num_cubes, should_raise, expected_message",
    [
        # Case 0: No cubes (0 instead of 2)
        (0, True, "Expected 2 cubes, found 0"),
        # Case 1: Too few cubes (1 instead of 2)
        (1, True, "Expected 2 cubes, found 1"),
        # Case 2: Correct number of cubes (2)
        (2, False, None),
        # Case 3: Too many cubes (3 instead of 2)
        (3, True, "Expected 2 cubes, found 3"),
    ],
)
def test_load_input_cubes_wrong_number_raises_error(
    num_cubes: int, should_raise: bool, expected_message: str
) -> None:
    """Test that load_input_cubes raises ValueError for wrong number of cubes.

    Args:
        num_cubes (int): Number of cubes to provide.
        should_raise (bool): Whether a ValueError should be raised.
        expected_message (str): Expected error message.

    Raises:
        AssertionError: If ValueError behavior does not match expectations.
    """
    cubes = input_cubes_basic()

    # Adjust cube list length
    if num_cubes < len(cubes):
        cubes = cubes[:num_cubes]
    elif num_cubes > len(cubes):
        # Add extra dummy cube
        extra = make_cube(np.full((5, 5), 10.0), "extra_cube", "1")
        cubes.append(extra)

    plugin = ConcreteFireWeatherIndex()

    if should_raise:
        with pytest.raises(ValueError, match=expected_message):
            plugin.load_input_cubes(CubeList(cubes))
    else:
        plugin.load_input_cubes(CubeList(cubes))
        assert isinstance(plugin.temperature, Cube)


def test_load_input_cubes_with_month_parameter() -> None:
    """Test load_input_cubes with REQUIRES_MONTH=True."""
    temp = make_cube(np.full((5, 5), 20.0), "air_temperature", "Celsius")
    precip = make_cube(
        np.full((5, 5), 1.0),
        "lwe_thickness_of_precipitation_amount",
        "mm",
        add_time_coord=True,
    )
    cubes = [temp, precip]

    plugin = ConcreteFireWeatherIndexWithMonth()
    plugin.load_input_cubes(CubeList(cubes), month=7)

    # Check month was set
    assert plugin.month == 7

    # Check cubes loaded correctly
    assert isinstance(plugin.temperature, Cube)
    assert isinstance(plugin.precipitation, Cube)


def test_load_input_cubes_missing_month_raises_error() -> None:
    """Test that missing month parameter raises ValueError when required."""
    temp = make_cube(np.full((5, 5), 20.0), "air_temperature", "Celsius")
    precip = make_cube(
        np.full((5, 5), 1.0), "lwe_thickness_of_precipitation_amount", "mm"
    )
    cubes = [temp, precip]

    plugin = ConcreteFireWeatherIndexWithMonth()

    with pytest.raises(
        ValueError, match="ConcreteFireWeatherIndexWithMonth requires a month parameter"
    ):
        plugin.load_input_cubes(CubeList(cubes))


@pytest.mark.parametrize(
    "month, should_raise, expected_message",
    [
        # Case 0: Valid month (January)
        (1, False, None),
        # Case 1: Valid month (July)
        (7, False, None),
        # Case 2: Valid month (December)
        (12, False, None),
        # Case 3: Month too low
        (0, True, "Month must be between 1 and 12, got 0"),
        # Case 4: Month too high
        (13, True, "Month must be between 1 and 12, got 13"),
        # Case 5: Negative month
        (-1, True, "Month must be between 1 and 12, got -1"),
    ],
)
def test_load_input_cubes_month_validation(
    month: int, should_raise: bool, expected_message: str
) -> None:
    """Test that month parameter is validated correctly.

    Args:
        month (int): Month value to test.
        should_raise (bool): Whether a ValueError should be raised.
        expected_message (str): Expected error message.

    Raises:
        AssertionError: If validation behavior does not match expectations.
    """
    temp = make_cube(np.full((5, 5), 20.0), "air_temperature", "Celsius")
    precip = make_cube(
        np.full((5, 5), 1.0), "lwe_thickness_of_precipitation_amount", "mm"
    )
    cubes = [temp, precip]

    plugin = ConcreteFireWeatherIndexWithMonth()

    if should_raise:
        with pytest.raises(ValueError, match=expected_message):
            plugin.load_input_cubes(CubeList(cubes), month=month)
    else:
        plugin.load_input_cubes(CubeList(cubes), month=month)
        assert plugin.month == month


@pytest.mark.parametrize(
    "standard_name, expected_attr_name",
    [
        # Case 0: Standard temperature name
        ("air_temperature", "temperature"),
        # Case 1: Precipitation with prefix and suffix
        ("lwe_thickness_of_precipitation_amount", "precipitation"),
        # Case 2: Relative humidity (no prefix/suffix)
        ("relative_humidity", "relative_humidity"),
        # Case 3: Wind speed (no prefix/suffix)
        ("wind_speed", "wind_speed"),
        # Case 4: Simple index name
        ("drought_code", "drought_code"),
    ],
)
def test_get_attribute_name_standard_conversion(
    standard_name: str, expected_attr_name: str
) -> None:
    """Test _get_attribute_name with standard name conversions.

    Args:
        standard_name (str): Standard name to convert.
        expected_attr_name (str): Expected attribute name.

    Raises:
        AssertionError: If attribute name does not match expectations.
    """
    plugin = ConcreteFireWeatherIndex()
    result = plugin._get_attribute_name(standard_name)
    assert result == expected_attr_name


def test_get_attribute_name_with_mappings() -> None:
    """Test _get_attribute_name with INPUT_ATTRIBUTE_MAPPINGS."""
    plugin = ConcreteFireWeatherIndexWithMappings()

    # Mapped name should use the mapping
    assert plugin._get_attribute_name("test_index") == "input_test_index"

    # Unmapped name should use standard conversion
    assert plugin._get_attribute_name("air_temperature") == "temperature"


def test_input_attribute_mappings_disambiguation() -> None:
    """Test INPUT_ATTRIBUTE_MAPPINGS allows input/output name disambiguation."""
    temp = make_cube(np.full((5, 5), 20.0), "air_temperature", "Celsius")
    input_index = make_cube(np.full((5, 5), 10.0), "test_index", "1")

    cubes = [temp, input_index]
    plugin = ConcreteFireWeatherIndexWithMappings()
    plugin.load_input_cubes(CubeList(cubes))

    # Check that mapping was applied
    assert hasattr(plugin, "input_test_index")
    assert isinstance(plugin.input_test_index, Cube)
    assert np.allclose(plugin.input_test_index.data, 10.0)

    # Check standard conversion still works
    assert hasattr(plugin, "temperature")
    assert np.allclose(plugin.temperature.data, 20.0)


@pytest.mark.parametrize(
    "output_value, shape",
    [
        # Case 0: Typical value and standard grid
        (85.0, (5, 5)),
        # Case 1: Zero value with different grid
        (0.0, (3, 4)),
        # Case 2: High value with larger grid
        (150.0, (10, 10)),
        # Case 3: Small value with small grid
        (5.0, (2, 2)),
        # Case 4: Decimal value
        (42.7, (5, 5)),
    ],
)
def test_make_output_cube_basic(output_value: float, shape: tuple[int, int]) -> None:
    """Test _make_output_cube creates cube with correct properties.

    Args:
        output_value (float): Value for output data.
        shape (tuple[int, int]): Shape of the grid.

    Raises:
        AssertionError: If output cube does not have expected properties.
    """
    cubes = input_cubes_basic(shape=shape)
    plugin = ConcreteFireWeatherIndex()
    plugin.load_input_cubes(CubeList(cubes))

    # Create output data
    output_data = np.full(shape, output_value, dtype=np.float64)

    # Call the method under test
    result_cube = plugin._make_output_cube(output_data)

    # Check cube properties
    assert isinstance(result_cube, Cube)
    assert result_cube.long_name == "test_index"
    assert result_cube.units == "1"
    assert result_cube.data.dtype == np.float32
    assert result_cube.data.shape == shape
    assert np.allclose(result_cube.data, output_value, atol=0.001)


def test_make_output_cube_with_template() -> None:
    """Test _make_output_cube with explicit template cube."""
    cubes = input_cubes_basic()
    plugin = ConcreteFireWeatherIndex()
    plugin.load_input_cubes(CubeList(cubes))

    # Create output data
    output_data = np.full((5, 5), 42.0, dtype=np.float64)

    # Use relative_humidity as template instead of temperature
    result_cube = plugin._make_output_cube(
        output_data, template_cube=plugin.relative_humidity
    )

    # Check cube was created correctly
    assert isinstance(result_cube, Cube)
    assert result_cube.long_name == "test_index"
    assert np.allclose(result_cube.data, 42.0)


def test_make_output_cube_preserves_forecast_reference_time() -> None:
    """Test that _make_output_cube preserves forecast_reference_time coordinate."""
    cubes = input_cubes_basic()
    plugin = ConcreteFireWeatherIndex()
    plugin.load_input_cubes(CubeList(cubes))

    output_data = np.full((5, 5), 50.0, dtype=np.float64)
    result_cube = plugin._make_output_cube(output_data)

    # Check forecast_reference_time was copied from template
    assert result_cube.coords("forecast_reference_time")
    # Check that forecast_reference_time has a valid timestamp
    assert result_cube.coord("forecast_reference_time").points[0] > 0


def test_make_output_cube_with_precipitation_updates_time_coords() -> None:
    """Test that _make_output_cube updates time coords from precipitation cube."""
    cubes = input_cubes_with_precip()
    plugin = ConcreteFireWeatherIndexWithPrecipitation()
    plugin.load_input_cubes(CubeList(cubes))

    output_data = np.full((5, 5), 100.0, dtype=np.float64)
    result_cube = plugin._make_output_cube(output_data)

    # Check that time coordinates were updated from precipitation
    time_coord = result_cube.coord("time")
    precip_time = plugin.precipitation.coord("time")

    # Time points should match
    assert np.allclose(time_coord.points, precip_time.points)

    # Time bounds should be removed in output
    assert time_coord.bounds is None

    # Check forecast_reference_time was updated
    frt_coord = result_cube.coord("forecast_reference_time")
    precip_frt = plugin.precipitation.coord("forecast_reference_time")
    assert np.allclose(frt_coord.points, precip_frt.points)


def test_make_output_cube_without_precipitation_no_time_update() -> None:
    """Test that _make_output_cube doesn't update time coords without precipitation."""
    cubes = input_cubes_basic()
    plugin = ConcreteFireWeatherIndex()
    plugin.load_input_cubes(CubeList(cubes))

    output_data = np.full((5, 5), 75.0, dtype=np.float64)
    result_cube = plugin._make_output_cube(output_data)

    # Should have forecast_reference_time from template
    assert result_cube.coords("forecast_reference_time")


def test_make_output_cube_adds_missing_forecast_reference_time() -> None:
    """Test that _make_output_cube adds forecast_reference_time when template lacks it."""
    cubes = input_cubes_with_precip()
    plugin = ConcreteFireWeatherIndexWithPrecipitation()
    plugin.load_input_cubes(CubeList(cubes))

    # Create a template cube without forecast_reference_time
    template_cube = Cube(
        np.full((5, 5), 20.0, dtype=np.float32),
        long_name="template_without_frt",
        units="1",
    )

    output_data = np.full((5, 5), 100.0, dtype=np.float64)
    result_cube = plugin._make_output_cube(output_data, template_cube=template_cube)

    # Check that forecast_reference_time was added from precipitation cube
    assert result_cube.coords("forecast_reference_time")
    frt_coord = result_cube.coord("forecast_reference_time")
    precip_frt = plugin.precipitation.coord("forecast_reference_time")
    assert np.allclose(frt_coord.points, precip_frt.points)


def test_process_complete_workflow() -> None:
    """Test the complete process workflow from cubes to output."""
    cubes = input_cubes_basic(temp_val=20.0, rh_val=50.0)
    plugin = ConcreteFireWeatherIndex()

    result = plugin.process(CubeList(cubes))

    # Check result is a cube
    assert isinstance(result, Cube)
    assert result.long_name == "test_index"
    assert result.units == "1"

    # Check calculation was performed correctly (temp + rh = 70)
    assert np.allclose(result.data, 70.0)


def test_process_with_month_parameter() -> None:
    """Test process method with month parameter."""
    temp = make_cube(np.full((5, 5), 10.0), "air_temperature", "Celsius")
    precip = make_cube(
        np.full((5, 5), 1.0),
        "lwe_thickness_of_precipitation_amount",
        "mm",
        add_time_coord=True,
    )
    cubes = [temp, precip]

    plugin = ConcreteFireWeatherIndexWithMonth()
    result = plugin.process(CubeList(cubes), month=3)

    # Check month was used in calculation (temp * month = 30)
    assert np.allclose(result.data, 30.0)


def test_process_with_precipitation_time_coords() -> None:
    """Test process method with precipitation updates time coordinates."""
    cubes = input_cubes_with_precip(temp_val=10.0, precip_val=5.0, rh_val=15.0)
    plugin = ConcreteFireWeatherIndexWithPrecipitation()

    result = plugin.process(CubeList(cubes))

    # Check calculation (10 + 5 + 15 = 30)
    assert np.allclose(result.data, 30.0)

    # Check time coordinates were updated from precipitation
    assert result.coords("time")
    time_coord = result.coord("time")
    # Bounds should be removed
    assert time_coord.bounds is None


def test_process_with_unit_conversion() -> None:
    """Test that process correctly handles unit conversion."""
    # Create cubes with non-standard units
    temp = make_cube(np.full((5, 5), 293.15), "air_temperature", "K")
    rh = make_cube(np.full((5, 5), 50.0), "relative_humidity", "%")

    cubes = [temp, rh]
    plugin = ConcreteFireWeatherIndex()

    result = plugin.process(CubeList(cubes))

    # Should convert Kelvin->Celsius (293.15->20) and %->1 (50->0.5)
    # Result should be 20 + 0.5 = 20.5
    assert np.allclose(result.data, 20.5, atol=0.1)


def test_input_attribute_mappings_in_process() -> None:
    """Test INPUT_ATTRIBUTE_MAPPINGS works in full process workflow."""
    temp = make_cube(np.full((5, 5), 15.0), "air_temperature", "Celsius")
    input_index = make_cube(np.full((5, 5), 25.0), "test_index", "1")

    cubes = [temp, input_index]
    plugin = ConcreteFireWeatherIndexWithMappings()

    result = plugin.process(CubeList(cubes))

    # Calculation should use mapped attribute (temp + input_test_index = 15 + 25 = 40)
    assert np.allclose(result.data, 40.0)
    assert result.long_name == "test_index"
