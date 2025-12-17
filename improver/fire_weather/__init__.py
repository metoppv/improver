# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Fire Weather Index System components."""

import warnings
from abc import abstractmethod
from typing import cast

import iris.exceptions
import numpy as np
from iris.cube import Cube, CubeList

from improver import BasePlugin


class FireWeatherIndexBase(BasePlugin):
    """
    Abstract base class for Fire Weather Index System calculations.

    This class provides common functionality for all fire weather index
    components, including:

    - Standardised cube loading and validation
    - Fixed unit conversions for all cube types (non-configurable)
    - Output cube creation
    - Process orchestration

    The Canadian Forest Fire Weather Index System requires specific units
    for all calculations. These are fixed and cannot be overridden:

    - Temperature: degrees Celsius (Celsius)
    - Precipitation: millimeters (mm)
    - Relative humidity: dimensionless fraction (1)
    - Wind speed: kilometers per hour (km/h)
    - All fire weather indices: dimensionless (1)

    Subclasses must define class attributes:

    - INPUT_CUBE_NAMES: List of standard names for required input cubes
    - OUTPUT_CUBE_NAME: Standard name for the output cube
    - REQUIRES_MONTH: Boolean indicating if month parameter is required

    Subclasses must implement:

    - _calculate(): Method that performs the actual calculation
    """

    # Fixed unit conversions for all cube types used in fire weather calculations
    # These units are required by the Canadian FWI System and cannot be changed
    _REQUIRED_UNITS: dict[str, str] = {
        "temperature": "Celsius",
        "precipitation": "mm",
        "relative_humidity": "1",
        "wind_speed": "km/h",
        # Fire weather indices are dimensionless
        "fine_fuel_moisture_content": "1",
        "duff_moisture_code": "1",
        "drought_code": "1",
        "initial_spread_index": "1",
        "build_up_index": "1",
        "canadian_forest_fire_weather_index": "1",
        "fire_severity_index": "1",
        # Disambiguated input indices (used by FFMC, DMC, DC, and ISI calculations)
        "input_ffmc": "1",
        "input_dmc": "1",
        "input_dc": "1",
    }

    # Class attributes to be overridden by subclasses
    INPUT_CUBE_NAMES: list[str] = []
    OUTPUT_CUBE_NAME: str = ""
    REQUIRES_MONTH: bool = False
    # Optional: mapping of input cube names to attribute names (for disambiguation)
    INPUT_ATTRIBUTE_MAPPINGS: dict[str, str] = {}

    # Valid ranges for input validation (attribute_name: (min, max))
    # None means no validation for that bound
    _VALID_RANGES: dict[str, tuple[float | None, float | None]] = {
        "temperature": (-100.0, 100.0),  # Reasonable temperature range in Celsius
        "precipitation": (0.0, None),  # Must be non-negative
        "relative_humidity": (0.0, 101.0),  # Percentage
        "wind_speed": (0.0, None),  # Must be non-negative
        "input_ffmc": (0.0, 101.0),  # Valid FFMC range
        "input_dmc": (0.0, None),  # DMC is non-negative
        "input_dc": (0.0, None),  # DC is non-negative
        "initial_spread_index": (0.0, 100.0),  # ISI valid range
        "build_up_index": (0.0, 500.0),  # BUI valid range
    }

    # Valid output ranges for warning checks (output_name: (min, max))
    # Minimum and maximum feasible values for each output index are drawn from
    # values reported in:
    # Wang, X., Oliver, J., Swystun, T., Hanes, C.C., Erni, S. and Flannigan,
    # M.D., 2023. Critical fire weather conditions during active fire spread
    # days in Canada. Science of the total environment, 869, p.161831.
    _OUTPUT_RANGES: dict[str, tuple[float, float | None]] = {
        "fine_fuel_moisture_content": (0.0, 101.0),
        "duff_moisture_code": (0.0, 400),
        "drought_code": (0.0, 1000),
        "initial_spread_index": (0.0, 100),
        "build_up_index": (0.0, 500),
        "canadian_forest_fire_weather_index": (0.0, 100),
        "fire_severity_index": (0.0, 100),
    }

    def load_input_cubes(self, cubes: tuple[Cube] | CubeList, month: int | None = None):
        """Loads the required input cubes for the calculation. These are stored
        internally as Cube objects.

        Args:
            cubes:
                Input cubes containing the necessary data.
            month:
                Month of the year (1-12), required only if REQUIRES_MONTH is True.
                Defaults to None.

        Raises:
            ValueError:
                If the number of cubes does not match the expected number, if
                month is required but not provided, or if month is out of range.
        """
        if len(cubes) != len(self.INPUT_CUBE_NAMES):
            raise ValueError(
                f"Expected {len(self.INPUT_CUBE_NAMES)} cubes, found {len(cubes)}"
            )

        if self.REQUIRES_MONTH:
            if month is None:
                raise ValueError(
                    f"{self.__class__.__name__} requires a month parameter"
                )
            if not (1 <= month <= 12):
                raise ValueError(f"Month must be between 1 and 12, got {month}")
            self.month = month

        # Load cubes by extracting them using their standard names
        loaded_cubes = tuple(
            cast(Cube, CubeList(cubes).extract_cube(n)) for n in self.INPUT_CUBE_NAMES
        )

        # Assign cubes to instance attributes and convert units in a single loop
        for cube, cube_name in zip(loaded_cubes, self.INPUT_CUBE_NAMES):
            attr_name = self._get_attribute_name(cube_name)
            setattr(self, attr_name, cube)

            # Convert to required units if defined
            if attr_name in self._REQUIRED_UNITS:
                cube.convert_units(self._REQUIRED_UNITS[attr_name])

            # Validate input ranges
            self._validate_input_range(cube, attr_name)

    def _get_attribute_name(self, standard_name: str) -> str:
        """Convert a cube standard name to an attribute name.

        Args:
            standard_name:
                The cube's standard name

        Returns:
            The attribute name to use for storing the cube

        Examples:
            "air_temperature" -> "temperature"
            "lwe_thickness_of_precipitation_amount" -> "precipitation"
            "fine_fuel_moisture_content" -> "input_ffmc" (if INPUT_ATTRIBUTE_MAPPINGS is set)
        """
        # Check class-specific mappings first (for disambiguation)
        if standard_name in self.INPUT_ATTRIBUTE_MAPPINGS:
            return self.INPUT_ATTRIBUTE_MAPPINGS[standard_name]

        # Strip common prefixes and suffixes to create cleaner attribute names
        name = standard_name.removeprefix("lwe_thickness_of_").removeprefix("air_")
        name = name.removesuffix("_amount")

        return name

    def _validate_input_range(self, cube: Cube, attr_name: str) -> None:
        """Validate that input data falls within expected physical ranges.

        Args:
            cube:
                The input cube to validate
            attr_name:
                The attribute name for the cube

        Raises:
            ValueError:
                If any values fall outside the valid range for this input type,
                or if data contains NaN or Inf values
        """
        if attr_name not in self._VALID_RANGES:
            return  # No validation defined for this input type

        min_val, max_val = self._VALID_RANGES[attr_name]
        data = cube.data

        # Check for NaN values
        if np.any(np.isnan(data)):
            raise ValueError(f"{attr_name} contains NaN (Not a Number) values")

        # Check for infinite values
        if np.any(np.isinf(data)):
            raise ValueError(f"{attr_name} contains infinite values")

        # Check minimum bound if defined
        if min_val is not None and np.any(data < min_val):
            actual_min = float(np.min(data))
            raise ValueError(
                f"{attr_name} contains values below valid minimum: "
                f"found {actual_min}, expected >= {min_val}"
            )

        # Check maximum bound if defined
        if max_val is not None and np.any(data > max_val):
            actual_max = float(np.max(data))
            raise ValueError(
                f"{attr_name} contains values above valid maximum: "
                f"found {actual_max}, expected <= {max_val}"
            )

    def _make_output_cube(
        self, data: np.ndarray, template_cube: Cube | None = None
    ) -> Cube:
        """Creates an output cube with specified data and metadata.

        For classes that use precipitation data (FFMC, DMC, DC), automatically
        updates the time coordinates from the precipitation cube to reflect the
        24-hour accumulation period.

        Args:
            data:
                The output data array
            template_cube:
                The cube to use as a template for metadata.
                If None, uses the first input cube. Defaults to None

        Returns:
            The output cube containing the output data with proper metadata
            and coordinates.
        """
        if template_cube is None:
            # Use first input cube as template
            first_attr = self._get_attribute_name(self.INPUT_CUBE_NAMES[0])
            template_cube = getattr(self, first_attr)

        output_cube = template_cube.copy(data=data.astype(np.float32))
        output_cube.rename(self.OUTPUT_CUBE_NAME)
        output_cube.units = "1"

        # If this class uses precipitation, update time coordinates from precipitation cube
        if hasattr(self, "precipitation"):
            # Update forecast_reference_time from precipitation cube
            frt_coord = self.precipitation.coord("forecast_reference_time").copy()
            try:
                output_cube.replace_coord(frt_coord)
            except iris.exceptions.CoordinateNotFoundError:
                output_cube.add_aux_coord(frt_coord)

            # Update time coordinate from precipitation cube (without bounds)
            time_coord = self.precipitation.coord("time").copy()
            time_coord.bounds = None
            try:
                output_cube.replace_coord(time_coord)
            except iris.exceptions.CoordinateNotFoundError:
                output_cube.add_aux_coord(time_coord)

        return output_cube

    @abstractmethod
    def _calculate(self) -> np.ndarray:
        """Perform the fire weather index calculation.

        This method must be implemented by subclasses to perform
        the specific calculation logic for that component.

        Raises:
            NotImplementedError:
                This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the _calculate method.")

    def process(self, cubes: tuple[Cube] | CubeList, month: int | None = None) -> Cube:
        """Calculate the fire weather index component.

        Args:
            cubes:
                Input cubes as specified by INPUT_CUBE_NAMES
            month:
                Month parameter (1-12), required only if REQUIRES_MONTH is True
                Defaults to None.

        Returns:
            The calculated output cube.

        Warns:
            UserWarning:
                If output values fall outside typical expected ranges
        """
        self.load_input_cubes(cubes, month)
        output_data = self._calculate()
        output_cube = self._make_output_cube(output_data)

        # Check if output values are within expected ranges
        self._validate_output_range(output_cube)

        return output_cube

    def _validate_output_range(self, output_cube: Cube) -> None:
        """Check if output values fall within expected ranges and issue warnings if not.

        Args:
            output_cube:
                The output cube to validate

        Warns:
            UserWarning:
                If output contains NaN, Inf, or values outside expected ranges
        """
        output_name = output_cube.name()

        if output_name not in self._OUTPUT_RANGES:
            return  # No validation defined for this output type

        min_val, max_val = self._OUTPUT_RANGES[output_name]
        data = output_cube.data

        # Check for NaN values
        if np.any(np.isnan(data)):
            warnings.warn(
                f"{output_name} contains NaN (Not a Number) values. "
                f"This indicates a calculation error or invalid input data.",
                UserWarning,
                stacklevel=3,
            )
            return

        # Check for infinite values
        if np.any(np.isinf(data)):
            warnings.warn(
                f"{output_name} contains infinite values. "
                f"This indicates a calculation error or invalid input data.",
                UserWarning,
                stacklevel=3,
            )
            return

        # Check for values outside expected range
        if np.any(data < min_val) or np.any(data > max_val):
            actual_min = float(np.min(data))
            actual_max = float(np.max(data))
            warnings.warn(
                f"{output_name} contains values outside feasible range "
                f"[{min_val}, {max_val}]: found range [{actual_min:.2f}, {actual_max:.2f}]. "
                f"This may indicate unusual conditions or invalid input data.",
                UserWarning,
                stacklevel=3,
            )
