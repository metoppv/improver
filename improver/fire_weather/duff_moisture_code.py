# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from datetime import datetime
from typing import cast

import numpy as np
from iris.cube import Cube, CubeList

from improver import BasePlugin


class DuffMoistureCode(BasePlugin):
    """
    Plugin to calculate the Duff Moisture Code (DMC) following
    the Canadian Forest Fire Weather Index System.

    The DMC is a numerical rating of the average moisture content of loosely
    compacted organic layers of moderate depth. It indicates fuel consumption
    in moderate duff layers and medium-size woody material.
    Higher values indicate drier conditions.

    This process is adapted directly from:
        Equations and FORTRAN Program for the
        Canadian Forest Fire Weather Index System
        (C.E. Van Wagner and T.L. Pickett, 1985).
        Page 6, Equations 11-16.

    Expected input units:
        - Temperature: degrees Celsius
        - Precipitation: mm (24-hour accumulation)
        - Relative humidity: percentage (0-100)
        - Previous DMC: dimensionless
        - Month: integer (1-12) for day length factor lookup
    """

    temperature: Cube
    precipitation: Cube
    relative_humidity: Cube
    input_dmc: Cube
    month: int
    previous_dmc: np.ndarray

    # Day length factors for DMC calculation (L_e values from Table 3)
    # Index 0 is unused, indices 1-12 correspond to months January-December
    DMC_DAY_LENGTH_FACTORS = [
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

    def load_input_cubes(self, cubes: tuple[Cube] | CubeList, month: int):
        """Loads the required input cubes for the DMC calculation. These
        are stored internally as Cube objects.

        Args:
            cubes (tuple[Cube] | CubeList): Input cubes containing the necessary data.
            month (int): Month of the year (1-12) for day length factor lookup.

        Raises:
            ValueError: If the number of cubes does not match the expected
                number (4), or if month is out of range.
        """
        names_to_extract = [
            "air_temperature",
            "lwe_thickness_of_precipitation_amount",
            "relative_humidity",
            "duff_moisture_code",
        ]
        if len(cubes) != len(names_to_extract):
            raise ValueError(
                f"Expected {len(names_to_extract)} cubes, found {len(cubes)}"
            )

        if not (1 <= month <= 12):
            raise ValueError(f"Month must be between 1 and 12, got {month}")

        self.month = month

        # Load the cubes into class attributes
        (
            self.temperature,
            self.precipitation,
            self.relative_humidity,
            self.input_dmc,
        ) = tuple(cast(Cube, CubeList(cubes).extract_cube(n)) for n in names_to_extract)

        # Ensure the cubes are set to the correct units
        self.temperature.convert_units("degC")
        self.precipitation.convert_units("mm")
        self.relative_humidity.convert_units("1")
        self.input_dmc.convert_units("1")

    def _perform_rainfall_adjustment(self):
        """Updates the previous DMC value based on available precipitation
        accumulation data for the previous 24 hours. This is done element-wise
        for each grid point.

        From Van Wagner and Pickett (1985), Page 6: Equations 11-15,
        and Steps 2a-2e corresponding to rainfall adjustment.
        """
        # Only adjust if precipitation > 1.5 mm
        precip_mask = self.precipitation.data > 1.5

        # Step 2a: Calculate effective rainfall via Equation 11
        effective_rain = 0.92 * self.precipitation.data - 1.27

        # Step 2b: Calculate initial moisture content from previous DMC via Equation 12
        moisture_content_initial = 20.0 + np.exp(5.6348 - self.previous_dmc / 43.43)

        # Step 2c: Calculate the slope variable based on previous DMC via Equations 13a, 13b, 13c
        # In the original algorithm, the slope variable is referred to as 'b'
        # VECTORIZATION NOTE: This structure matches the original algorithm while being vectorized
        # Clip previous_dmc to avoid log(0) warnings in equations 13b and 13c
        dmc_clipped = np.maximum(self.previous_dmc, 1e-10)
        slope_variable = np.where(
            self.previous_dmc <= 33.0,
            # Equation 13a: DMC <= 33
            100.0 / (0.5 + 0.3 * self.previous_dmc),
            np.where(
                self.previous_dmc <= 65.0,
                # Equation 13b: 33 < DMC <= 65
                14.0 - 1.3 * np.log(dmc_clipped),
                # Equation 13c: DMC > 65
                6.2 * np.log(dmc_clipped) - 17.2,
            ),
        )

        # Step 2d: Calculate moisture content after rain via Equation 14
        # Protect against division by zero (though mathematically unlikely)
        denominator = 48.77 + slope_variable * effective_rain
        moisture_content_after_rain = moisture_content_initial + (
            1000.0 * effective_rain
        ) / np.maximum(denominator, 1e-10)

        # Step 2e: Calculate DMC after rain via Equation 15
        # This is modified to avoid log of zero or negative values
        log_arg = np.clip(moisture_content_after_rain - 20.0, 1e-10, None)
        dmc_after_rain = 244.72 - 43.43 * np.log(log_arg)

        # Apply lower bound of 0
        dmc_after_rain = np.maximum(dmc_after_rain, 0.0)

        # Update previous_dmc where precipitation > 1.5
        self.previous_dmc = np.where(precip_mask, dmc_after_rain, self.previous_dmc)

    def _calculate_drying_rate(self) -> np.ndarray:
        """Calculates the drying rate for DMC. This is multiplied by 100 for
        computational efficiency in the final DMC calculation. The original
        algorithm calculates K and then multilies it by 100 in the DMC equation.

        From Van Wagner and Pickett (1985), Page 6: Equation 16, Steps 3 & 4.

        Returns:
            np.ndarray: The drying rate value.
        """
        # Apply temperature lower bound of -1.1°C
        temp_adjusted = np.maximum(self.temperature.data, -1.1)

        # Step 3: Get day length factor for current month
        day_length_factor = self.DMC_DAY_LENGTH_FACTORS[self.month]

        # Step 4: Calculate drying rate via Equation 16
        drying_rate = (
            1.894
            * (temp_adjusted + 1.1)
            * (100.0 - self.relative_humidity.data)
            * day_length_factor
            * 1e-4
        )

        return drying_rate

    def _calculate_dmc(self, drying_rate: np.ndarray) -> np.ndarray:
        """Calculates the Duff Moisture Code from previous DMC and drying rate.
        Note that the drying rate is expected to be pre-multiplied by 100
        for computational efficiency. This mathematically matches the original
        algorithm, but is more efficient to implement this way.

        From Van Wagner and Pickett (1985), Page 6: Equation 16.

        Args:
            drying_rate (np.ndarray): The drying rate (RK).

        Returns:
            np.ndarray: The calculated DMC value.
        """
        # Equation 16: Calculate DMC
        dmc = self.previous_dmc + drying_rate

        # Apply lower bound of 0
        dmc = np.maximum(dmc, 0.0)

        return dmc

    def _make_dmc_cube(self, dmc_data: np.ndarray) -> Cube:
        """Converts a DMC data array into an iris.cube.Cube object
        with relevant metadata copied from the input DMC cube, and updated
        time coordinates from the precipitation cube. Time bounds are
        removed from the output.

        Args:
            dmc_data (np.ndarray): The DMC data

        Returns:
            Cube: An iris.cube.Cube containing the DMC data with updated
                metadata and coordinates.
        """
        dmc_cube = self.input_dmc.copy(data=dmc_data.astype(np.float32))

        # Update forecast_reference_time from precipitation cube
        frt_coord = self.precipitation.coord("forecast_reference_time").copy()
        dmc_cube.replace_coord(frt_coord)

        # Update time coordinate from precipitation cube (without bounds)
        time_coord = self.precipitation.coord("time").copy()
        time_coord.bounds = None
        dmc_cube.replace_coord(time_coord)

        return dmc_cube

    def process(
        self,
        cubes: tuple[Cube] | CubeList,
        month: int | None = None,
    ) -> Cube:
        """Calculate the Duff Moisture Code (DMC).

        Args:
            cubes (Cube | CubeList): Input cubes containing:
                air_temperature: Temperature in degrees Celsius
                lwe_thickness_of_precipitation_amount: 24-hour precipitation in mm
                relative_humidity: Relative humidity as a percentage (0-100)
                duff_moisture_code: Previous day's DMC value
            month (int | None): Month of the year (1-12) for day length factor lookup.
                If None, defaults to the current month.

        Returns:
            Cube: The calculated DMC values for the current day.
        """
        if month is None:
            month = datetime.now().month

        self.load_input_cubes(cubes, month)

        # Step 1: Set today's DMC value to the previous day's DMC value
        self.previous_dmc = self.input_dmc.data.copy()

        # Step 2: Perform rainfall adjustment, if precipitation > 1.5 mm
        self._perform_rainfall_adjustment()

        # Steps 3 & 4: Calculate drying rate
        drying_rate = self._calculate_drying_rate()

        # Step 5: Calculate DMC from adjusted previous DMC and drying rate
        output_dmc = self._calculate_dmc(drying_rate)

        # Convert DMC data to a cube and return
        dmc_cube = self._make_dmc_cube(output_dmc)

        return dmc_cube
