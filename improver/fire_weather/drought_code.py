# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from datetime import datetime
from typing import cast

import numpy as np
from iris.cube import Cube, CubeList

from improver import BasePlugin


class DroughtCode(BasePlugin):
    """
    Plugin to calculate the Drought Code (DC) following
    the Canadian Forest Fire Weather Index System.

    The DC is a numerical rating of the average moisture content of deep,
    compact organic layers. It is a useful indicator of seasonal drought
    effects on forest fuels and the amount of smoldering in deep duff layers
    and large logs. Higher values indicate drier conditions.

    This process is adapted directly from:
        Equations and FORTRAN Program for the
        Canadian Forest Fire Weather Index System
        (C.E. Van Wagner and T.L. Pickett, 1985).
        Pages 6-7, Equations 18-23.

    Expected input units:
        - Temperature: degrees Celsius
        - Precipitation: mm (24-hour accumulation)
        - Previous DC: dimensionless
        - Month: integer (1-12) for day length factor lookup
    """

    temperature: Cube
    precipitation: Cube
    input_dc: Cube
    month: int
    previous_dc: np.ndarray

    # Day length factors for DC calculation (L_f values from Table 4)
    # Index 0 is unused, indices 1-12 correspond to months January-December
    DC_DAY_LENGTH_FACTORS = [
        0.0,  # Placeholder for index 0
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

    def load_input_cubes(self, cubes: tuple[Cube] | CubeList, month: int):
        """Loads the required input cubes for the DC calculation. These
        are stored internally as Cube objects.

        Args:
            cubes (tuple[Cube] | CubeList): Input cubes containing the necessary data.
            month (int): Month of the year (1-12) for day length factor lookup.

        Raises:
            ValueError: If the number of cubes does not match the expected
                number (3), or if month is out of range.
        """
        names_to_extract = [
            "air_temperature",
            "lwe_thickness_of_precipitation_amount",
            "drought_code",
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
            self.input_dc,
        ) = tuple(cast(Cube, CubeList(cubes).extract_cube(n)) for n in names_to_extract)

        # Ensure the cubes are set to the correct units
        self.temperature.convert_units("degC")
        self.precipitation.convert_units("mm")
        self.input_dc.convert_units("1")

    def _perform_rainfall_adjustment(self):
        """Updates the previous DC value based on available precipitation
        accumulation data for the previous 24 hours. This is done element-wise
        for each grid point.

        From Van Wagner and Pickett (1985), Pages 6-7: Equations 18-21,
        and Steps 2a-2d corresponding to rainfall adjustment.
        """
        # Only adjust if precipitation > 2.8 mm
        precip_mask = self.precipitation.data > 2.8

        # Step 2a: Calculate effective rainfall via Equation 18
        effective_rain = 0.83 * self.precipitation.data - 1.27

        # Step 2b: Calculate moisture equivalent of the previous DC via Equation 19
        # Protect against negative DC values (though unlikely)
        dc_clipped = np.maximum(self.previous_dc, 0.0)
        moisture_equivalent = 800.0 * np.exp(-dc_clipped / 400.0)

        # Step 2c: Calculate the moisture equivalent with rainfall via Equation 20
        # Protect against division by zero in the rain term
        rain_effect = moisture_equivalent + 3.937 * effective_rain
        rain_effect = np.clip(rain_effect, a_min=1e-10, a_max=None)

        # Step 2d: Calculate DC after rain via Equation 21
        dc_after_rain = 400.0 * np.log(800.0 / rain_effect)

        # Step 2d: Apply lower bound of 0
        dc_after_rain = np.maximum(dc_after_rain, 0.0)

        # Update previous_dc where precipitation > 2.8
        self.previous_dc = np.where(precip_mask, dc_after_rain, self.previous_dc)

    def _calculate_potential_evapotranspiration(self) -> np.ndarray:
        """Calculates the potential evapotranspiration adjusted for day length.
        This represents the moisture loss from deep layers due to evaporation
        and transpiration.

        From Van Wagner and Pickett (1985), Pages 6-7: Equation 22, Steps 3 & 4.

        Returns:
            np.ndarray: The potential evapotranspiration value.
        """
        # Apply temperature lower bound of -2.8°C
        temp_adjusted = np.maximum(self.temperature.data, -2.8)

        # Step 3: Get day length factor for current month
        day_length_factor = self.DC_DAY_LENGTH_FACTORS[self.month]

        # Step 4: Calculate potential evapotranspiration via Equation 22
        potential_evapotranspiration = 0.36 * (temp_adjusted + 2.8) + day_length_factor

        return potential_evapotranspiration

    def _calculate_dc(self, potential_evapotranspiration: np.ndarray) -> np.ndarray:
        """Calculates the Drought Code from previous DC and potential evapotranspiration.

        From Van Wagner and Pickett (1985), Page 7: Equation 23.

        Args:
            potential_evapotranspiration (np.ndarray): The potential evapotranspiration.

        Returns:
            np.ndarray: The calculated DC value.
        """
        # Equation 23: Calculate DC
        dc = self.previous_dc + 0.5 * potential_evapotranspiration

        # Apply lower bound of 0
        dc = np.maximum(dc, 0.0)

        return dc

    def _make_dc_cube(self, dc_data: np.ndarray) -> Cube:
        """Converts a DC data array into an iris.cube.Cube object
        with relevant metadata copied from the input DC cube, and updated
        time coordinates from the precipitation cube. Time bounds are
        removed from the output.

        Args:
            dc_data (np.ndarray): The DC data

        Returns:
            Cube: An iris.cube.Cube containing the DC data with updated
                metadata and coordinates.
        """
        dc_cube = self.input_dc.copy(data=dc_data.astype(np.float32))

        # Update forecast_reference_time from precipitation cube
        frt_coord = self.precipitation.coord("forecast_reference_time").copy()
        dc_cube.replace_coord(frt_coord)

        # Update time coordinate from precipitation cube (without bounds)
        time_coord = self.precipitation.coord("time").copy()
        time_coord.bounds = None
        dc_cube.replace_coord(time_coord)

        return dc_cube

    def process(
        self,
        cubes: tuple[Cube] | CubeList,
        month: int | None = None,
    ) -> Cube:
        """Calculate the Drought Code (DC).

        Args:
            cubes (Cube | CubeList): Input cubes containing:
                air_temperature: Temperature in degrees Celsius
                lwe_thickness_of_precipitation_amount: 24-hour precipitation in mm
                drought_code: Previous day's DC value
            month (int | None): Month of the year (1-12) for day length factor lookup.
                If None, defaults to the current month.

        Returns:
            Cube: The calculated DC values for the current day.
        """
        if month is None:
            month = datetime.now().month

        self.load_input_cubes(cubes, month)

        # Step 1: Set today's DC value to the previous day's DC value
        self.previous_dc = self.input_dc.data.copy()

        # Step 2: Perform rainfall adjustment, if precipitation > 2.8 mm
        self._perform_rainfall_adjustment()

        # Steps 3 & 4: Calculate potential evapotranspiration
        potential_evapotranspiration = self._calculate_potential_evapotranspiration()

        # Step 5: Calculate DC from adjusted previous DC and potential evapotranspiration
        output_dc = self._calculate_dc(potential_evapotranspiration)

        # Convert DC data to a cube and return
        dc_cube = self._make_dc_cube(output_dc)

        return dc_cube
