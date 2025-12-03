# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

import numpy as np
from iris.cube import Cube

from improver.fire_weather import FireWeatherIndexBase


class DroughtCode(FireWeatherIndexBase):
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

    INPUT_CUBE_NAMES = [
        "air_temperature",
        "lwe_thickness_of_precipitation_amount",
        "drought_code",
    ]
    OUTPUT_CUBE_NAME = "drought_code"
    REQUIRES_MONTH = True
    # Disambiguate input DC (yesterday's value) from output DC (today's calculated value)
    INPUT_ATTRIBUTE_MAPPINGS = {"drought_code": "input_dc"}

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

    def _calculate(self) -> np.ndarray:
        """Calculate the Drought Code (DC).

        Returns:
            np.ndarray: The calculated DC values for the current day.
        """
        # Step 1: Set today's DC value to the previous day's DC value
        self.previous_dc = self.input_dc.data.copy()

        # Step 2: Perform rainfall adjustment, if precipitation > 2.8 mm
        self._perform_rainfall_adjustment()

        # Steps 3 & 4: Calculate potential evapotranspiration
        potential_evapotranspiration = self._calculate_potential_evapotranspiration()

        # Step 5: Calculate DC from adjusted previous DC and potential evapotranspiration
        dc = self._calculate_dc(potential_evapotranspiration)

        return dc

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
