# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import numpy as np
from iris.cube import Cube

from improver.fire_weather import FireWeatherIndexBase


class InitialSpreadIndex(FireWeatherIndexBase):
    """
    Plugin to calculate the Initial Spread Index (ISI) following
    the Canadian Forest Fire Weather Index System.

    The ISI is a numerical rating of the expected rate of fire spread.
    It combines the effects of wind and the Fine Fuel Moisture Code (FFMC)
    on the rate of spread without the influence of variable quantities of fuel.

    This process is adapted directly from:
        Equations and FORTRAN Program for the
        Canadian Forest Fire Weather Index System
        (C.E. Van Wagner and T.L. Pickett, 1985).
        Pages 7-8, Equations 24-26.

    Expected input units:
        - Wind speed: km/h
        - Fine Fuel Moisture Code (FFMC): dimensionless (0-101)
    """

    INPUT_CUBE_NAMES = ["wind_speed", "fine_fuel_moisture_content"]
    OUTPUT_CUBE_NAME = "initial_spread_index"
    # Valid output ranges for warning checks (output_name: (min, max))
    # Minimum and maximum feasible values for each output index are drawn from
    # values reported in:
    # Wang, X., Oliver, J., Swystun, T., Hanes, C.C., Erni, S. and Flannigan,
    # M.D., 2023. Critical fire weather conditions during active fire spread
    # days in Canada. Science of the total environment, 869, p.161831.
    VALID_OUTPUT_RANGE = (0.0, 100)
    # Disambiguate input FFMC from the output (ISI doesn't output FFMC, but uses naming consistency)
    INPUT_ATTRIBUTE_MAPPINGS = {"fine_fuel_moisture_content": "input_ffmc"}

    wind_speed: Cube
    input_ffmc: Cube
    moisture_content: np.ndarray

    def _calculate(self) -> np.ndarray:
        """Calculates the Initial Spread Index (ISI) from wind and FFMC.

        This uses Steps 1 & 2 from Van Wagner and Pickett (1985), page 8.

        Returns:
            The calculated ISI values.
        """
        # Calculate fine fuel moisture content from FFMC
        self._calculate_fine_fuel_moisture()

        # Step 1: Calculate wind function and spread factor
        wind_function = self._calculate_wind_function()
        spread_factor = self._calculate_spread_factor()

        # Step 2: Calculate ISI
        initial_spread_index = self._calculate_isi(spread_factor, wind_function)

        return initial_spread_index

    def _calculate_fine_fuel_moisture(self):
        """Calculates the moisture content from the FFMC value.

        From Van Wagner and Pickett (1985), Page 5: Equation 1.
        """
        self.moisture_content = (
            147.2 * (101.0 - self.input_ffmc.data) / (59.5 + self.input_ffmc.data)
        )

    def _calculate_wind_function(self) -> np.ndarray:
        """Calculates the wind function component of ISI.

        From Van Wagner and Pickett (1985), Page 7: Equation 24.

        Returns:
            The wind function values.
        """
        wind_function = np.exp(0.05039 * self.wind_speed.data)
        return wind_function

    def _calculate_spread_factor(self) -> np.ndarray:
        """Calculates the spread factor component for ISI.

        From Van Wagner and Pickett (1985), Page 7: Equation 25.

        Returns:
            The spread factor values.
        """
        # Equation 25: Calculate the spread factor (SF)
        spread_factor = (
            91.9
            * np.exp(self.moisture_content * -0.1386)
            * (1.0 + (self.moisture_content**5.31) / 4.93e7)
        )
        return spread_factor

    def _calculate_isi(
        self, spread_factor: np.ndarray, wind_function: np.ndarray
    ) -> np.ndarray:
        """Calculates the Initial Spread Index (ISI).

        From Van Wagner and Pickett (1985), Page 7: Equation 26.

        Args:
            spread_factor:
                The spread factor values.
            wind_function:
                The wind function values.

        Returns:
            The calculated ISI values.
        """
        # Equation 26: Calculate the Initial Spread Index (ISI)
        initial_spread_index = 0.208 * spread_factor * wind_function
        return initial_spread_index
