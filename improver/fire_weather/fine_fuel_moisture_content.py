# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import numpy as np
from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.utilities.common_input_handle import as_cubelist


class FineFuelMoistureContent(BasePlugin):
    """
    Plugin to calculate the Fine Fuel Moisture Code (FFMC) following
    the Canadian Forest Fire Weather Index System.
    """

    def __init__(self):
        self.temperature = None
        self.precipitation = None
        self.relative_humidity = None
        self.wind_speed = None
        self.input_ffmc = None
        self.moisture_content = None

    def load_input_cubes(self, cubes: Cube | CubeList):
        """Loads the required input cubes for the FFMC calculation.

        Args:
            cubes (Cube | CubeList): Input cubes containing the necessary data.

        Raises:
            ValueError: If the number of cubes does not match the expected
                number.
        """
        cubes = as_cubelist(*cubes)
        names_to_extract = [
            "temperature",
            "precipitation",
            "relative_humidity",
            "wind_speed",
            "input_ffmc",
        ]
        if len(cubes) != len(names_to_extract):
            raise ValueError(
                f"Expected {len(names_to_extract)} cubes, found {len(cubes)}"
            )

        (
            self.temperature,
            self.precipitation,
            self.relative_humidity,
            self.wind_speed,
            self.input_ffmc,
        ) = tuple(CubeList(cubes).extract_cube(n) for n in names_to_extract)

    def _calculate_moisture_content(self):
        """Calculates the moisture content for a given input value of the Fine
        Fuel Moisture Content"""
        self.moisture_content = (
            147.2 * (101.0 - self.input_ffmc) / (59.5 + self.input_ffmc)
        )

    def _perform_rainfall_adjustment(self):
        """Updates the moisture content value based on available precipitaion
        accumulation data for the previous 24 hours. This is done element-wise
        for each grid point.
        """
        precip_mask = self.precipitation.data > 0.5
        r_f = self.precipitation - 0.5
        adjustment = (
            42.5
            * r_f
            * np.exp(-100.0 / (251.0 - self.moisture_content))
            * (1 - np.exp(-6.93 / r_f))
        )
        # Only apply adjustment where precipitation > 0.5
        self.moisture_content = np.where(
            precip_mask, self.moisture_content, self.moisture_content + adjustment
        )
        # Cap at 250.0
        self.moisture_content = np.minimum(self.moisture_content, 250.0)

    def _calculate_drying_phase(self) -> Cube:
        """Calculates the drying phase for the current environmental conditions
        (relative humidity, and temperature)

        Returns:
            Cube: The drying phase value.
        """
        E_d = (
            0.942 * self.relative_humidity**0.679
            + 11 * np.exp((self.relative_humidity - 100) / 10)
            + 0.18
            * (21.1 - self.temperature)
            * (1 - np.exp(-0.115 * self.relative_humidity))
        )
        return E_d

    def _calculate_moisture_content_through_drying_rate(
        self,
        E_d: Cube,
    ):
        """Calculates the moisture content through the drying rate.

        Args:
            E_d (Cube): The current drying phase value.
        """
        # Drying rate
        k_o = 0.424 * (1 - (self.relative_humidity / 100.0) ** 1.7) + 0.0694 * np.sqrt(
            self.wind_speed
        ) * (1 - (self.relative_humidity / 100.0) ** 8)
        k_d = k_o * 0.581 * np.exp(0.0365 * self.temperature)
        self.moisture_content = E_d + (self.moisture_content - E_d) * 10 ** (-k_d)

    def _calculate_moisture_content_through_wetting_equilibrium(
        self,
    ):
        """Calculates the moisture content through the wetting equilibrium."""
        E_w = (
            0.618 * self.relative_humidity**0.753
            + 10.0 * np.exp((self.relative_humidity - 100.0) / 10.0)
            + 0.18
            * (21.1 - self.temperature)
            * (1 - np.exp(-0.115 * self.relative_humidity))
        )
        k_l = 0.424 * (
            1 - ((100.0 - self.relative_humidity) / 100.0) ** 1.7
        ) + 0.0694 * np.sqrt(self.wind_speed) * (
            1 - ((100.0 - self.relative_humidity) / 100.0) ** 8
        )
        k_w = k_l * 0.581 * np.exp(0.0365 * self.temperature)
        self.moisture_content = E_w - (E_w - self.moisture_content) * 10 ** (-k_w)

    def _calculate_ffmc_from_moisture_content(self) -> Cube:
        """Calculates the Fine Fuel Moisture Content (FFMC) from the moisture
        content.

        Returns:
            float: The calculated FFMC value.
        """
        return 59.5 * (250.0 - self.moisture_content) / (147.2 + self.moisture_content)

    def process(
        self,
        *cubes: Cube | CubeList,
    ) -> Cube:
        """
        Calculate the Fine Fuel Moisture Code (FFMC).

        Args:
            cubes (Cube | CubeList): Input cubes containing:
                - Temperature
                - Precipitation accumulation over the previous 24 hours.
                - Relative humidity
                - Wind speed
                - Previous day's FFMC value.

        Returns:
            float: Calculated FFMC value.
        """
        self.load_input_cubes(cubes)

        # Step 1: Convert previous FFMC to moisture content
        self._calculate_moisture_content()

        # Step 2: Rainfall adjustment
        self._perform_rainfall_adjustment()

        # Step 3: Drying phase
        E_d = self._calculate_drying_phase()

        moisture_content_mask = self.moisture_content < E_d

        moisture_content_via_drying = (
            self._calculate_moisture_content_through_drying_rate(E_d)
        )
        moisture_content_via_wetting = (
            self._calculate_moisture_content_through_wetting_equilibrium()
        )
        self.moisture_content = np.where(
            moisture_content_mask,
            moisture_content_via_drying,
            moisture_content_via_wetting,
        )

        # Step 4: Convert moisture content back to FFMC
        output_ffmc = self._calculate_ffmc_from_moisture_content()
        return output_ffmc
