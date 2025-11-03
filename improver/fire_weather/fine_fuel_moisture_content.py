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

        # Load the cubes into class attributes
        (
            self.temperature,
            self.precipitation,
            self.relative_humidity,
            self.wind_speed,
            self.input_ffmc,
        ) = tuple(CubeList(cubes).extract_cube(n) for n in names_to_extract)

        # Ensure the cubes are set to the correct units
        self.temperature.convert_units("degC")
        self.precipitation.convert_units("mm")
        self.relative_humidity.convert_units(1)
        self.wind_speed.convert_units("km/h")
        self.input_ffmc.convert_units(1)

    def _calculate_moisture_content(self):
        """Calculates the previous day's moisture content for a given input value
        of the Fine Fuel Moisture Content, and initialises the moisture_content
        attribute to that value.
        """
        self.initial_moisture_content = (
            147.2 * (101.0 - self.input_ffmc) / (59.5 + self.input_ffmc)
        )
        self.moisture_content = self.initial_moisture_content.copy()

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

    def _perform_rainfall_adjustment(self):
        """
        Updates the moisture content value based on available precipitation
        accumulation data for the previous 24 hours. This is done element-wise
        for each grid point, matching the logic in the linear version.
        """
        precip_mask = self.precipitation > 0.5
        r_f = self.precipitation - 0.5

        # Calculate both adjustment terms
        adjustment1 = (
            42.5
            * r_f
            * np.exp(-100.0 / (251.0 - self.moisture_content))
            * (1 - np.exp(-6.93 / r_f))
        )
        adjustment2 = 0.0015 * (self.moisture_content - 150.0) ** 2 * np.sqrt(r_f)

        # Where moisture_content <= 150, use adjustment1
        mask_lte_150 = precip_mask & self.moisture_content <= 150.0
        self.moisture_content = np.where(
            mask_lte_150,
            self.moisture_content + adjustment1[mask_lte_150],
            self.moisture_content,
        )
        # Where moisture_content > 150, use adjustment1 + adjustment2
        mask_gt_150 = precip_mask & self.moisture_content > 150.0
        self.moisture_content = np.where(
            mask_gt_150,
            self.moisture_content + adjustment1[mask_gt_150] + adjustment2[mask_gt_150],
            self.moisture_content,
        )
        # Where moisture_content > 250, cap at 250
        mask_gt_250 = precip_mask & self.moisture_content > 250.0
        self.moisture_content = np.where(mask_gt_250, 250.0, self.moisture_content)

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

    def _calculate_wetting_phase(self) -> Cube:
        """Calculates the wetting phase for the current environmental conditions
        (relative humidity, and temperature)

        Returns:
            Cube: The wetting phase value.
        """
        E_w = (
            0.618 * self.relative_humidity**0.753
            + 10.0 * np.exp((self.relative_humidity - 100.0) / 10.0)
            + 0.18
            * (21.1 - self.temperature)
            * (1 - np.exp(-0.115 * self.relative_humidity))
        )
        return E_w

    def _calculate_moisture_content_through_wetting_equilibrium(
        self,
        E_w: Cube,
    ):
        """Calculates the moisture content through the wetting equilibrium.

        Args:
            E_w (Cube): The current wetting phase value.
        """
        k_l = 0.424 * (
            1 - ((100.0 - self.relative_humidity) / 100.0) ** 1.7
        ) + 0.0694 * np.sqrt(self.wind_speed) * (
            1 - ((100.0 - self.relative_humidity) / 100.0) ** 8
        )
        k_w = k_l * 0.581 * np.exp(0.0365 * self.temperature)
        self.moisture_content = E_w - (E_w - self.moisture_content) * 10 ** (-k_w)

    def _calculate_ffmc_from_moisture_content(self, E_d, E_w) -> Cube:
        """Calculates the Fine Fuel Moisture Content (FFMC) from the moisture
        content.

        Args:
            E_d (Cube): The current drying phase values.
            E_w (Cube): The current wetting phase values.

        Returns:
            Cube: The calculated FFMC value.
        """
        # FFMC calculation
        ffmc = 59.5 * (250.0 - self.moisture_content) / (147.2 + self.moisture_content)
        return ffmc

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

        # Step 3: Drying and wetting phase
        E_d = self._calculate_drying_phase()
        E_w = self._calculate_wetting_phase()

        # Produce masks for different phases
        mask_drying = self.moisture_content < E_d
        mask_wetting = self.moisture_content > E_w
        # ! Commented out for ruff while still being considered
        # mask_considerable = (self.moisture_content <= E_d) & (
        #    self.moisture_content >= E_w
        # )

        moisture_content_via_drying = (
            self._calculate_moisture_content_through_drying_rate(E_d)
        )
        moisture_content_via_wetting = (
            self._calculate_moisture_content_through_wetting_equilibrium(E_w)
        )
        # Combine all cases: considerable wetting/drying keeps value
        self.moisture_content = np.where(
            mask_wetting,
            moisture_content_via_wetting,
            np.where(
                mask_drying,
                moisture_content_via_drying,
                self.moisture_content,  # considerable wetting/drying: keep value
            ),
        )

        # Step 4: Convert moisture content back to FFMC
        output_ffmc = self._calculate_ffmc_from_moisture_content(E_d, E_w)
        return output_ffmc
