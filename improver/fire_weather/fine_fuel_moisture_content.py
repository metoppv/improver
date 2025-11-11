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

    This process is adapted directly from Equations and FORTRAN Program  for the
    Canadian Forest Fire Weather Index System
    (C.E. Van Wagner and T.L. Pickett, 1985).
    Page 5, Equations 1-10.
    """

    def load_input_cubes(self, cubes: Cube | CubeList):
        """Loads the required input cubes for the FFMC calculation.

        Args:
            cubes (Cube | CubeList): Input cubes containing the necessary data.

        Raises:
            ValueError: If the number of cubes does not match the expected
                number.
        """
        # ! Can we allow anything other than a cube here?
        # ? Should we?
        cubes = as_cubelist(*cubes)
        names_to_extract = [
            "air_temperature",
            "lwe_thickness_of_precipitation_amount",
            "relative_humidity",
            "wind_speed",
            "fine_fuel_moisture_content",
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

        # Convert all inputted cubes to their data values for processing
        self.temperature = self.temperature.data
        self.precipitation = self.precipitation.data
        self.relative_humidity = self.relative_humidity.data
        self.wind_speed = self.wind_speed.data
        self.input_ffmc = self.input_ffmc.data

    def _calculate_moisture_content(self):
        """Calculates the previous day's moisture content for a given input value
        of the Fine Fuel Moisture Content, and initialises the moisture_content
        attribute to that value.

        From Van Wagner and Pickett (1985), Page 5: Equation 1, Steps 1 & 2.
        """
        # Steps 1 & 2: Calculate the previous day's moisture content
        self.initial_moisture_content = (
            147.2 * (101.0 - self.input_ffmc) / (59.5 + self.input_ffmc)
        )
        # Initialise today's moisture content to the previous day's value
        self.moisture_content = self.initial_moisture_content.copy()

    def _perform_rainfall_adjustment(self):
        """Updates the moisture content value based on available precipitaion
        accumulation data for the previous 24 hours. This is done element-wise
        for each grid point.

        From Van Wagner and Pickett (1985), Page 5: Equations 2, 3a, 3b, and Steps 3a, 3b, 3c.
        """
        # Step 3a: Check where precipitation > 0.5
        precip_mask = self.precipitation > 0.5
        # Set the rainfall value, adjusted for the threshold but bounded to >= 0.0
        r_f = self.precipitation.copy() - 0.5
        # Bound to zero to avoid negative values where the measurement is close
        r_f = np.maximum(r_f, 0.0)
        # Set values to np.nan where precipitation <= 0.5 to avoid unnecessary calculations
        r_f = np.where(precip_mask, r_f, np.nan)

        # Step 3b: Calculate the moisture content from the rainfall and initial moisture content
        # Equation 3a:
        adjustment1 = (
            42.5
            * r_f
            * np.exp(-100.0 / (251.0 - self.moisture_content))
            * (1 - np.exp(-6.93 / r_f))
        )
        # Equation 3b:
        adjustment2 = adjustment1 + 0.0015 * (
            self.moisture_content - 150.0
        ) ** 2 * np.sqrt(r_f)

        # Step 3bi: Where moisture_content <= 150, use adjustment1
        mask_lte_150 = np.logical_and(precip_mask, self.moisture_content <= 150.0)
        self.moisture_content = np.where(
            mask_lte_150,
            self.moisture_content + adjustment1,
            self.moisture_content,
        )

        # Step 3bii: Where moisture_content > 150, use adjustment1 + adjustment2
        mask_gt_150 = np.logical_and(precip_mask, self.moisture_content > 150.0)
        self.moisture_content = np.where(
            mask_gt_150,
            self.moisture_content + adjustment2,
            self.moisture_content,
        )

        # Where moisture_content > 250, cap at 250
        # Note: This step is not given directly in the process on page 5, but is listed at the
        # bottom of that page in a list of restrictions on the FFMC calculation.
        mask_gt_250 = np.logical_and(precip_mask, self.moisture_content > 250.0)
        self.moisture_content = np.where(mask_gt_250, 250.0, self.moisture_content)

    def _calculate_drying_phase(self) -> Cube:
        """Calculates the drying phase for the current environmental conditions
        (relative humidity, and temperature)

        From Van Wagner and Pickett (1985), Page 5: Equation 4, and Step 4.

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

        From Van Wagner and Pickett (1985), Page 5: Equations 6a, 6b, 8, and Steps 5a, 5b.

        Args:
            E_d (Cube): The current drying phase value.
        """
        # Equation 6a:
        k_o = 0.424 * (1 - (self.relative_humidity / 100.0) ** 1.7) + 0.0694 * np.sqrt(
            self.wind_speed
        ) * (1 - (self.relative_humidity / 100.0) ** 8)

        # Equation 6b:
        k_d = k_o * 0.581 * np.exp(0.0365 * self.temperature)

        # Equation 8:
        new_moisture_content = E_d + (self.moisture_content - E_d) * 10 ** (-k_d)

        # Steps 5a & 5b: Update moisture content where drying occurs
        self.moisture_content = np.where(
            self.moisture_content < E_d, new_moisture_content, self.moisture_content
        )

    def _calculate_wetting_phase(self) -> Cube:
        """Calculates the wetting phase for the current environmental conditions
        (relative humidity, and temperature)

        From Van Wagner and Pickett (1985), Page 5: Equation 5, and Step 6.

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

        From Van Wagner and Pickett (1985), Page 5: Equations 7a, 7b, 9, and Steps 7a, 7b.

        Args:
            E_w (Cube): The current wetting phase value.
        """
        # Equation 7a:
        k_l = 0.424 * (
            1 - ((100.0 - self.relative_humidity) / 100.0) ** 1.7
        ) + 0.0694 * np.sqrt(self.wind_speed) * (
            1 - ((100.0 - self.relative_humidity) / 100.0) ** 8
        )

        # Equation 7b:
        k_w = k_l * 0.581 * np.exp(0.0365 * self.temperature)

        # Equation 9:
        new_moisture_content = E_w - (E_w - self.moisture_content) * 10 ** (-k_w)

        # Steps 7a & 7b: Update moisture content where wetting occurs
        self.moisture_content = np.where(
            self.moisture_content > E_w, new_moisture_content, self.moisture_content
        )

    def _calculate_ffmc_from_moisture_content(self, E_d, E_w) -> Cube:
        """Calculates the Fine Fuel Moisture Content (FFMC) from the moisture
        content.

        From Van Wagner and Pickett (1985), Page 5: Equation 10, Steps 8 and 9.

        Args:
            E_d (Cube): The current drying phase values.
            E_w (Cube): The current wetting phase values.

        Returns:
            Cube: The calculated FFMC value.
        """
        # Step 8: Replace previous day's moisture content where the moisture content has changed by a significant amount
        condition_mask = np.logical_and(
            self.moisture_content <= E_d, self.moisture_content >= E_w
        )
        self.moisture_content = np.where(
            condition_mask,
            self.moisture_content,
            self.initial_moisture_content,
        )
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

        self._calculate_moisture_content()

        self._perform_rainfall_adjustment()

        E_d = self._calculate_drying_phase()
        E_w = self._calculate_wetting_phase()

        self._calculate_moisture_content_through_drying_rate(E_d)
        self._calculate_moisture_content_through_wetting_equilibrium(E_w)

        output_ffmc = self._calculate_ffmc_from_moisture_content(E_d, E_w)

        return output_ffmc
