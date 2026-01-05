# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import numpy as np
from iris.cube import Cube

from improver.fire_weather import FireWeatherIndexBase


class FineFuelMoistureContent(FireWeatherIndexBase):
    """
    Plugin to calculate the Fine Fuel Moisture Code (FFMC) following
    the Canadian Forest Fire Weather Index System.

    The FFMC is a numerical rating of the moisture content of litter and other
    fine fuels, representing the relative ease of ignition and flammability of fine fuel.
    Values range from 0-101, with higher values indicating drier conditions.

    This process is adapted directly from:
        Equations and FORTRAN Program for the
        Canadian Forest Fire Weather Index System
        (C.E. Van Wagner and T.L. Pickett, 1985).
        Page 5, Equations 1-10.

    Expected input units:
        - Temperature: degrees Celsius
        - Precipitation: mm (24-hour accumulation)
        - Relative humidity: percentage (0-100)
        - Wind speed: km/h
        - Previous FFMC: dimensionless (0-101)
    """

    INPUT_CUBE_NAMES = [
        "air_temperature",
        "lwe_thickness_of_precipitation_amount",
        "relative_humidity",
        "wind_speed",
        "fine_fuel_moisture_content",
    ]
    OUTPUT_CUBE_NAME = "fine_fuel_moisture_content"
    # Disambiguate input FFMC (yesterday's value) from output FFMC (today's calculated value)
    INPUT_ATTRIBUTE_MAPPINGS = {"fine_fuel_moisture_content": "input_ffmc"}

    temperature: Cube
    precipitation: Cube
    relative_humidity: Cube
    wind_speed: Cube
    input_ffmc: Cube
    initial_moisture_content: np.ndarray
    moisture_content: np.ndarray

    def _calculate(self) -> np.ndarray:
        """Calculate the Fine Fuel Moisture Code (FFMC).

        Returns:
            The calculated FFMC values for the current day.
        """
        # Step 1 & 2: Calculate the previous day's moisture content
        self._calculate_moisture_content()

        # Step 3: Perform rainfall adjustment
        self._perform_rainfall_adjustment()

        # Step 4: Calculate Equilibrium Moisture Content for the drying phase
        E_d = self._calculate_EMC_for_drying_phase()

        # VECTORIZATION NOTE: The original algorithm in Van Wagner and Pickett (1985)
        # is applied to each point linearly. We are applying this to Cubes. This means
        # that the order of Steps 5-7 is not identical to the original algorithm.

        # Step 5a & 5b: Calculate moisture content through drying rate
        # VECTORIZATION NOTE: We calculate drying for all points, then apply selectively.
        # This differs from the scalar algorithm which only calculates when moisture content > E_d.
        moisture_content_from_drying = (
            self._calculate_moisture_content_through_drying_rate(E_d)
        )
        mask_wetting = self.moisture_content > E_d

        # Step 6: Calculate Equilibrium Moisture Content for the wetting phase
        # VECTORIZATION NOTE: We calculate E_w for all points for efficiency,
        # though it's only needed where the moisture content < E_d
        E_w = self._calculate_EMC_for_wetting_phase()

        # Step 7a & 7b: Calculate moisture content through wetting equilibrium
        # VECTORIZATION NOTE: We calculate wetting for all points, then apply selectively.
        # This differs from the scalar algorithm which only calculates when moisture content < E_d
        # AND moisture content < E_w.
        moisture_content_from_wetting = (
            self._calculate_moisture_content_through_wetting_equilibrium(E_w)
        )
        mask_drying = (
            self.moisture_content < E_d
        )  # Identifies where wetting might apply
        mask_apply_wetting = np.logical_and(mask_drying, self.moisture_content < E_w)

        # Step 5b, 7b, 8: Apply the appropriate transformation using mutually exclusive logic.
        # VECTORIZATION NOTE: We use nested np.where to implement the three-way
        # if-elseif-else structure from the scalar algorithm, ensuring only one
        # transformation is applied per grid point.
        self.moisture_content = np.where(
            mask_wetting,
            # If moisture content > E_d: apply drying (Step 5b, Equation 8)
            moisture_content_from_drying,
            np.where(
                mask_apply_wetting,
                # Else if (moisture_content < E_d) AND (moisture_content < E_w): apply wetting (Step 7b, Equation 9)
                moisture_content_from_wetting,
                # Else: no change, (Step 8: E_d >= moisture_content >= E_w)
                self.moisture_content,
            ),
        )

        # Step 9: Calculate Fine Fuel Moisture Content (FFMC) from moisture content
        ffmc = self._calculate_ffmc_from_moisture_content()

        return ffmc

    def _calculate_moisture_content(self):
        """Calculates the previous day's moisture content for a given input value
        of the Fine Fuel Moisture Content, and initialises the moisture_content
        attribute to that value.

        From Van Wagner and Pickett (1985), Page 5: Equation 1, Steps 1 & 2.
        """
        # Steps 1 & 2: Calculate the previous day's moisture content
        self.initial_moisture_content = (
            147.2 * (101.0 - self.input_ffmc.data) / (59.5 + self.input_ffmc.data)
        )
        # Initialise today's moisture content to the previous day's value
        self.moisture_content = self.initial_moisture_content.copy()

    def _perform_rainfall_adjustment(self):
        """Updates the moisture content value based on available precipitation
        accumulation data for the previous 24 hours. This is done element-wise
        for each grid point.

        Modifies self.moisture_content in place. Where precipitation > 0.5 mm,
        increases moisture content based on rainfall amount and current moisture
        level. Caps moisture content at 250 (dimensionless).

        From Van Wagner and Pickett (1985), Page 5: Equations 2, 3a, 3b,
        and Steps 3a, 3b, 3c.
        """
        # Step 3a: Check where precipitation > 0.5
        precip_mask = self.precipitation.data > 0.5
        # Equation 2: Set the rainfall value, adjusted for the threshold but
        # bounded to >= 0.0 to avoid negative values where the measurement is close
        r_f = self.precipitation.data.copy() - 0.5
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

        # Set the mask conditions for moisture content
        # Note these modify in-place so must be found before application
        mask_lte_150 = np.logical_and(precip_mask, self.moisture_content <= 150.0)
        mask_gt_150 = np.logical_and(precip_mask, self.moisture_content > 150.0)

        # Step 3bi: Where moisture_content <= 150, use adjustment1
        self.moisture_content = np.where(
            mask_lte_150,
            self.moisture_content + adjustment1,
            self.moisture_content,
        )

        # Step 3bii: Where moisture_content > 150, use adjustment1 + adjustment2
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

    def _calculate_EMC_for_drying_phase(self) -> np.ndarray:
        """Calculates the Equilibrium Moisture Content (EMC) for the drying phase
        under current environmental conditions (relative humidity, and temperature)

        From Van Wagner and Pickett (1985), Page 5: Equation 4, and Step 4.

        Returns:
            The Equilibrium Moisture Content for the drying phase (E_d).Array
            shape matches the input cube data shape. Values are in moisture
            content units (dimensionless).
        """
        # Equation 4: Calculate EMC for drying phase (E_d)
        E_d = (
            0.942 * self.relative_humidity.data**0.679
            + 11 * np.exp((self.relative_humidity.data - 100) / 10)
            + 0.18
            * (21.1 - self.temperature.data)
            * (1 - np.exp(-0.115 * self.relative_humidity.data))
        )
        return E_d

    def _calculate_moisture_content_through_drying_rate(
        self,
        E_d: np.ndarray,
    ) -> np.ndarray:
        """Calculates the moisture content through the drying rate.

        From Van Wagner and Pickett (1985), Page 5: Equations 6a, 6b, 8, and Step 5.

        Args:
            E_d:
                The Equilibrium Moisture Content for the drying phase.

        Returns:
            Array of moisture content (dimensionless) with drying applied at all
            grid points. Shape matches input cube data shape.
        """
        # Equation 6a: Calculate the log drying rate intermediate step
        k_o = 0.424 * (
            1 - (self.relative_humidity.data / 100.0) ** 1.7
        ) + 0.0694 * np.sqrt(self.wind_speed.data) * (
            1 - (self.relative_humidity.data / 100.0) ** 8
        )

        # Equation 6b: Calculate the log drying rate
        k_d = k_o * 0.581 * np.exp(0.0365 * self.temperature.data)

        # Equation 8: Calculate the new moisture content via drying
        new_moisture_content = E_d + (self.moisture_content - E_d) * 10 ** (-k_d)

        return new_moisture_content

    def _calculate_EMC_for_wetting_phase(self) -> np.ndarray:
        """Calculates the Equilibrium Moisture Content (EMC) for the wetting phase
        under current environmental conditions (relative humidity, and temperature)

        From Van Wagner and Pickett (1985), Page 5: Equation 5, and Step 6.

        Returns:
            The Equilibrium Moisture Content for the wetting phase (E_w). Array
            shape matches the input cube data shape. Values are in moisture
            content units (dimensionless).
        """
        # Equation 5: Calculate the EMC for the wetting phase (E_w)
        E_w = (
            0.618 * self.relative_humidity.data**0.753
            + 10.0 * np.exp((self.relative_humidity.data - 100.0) / 10.0)
            + 0.18
            * (21.1 - self.temperature.data)
            * (1 - np.exp(-0.115 * self.relative_humidity.data))
        )
        return E_w

    def _calculate_moisture_content_through_wetting_equilibrium(
        self,
        E_w: np.ndarray,
    ) -> np.ndarray:
        """Calculates the moisture content through the wetting equilibrium.

        From Van Wagner and Pickett (1985), Page 5: Equations 7a, 7b, 9, and Step 7.

        Args:
            E_w:
                The Equilibrium Moisture Content for the wetting phase.

        Returns:
            Array of moisture content (dimensionless) with wetting applied at all
            grid points. Shape matches input cube data shape.
        """
        # Equation 7a: Calculate the log wetting rate intermediate step
        k_l = 0.424 * (
            1 - ((100.0 - self.relative_humidity.data) / 100.0) ** 1.7
        ) + 0.0694 * np.sqrt(self.wind_speed.data) * (
            1 - ((100.0 - self.relative_humidity.data) / 100.0) ** 8
        )

        # Equation 7b: Calculate the log wetting rate intermediate step
        k_w = k_l * 0.581 * np.exp(0.0365 * self.temperature.data)

        # Equation 9: Calculate the new moisture content via wetting
        new_moisture_content = E_w - (E_w - self.moisture_content) * 10 ** (-k_w)

        return new_moisture_content

    def _calculate_ffmc_from_moisture_content(self) -> np.ndarray:
        """Calculates the Fine Fuel Moisture Content (FFMC) from the moisture
        content.

        From Van Wagner and Pickett (1985), Page 5: Equation 10, and Step 9.

        Returns:
            The calculated FFMC values (dimensionless, range 0-101). Array shape
            matches input cube data shape.
        """
        # Equation 10: Calculate FFMC from moisture content
        ffmc = 59.5 * (250.0 - self.moisture_content) / (147.2 + self.moisture_content)
        return ffmc
