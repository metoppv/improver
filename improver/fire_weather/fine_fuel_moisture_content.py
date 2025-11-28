# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from typing import cast

import numpy as np
from iris.cube import Cube, CubeList

from improver import BasePlugin


class FineFuelMoistureContent(BasePlugin):
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

    temperature: Cube
    precipitation: Cube
    relative_humidity: Cube
    wind_speed: Cube
    input_ffmc: Cube
    initial_moisture_content: np.ndarray
    moisture_content: np.ndarray

    def load_input_cubes(self, cubes: tuple[Cube] | CubeList):
        """Loads the required input cubes for the FFMC calculation. These
        are stored internally as Cube objects.

        Args:
            cubes (tuple[Cube] | CubeList): Input cubes containing the necessary data.

        Raises:
            ValueError: If the number of cubes does not match the expected
                number (5).
        """
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
        ) = tuple(cast(Cube, CubeList(cubes).extract_cube(n)) for n in names_to_extract)

        # Ensure the cubes are set to the correct units
        self.temperature.convert_units("degC")
        self.precipitation.convert_units("mm")
        self.relative_humidity.convert_units("1")
        self.wind_speed.convert_units("km/h")
        self.input_ffmc.convert_units("1")

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
            np.ndarray: The drying phase value.
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
            E_d (np.ndarray): The Equilibrium Moisture Content for the drying phase.

        Returns:
            np.ndarray: Array of moisture content with drying applied at all grid points.
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
            np.ndarray: The Equilibrium Moisture Content for the wetting phase.
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
            E_w (np.ndarray): The Equilibrium Moisture Content for the wetting phase.

        Returns:
            np.ndarray: Array of moisture content with wetting applied at all grid points.
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
            np.ndarray: The calculated FFMC value.
        """
        # Equation 10: Calculate FFMC from moisture content
        ffmc = 59.5 * (250.0 - self.moisture_content) / (147.2 + self.moisture_content)
        return ffmc

    def _make_ffmc_cube(self, ffmc_data: np.ndarray) -> Cube:
        """Converts an FFMC data array into an iris.cube.Cube object
        with relevant metadata copied from the input FFMC cube, and updated
        time coordinates from the precipitation cube. Time bounds are
        removed from the output.

        Args:
            ffmc_data (np.ndarray): The FFMC data

        Returns:
            Cube: An iris.cube.Cube containing the FFMC data with updated
                metadata and coordinates.
        """
        ffmc_cube = self.input_ffmc.copy(data=ffmc_data.astype(np.float32))

        # Update forecast_reference_time from precipitation cube
        frt_coord = self.precipitation.coord("forecast_reference_time").copy()
        ffmc_cube.replace_coord(frt_coord)

        # Update time coordinate from precipitation cube (without bounds)
        time_coord = self.precipitation.coord("time").copy()
        time_coord.bounds = None
        ffmc_cube.replace_coord(time_coord)

        return ffmc_cube

    def process(
        self,
        cubes: tuple[Cube] | CubeList,
    ) -> Cube:
        """Calculate the Fine Fuel Moisture Code (FFMC).

        Args:
            cubes (Cube | CubeList): Input cubes containing:
                air_temperature: Temperature in degrees Celsius
                lwe_thickness_of_precipitation_amount: 24-hour precipitation in mm
                relative_humidity: Relative humidity as a percentage (0-100)
                wind_speed: Wind speed in km/h
                fine_fuel_moisture_content: Previous day's FFMC value

        Returns:
            Cube: The calculated FFMC values for the current day.
        """
        self.load_input_cubes(cubes)

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
        output_ffmc = self._calculate_ffmc_from_moisture_content()

        # Convert FFMC data to a cube and return
        ffmc_cube = self._make_ffmc_cube(output_ffmc)

        return ffmc_cube
