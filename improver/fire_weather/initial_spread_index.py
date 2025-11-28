# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from typing import cast

import numpy as np
from iris.cube import Cube, CubeList

from improver import BasePlugin


class InitialSpreadIndex(BasePlugin):
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

    wind_speed: Cube
    input_ffmc: Cube
    fine_fuel_moisture: np.ndarray
    wind_function: np.ndarray

    def load_input_cubes(self, cubes: tuple[Cube] | CubeList):
        """Loads the required input cubes for the ISI calculation. These
        are stored internally as Cube objects.

        Args:
            cubes (tuple[Cube] | CubeList): Input cubes containing the necessary data.

        Raises:
            ValueError: If the number of cubes does not match the expected
                number (2).
        """
        names_to_extract = [
            "wind_speed",
            "fine_fuel_moisture_content",
        ]
        if len(cubes) != len(names_to_extract):
            raise ValueError(
                f"Expected {len(names_to_extract)} cubes, found {len(cubes)}"
            )

        # Load the cubes into class attributes
        (
            self.wind_speed,
            self.input_ffmc,
        ) = tuple(cast(Cube, CubeList(cubes).extract_cube(n)) for n in names_to_extract)

        # Ensure the cubes are set to the correct units
        self.wind_speed.convert_units("km/h")
        self.input_ffmc.convert_units("1")

    def _calculate_fine_fuel_moisture(self):
        """Calculates the moisture content from the FFMC value.

        From Van Wagner and Pickett (1985), Page 5: Equation 1.
        """
        # Equation 24: Calculate fine fuel moisture content from FFMC
        self.moisture_content = (
            147.2 * (101.0 - self.input_ffmc.data) / (59.5 + self.input_ffmc.data)
        )

    def _calculate_wind_function(self):
        """Calculates the wind function component of ISI.

        From Van Wagner and Pickett (1985), Page 7: Equation 24.
        """
        # Equation 26: Calculate wind function
        wind_function = np.exp(0.05039 * self.wind_speed.data)
        return wind_function

    def _calculate_spread_factor(self):
        """Calculates the spread factor component for ISI.

        From Van Wagner and Pickett (1985), Page 7: Equation 25.
        Note: The Fortran implementation pre-multiplies the 0.208 coefficient
        from equation 26 into this calculation (91.9 * 0.208 = 19.115).

        """
        # Equation 25: Calculate the spread factor (SF)
        # Using 19.115 instead of 91.9 to match Fortran implementation
        # which pre-incorporates the 0.208 coefficient from eq 26
        spread_factor = (
            19.115
            * np.exp(self.moisture_content * -0.1386)
            * (1.0 + (self.moisture_content**5.31) / 4.93e7)
        )
        return spread_factor

    def _calculate_isi(
        self, spread_factor: np.ndarray, wind_function: np.ndarray
    ) -> np.ndarray:
        """Calculates the Initial Spread Index (ISI).

        From Van Wagner and Pickett (1985), Page 7: Equation 26.
        Note: The 0.208 coefficient has been pre-incorporated into the
        spread_factor calculation (19.115 instead of 91.9) to match
        the Fortran reference implementation.

        Args:
            spread_factor (np.ndarray): The spread factor values.
            wind_function (np.ndarray): The wind function values.

        Returns:
            np.ndarray: The calculated ISI values.
        """
        # Equation 26: Calculate ISI (0.208 pre-incorporated in spread_factor)
        initial_spread_index = spread_factor * wind_function
        return initial_spread_index

    def _make_isi_cube(self, isi_data: np.ndarray) -> Cube:
        """Converts an ISI data array into an iris.cube.Cube object
        with relevant metadata copied from the input FFMC cube.

        Args:
            isi_data (np.ndarray): The ISI data

        Returns:
            Cube: An iris.cube.Cube containing the ISI data with updated
                metadata and coordinates.
        """
        isi_cube = self.input_ffmc.copy(data=isi_data.astype(np.float32))

        # Update the cube name and metadata
        isi_cube.rename("initial_spread_index")
        isi_cube.units = "1"

        return isi_cube

    def process(
        self,
        cubes: tuple[Cube] | CubeList,
    ) -> Cube:
        """Calculate the Initial Spread Index (ISI).

        This uses Steps 1 & 2 from Van Wagner and Pickett (1985), page 8.

        Args:
            cubes (Cube | CubeList): Input cubes containing:
                wind_speed: Wind speed in km/h
                fine_fuel_moisture_content: FFMC value (0-101)

        Returns:
            Cube: The calculated ISI values.
        """
        self.load_input_cubes(cubes)

        # Calculate fine fuel moisture content from FFMC
        self._calculate_fine_fuel_moisture()

        # Step 1: Calculate wind function and spread factor
        wind_function = self._calculate_wind_function()
        spread_factor = self._calculate_spread_factor()

        # Step 3: Calculate ISI
        output_isi = self._calculate_isi(spread_factor, wind_function)

        # Convert ISI data to a cube and return
        isi_cube = self._make_isi_cube(output_isi)

        return isi_cube
