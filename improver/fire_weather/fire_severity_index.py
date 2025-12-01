# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugin to calculate the Fire Severity Index (Daily Severity Rating)."""

from typing import cast

import numpy as np
from iris.cube import Cube, CubeList

from improver import BasePlugin


class FireSeverityIndex(BasePlugin):
    """
    Plugin to calculate the Fire Severity Index, also known as the
    Daily Severity Rating (DSR).

    The DSR provides a numerical rating of the difficulty of controlling fires.
    It is derived from the Fire Weather Index (FWI) and represents the daily
    fire load and the expected effort required for fire suppression.

    This process is adapted directly from:
        Equations and FORTRAN Program for the
        Canadian Forest Fire Weather Index System
        (C.E. Van Wagner and T.L. Pickett, 1985).
        Page 8, Equation 31.

    Expected input units:
        - Fire Weather Index (FWI): dimensionless
    """

    fire_weather_index: Cube

    def load_input_cubes(self, cubes: tuple[Cube] | CubeList):
        """Loads the required input cube for the DSR calculation. This
        is stored internally as a Cube object.

        Args:
            cubes (tuple[Cube] | CubeList): Input cubes containing the necessary data.

        Raises:
            ValueError: If the number of cubes does not match the expected
                number (1).
        """
        names_to_extract = [
            "canadian_forest_fire_weather_index",
        ]
        if len(cubes) != len(names_to_extract):
            raise ValueError(
                f"Expected {len(names_to_extract)} cubes, found {len(cubes)}"
            )

        # Load the cube into class attribute
        (self.fire_weather_index,) = tuple(
            cast(Cube, CubeList(cubes).extract_cube(n)) for n in names_to_extract
        )

        # Ensure the cube is set to the correct units
        self.fire_weather_index.convert_units("1")

    def _calculate_dsr(self) -> np.ndarray:
        """Calculates the Daily Severity Rating (DSR) from FWI.

        From Van Wagner and Pickett (1985), Page 8: Equation 31.

        Equation 31:
            DSR = 0.0272 * FWI^1.77

        Returns:
            np.ndarray: The calculated DSR values.
        """
        fwi_data = self.fire_weather_index.data

        # Equation 31: DSR = 0.0272 * FWI^1.77
        dsr = 0.0272 * fwi_data**1.77

        return dsr

    def _make_dsr_cube(self, dsr_data: np.ndarray) -> Cube:
        """Converts a DSR data array into an iris.cube.Cube object
        with relevant metadata copied from the input FWI cube.

        Args:
            dsr_data (np.ndarray): The DSR data

        Returns:
            Cube: An iris.cube.Cube containing the DSR data with updated
                metadata and coordinates.
        """
        dsr_cube = self.fire_weather_index.copy(data=dsr_data.astype(np.float32))

        # Update the cube name and metadata
        dsr_cube.rename("fire_severity_index")
        dsr_cube.units = "1"

        return dsr_cube

    def process(
        self,
        cubes: tuple[Cube] | CubeList,
    ) -> Cube:
        """Calculate the Fire Severity Index (Daily Severity Rating).

        Args:
            cubes (Cube | CubeList): Input cube containing:
                canadian_forest_fire_weather_index: FWI value (dimensionless)

        Returns:
            Cube: The calculated DSR values.
        """
        self.load_input_cubes(cubes)

        # Calculate DSR
        output_dsr = self._calculate_dsr()

        # Convert DSR data to a cube and return
        dsr_cube = self._make_dsr_cube(output_dsr)

        return dsr_cube
