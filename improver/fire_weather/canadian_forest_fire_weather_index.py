# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugin to calculate the Canadian Forest Fire Weather Index (FWI)."""

from typing import cast

import numpy as np
from iris.cube import Cube, CubeList

from improver import BasePlugin


class CanadianForestFireWeatherIndex(BasePlugin):
    """
    Plugin to calculate the Canadian Forest Fire Weather Index (FWI).

    The FWI combines the Initial Spread Index (ISI) and the Build Up Index (BUI)
    to provide a numerical rating of fire intensity. It represents the rate of
    fire spread and the amount of available fuel.

    This process is adapted directly from:
        Equations and FORTRAN Program for the
        Canadian Forest Fire Weather Index System
        (C.E. Van Wagner and T.L. Pickett, 1985).
        Pages 7-8, Equations 28-30.

    Expected input units:
        - Initial Spread Index (ISI): dimensionless
        - Build Up Index (BUI): dimensionless
    """

    initial_spread_index: Cube
    build_up_index: Cube

    def load_input_cubes(self, cubes: tuple[Cube] | CubeList):
        """Loads the required input cubes for the FWI calculation. These
        are stored internally as Cube objects.

        Args:
            cubes (tuple[Cube] | CubeList): Input cubes containing the necessary data.

        Raises:
            ValueError: If the number of cubes does not match the expected
                number (2).
        """
        names_to_extract = [
            "initial_spread_index",
            "build_up_index",
        ]
        if len(cubes) != len(names_to_extract):
            raise ValueError(
                f"Expected {len(names_to_extract)} cubes, found {len(cubes)}"
            )

        # Load the cubes into class attributes
        (
            self.initial_spread_index,
            self.build_up_index,
        ) = tuple(cast(Cube, CubeList(cubes).extract_cube(n)) for n in names_to_extract)

        # Ensure the cubes are set to the correct units
        self.initial_spread_index.convert_units("1")
        self.build_up_index.convert_units("1")

    def _calculate_fwi(self, extrapolated_DMF) -> np.ndarray:
        """Calculates the Fire Weather Index (FWI) from the Initial Spread Index (ISI)
        and the extrapolated Duff Moisture Function (extrapolated_DMF).

        From Van Wagner and Pickett (1985), Page 7-8: Equations 29-30.

        Returns:
            np.ndarray: The calculated FWI values.
        """
        isi_data = self.initial_spread_index.data

        # Equation 29: Calculate "B-Scale" FWI (intermmediate value)
        fwi_Bscale = 0.1 * isi_data * extrapolated_DMF

        # Equations 30a and 30b: Calculate the final "S-Scale"
        # When B > 1, use equation 30a; otherwise use equation 30b
        with np.errstate(divide="ignore", invalid="ignore"):
            fwi_Sscale = 2.72 * (0.434 * np.log(fwi_Bscale)) ** 0.647
            fwi_30a = np.exp(fwi_Sscale)

        fwi = np.where(fwi_Bscale > 1.0, fwi_30a, fwi_Bscale)

        return fwi

    def _calculate_extrapolated_duff_moisture_function(self) -> np.ndarray:
        """Calculates the extrapolated Duff Moisture Function (extrapolated_DMF)
        from the Build Up Index (BUI).

        From Van Wagner and Pickett (1985), Page 7-8: Equations 28a and 28b.

        Returns:
            np.ndarray: The calculated extrapolated_DMF values.
        """
        bui_data = self.build_up_index.data

        # Calculate extrapolated_DMF using equations 28a and 28b
        extrapolated_DMF_28a = 0.626 * bui_data**0.809 + 2.0
        extrapolated_DMF_28b = 1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui_data))

        # Condition: BUI <= 80 uses equation 28a, BUI > 80 uses equation 28b
        extrapolated_DMF = np.where(
            bui_data <= 80.0, extrapolated_DMF_28a, extrapolated_DMF_28b
        )

        return extrapolated_DMF

    def _make_fwi_cube(self, fwi_data: np.ndarray) -> Cube:
        """Converts an FWI data array into an iris.cube.Cube object
        with relevant metadata copied from the input ISI cube.

        Args:
            fwi_data (np.ndarray): The FWI data

        Returns:
            Cube: An iris.cube.Cube containing the FWI data with updated
                metadata and coordinates.
        """
        fwi_cube = self.initial_spread_index.copy(data=fwi_data.astype(np.float32))

        # Update the cube name and metadata
        fwi_cube.rename("canadian_forest_fire_weather_index")
        fwi_cube.units = "1"

        return fwi_cube

    def process(
        self,
        cubes: tuple[Cube] | CubeList,
    ) -> Cube:
        """Calculate the Fire Weather Index (FWI).

        From Van Wagner and Pickett (1985), Page 8: Steps 4-5

        Args:
            cubes (Cube | CubeList): Input cubes containing:
                initial_spread_index: ISI value (dimensionless)
                build_up_index: BUI value (dimensionless)

        Returns:
            Cube: The calculated FWI values.
        """
        self.load_input_cubes(cubes)

        # Step 4: Calculate extrapolated Duff Moisture Function
        extrapolated_DMF = self._calculate_extrapolated_duff_moisture_function()

        # Step 5: Calculate FWI
        output_fwi = self._calculate_fwi(extrapolated_DMF)

        # Convert FWI data to a cube and return
        fwi_cube = self._make_fwi_cube(output_fwi)

        return fwi_cube
