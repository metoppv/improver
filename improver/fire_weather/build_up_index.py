# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from typing import cast

import numpy as np
from iris.cube import Cube, CubeList

from improver import BasePlugin


class BuildUpIndex(BasePlugin):
    """
    Plugin to calculate the Build Up Index (BUI) following
    the Canadian Forest Fire Weather Index System.

    The BUI is a numerical rating of the total amount of fuel available
    for combustion. It combines the Duff Moisture Code (DMC) and the
    Drought Code (DC) to represent the fuel buildup.

    This process is adapted directly from:
        Equations and FORTRAN Program for the
        Canadian Forest Fire Weather Index System
        (C.E. Van Wagner and T.L. Pickett, 1985).
        Page 7, Equations 27a-27b.

    Expected input units:
        - Duff Moisture Code (DMC): dimensionless
        - Drought Code (DC): dimensionless
    """

    duff_moisture_code: Cube
    drought_code: Cube

    def load_input_cubes(self, cubes: tuple[Cube] | CubeList):
        """Loads the required input cubes for the BUI calculation. These
        are stored internally as Cube objects.

        Args:
            cubes (tuple[Cube] | CubeList): Input cubes containing the necessary data.

        Raises:
            ValueError: If the number of cubes does not match the expected
                number (2).
        """
        names_to_extract = [
            "duff_moisture_code",
            "drought_code",
        ]
        if len(cubes) != len(names_to_extract):
            raise ValueError(
                f"Expected {len(names_to_extract)} cubes, found {len(cubes)}"
            )

        # Load the cubes into class attributes
        (
            self.duff_moisture_code,
            self.drought_code,
        ) = tuple(cast(Cube, CubeList(cubes).extract_cube(n)) for n in names_to_extract)

        # Ensure the cubes are set to the correct units
        self.duff_moisture_code.convert_units("1")
        self.drought_code.convert_units("1")

    def _calculate_bui(self) -> np.ndarray:
        """Calculates the Build Up Index (BUI) from DMC and DC.

        From Van Wagner and Pickett (1985), Page 7: Equations 27a-27b.

        Returns:
            np.ndarray: The calculated BUI values.
        """
        dmc_data = self.duff_moisture_code.data
        dc_data = self.drought_code.data

        # Condition 1: If both DMC and DC are zero, set BUI = 0
        both_zero = (dmc_data == 0.0) & (dc_data == 0.0)

        # Condition 2: If DMC <= 0.4 * DC use equation 27a
        use_27a = dmc_data <= 0.4 * dc_data

        # Calculate equations, suppressing divide-by-zero warnings
        # (the np.where will handle the zero cases correctly)
        with np.errstate(divide="ignore", invalid="ignore"):
            bui_27a = (0.8 * dmc_data * dc_data) / (dmc_data + 0.4 * dc_data)

            # Condition 3: If DMC > 0.4 * DC use equation 27b
            bui_27b = dmc_data - (
                1.0 - (0.8 * dc_data / (dmc_data + 0.4 * dc_data))
            ) * (0.92 + (0.0114 * dmc_data) ** 1.7)

        # Apply conditions using np.where:
        # 1. If both_zero: BUI = 0
        # 2. Elif use_27a (DMC <= 0.4*DC): BUI = bui_27a
        # 3. Else (DMC > 0.4*DC): BUI = bui_27b
        bui = np.where(both_zero, 0.0, np.where(use_27a, bui_27a, bui_27b))

        # Ensure BUI is never negative
        bui = np.clip(bui, 0.0, None)

        return bui

    def _make_bui_cube(self, bui_data: np.ndarray) -> Cube:
        """Converts a BUI data array into an iris.cube.Cube object
        with relevant metadata copied from the input DMC cube.

        Args:
            bui_data (np.ndarray): The BUI data

        Returns:
            Cube: An iris.cube.Cube containing the BUI data with updated
                metadata and coordinates.
        """
        bui_cube = self.duff_moisture_code.copy(data=bui_data.astype(np.float32))

        # Update the cube name and metadata
        bui_cube.rename("build_up_index")
        bui_cube.units = "1"

        return bui_cube

    def process(
        self,
        cubes: tuple[Cube] | CubeList,
    ) -> Cube:
        """Calculate the Build Up Index (BUI). This implements page 8, Step 4
        of van Wagner and Pickett (1985).

        Args:
            cubes (Cube | CubeList): Input cubes containing:
                duff_moisture_code: DMC value (dimensionless)
                drought_code: DC value (dimensionless)

        Returns:
            Cube: The calculated BUI values.
        """
        self.load_input_cubes(cubes)

        # Step 4:Calculate BUI
        output_bui = self._calculate_bui()

        # Convert BUI data to a cube and return
        bui_cube = self._make_bui_cube(output_bui)

        return bui_cube
