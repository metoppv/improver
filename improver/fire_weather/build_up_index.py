# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import numpy as np
from iris.cube import Cube

from improver.fire_weather import FireWeatherIndexBase


class BuildUpIndex(FireWeatherIndexBase):
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

    INPUT_CUBE_NAMES = ["duff_moisture_code", "drought_code"]
    OUTPUT_CUBE_NAME = "build_up_index"
    # Valid output ranges for warning checks (output_name: (min, max))
    # Minimum and maximum feasible values for each output index are drawn from
    # values reported in:
    # Wang, X., Oliver, J., Swystun, T., Hanes, C.C., Erni, S. and Flannigan,
    # M.D., 2023. Critical fire weather conditions during active fire spread
    # days in Canada. Science of the total environment, 869, p.161831.
    VALID_OUTPUT_RANGE = (0.0, 500)
    # Map input cube names to internal attribute names for consistency
    INPUT_ATTRIBUTE_MAPPINGS = {
        "duff_moisture_code": "input_dmc",
        "drought_code": "input_dc",
    }

    input_dmc: Cube
    input_dc: Cube

    def _calculate(self) -> np.ndarray:
        """Calculates the Build Up Index (BUI) from DMC and DC.

        From Van Wagner and Pickett (1985), Page 7: Equations 27a-27b.

        Returns:
            The calculated BUI values.
        """
        dmc_data = self.input_dmc.data
        dc_data = self.input_dc.data

        # Condition 1: If both DMC and DC are zero, set BUI = 0
        both_zero = np.isclose(dmc_data, 0.0, atol=1e-7) & np.isclose(
            dc_data, 0.0, atol=1e-7
        )

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
