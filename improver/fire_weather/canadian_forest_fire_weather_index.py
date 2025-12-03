# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugin to calculate the Canadian Forest Fire Weather Index (FWI)."""

import numpy as np
from iris.cube import Cube

from improver.fire_weather import FireWeatherIndexBase


class CanadianForestFireWeatherIndex(FireWeatherIndexBase):
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

    INPUT_CUBE_NAMES = ["initial_spread_index", "build_up_index"]
    OUTPUT_CUBE_NAME = "canadian_forest_fire_weather_index"

    initial_spread_index: Cube
    build_up_index: Cube

    def _calculate(self) -> np.ndarray:
        """Calculate the Fire Weather Index (FWI).

        From Van Wagner and Pickett (1985), Page 8: Steps 4-6

        Returns:
            np.ndarray: The calculated FWI values.
        """
        # Step 4: Calculate extrapolated Duff Moisture Function
        extrapolated_DMF = self._calculate_extrapolated_duff_moisture_function()

        # Steps 5 & 6: Calculate FWI
        fwi = self._calculate_fwi(extrapolated_DMF)

        return fwi

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
