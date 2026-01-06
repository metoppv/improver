# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugin to calculate the Fire Severity Index (Daily Severity Rating)."""

import numpy as np
from iris.cube import Cube

from improver.fire_weather import FireWeatherIndexBase


class FireSeverityIndex(FireWeatherIndexBase):
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

    INPUT_CUBE_NAMES = ["canadian_forest_fire_weather_index"]
    OUTPUT_CUBE_NAME = "fire_severity_index"

    canadian_forest_fire_weather_index: Cube

    def _calculate(self) -> np.ndarray:
        """Calculates the Daily Severity Rating (DSR) from FWI.

        From Van Wagner and Pickett (1985), Page 8: Equation 31.

        Returns:
            The calculated DSR values.
        """
        fwi_data = self.canadian_forest_fire_weather_index.data

        # Equation 31: DSR = 0.0272 * FWI^1.77
        dsr = 0.0272 * fwi_data**1.77

        return dsr
