# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to contain Condensation trail calculations."""

import numpy as np

from improver import BasePlugin
from improver.constants import EARTH_REPSILON


class CondensationTrailFormation(BasePlugin):
    """Plugin to calculate whether a condensation trail (contrail) will form
    based on a given set of atmospheric conditions. The calculations require
    cubes of the following data:
        - Temperature
        - Pressure
        - Relative Humidity

    Alongside constants including the ratio of the molecular masses of water
    and air (EARTH_REPSILON), and defined values for the engine contrail factor.

    References:
        - Schrader, M.L., 1997. Calculations of aircraft contrail formation
          critical temperatures. Journal of Applied Meteorology, 36(12),
          pp.1725-1729.
    """

    def __init__(self):
        """Initialise the plugin."""
        self._engine_contrail_factors: np.ndarray = np.array(
            [1, 2, 3], dtype=np.float32
        )  # Placeholder values

    def calculate_engine_mixing_ratios(self, pressure_levels: np.ndarray) -> np.ndarray:
        """
        Calculate the mixing ratio of the atmosphere and aircraft exhaust. This
        calculation uses EARTH_REPSILON, which is the ratio of the molecular
        weights of water and air on Earth.

        Args:
            pressure (float): The pressure of the atmosphere provided in units:
                Pascals.
            engine_contrail_factor (float): The engine contrail factor, provided in
                units: kg/kg/K.

        Returns:
            float: The mixing ratio of the atmosphere and aircraft exhaust,
                provided in units: P/K.
        """
        return (
            pressure_levels[np.newaxis, :]
            * self._engine_contrail_factors[:, np.newaxis]
            / EARTH_REPSILON
        )
