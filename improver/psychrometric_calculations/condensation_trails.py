# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to contain Condensation trail calculations."""

from typing import Union

import numpy as np
from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.constants import EARTH_REPSILON
from improver.utilities.common_input_handle import as_cubelist


class CondensationTrailFormation(BasePlugin):
    """Plugin to calculate whether a condensation trail (contrail) will
    form based on a given set of atmospheric conditions.

    The calculations require cubes of the following data:

    - Temperature
    - Pressure
    - Relative Humidity

    Alongside constants including the ratio of the molecular masses of
    water and air (EARTH_REPSILON), and defined values for the engine
    contrail factor.

    References:
        Schrader, M.L., 1997. Calculations of aircraft contrail
        formation critical temperatures. Journal of Applied
        Meteorology, 36(12), pp.1725-1729.
    """

    def __init__(self, engine_contrail_factors: list = [3e-5, 3.4e-5, 3.9e-5]):
        """Initialsies the Class"""

        self._engine_contrail_factors: np.ndarray = np.array(
            engine_contrail_factors, dtype=np.float32
        )

    def calculate_engine_mixing_ratios(self) -> np.ndarray:
        """
        Calculate the mixing ratio of the atmosphere and aircraft
        exhaust (Schrader, 1997). This calculation uses
        EARTH_REPSILON, which is the ratio of the molecular
        weights of water and air on Earth.

        Returns:
            float: The mixing ratio of the atmosphere and aircraft
                exhaust, provided in units: P/K.
        """
        return (
            self.pressure_levels[np.newaxis, :]
            * self._engine_contrail_factors[:, np.newaxis]
            / EARTH_REPSILON
        )

    def process(self, *cubes: Union[Cube, CubeList]) -> Cube:
        """
        Main entry point of this class

        Args:
            cubes
                air_temperature:
                    Cube of the temperature on pressure levels.
                relative_humidity:
                    Cube of the relative humidity on pressure levels.
                geopotential_height:
                    Cube of the height above sea level on pressure levels.
        Returns:
            Cube of heights above sea level at which contrails will form.
        """
        cubes = as_cubelist(*cubes)
        (self.temperature, self.humidity, self.height) = CubeList(cubes).extract(
            ["air_temperature", "relative_humidity", "geopotential_height"]
        )

        # Get the pressure levels from the first cube
        self.pressure_levels = self.temperature.coord("pressure").points

        # Calculate the mixing ratios
        engine_mixing_ratios = self.calculate_engine_mixing_ratios()

        # Placeholder return to silence my type checker
        return_cube = Cube(
            engine_mixing_ratios,
            long_name="engine_mixing_ratios",
            units="Pa/K",
        )

        return return_cube
