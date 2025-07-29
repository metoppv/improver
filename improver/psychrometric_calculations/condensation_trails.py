# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to contain Condensation trail formation calculations."""

from typing import Union

import numpy as np
from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.constants import EARTH_REPSILON
from improver.utilities.common_input_handle import as_cubelist


class CondensationTrailFormation(BasePlugin):
    """Plugin to calculate whether a condensation trail (contrail) will
    form based on a given set of atmospheric conditions.

    The calculations require the following data:

    - Temperature on pressure levels.
    - Relative Humidity on pressure levels.

    Alongside constants including the ratio of the molecular masses of
    water and air (EARTH_REPSILON), and defined values for the engine
    contrail factor.

    References:
        Schrader, M.L., 1997. Calculations of aircraft contrail
        formation critical temperatures. Journal of Applied
        Meteorology, 36(12), pp.1725-1729.
    """

    temperature = None
    humidity = None
    pressure_levels = None
    engine_mixing_ratios = None

    def __init__(self, engine_contrail_factors: list = [3e-5, 3.4e-5, 3.9e-5]):
        """Initialises the Class

        Args:
            engine_contrail_factors (list, optional):
                List of engine contrail factors to use in the
                calculations. Defaults to [3e-5, 3.4e-5, 3.9e-5].
                These values are for Non-, Low-, and High-Bypass
                engines from Schrader (1997). The units are
                kg/kg/K.
        """

        self._engine_contrail_factors: np.ndarray = np.array(
            engine_contrail_factors, dtype=np.float32
        )

    def _calculate_engine_mixing_ratios(
        self, pressure_levels: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the mixing ratio of the atmosphere and aircraft
        exhaust (Schrader, 1997). This calculation uses
        EARTH_REPSILON, which is the ratio of the molecular
        weights of water and air on Earth.

        Returns:
            np.ndarray: The mixing ratio of the atmosphere and aircraft
                exhaust, provided in units: Pa/K.
        """
        return (
            pressure_levels[np.newaxis, :]
            * self._engine_contrail_factors[:, np.newaxis]
            / EARTH_REPSILON
        )

    def process_from_arrays(
        self, temperature: np.ndarray, humidity: np.ndarray, pressure_levels: np.ndarray
    ) -> np.ndarray:
        """
        Main entry point of this class for data as Numpy arrays

        Process the temperature, humidity and pressure data to calculate the
        contrails data.

        Args:
            temperature (np.ndarray): Temperature data on pressure levels where pressure is the leading axis (K).
            humidity (np.ndarray): Relative humidity data on pressure levels where pressure is the leading axis (kg kg-1).
            pressure_levels (np.ndarray): Pressure levels (Pa).

        Returns:
            np.ndarray: The calculated engine mixing ratios on pressure levels (Pa/K).
            This is a placeholder until the full contrail formation logic is implemented.
        """
        self.temperature = temperature
        self.humidity = humidity
        self.pressure_levels = pressure_levels
        self.engine_mixing_ratios = self._calculate_engine_mixing_ratios(
            self.pressure_levels
        )
        return self.engine_mixing_ratios

    def process(self, *cubes: Union[Cube, CubeList]) -> Cube:
        """
        Main entry point of this class for data as iris.Cubes

        Args:
            cubes
                air_temperature:
                    Cube of the temperature on pressure levels.
                relative_humidity:
                    Cube of the relative humidity on pressure levels.

        Returns:
            Cube of heights above sea level at which contrails will form.
        """
        cubes = as_cubelist(*cubes)
        (temperature_cube, humidity_cube) = CubeList(cubes).extract(
            ["air_temperature", "relative_humidity"]
        )
        temperature_cube.convert_units("K")
        humidity_cube.convert_units("kg kg-1")

        # Get the pressure levels from the first cube
        pressure_coord = temperature_cube.coord("pressure")
        pressure_coord.convert_units("Pa")

        # Calculate contrail formation using numpy arrays
        _ = self.process_from_arrays(
            temperature_cube.data, humidity_cube.data, pressure_coord.points
        )

        # Placeholder return to silence my type checker
        return_cube = Cube(
            self.engine_mixing_ratios,
            long_name="engine_mixing_ratios",
            units="Pa K-1",
        )

        return return_cube
