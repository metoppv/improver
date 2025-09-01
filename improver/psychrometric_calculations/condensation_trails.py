# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to contain Condensation trail formation calculations."""

from typing import Tuple, Union

import numpy as np
from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.constants import EARTH_REPSILON
from improver.generate_ancillaries.generate_svp_derivative_table import (
    SaturatedVapourPressureTable,
    SaturatedVapourPressureTableDerivative,
)
from improver.psychrometric_calculations.psychrometric_calculations import (
    calculate_svp_in_air,
)
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
    critical_temperatures = None
    critical_intercepts = None

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

    def _find_local_vapour_pressure(self, pressure_levels: np.ndarray) -> np.ndarray:
        """
        Calculate the local vapour pressure (svp) at the given pressure levels using the temperature and pressure data.

        Args:
            pressure_levels (np.ndarray): Pressure levels (Pa).

        Returns:
            np.ndarray: The localised vapour pressure at the given
                pressure levels (Pa).
        """
        # Pressure levels has to be reshaped to match the temperature and humidity dimensions
        pressure_levels_reshaped = np.reshape(
            pressure_levels,
            (len(pressure_levels),) + (1,) * (self.temperature.ndim - 1),
        )
        svp = calculate_svp_in_air(
            temperature=self.temperature, pressure=pressure_levels_reshaped
        )
        return self.relative_humidity * svp

    @staticmethod
    def _critical_temperatures_all_rh(
        mixing_ratio: float, svp_derivative_table: Cube, svp_table: Cube
    ) -> Tuple[np.ndarray, float]:
        """
        For a given engine mixing ratio (at a constant pressure level) return the critical
        temperatures and critical intercept from the tangent with the SVP curve for all
        relative humidities.
        """

        def find_nearest(array, value) -> Tuple[np.ndarray, int]:
            """Return array element and index of nearest element to a given value"""
            idx = (np.abs(array - value)).argmin()
            return array[idx], idx

        temperature_table = svp_derivative_table.coord("air_temperature").points

        # de_s/dT = m is satisifed at the critical temperature, Tc, at 100% RH (Schrader 1997, section 3)
        svp_derivative_Tc_RH100, idx = find_nearest(
            svp_derivative_table.data, mixing_ratio
        )
        Tc_RH100 = temperature_table[idx]

        # saturation vapour pressure at critical temperature
        svp_Tc_RH100 = svp_table[idx].data

        # critical intercept (eqn. 2 of the ticket instructions)
        Ic = svp_Tc_RH100 - svp_derivative_Tc_RH100 * Tc_RH100

        # tangent to SVP curve with gradient m
        y_tangent = svp_derivative_Tc_RH100 * temperature_table + Ic

        # Tc at 0% RH given by point where tangent crosses e = 0
        _, idx = find_nearest(y_tangent, 0)
        Tc_RH0 = temperature_table[idx]

        mask = (temperature_table >= Tc_RH0) & (temperature_table <= Tc_RH100)
        Tc_all = temperature_table[mask]

        RH_all = np.zeros(Tc_all.size)

        # calculate the relative humidities at each critical temperature
        for Tci, Tc in enumerate(Tc_all):
            T, Ti = find_nearest(temperature_table, Tc)

            e = y_tangent[Ti]

            esat = svp_table[Ti].data

            RH_all[Tci] = e / esat

        return RH_all, Tc_all, Ic

    def _calculate_critical_temperatures(
        self,
        pressure_levels: np.ndarray,
        engine_mixing_ratios: np.ndarray,
        relative_humidity: Cube,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get critical temperatures and intercepts for the relative humidity
        and pressure level data contained within the contrail class
        """

        critical_temperatures = np.zeros(
            ((engine_mixing_ratios.shape[0],) + relative_humidity.shape)
        )
        critical_intercepts = np.zeros(critical_temperatures.shape[:2])

        # TODO: pass 'water_only=True' flag after PR approved
        svp = SaturatedVapourPressureTable().process()
        svp_derivative = SaturatedVapourPressureTableDerivative().process()

        for cfi in range(engine_mixing_ratios.shape[0]):
            for pi in range(pressure_levels.size):
                rh, tc, ic = CondensationTrailFormation._critical_temperatures_all_rh(
                    engine_mixing_ratios[cfi, pi], svp_derivative, svp
                )

                # lookup critical temperature for a given relative humidity
                critical_temperatures[cfi, pi, :, :] = np.interp(
                    relative_humidity[pi, :, :].data, rh, tc
                )

                critical_intercepts[cfi, pi] = ic

        return critical_temperatures, critical_intercepts

    def process_from_arrays(
        self,
        temperature: np.ndarray,
        relative_humidity: np.ndarray,
        pressure_levels: np.ndarray,
    ) -> np.ndarray:
        """
        Main entry point of this class for data as Numpy arrays

        Process the temperature, humidity and pressure data to calculate the
        contrails data.

        Args:
            temperature (np.ndarray): Temperature data on pressure levels where pressure is the leading axis (K).
            relative_humidity (np.ndarray): Relative humidity data on pressure levels where pressure is the leading axis (kg kg-1).
            pressure_levels (np.ndarray): Pressure levels (Pa).

        Returns:
            np.ndarray: The calculated engine mixing ratios on pressure levels (Pa/K).
            This is a placeholder until the full contrail formation logic is implemented.
        """
        self.temperature = temperature
        self.relative_humidity = relative_humidity
        self.pressure_levels = pressure_levels
        self.engine_mixing_ratios = self._calculate_engine_mixing_ratios(
            self.pressure_levels
        )
        self.local_vapour_pressure = self._find_local_vapour_pressure(
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

        self.critical_temperatures, self.critical_intercepts = (
            self._calculate_critical_temperatures(
                pressure_levels=pressure_coord.points,
                engine_mixing_ratios=self.engine_mixing_ratios,
                relative_humidity=humidity_cube,
            )
        )

        # Placeholder return to silence my type checker
        return_cube = Cube(
            self.engine_mixing_ratios,
            long_name="engine_mixing_ratios",
            units="Pa K-1",
        )

        return return_cube
