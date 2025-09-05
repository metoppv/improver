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
    def _critical_temperatures_all_rh_as_arrays(
        mixing_ratios: np.ndarray, svp_derivative_table: Cube, svp_table: Cube
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For engine mixing ratios at a given engine contrail factor, return the critical
        temperatures and critical intercepts from the tangents with the SVP curve for all
        relative humidities.
        """

        temperature_table = svp_derivative_table.coord("air_temperature").points

        # de_s/dT = m is satisifed at the critical temperature, Tc, at 100% RH (Schrader 1997, section 3)
        indices = [
            (np.abs(svp_derivative_table.data - m)).argmin() for m in mixing_ratios
        ]
        svp_derivative_Tc_RH100 = svp_derivative_table[indices].data
        Tc_RH100 = temperature_table[indices]

        # saturation vapour pressures at critical temperatures
        svp_Tc_RH100 = svp_table[indices].data

        # critical intercepts (eqn. 2 of the ticket instructions)
        Ic = svp_Tc_RH100 - svp_derivative_Tc_RH100 * Tc_RH100

        # tangents to SVP curve, with gradients equal to mixing ratios
        y_tangent = (
            svp_derivative_Tc_RH100[:, np.newaxis] * temperature_table[np.newaxis, :]
            + Ic[:, np.newaxis]
        )

        # Tc at 0% RH given by point where tangent crosses e = 0
        indices = [
            np.abs(y_tangent[mi, :]).argmin() for mi in range(mixing_ratios.shape[0])
        ]
        Tc_RH0 = temperature_table[indices]

        # critical temperatures for all pressure levels and relative humidites (0 to 100)
        Tc_all = np.array(
            [
                np.linspace(Tc_RH0[mi], Tc_RH100[mi], num=100)
                for mi in range(mixing_ratios.shape[0])
            ]
        )

        # vapour pressure interpolated from tangent
        e = np.array(
            [
                np.interp(Tc_all[mi, :], temperature_table, y_tangent[mi, :])
                for mi in range(mixing_ratios.shape[0])
            ]
        )

        # svp interpolated from curve
        esat = np.array(
            [
                np.interp(Tc_all[mi, :], temperature_table, svp_table.data)
                for mi in range(mixing_ratios.shape[0])
            ]
        )

        # relative humidity for each critical temperature
        RH_all = e / esat

        return RH_all, Tc_all, Ic

    def _calculate_critical_temperatures_as_arrays(
        self,
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
        svp = SaturatedVapourPressureTable(t_increment=0.01).process()
        svp_derivative = SaturatedVapourPressureTableDerivative(
            t_increment=0.01
        ).process()

        for cfi in range(engine_mixing_ratios.shape[0]):
            # compute critical temperatures for all relative humidities over pressure levels
            rh, tc, critical_intercepts[cfi, :] = (
                CondensationTrailFormation._critical_temperatures_all_rh_as_arrays(
                    engine_mixing_ratios[cfi, :], svp_derivative, svp
                )
            )

            # lookup critical temperatures for stored relative humidities
            critical_temperatures[cfi, :, :, :] = np.array(
                [
                    np.interp(relative_humidity[mi, :, :].data, rh[mi, :], tc[mi, :])
                    for mi in range(engine_mixing_ratios.shape[1])
                ]
            )

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
            self._calculate_critical_temperatures_as_arrays(
                self.engine_mixing_ratios,
                humidity_cube,
            )
        )

        # Placeholder return to silence my type checker
        return_cube = Cube(
            self.engine_mixing_ratios,
            long_name="engine_mixing_ratios",
            units="Pa K-1",
        )

        return return_cube
