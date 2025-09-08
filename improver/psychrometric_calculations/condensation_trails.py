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
    def _critical_temperatures_for_given_contrail_factor(
        mixing_ratios: np.ndarray,
        relative_humidity: np.ndarray,
        svp: np.ndarray,
        svp_derivative: np.ndarray,
        temperature: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For engine mixing ratios on pressure levels for a given engine contrail factor,
        return the critical temperatures and critical intercepts of contrail formation.

        These are calculated for each pressure level by drawing a tangent to the saturation
        vapour pressure curve, with a gradient equal to the engine mixing ratio.

        Args:
            mixing_ratios (np.ndarray): Engine mixing ratios on pressure levels for a given engine contrail factor (Pa/K).
            relative_humidity (np.ndarray): Relative humidity on pressure levels (kg kg-1).
            svp (np.ndarray): Lookup table of saturation vapour pressure with respect to water (Pa).
            svp_derivative (np.ndarray): Lookup table of the first derivative of saturation vapour pressure with respect to water (Pa/K).
            temeprature (np.ndarray): Air temperatures corresponding to both lookup tables (K).

        Returns:
            np.ndarray: The critical temperatures at which contrails may form, on pressure levels (K).
            np.ndarray: The critical intercepts at which contrails may form, on pressure levels (Pa).

        """
        num_pressure_levels = mixing_ratios.shape[0]

        # maximum critical temperature (at 100% relative humidity) is given by
        # the point on the SVP derivative curve that is equal to the engine mixing
        # ratio
        ind_max = [(np.abs(svp_derivative - m)).argmin() for m in mixing_ratios]
        tangent_gradient = svp_derivative[ind_max]
        critical_temperature_maximum = temperature[ind_max]

        # tangent to SVP curve, with gradient equal to the engine mixing ratio
        critical_intercepts = (
            svp[ind_max] - tangent_gradient * critical_temperature_maximum
        )
        tangent_vapour_pressure = (
            tangent_gradient[:, np.newaxis] * temperature[np.newaxis, :]
            + critical_intercepts[:, np.newaxis]
        )

        # minimum critical temperature (at 0% relative humidity) is given by
        # point at which the tangent crosses the line of zero vapour pressure
        ind_min = [
            np.abs(tangent_vapour_pressure[i, :]).argmin()
            for i in range(num_pressure_levels)
        ]
        critical_temperature_minimum = temperature[ind_min]

        # critical temperature for all relative humidites (0% to 100%)
        critical_temperature_all_relative_humidities = np.array(
            [
                np.linspace(
                    critical_temperature_minimum[i],
                    critical_temperature_maximum[i],
                    num=100,
                )
                for i in range(num_pressure_levels)
            ]
        )

        # vapour pressure at critical temperature
        e = np.array(
            [
                np.interp(
                    critical_temperature_all_relative_humidities[i, :],
                    temperature,
                    tangent_vapour_pressure[i, :],
                )
                for i in range(num_pressure_levels)
            ]
        )

        # saturation vapour pressure at critical temperature
        esat = np.array(
            [
                np.interp(
                    critical_temperature_all_relative_humidities[i, :],
                    temperature,
                    svp,
                )
                for i in range(num_pressure_levels)
            ]
        )

        # critical temperature at given relative humidities
        critical_temperatures = np.array(
            [
                np.interp(
                    relative_humidity[i, :, :],
                    e[i, :] / esat[i, :],
                    critical_temperature_all_relative_humidities[i, :],
                )
                for i in range(num_pressure_levels)
            ]
        )

        return critical_temperatures, critical_intercepts

    @staticmethod
    def _calculate_critical_temperatures(
        engine_mixing_ratios: np.ndarray,
        relative_humidity: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the critical temperatures and intercepts on pressure levels, for all engine contrail factors.

        Args:
            engine_mixing_ratios (np.ndarray): Engine mixing ratios on pressure levels for all engine contrail factors (Pa/K).
            relative_humidity (np.ndarray): Relative humidity on pressure levels (kg/kg).

        Returns:
            np.ndarray: The critical temperatures at which contrails may form, on pressure levels, for all engine contrail factors (K).
            np.ndarray: The critical intercepts at which contrails may form, on pressure levels, for all engine contrail factors (K).
        """
        critical_temperatures = np.zeros(
            ((engine_mixing_ratios.shape[0],) + relative_humidity.shape)
        )
        critical_intercepts = np.zeros(critical_temperatures.shape[:2])

        # TODO: pass 'water_only=True' flag after PR approved
        svp_table = SaturatedVapourPressureTable(183.15, 253.15, 0.01).process()
        svp_derivative_table = SaturatedVapourPressureTableDerivative(
            183.15, 253.15, 0.01
        ).process()

        for i in range(engine_mixing_ratios.shape[0]):
            critical_temperatures[i], critical_intercepts[i] = (
                CondensationTrailFormation._critical_temperatures_for_given_contrail_factor(
                    engine_mixing_ratios[i],
                    relative_humidity.data,
                    svp_table.data,
                    svp_derivative_table.data,
                    svp_table.coord("air_temperature").points,
                )
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
            relative_humidity (np.ndarray): Relative humidity data on pressure levels where pressure is the leading axis (kg/kg).
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
        self.critical_temperatures, self.critical_intercepts = (
            CondensationTrailFormation._calculate_critical_temperatures(
                self.engine_mixing_ratios,
                self.relative_humidity,
            )
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
