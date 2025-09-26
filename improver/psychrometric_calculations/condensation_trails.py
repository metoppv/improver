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
    SaturatedVapourPressureDerivativeTable,
    SaturatedVapourPressureTable,
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

    .. include:: extended_documentation/psychrometric_calculations/condensation_trails/appleman_diagram.rst

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
            np.ndarray: The mixing ratio of the atmosphere and aircraft exhaust, provided in units: Pa/K.
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
            np.ndarray: The localised vapour pressure at the given pressure levels (Pa).
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

    def _critical_temperatures_and_intercepts_for_given_contrail_factor(
        self,
        engine_mixing_ratios_for_contrail_factor: np.ndarray,
        svp_table: Cube,
        svp_derivative_table: Cube,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the critical temperatures and critical intercepts on pressure levels
        for a single engine contrail factor.

        These are calculated at each pressure level by drawing a tangent to the saturation
        vapour pressure curve with respect to water. The tangent gradient is equal to the
        engine mixing ratio.

        .. include:: extended_documentation/psychrometric_calculations/condensation_trails/critical_temperatures.rst

        Args:
            engine_mixing_ratios_for_contrail_factor (np.ndarray): Engine mixing ratios for a single contrail factor on pressure levels. Pressure is the leading axis (Pa/K).
            svp (iris.cube.Cube): Lookup table of saturation vapour pressure with respect to water (Pa).
            svp_derivative (iris.cube.Cube): Lookup table of the first derivative of saturation vapour pressure with respect to water (Pa/K).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays of critical temperatures on pressure levels (K), and critical intercepts on pressure levels (Pa).

        """
        temperature_from_svp_table = svp_table.coord("air_temperature").points

        # maximum critical temperature (at 100% relative humidity) is given by
        # the point on the SVP derivative curve that is equal to the engine mixing
        # ratio
        ind_max = np.abs(
            svp_derivative_table.data
            - engine_mixing_ratios_for_contrail_factor[:, np.newaxis]
        ).argmin(axis=1)
        tangent_gradient = svp_derivative_table.data[ind_max]
        critical_temperature_maximum = temperature_from_svp_table[ind_max]

        # tangent to SVP curve, with gradient equal to the engine mixing ratio
        critical_intercepts = (
            svp_table.data[ind_max] - tangent_gradient * critical_temperature_maximum
        )
        tangent_vapour_pressure = (
            tangent_gradient[:, np.newaxis] * temperature_from_svp_table[np.newaxis, :]
            + critical_intercepts[:, np.newaxis]
        )

        # minimum critical temperature (at 0% relative humidity) is given by
        # point at which the tangent crosses the line of zero vapour pressure
        ind_min = np.abs(tangent_vapour_pressure).argmin(axis=1)
        critical_temperature_minimum = temperature_from_svp_table[ind_min]

        # critical temperature for all relative humidites (0% to 100%)
        critical_temperature_all_relative_humidities = np.linspace(
            critical_temperature_minimum, critical_temperature_maximum, num=100, axis=1
        )

        # vapour pressure at critical temperature
        e = np.array(
            [
                np.interp(
                    critical_temperature_all_relative_humidities[i],
                    temperature_from_svp_table,
                    tangent_vapour_pressure[i],
                )
                for i in range(critical_temperature_all_relative_humidities.shape[0])
            ]
        )

        # saturation vapour pressure at critical temperature
        esat = np.array(
            [
                np.interp(
                    critical_temperature_all_relative_humidities[i],
                    temperature_from_svp_table,
                    svp_table.data,
                )
                for i in range(critical_temperature_all_relative_humidities.shape[0])
            ]
        )

        # critical temperature at given relative humidities
        critical_temperatures = np.array(
            [
                np.interp(
                    self.relative_humidity[i],
                    e[i] / esat[i],
                    critical_temperature_all_relative_humidities[i],
                )
                for i in range(critical_temperature_all_relative_humidities.shape[0])
            ]
        )
        return critical_temperatures, critical_intercepts

    def _calculate_critical_temperatures_and_intercepts(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the critical temperatures and intercepts on pressure levels for all engine contrail factors.

        Returns:
            Tuple[np.ndarray, np.ndarry]: Arrays of critical temperatures on pressure levels for all engine contrail factors (K), and critical intercepts on pressure levels for all engine contrail factors (Pa).
        """
        if not (
            isinstance(self.engine_mixing_ratios, np.ndarray)
            and isinstance(self.relative_humidity, np.ndarray)
        ):
            raise TypeError(
                f"Incorrect types for engine_mixing_ratios (expected {np.ndarray}, got {type(self.engine_mixing_ratios)}) "
                f"and relative_humidity (expected {np.ndarray}, got {type(self.relative_humidity)})."
            )
        if self.engine_mixing_ratios.ndim != 2:
            raise ValueError(
                f"Incorrect number of dimensions for engine_mixing_ratios (expected 2, got {self.engine_mixing_ratios.ndim:d})."
            )
        if self.relative_humidity.ndim != 2 and self.relative_humidity.ndim != 3:
            raise ValueError(
                f"Incorrect number of dimensions for relative_humidity (expected 2 or 3, got {self.relative_humidity.ndim:d})."
            )
        if self.engine_mixing_ratios.shape[1] != self.relative_humidity.shape[0]:
            raise ValueError(
                f"Mismatch between dimension 1 of engine_mixing_ratios ({self.engine_mixing_ratios.shape[1]:d}) and dimension 0 "
                f"of relative_humidity ({self.relative_humidity.shape[0]:d})."
            )

        critical_temperatures = np.zeros(
            ((self.engine_mixing_ratios.shape[0],) + self.relative_humidity.shape),
            dtype=np.float32,
        )
        critical_intercepts = np.zeros(
            critical_temperatures.shape[:2], dtype=np.float32
        )

        svp_table = SaturatedVapourPressureTable(
            183.15, 253.15, water_only=True
        ).process()
        svp_derivative_table = SaturatedVapourPressureDerivativeTable(
            183.15, 253.15, water_only=True
        ).process()

        # loop over engine contrail factors
        for i, engine_mixing_ratios_for_contrail_factor in enumerate(
            self.engine_mixing_ratios
        ):
            critical_temperatures[i], critical_intercepts[i] = (
                self._critical_temperatures_and_intercepts_for_given_contrail_factor(
                    engine_mixing_ratios_for_contrail_factor,
                    svp_table,
                    svp_derivative_table,
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
            self._calculate_critical_temperatures_and_intercepts()
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
