# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Module to contain wet-bulb temperature plugins."""
from typing import List, Union

import iris
import numpy as np
from cf_units import Unit
from iris.cube import Cube, CubeList
from numpy import ndarray

from improver import BasePlugin
from improver import constants as consts
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.psychrometric_calculations.psychrometric_calculations import (
    _calculate_latent_heat,
    saturated_humidity,
)
from improver.utilities.cube_checker import check_cube_coordinates
from improver.utilities.mathematical_operations import Integration


class WetBulbTemperature(BasePlugin):
    """
    A plugin to calculate wet bulb temperatures from air temperature, relative
    humidity, and pressure data. Calculations are performed using a Newton
    iterator, with saturated vapour pressures drawn from a lookup table using
    linear interpolation.

    The svp_table used in this plugin is imported (see top of file). It is a
    table of saturated vapour pressures calculated for a range of temperatures.
    The import also brings in attributes that describe the range of
    temperatures covered by the table and the increments in the table.

    References:
        Met Office UM Documentation Paper 080, UM Version 10.8,
        last updated 2014-12-05.
    """

    def __init__(self, precision: float = 0.005, model_id_attr: str = None) -> None:
        """
        Initialise class.

        Args:
            precision:
                The precision to which the Newton iterator must converge before
                returning wet bulb temperatures.
            model_id_attr (str):
                Name of the attribute used to identify the source model for blending.

        """
        self.precision = precision
        self.maximum_iterations = 20
        self.model_id_attr = model_id_attr

    @staticmethod
    def _slice_inputs(temperature, relative_humidity, pressure):
        """Create iterable or iterator over cubes on which to calculate
        wet bulb temperature"""
        vertical_coords = None
        try:
            vertical_coords = [
                cube.coord(axis="z").name()
                for cube in [temperature, relative_humidity, pressure]
                if cube.coord_dims(cube.coord(axis="z")) != ()
            ]
        except iris.exceptions.CoordinateNotFoundError:
            pass

        if not vertical_coords:
            slices = [(temperature, relative_humidity, pressure)]
        else:
            if len(set(vertical_coords)) > 1 or len(vertical_coords) != 3:
                raise ValueError(
                    "WetBulbTemperature: Cubes have differing vertical coordinates."
                )
            (level_coord,) = set(vertical_coords)
            temperature_over_levels = temperature.slices_over(level_coord)
            relative_humidity_over_levels = relative_humidity.slices_over(level_coord)
            pressure_over_levels = pressure.slices_over(level_coord)
            slices = zip(
                temperature_over_levels,
                relative_humidity_over_levels,
                pressure_over_levels,
            )

        return slices

    @staticmethod
    def _calculate_specific_heat(mixing_ratio: ndarray) -> ndarray:
        """
        Calculate the specific heat capacity for moist air by combining that of
        dry air and water vapour in proportion given by the specific humidity.

        Args:
            mixing_ratio:
                Array of specific humidity (fractional).

        Returns:
            Specific heat capacity of moist air (J kg-1 K-1).
        """
        specific_heat = (
            -1.0 * mixing_ratio + 1.0
        ) * consts.CP_DRY_AIR + mixing_ratio * consts.CP_WATER_VAPOUR
        return specific_heat

    @staticmethod
    def _calculate_enthalpy(
        mixing_ratio: ndarray,
        specific_heat: ndarray,
        latent_heat: ndarray,
        temperature: ndarray,
    ) -> ndarray:
        """
        Calculate the enthalpy (total energy per unit mass) of air (J kg-1).

        Method from referenced UM documentation.

        References:
            Met Office UM Documentation Paper 080, UM Version 10.8,
            last updated 2014-12-05.

        Args:
            mixing_ratio:
                Array of mixing ratios.
            specific_heat:
                Array of specific heat capacities of moist air (J kg-1 K-1).
            latent_heat:
                Array of latent heats of condensation of water vapour
                (J kg-1).
            temperature:
                Array of air temperatures (K).

        Returns:
           Array of enthalpy values calculated at the same points as the
           input cubes (J kg-1).
        """
        enthalpy = latent_heat * mixing_ratio + specific_heat * temperature
        return enthalpy

    @staticmethod
    def _calculate_enthalpy_gradient(
        mixing_ratio: ndarray,
        specific_heat: ndarray,
        latent_heat: ndarray,
        temperature: ndarray,
    ) -> ndarray:
        """
        Calculate the enthalpy gradient with respect to temperature.

        Method from referenced UM documentation.

        Args:
            mixing_ratio:
                Array of mixing ratios.
            specific_heat:
                Array of specific heat capacities of moist air (J kg-1 K-1).
            latent_heat:
                Array of latent heats of condensation of water vapour
                (J kg-1).
            temperature:
                Array of temperatures (K).

        Returns:
            Array of the enthalpy gradient with respect to temperature.
        """
        numerator = mixing_ratio * latent_heat * latent_heat
        denominator = consts.R_WATER_VAPOUR * temperature * temperature
        return numerator / denominator + specific_heat

    def _calculate_wet_bulb_temperature(
        self, pressure: ndarray, relative_humidity: ndarray, temperature: ndarray
    ) -> ndarray:
        """
        Calculate an array of wet bulb temperatures from inputs in
        the correct units.

        A Newton iterator is used to minimise the gradient of enthalpy
        against temperature. Assumes that the variation of latent heat with
        temperature can be ignored.

        Args:
            pressure:
                Array of air Pressure (Pa).
            relative_humidity:
                Array of relative humidities (1).
            temperature:
                Array of air temperature (K).

        Returns:
            Array of wet bulb temperature (K).

        """
        # Initialise psychrometric variables
        wbt_data_upd = wbt_data = temperature.flatten()
        pressure = pressure.flatten()

        latent_heat = _calculate_latent_heat(wbt_data)
        saturation_mixing_ratio = saturated_humidity(wbt_data, pressure)
        mixing_ratio = relative_humidity.flatten() * saturation_mixing_ratio
        specific_heat = self._calculate_specific_heat(mixing_ratio)
        enthalpy = self._calculate_enthalpy(
            mixing_ratio, specific_heat, latent_heat, wbt_data
        )
        del mixing_ratio

        # Iterate to find the wet bulb temperature, using temperature as first
        # guess
        iteration = 0
        to_update = np.arange(temperature.size)
        update_to_update = slice(None)
        while to_update.size and iteration < self.maximum_iterations:

            if iteration > 0:
                wbt_data_upd = wbt_data[to_update]
                pressure = pressure[update_to_update]
                specific_heat = specific_heat[update_to_update]
                latent_heat = latent_heat[update_to_update]
                enthalpy = enthalpy[update_to_update]
                saturation_mixing_ratio = saturated_humidity(wbt_data_upd, pressure)

            enthalpy_new = self._calculate_enthalpy(
                saturation_mixing_ratio, specific_heat, latent_heat, wbt_data_upd
            )
            enthalpy_gradient = self._calculate_enthalpy_gradient(
                saturation_mixing_ratio, specific_heat, latent_heat, wbt_data_upd
            )
            delta_wbt = (enthalpy - enthalpy_new) / enthalpy_gradient

            # Increment wet bulb temperature at points which have not converged
            update_to_update = np.abs(delta_wbt) > self.precision
            to_update = to_update[update_to_update]
            wbt_data[to_update] += delta_wbt[update_to_update]

            iteration += 1

        return wbt_data.reshape(temperature.shape)

    def create_wet_bulb_temperature_cube(
        self, temperature: Cube, relative_humidity: Cube, pressure: Cube
    ) -> Cube:
        """
        Creates a cube of wet bulb temperature values

        Args:
            temperature:
                Cube of air temperatures.
            relative_humidity:
                Cube of relative humidities.
            pressure:
                Cube of air pressures.

        Returns:
            Cube of wet bulb temperature (K).
        """
        temperature.convert_units("K")
        relative_humidity.convert_units(1)
        pressure.convert_units("Pa")
        wbt_data = self._calculate_wet_bulb_temperature(
            pressure.data, relative_humidity.data, temperature.data
        )

        attributes = generate_mandatory_attributes(
            [temperature, relative_humidity, pressure], model_id_attr=self.model_id_attr
        )
        wbt = create_new_diagnostic_cube(
            "wet_bulb_temperature", "K", temperature, attributes, data=wbt_data
        )
        return wbt

    def process(self, cubes: Union[List[Cube], CubeList]) -> Cube:
        """
        Call the calculate_wet_bulb_temperature function to calculate wet bulb
        temperatures. This process function splits input cubes over vertical
        levels to mitigate memory issues when trying to operate on multi-level
        data.

        Args:
            cubes:
                containing:
                    temperature:
                        Cube of air temperatures.
                    relative_humidity:
                        Cube of relative humidities.
                    pressure:
                        Cube of air pressures.

        Returns:
            Cube of wet bulb temperature (K).
        """
        names_to_extract = ["air_temperature", "relative_humidity", "air_pressure"]
        if len(cubes) != len(names_to_extract):
            raise ValueError(
                f"Expected {len(names_to_extract)} cubes, found {len(cubes)}"
            )

        temperature, relative_humidity, pressure = tuple(
            CubeList(cubes).extract_cube(n) for n in names_to_extract
        )

        slices = self._slice_inputs(temperature, relative_humidity, pressure)

        cubelist = iris.cube.CubeList([])
        for t_slice, rh_slice, p_slice in slices:
            cubelist.append(
                self.create_wet_bulb_temperature_cube(
                    t_slice.copy(), rh_slice.copy(), p_slice.copy()
                )
            )
        wet_bulb_temperature = cubelist.merge_cube()

        # re-promote any scalar coordinates lost in slice / merge
        wet_bulb_temperature = check_cube_coordinates(temperature, wet_bulb_temperature)

        return wet_bulb_temperature


class WetBulbTemperatureIntegral(BasePlugin):
    """Calculate a wet-bulb temperature integral."""

    def __init__(self, model_id_attr: str = None):
        """Initialise class."""
        self.model_id_attr = model_id_attr
        self.integration_plugin = Integration("height")

    def process(self, wet_bulb_temperature: Cube) -> Cube:
        """
        Calculate the vertical integral of wet bulb temperature from the input
        wet bulb temperatures on height levels.

        Args:
            wet_bulb_temperature:
                Cube of wet bulb temperatures on height levels.

        Returns:
            Cube of wet bulb temperature integral (Kelvin-metres).
        """
        wbt = wet_bulb_temperature.copy()
        wbt.convert_units("degC")
        wbt.coord("height").convert_units("m")
        # Touch the data to ensure it is not lazy
        # otherwise vertical interpolation is slow
        wbt.data
        wet_bulb_temperature_integral = self.integration_plugin(wbt)
        if self.model_id_attr:
            wet_bulb_temperature_integral.attributes[
                self.model_id_attr
            ] = wbt.attributes[self.model_id_attr]
        # although the integral is computed over degC the standard unit is
        # 'K m', and these are equivalent
        wet_bulb_temperature_integral.units = Unit("K m")
        return wet_bulb_temperature_integral
