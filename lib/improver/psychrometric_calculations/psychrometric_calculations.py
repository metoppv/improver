# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
"""Module to contain Psychrometric Calculations."""

import warnings
from cf_units import Unit
import improver.constants as cc
import numpy as np

from improver.psychrometric_calculations import svp_table


class Utilities(object):

    """
    Utilities for pschrometric calculations.
    """

    def __init__(self):
        """
        Initialise class.
        """
        pass

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<Utilities>')
        return result

    @staticmethod
    def specific_heat_of_moist_air(mixing_ratio):
        """
        Calculate the specific heat capacity for moist air by combining that of
        dry air and water vapour in proportion given by the specific humidity.

        Args:
            mixing_ratio : iris.cube.Cube
                Cube of specific humidity (fractional).
        Returns:
            iris.cube.Cube
                Specific heat capacity of moist air (J kg-1 K-1).
        """
        specific_heat = ((-1.*mixing_ratio + 1.) * cc.U_CP_DRY_AIR +
                         mixing_ratio * cc.U_CP_WATER_VAPOUR)
        specific_heat.rename('specific_heat')
        return specific_heat

    @staticmethod
    def latent_heat_of_condensation(temperature_input):
        """
        Calculate a temperature adjusted latent heat of condensation for water
        vapour using the relationship employed by the UM.

        Args:
            temperature : iris.cube.Cube
                A cube of air temperatures (Celsius, converted if not).
        Returns:
            iris.cube.Cube
                Temperature adjusted latent heat of condesation (J kg-1).
        """
        temperature = temperature_input.copy()
        temperature.convert_units('celsius')
        latent_heat = (-1. * cc.U_LATENT_HEAT_T_DEPENDENCE * temperature +
                       cc.U_LH_CONDENSATION_WATER)
        latent_heat.units = cc.U_LH_CONDENSATION_WATER.units
        latent_heat.rename('latent_heat_of_condensation')
        return latent_heat

    @staticmethod
    def calculate_enthalpy(mixing_ratio, specific_heat, latent_heat,
                           temperature):
        """
        Calculate the enthalpy (total energy per unit mass) of air (J kg-1).

        Args:
            mixing_ratio : iris.cube.Cube
                Cube of mixing ratios.
            specific_heat : iris.cube.Cube
                Cube of specific heat capacities of moist air (J kg-1 K-1).
            latent_heat : iris.cube.Cube
                Cube of latent heats of condensation of water vapour
                (J kg-1).
            temperature : iris.cube.Cube
                A cube of air temperatures (K).
        Returns:
           enthalpy : iris.cube.Cube
               A cube of enthalpy values calculated at the same points as the
               input cubes (J kg-1).
        """
        enthalpy = latent_heat * mixing_ratio + specific_heat * temperature
        enthalpy.rename('enthalpy_of_air')
        return enthalpy

    @staticmethod
    def calculate_d_enthalpy_dt(mixing_ratio, specific_heat,
                                latent_heat, wet_bulb_temperature):
        """
        Calculate the enthalpy gradient with respect to temperature.

        Args:

        Returns:

        """
        wet_bulb_temperature.convert_units('K')
        numerator = (mixing_ratio * latent_heat ** 2)
    #    R_term = np.max(np.finfo(cc.R_WATER_VAPOUR).eps,
    #                    cc.R_WATER_VAPOUR * wet_bulb_temperature ** 2)
        denominator = cc.U_R_WATER_VAPOUR * wet_bulb_temperature ** 2
        return numerator/denominator + specific_heat


class WetBulbTemperature(object):

    """
    Functions to calculate the wet bulb temperature.
    """

    def __init__(self, precision=0.005):
        """
        Initialise class.

        Args:
            svp_table : iris.cube.Cube
                A table of saturated vapour pressures calculated for a range
                of temperatures. The cube attributes should describe the range
                of temperature covered by the table with 'temperature_minimum'
                and 'temperature_maximum' keys. The increments in the table
                should also be described with a 'temperature_increment'
                attribute.
            precision: float
                The precision to which the Newton iterator must converge before
                returning wet bulb temperatures.
        """
        self.precision = precision

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<WetBulbTemperature: precision: {}>'.format(self.precision))
        return result

    @staticmethod
    def _check_range(cube, low, high):
        """Function to wrap functionality for throwing out temperatures
        too low or high for a method to use safely.

        Args:
            cube : iris.cube.Cube
                A cube of temperature.

            low : int or float
                Lowest allowable temperature for check

            high : int or float
                Highest allowable temperature for check

        Raises:
            UserWarning : If any of the values in cube.data are outside the
                          bounds set by the low and high variables.
        """
        if cube.data.max() > high or cube.data.min() < low:
            emsg = ("Wet bulb temperatures are being calculated for conditions"
                    " beyond the valid range of the saturated vapour pressure"
                    " lookup table (< {}K or > {}K). Input cube has\n"
                    "Lowest temperature = {}\nHighest temperature = {}")
            warnings.warn(emsg.format(low, high, cube.data.min(),
                                      cube.data.max()))

    def _lookup_svp(self, temperature):
        """
        Looks up a value for the saturation vapour pressure of water vapour
        using the temperature and a table of values. These tabulated values
        have been calculated using the utilties.ancillary_creation
        SaturatedVapourPressureTable plugin that uses the Goff-Gratch method.

        Args:
            temperature : iris.cube.Cube
                A cube of air temperatures (K).
        Returns:
            svp : iris.cube.Cube
                A cube of saturated vapour pressures (Pa).
        """
        T_min = svp_table.T_MIN
        T_max = svp_table.T_MAX
        delta_T = svp_table.T_INCREMENT
        self._check_range(temperature, T_min, T_max)
        temperatures = temperature.data
        T_clipped = np.clip(temperatures, T_min, T_max)

        # Note the indexing below differs by -1 compared with the UM due to
        # Python vs. Fortran indexing.
        table_position = (T_clipped - T_min + delta_T)/delta_T - 1.
        table_index = table_position.astype(int)
        interpolation_factor = table_position - table_index
        svps = ((1.0 - interpolation_factor) * svp_table.DATA[table_index] +
                interpolation_factor * svp_table.DATA[table_index + 1])

        svp = temperature.copy(data=svps)
        svp.units = Unit('Pa')
        svp.rename("saturated_vapour_pressure")
        return svp

    @staticmethod
    def _pressure_correct_svp(svp, temperature, pressure):
        """
        Convert saturated vapour pressure in a pure water vapour system into
        the saturated vapour pressure in air.

        Args:
            temperature : iris.cube.Cube
                A cube of air temperatures (K, converted to Celsius).
            pressure : iris.cube.Cube
                Cube of pressure (Pa).

        Returns:
            svp : iris.cube.Cube
                Cube of saturated vapour pressure of air (Pa).
        """
        temp = temperature.copy()
        temp.convert_units('celsius')

        correction = 1. + 1E-8 * pressure.data * (4.5 + 6E-4 * temp.data ** 2)
        svp.data = svp.data*correction
        return svp

    def _mixing_ratio(self, temperature, pressure):
        """Function to compute the mixing ratio given temperature, relative
        humidity, and pressure.

        Args:
            temperature : iris.cube.Cube
                Cube of air temperature (K).
            pressure : iris.cube.Cube
                Cube of air pressure (Pa).

        Returns
            mixing_ratio : iris.cube.Cube
                Cube of mixing ratios.

        References:
            ASHRAE Fundamentals handbook (2005) Equation 22, 24, p6.8
        """
        svp = self._lookup_svp(temperature)
        svp = self._pressure_correct_svp(svp, temperature, pressure)

        # Calculation
        result_numer = (cc.EARTH_REPSILON * svp.data)
        # This max pressure term may be redundant, but we should check in very
        # low pressure environments.
        max_pressure_term = np.maximum(svp.data, pressure.data)
        result_denom = (max_pressure_term - ((1. - cc.EARTH_REPSILON) *
                                             svp.data))
        mixing_ratio = temperature.copy(data=result_numer / result_denom)

        # Tidying up cube
        mixing_ratio.rename("mixing_ratio")
        mixing_ratio.units = Unit("1")
        return mixing_ratio

    def calculate_wet_bulb_temperature(self, temperature, relative_humidity,
                                       pressure):
        """

        Args:
            temperature : iris.cube.Cube
                Cube of air temperatures (K).
            relative_humidity : iris.cube.Cube
                Cube of relative humidities (%, converted to fractional).
            pressure : iris.cube.Cube
                Cube of air pressures (Pa).

        Returns
            wbt : iris.cube.Cube
                Cube of wet bulb temperature (K).

        """
        precision = np.full(temperature.data.shape, self.precision)

        # Set units of input diagnostics.
        relative_humidity.convert_units(1)
        pressure.convert_units('Pa')
        temperature.convert_units('K')

        # Calculate mixing ratios.
        saturation_mixing_ratio = self._mixing_ratio(temperature, pressure)
        mixing_ratio = relative_humidity * saturation_mixing_ratio
        # Calculate specific and latent heats.
        specific_heat = Utilities.specific_heat_of_moist_air(mixing_ratio)
        latent_heat = Utilities.latent_heat_of_condensation(temperature)

        # Calculate enthalpy.
        g_tw = Utilities.calculate_enthalpy(mixing_ratio, specific_heat,
                                            latent_heat, temperature)
        # Use air temperature as a first guess for wet bulb temperature.
        wbt = temperature.copy()
        wbt.rename('wet_bulb_temperature')
        delta_wbt = temperature.copy(data=(10. * precision))
        delta_wbt_history = temperature.copy(data=(5. * precision))

        # Iterate to find the wet bulb temperature
        while (np.abs(delta_wbt.data) > precision).any():
            g_tw_new = Utilities.calculate_enthalpy(
                saturation_mixing_ratio, specific_heat, latent_heat, wbt)
            dg_dt = Utilities.calculate_d_enthalpy_dt(
                saturation_mixing_ratio, specific_heat, latent_heat, wbt)
            delta_wbt = (g_tw - g_tw_new) / dg_dt

            # Only change values at those points yet to converge to avoid
            # oscillating solutions (the now fixed points are still calculated
            # unfortunately).
            unfinished = np.where(np.abs(delta_wbt.data) > precision)
            wbt.data[unfinished] = (wbt.data[unfinished] +
                                    delta_wbt.data[unfinished])

            # If the errors are identical between two iterations, stop.
            if np.array_equal(delta_wbt.data, delta_wbt_history.data):
                print 'No further refinement occuring'
                break
            delta_wbt_history = delta_wbt

            # Recalculate the saturation mixing ratio
            saturation_mixing_ratio = self._mixing_ratio(
                wbt, pressure)

        return wbt
