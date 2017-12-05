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

import numpy as np
import iris
from stratify import interpolate
from scipy.interpolate import griddata
from cf_units import Unit

from improver.psychrometric_calculations import svp_table
from improver.utilities.cube_checker import check_cube_coordinates
from improver.utilities.mathematical_operations import Integration
import improver.constants as cc


class Utilities(object):

    """
    Utilities for psychrometric calculations.
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
            mixing_ratio (iris.cube.Cube):
                Cube of specific humidity (fractional).
        Returns:
            iris.cube.Cube:
                Specific heat capacity of moist air (J kg-1 K-1).
        """
        specific_heat = ((-1.*mixing_ratio + 1.) * cc.U_CP_DRY_AIR +
                         mixing_ratio * cc.U_CP_WATER_VAPOUR)
        specific_heat.rename('specific_heat_capacity_of_moist_air')
        return specific_heat

    @staticmethod
    def latent_heat_of_condensation(temperature_input):
        """
        Calculate a temperature adjusted latent heat of condensation for water
        vapour using the relationship employed by the UM.

        Args:
            temperature_input (iris.cube.Cube):
                A cube of air temperatures (Celsius, converted if not).
        Returns:
            iris.cube.Cube:
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

        Method from referenced UM documentation.

        References:
            Met Office UM Documentation Paper 080, UM Version 10.8,
            last updated 2014-12-05.

        Args:
            mixing_ratio (iris.cube.Cube):
                Cube of mixing ratios.
            specific_heat (iris.cube.Cube):
                Cube of specific heat capacities of moist air (J kg-1 K-1).
            latent_heat (iris.cube.Cube):
                Cube of latent heats of condensation of water vapour
                (J kg-1).
            temperature (iris.cube.Cube):
                A cube of air temperatures (K).
        Returns:
           enthalpy (iris.cube.Cube):
               A cube of enthalpy values calculated at the same points as the
               input cubes (J kg-1).
        """
        enthalpy = latent_heat * mixing_ratio + specific_heat * temperature
        enthalpy.rename('enthalpy_of_air')
        return enthalpy

    @staticmethod
    def calculate_d_enthalpy_dt(mixing_ratio, specific_heat,
                                latent_heat, temperature_input):
        """
        Calculate the enthalpy gradient with respect to temperature.

        Method from referenced UM documentation.

        References:
            Met Office UM Documentation Paper 080, UM Version 10.8,
            last updated 2014-12-05.

        Args:
            mixing_ratio (iris.cube.Cube):
                Cube of mixing ratios.
            specific_heat (iris.cube.Cube):
                Cube of specific heat capacities of moist air (J kg-1 K-1).
            latent_heat (iris.cube.Cube):
                Cube of latent heats of condensation of water vapour
                (J kg-1).
            temperature_input (iris.cube.Cube):
                A cube of temperatures (K, or converted).

        Returns:
            iris.cube.Cube:
                A cube of the enthalpy gradient with respect to temperature.
        """
        temperature = temperature_input.copy()
        temperature.convert_units('K')
        numerator = (mixing_ratio * latent_heat ** 2)
        denominator = cc.U_R_WATER_VAPOUR * temperature ** 2
        return numerator/denominator + specific_heat

    @staticmethod
    def saturation_vapour_pressure_goff_gratch(temperature):
        """
        Saturation Vapour pressure in a water vapour system calculated using
        the Goff-Gratch Equation (WMO standard method).

        Args:
            temperature (iris.cube.Cube):
                Cube of temperature which will be converted to Kelvin
                prior to calculation. Valid from 173K to 373K

        Returns:
            svp (iris.cube.Cube):
                Cube containing the saturation vapour pressure of a pure
                water vapour system. A correction must be applied to the data
                when used to convert this to the SVP in air; see the
                WetBulbTemperature._pressure_correct_svp function.

        References:
            Numerical data and functional relationships in science and
            technology. New series. Group V. Volume 4. Meteorology.
            Subvolume b. Physical and chemical properties of the air, P35.
        """
        constants = {1: 10.79574,
                     2: 5.028,
                     3: 1.50475E-4,
                     4: -8.2969,
                     5: 0.42873E-3,
                     6: 4.76955,
                     7: 0.78614,
                     8: -9.09685,
                     9: 3.56654,
                     10: 0.87682,
                     11: 0.78614}
        triple_pt = cc.TRIPLE_PT_WATER

        # Values for which method is considered valid (see reference).
        WetBulbTemperature.check_range(temperature, 173., 373.)

        data = temperature.data.copy()
        for cell in np.nditer(data, op_flags=['readwrite']):
            if cell > triple_pt:
                n0 = constants[1] * (1. - triple_pt / cell)
                n1 = constants[2] * np.log10(cell / triple_pt)
                n2 = constants[3] * (1. - np.power(10.,
                                                   (constants[4] *
                                                    (cell / triple_pt - 1.))))
                n3 = constants[5] * (np.power(10., (constants[6] *
                                                    (1. - triple_pt / cell))) -
                                     1.)
                log_es = n0 - n1 + n2 + n3 + constants[7]
                cell[...] = (np.power(10., log_es))
            else:
                n0 = constants[8] * ((triple_pt / cell) - 1.)
                n1 = constants[9] * np.log10(triple_pt / cell)
                n2 = constants[10] * (1. - (cell / triple_pt))
                log_es = n0 - n1 + n2 + constants[11]
                cell[...] = (np.power(10., log_es))

        # Create SVP cube
        svp = iris.cube.Cube(
            data, long_name='saturated_vapour_pressure', units='hPa')
        # Output of the Goff-Gratch is in hPa, but we want to return in Pa.
        svp.convert_units('Pa')
        return svp


class WetBulbTemperature(object):

    """
    A plugin to calculate wet bulb temperatures from air temperature, relative
    humidity, and pressure data. Calculations are performed using a Newton
    iterator, with saturated vapour pressures drawn from a lookup table using
    linear interpolation.

    The svp_table used in this plugin is imported (see top of file). It is a
    table of saturated vapour pressures calculated for a range of temperatures.
    The import also brings in attributes that describe the range of
    temperatures covered by the table and the increments in the table.

    """
    def __init__(self, precision=0.005):
        """
        Initialise class.

        Args:
            precision (float):
                The precision to which the Newton iterator must converge before
                returning wet bulb temperatures.
        """
        self.precision = precision

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<WetBulbTemperature: precision: {}>'.format(self.precision))
        return result

    @staticmethod
    def check_range(cube, low, high):
        """Function to wrap functionality for throwing out temperatures
        too low or high for a method to use safely.

        Args:
            cube (iris.cube.Cube):
                A cube of temperature.

            low (int or float):
                Lowest allowable temperature for check

            high (int or float):
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
            temperature (iris.cube.Cube):
                A cube of air temperatures (K).
        Returns:
            svp (iris.cube.Cube):
                A cube of saturated vapour pressures (Pa).
        """
        T_min = svp_table.T_MIN
        T_max = svp_table.T_MAX
        delta_T = svp_table.T_INCREMENT
        self.check_range(temperature, T_min, T_max)
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

        Method from referenced dcumentation.

        References:
            Atmosphere-Ocean Dynamics, Adrian E. Gill, International Geophysics
            Series, Vol. 30; Equation A4.7.

        Args:
            svp (iris.cube.Cube):
                A cube of saturated vapour pressures (Pa).
            temperature (iris.cube.Cube):
                A cube of air temperatures (K, converted to Celsius).
            pressure (iris.cube.Cube):
                Cube of pressure (Pa).

        Returns:
            svp (iris.cube.Cube):
                The input cube of saturated vapour pressure of air (Pa) is
                modified by the pressure correction.
        """
        temp = temperature.copy()
        temp.convert_units('celsius')

        correction = (1. + 1.0E-8 * pressure.data *
                      (4.5 + 6.0E-4 * temp.data ** 2))
        svp.data = svp.data*correction
        return svp

    def _calculate_mixing_ratio(self, temperature, pressure):
        """Function to compute the mixing ratio given temperature and pressure.

        Args:
            temperature (iris.cube.Cube):
                Cube of air temperature (K).
            pressure (iris.cube.Cube):
                Cube of air pressure (Pa).

        Returns
            mixing_ratio (iris.cube.Cube):
                Cube of mixing ratios.

        Method from referenced documentation. Note that EARTH_REPSILON is
        simply given as an unnamed constant in the reference (0.62198).

        References:
            ASHRAE Fundamentals handbook (2005) Equation 22, 24, p6.8
        """
        svp = self._lookup_svp(temperature)
        svp = self._pressure_correct_svp(svp, temperature, pressure)

        # Calculation
        result_numer = (cc.EARTH_REPSILON * svp.data)
        max_pressure_term = np.maximum(svp.data, pressure.data)
        result_denom = (max_pressure_term - ((1. - cc.EARTH_REPSILON) *
                                             svp.data))
        mixing_ratio = temperature.copy(data=result_numer / result_denom)

        # Tidying up cube
        mixing_ratio.rename("humidity_mixing_ratio")
        mixing_ratio.units = Unit("1")
        return mixing_ratio

    def calculate_wet_bulb_temperature(self, temperature, relative_humidity,
                                       pressure):
        """
        Perform the calculation of wet bulb temperatures. A Newton iterator is
        used to minimise the gradient of enthalpy against temperature.

        Args:
            temperature (iris.cube.Cube):
                Cube of air temperatures (K).
            relative_humidity (iris.cube.Cube):
                Cube of relative humidities (%, converted to fractional).
            pressure (iris.cube.Cube):
                Cube of air pressures (Pa).

        Returns:
            wbt (iris.cube.Cube):
                Cube of wet bulb temperature (K).

        """
        precision = np.full(temperature.data.shape, self.precision)

        # Set units of input diagnostics.
        relative_humidity.convert_units(1)
        pressure.convert_units('Pa')
        temperature.convert_units('K')

        # Calculate mixing ratios.
        saturation_mixing_ratio = self._calculate_mixing_ratio(temperature,
                                                               pressure)
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
        max_iterations = 20
        iteration = 0

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
            if (np.array_equal(delta_wbt.data, delta_wbt_history.data) or
                    iteration > max_iterations):
                warnings.warn('No further refinement occuring; breaking out '
                              'of Newton iterator and returning result.')
                break
            delta_wbt_history = delta_wbt
            iteration += 1

            # Recalculate the saturation mixing ratio
            saturation_mixing_ratio = self._calculate_mixing_ratio(
                wbt, pressure)

        return wbt

    def process(self, temperature, relative_humidity, pressure):
        """
        Call the calculate_wet_bulb_temperature function to calculate wet bulb
        temperatures. This process function splits input cubes over vertical
        levels to mitigate memory issues when trying to operate on multi-level
        data.

        Args:
            temperature (iris.cube.Cube):
                Cube of air temperatures (K).
            relative_humidity (iris.cube.Cube):
                Cube of relative humidities (%, converted to fractional).
            pressure (iris.cube.Cube):
                Cube of air pressures (Pa).

        Returns:
            wet_bulb_temperature (iris.cube.Cube):
                Cube of wet bulb temperature (K).
        """
        try:
            vertical_coords = [cube.coord(axis='z').name() for cube in
                               [temperature, relative_humidity, pressure]
                               if cube.coord_dims(cube.coord(axis='z')) != ()]
        except iris.exceptions.CoordinateNotFoundError:
            vertical_coords = []

        if len(vertical_coords) == 3 and len(set(vertical_coords)) == 1:
            level_coord, = set(vertical_coords)
            temperature_over_levels = temperature.slices_over(level_coord)
            relative_humidity_over_levels = relative_humidity.slices_over(
                level_coord)
            pressure_over_levels = pressure.slices_over(level_coord)
            slices = zip(temperature_over_levels,
                         relative_humidity_over_levels, pressure_over_levels)
        elif len(vertical_coords) > 0 and len(set(vertical_coords)) != 1:
            raise ValueError('WetBulbTemperature: Cubes have differing '
                             'vertical coordinates.')
        else:
            slices = [(temperature, relative_humidity, pressure)]

        cubelist = iris.cube.CubeList([])
        for t_slice, rh_slice, p_slice in slices:
            cubelist.append(self.calculate_wet_bulb_temperature(
                t_slice, rh_slice, p_slice))

        wet_bulb_temperature = cubelist.merge_cube()
        wet_bulb_temperature = check_cube_coordinates(temperature,
                                                      wet_bulb_temperature)
        return wet_bulb_temperature


class WetBulbTemperatureIntegral(object):
    """Calculate  a wet-bulb temperature integral."""

    def __init__(self, precision=0.005, coord_name_to_integrate="height",
                 start_point=None, end_point=None,
                 direction_of_integration="negative"):
        """
        Initialise class.

        Keyword Args:
            precision (float):
                The precision to which the Newton iterator must converge
                before returning wet bulb temperatures.
            coord_name_to_integrate (string):
                Name of the coordinate to be integrated.
            start_point (float or None):
                Point at which to start the integration.
                Default is None. If start_point is None, integration starts
                from the first available point.
            end_point (float or None):
                Point at which to end the integration.
                Default is None. If end_point is None, integration will
                continue until the last available point.
            direction_of_integration (string):
                Description of the direction in which to integrate.
                Options are 'positive' or 'negative'.
                'positive' corresponds to the values within the array
                increasing as the array index increases.
                'negative' corresponds to the values within the array
                decreasing as the array index increases.
        """
        self.wet_bulb_temperature_plugin = (
            WetBulbTemperature(precision=precision))
        self.integration_plugin = Integration(
            coord_name_to_integrate, start_point=start_point,
            end_point=end_point,
            direction_of_integration=direction_of_integration)
        self.coord_name_to_integrate = coord_name_to_integrate

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<WetBulbTemperatureIntegral: {}, {}>'.format(
            self.wet_bulb_temperature_plugin,
            self.integration_plugin))
        return result

    def process(self, temperature, relative_humidity, pressure):
        """
        Calculate the wet bulb temperature integral by firstly calculating
        the wet bulb temperature from the inputs provided, and then
        calculating the vertical integral of the wet bulb temperature.

        Args:
            temperature (iris.cube.Cube):
                Cube of air temperatures (K).
            relative_humidity (iris.cube.Cube):
                Cube of relative humidities (%, converted to fractional).
            pressure (iris.cube.Cube):
                Cube of air pressures (Pa).

        Returns:
            wet_bulb_temperature_integral (iris.cube.Cube):
                Cube of wet bulb temperature integral (Kelvin-metres).
        """
        # Calculate wet-bulb temperature.
        wet_bulb_temperature = (
            self.wet_bulb_temperature_plugin.process(
                temperature, relative_humidity, pressure))
        # Convert to Celsius
        wet_bulb_temperature.convert_units('celsius')
        # Integrate.
        wet_bulb_temperature_integral = (
            self.integration_plugin.process(wet_bulb_temperature))
        wet_bulb_temperature_integral.rename("wet_bulb_temperature_integral")
        units_string = "K {}".format(
            wet_bulb_temperature.coord(self.coord_name_to_integrate).units)
        wet_bulb_temperature_integral.units = Unit(units_string)
        return wet_bulb_temperature_integral


class FallingSnowLevel(object):
    """Calculate a field of continuous falling snow level."""

    def __init__(self, precision=0.005, falling_level_threshold=90.0):
        """
        Initialise class.

        Keyword Args:
            precision (float):
                The precision to which the Newton iterator must converge
                before returning wet bulb temperatures.
            falling_level_threshold (float):
                The cutoff threshold for the Wet-bulb integral used
                to calculate the falling snow level.We are integrating to the
                threshold that is presumed to indicate
                the level at which snow has melted back to rain. Above
                this level we should have falling snow.

        """
        self.precision = precision
        self.wet_bulb_integral_plugin = (
            WetBulbTemperatureIntegral(precision=precision))
        self.falling_level_threshold = falling_level_threshold
        self.missing_data = -300.0

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<FallingSnowLevel: precision:'
                  '{}, falling_level_threshold:{}>'.format(
                      self.precision,
                      self.falling_level_threshold))
        return result

    def find_falling_level(self, wb_int_data, orog_data, height_points):
        """
        Find the falling snow level by finding the level of the wet-bulb
        integral data at the required threshold. Wet-bulb integral data
        is only available above ground level and there may be an insufficient
        number of levels in the input data, in which case the required
        threshold may lie outside the Wet-bulb integral data and the value
        at that point will be set to np.nan.

        Args:
            wb_int_data (np.array):
                Wet bulb integral data on heights
            orog_data (np.array):
                Orographic data
            heights (np.array):
                heights agl

        Returns:
            snow_level_data (np.array):
                Falling snow level data asl.

        """
        # Create cube of heights above sea level for each height in
        # the wet bulb integral cube.
        asl = wb_int_data.copy()
        for i, height in enumerate(height_points):
            asl[i, ::] = orog_data + height

        # Calculate falling snow level above sea level by
        # finding the level corresponding to the falling_level_threshold.
        # Interpolate returns an array with height indice
        #  for falling_level_threshold so we take the 0 index
        snow_level_data = interpolate(np.array([self.falling_level_threshold]),
                                      wb_int_data, asl, axis=0)[0]

        return snow_level_data

    def fill_in_missing_data(self, snow_level_data, orog_data,
                             highest_wb_int_data, highest_height):
        """
        Fill in missing data. Wet-bulb integral data
        is only available above ground level and there may be an insufficient
        number of levels in the input data, in which case the falling_level
        will have been set to np.nan.

        This function fills in the missing data by firstly setting
        sea-level points where all integral values are already above
        the falling_level_threshold to the highest_height.
        Any remaining missing sea_points are set to 0.

        The data are then interpolated linearly across the grid.

        For any remaining missing points we check to see if they
        are already above the falling_level_threshold. If they are
        we set them to orography + highest_height

        For the remaining points we are not sure what exactly to do with
        them so we set them to missing_data (value less than 0) for now.
        We will need to look at better ways to deal with this data later.

        Args:
            snow_level_data (np.array):
                Falling snow level data (m).
            orog_data (np.array):
                Orographic data (m)
            highest_wb_int_data (np.array):
                Wet bulb integral data on highest level (K m).
            highest_height (float):
                Highest height at which the integral starts (m).

        Returns:
            snow_level_updated (np.array):
                Falling snow level data with missing data added.

        """
        # Firstly find sea_points where all integral values
        # are already above the falling_level_threshold
        # Set these to highest_height
        sea_points_not_freezing = np.where(
            np.isnan(snow_level_data) &
            (orog_data == 0.0) &
            (highest_wb_int_data >
             self.falling_level_threshold))
        snow_level_data[sea_points_not_freezing] = highest_height
        # Now find any remaining sea_points and set these to 0
        sea_points = np.where(np.isnan(snow_level_data) &
                              (orog_data == 0.0))
        snow_level_data[sea_points] = 0.0
        # Interpolate linearly across the remaining points
        points = np.where(np.isfinite(snow_level_data))
        values = snow_level_data[points]
        ynum, xnum = snow_level_data.shape
        (y_points, x_points) = np.mgrid[0:ynum, 0:xnum]
        snow_level_updated = griddata(points, values, (y_points, x_points),
                                      method='linear')
        # For any remaining missing points check to see if they
        # are already above the falling_level_threshold. If they are
        # set them to orography + highest_height
        above_freezing_points = np.where(
            np.isnan(snow_level_updated) &
            (highest_wb_int_data >
             self.falling_level_threshold))
        snow_level_updated[above_freezing_points] = (
            orog_data[above_freezing_points] +
            highest_height)
        # For any remaining points we are not sure what to do so
        # will set them to missing_data (value less than 0) for now.
        # We will need to look at better ways to deal with this data later.
        remaining_points = np.where(np.isnan(snow_level_updated))
        snow_level_updated[remaining_points] = self.missing_data
        return snow_level_updated

    def process(self, temperature, relative_humidity, pressure, orog):
        """
        Calculate the wet bulb temperature integral by firstly calculating
        the wet bulb temperature from the inputs provided, and then
        calculating the vertical integral of the wet bulb temperature.
        Find the falling_snow_level by finding the height above sea level
        correspoinding to the falling_level_threshold in the integral data.
        Fill in missing data appropriately.

        Args:
            temperature (iris.cube.Cube):
                Cube of air temperatures (K).
            relative_humidity (iris.cube.Cube):
                Cube of relative humidities (%, converted to fractional).
            pressure (iris.cube.Cube):
                Cube of air pressures (Pa).
            orog (iris.cube.Cube):
                Cube of orography (m).

        Returns:
            falling_snow_level (iris.cube.Cube):
                Cube of Falling Snow Level above sea level (asl).
        """

        # Calculate wet-bulb temperature integral.
        wet_bulb_integral = (
            self.wet_bulb_integral_plugin.process(
                temperature, relative_humidity, pressure))
        # Find highest height from height bounds.
        # If these are set to None then use the heights in temperature.
        height_bounds = wet_bulb_integral.coord('height').bounds
        if height_bounds is None:
            heights = temperature.coord('height').points
            highest_height = heights[-1]
        else:
            highest_height = height_bounds[0][-1]

        # Firstly we need to slice over height, x and y
        x_coord = wet_bulb_integral.coord(axis='x').name()
        y_coord = wet_bulb_integral.coord(axis='y').name()
        for orog_cube in orog.slices([y_coord, x_coord]):
            orog_data = orog_cube.data

        snow = iris.cube.CubeList([])
        slice_list = ['height', y_coord, x_coord]
        for wb_integral in wet_bulb_integral.slices(slice_list):

            height_points = wb_integral.coord('height').points
            # Calculate falling snow level above sea level.
            snow_cube = wb_integral[0]
            snow_cube.rename('falling_snow_level_asl')
            snow_cube.units = 'm'
            snow_cube.remove_coord('height')

            snow_cube.data = self.find_falling_level(wb_integral.data,
                                                     orog_data,
                                                     height_points)
            # Interpolate missing data
            snow_cube.data = self.fill_in_missing_data(snow_cube.data,
                                                       orog_data,
                                                       wb_integral.data[0, ::],
                                                       highest_height)

            snow.append(snow_cube)

        falling_snow_level = snow.merge_cube()
        return falling_snow_level
