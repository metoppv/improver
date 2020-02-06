# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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

import iris
import numpy as np
from cf_units import Unit
from iris.cube import CubeList
from scipy.interpolate import griddata
from scipy.spatial.qhull import QhullError
from scipy.stats import linregress
from stratify import interpolate

import improver.constants as consts
from improver import BasePlugin
from improver.metadata.utilities import (
    generate_mandatory_attributes, create_new_diagnostic_cube)
from improver.psychrometric_calculations import svp_table
from improver.utilities.cube_checker import check_cube_coordinates
from improver.utilities.cube_manipulation import sort_coord_in_cube
from improver.utilities.mathematical_operations import Integration
from improver.utilities.spatial import (
    OccurrenceWithinVicinity, convert_number_of_grid_cells_into_distance)


def _svp_from_lookup(temperature):
    """
    Gets value for saturation vapour pressure in a pure water vapour system
    from a pre-calculated lookup table. Interpolates linearly between points in
    the table to the temperatures required.

    Args:
        temperature (numpy.ndarray):
            Array of air temperatures (K).
    Returns:
        numpy.ndarray:
            Array of saturated vapour pressures (Pa).
    """
    # where temperatures are outside the SVP table range, clip data to
    # within the available range
    T_clipped = np.clip(temperature, svp_table.T_MIN,
                        svp_table.T_MAX - svp_table.T_INCREMENT)

    # interpolate between bracketing values
    table_position = (T_clipped - svp_table.T_MIN) / svp_table.T_INCREMENT
    table_index = table_position.astype(int)
    interpolation_factor = table_position - table_index
    return ((1.0 - interpolation_factor) * svp_table.DATA[table_index] +
            interpolation_factor * svp_table.DATA[table_index + 1])


def calculate_svp_in_air(temperature, pressure):
    """
    Calculates the saturation vapour pressure in air.  Looks up the saturation
    vapour pressure in a pure water vapour system, and pressure-corrects the
    result to obtain the saturation vapour pressure in air.

    Args:
        temperature (numpy.ndarray):
            Array of air temperatures (K).
        pressure (numpy.ndarray):
            Array of pressure (Pa).

    Returns:
        numpy.ndarray:
            Saturation vapour pressure in air (Pa).

    References:
        Atmosphere-Ocean Dynamics, Adrian E. Gill, International Geophysics
        Series, Vol. 30; Equation A4.7.
    """
    svp = _svp_from_lookup(temperature)
    temp_celcius = temperature.copy() + consts.ABSOLUTE_ZERO
    correction = (1. + 1.0E-8 * pressure * (4.5 + 6.0E-4 * temp_celcius ** 2))
    return svp * correction.astype(np.float32)


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
        self.maximum_iterations = 20

    @staticmethod
    def _slice_inputs(temperature, relative_humidity, pressure):
        """Create iterable or iterator over cubes on which to calculate
        wet bulb temperature"""
        vertical_coords = None
        try:
            vertical_coords = [cube.coord(axis='z').name() for cube in
                               [temperature, relative_humidity, pressure]
                               if cube.coord_dims(cube.coord(axis='z')) != ()]
        except iris.exceptions.CoordinateNotFoundError:
            pass

        if not vertical_coords:
            slices = [(temperature, relative_humidity, pressure)]
        else:
            if len(set(vertical_coords)) > 1 or len(vertical_coords) != 3:
                raise ValueError('WetBulbTemperature: Cubes have differing '
                                 'vertical coordinates.')
            level_coord, = set(vertical_coords)
            temperature_over_levels = temperature.slices_over(level_coord)
            relative_humidity_over_levels = relative_humidity.slices_over(
                level_coord)
            pressure_over_levels = pressure.slices_over(level_coord)
            slices = zip(temperature_over_levels,
                         relative_humidity_over_levels, pressure_over_levels)

        return slices

    @staticmethod
    def _calculate_latent_heat(temperature):
        """
        Calculate a temperature adjusted latent heat of condensation for water
        vapour using the relationship employed by the UM.

        Args:
            temperature (np.ndarray):
                Array of air temperatures (K).
        Returns:
            np.ndarray:
                Temperature adjusted latent heat of condensation (J kg-1).
        """
        temp_celcius = temperature + consts.ABSOLUTE_ZERO
        latent_heat = (-1. * consts.LATENT_HEAT_T_DEPENDENCE * temp_celcius +
                       consts.LH_CONDENSATION_WATER)
        return latent_heat

    @staticmethod
    def _calculate_mixing_ratio(temperature, pressure):
        """Function to compute the mixing ratio given temperature and pressure.

        Args:
            temperature (numpy.ndarray):
                Array of air temperature (K).
            pressure (numpy.ndarray):
                Array of air pressure (Pa).

        Returns
            numpy.ndarray:
                Array of mixing ratios.

        Method from referenced documentation. Note that EARTH_REPSILON is
        simply given as an unnamed constant in the reference (0.62198).

        References:
            ASHRAE Fundamentals handbook (2005) Equation 22, 24, p6.8
        """
        svp = calculate_svp_in_air(temperature, pressure)
        numerator = (consts.EARTH_REPSILON * svp)
        denominator = (np.maximum(svp, pressure) - (
            (1. - consts.EARTH_REPSILON) * svp))
        return numerator / denominator

    @staticmethod
    def _calculate_specific_heat(mixing_ratio):
        """
        Calculate the specific heat capacity for moist air by combining that of
        dry air and water vapour in proportion given by the specific humidity.

        Args:
            mixing_ratio (numpy.ndarray):
                Array of specific humidity (fractional).
        Returns:
            numpy.ndarray:
                Specific heat capacity of moist air (J kg-1 K-1).
        """
        specific_heat = ((-1.*mixing_ratio + 1.) * consts.CP_DRY_AIR
                         + mixing_ratio * consts.CP_WATER_VAPOUR)
        return specific_heat

    @staticmethod
    def _calculate_enthalpy(
            mixing_ratio, specific_heat, latent_heat, temperature):
        """
        Calculate the enthalpy (total energy per unit mass) of air (J kg-1).

        Method from referenced UM documentation.

        References:
            Met Office UM Documentation Paper 080, UM Version 10.8,
            last updated 2014-12-05.

        Args:
            mixing_ratio (numpy.ndarray):
                Array of mixing ratios.
            specific_heat (numpy.ndarray):
                Array of specific heat capacities of moist air (J kg-1 K-1).
            latent_heat (numpy.ndarray):
                Array of latent heats of condensation of water vapour
                (J kg-1).
            temperature (numpy.ndarray):
                Array of air temperatures (K).
        Returns:
           numpy.ndarray:
               Array of enthalpy values calculated at the same points as the
               input cubes (J kg-1).
        """
        enthalpy = latent_heat * mixing_ratio + specific_heat * temperature
        return enthalpy

    @staticmethod
    def _calculate_enthalpy_gradient(
            mixing_ratio, specific_heat, latent_heat, temperature):
        """
        Calculate the enthalpy gradient with respect to temperature.

        Method from referenced UM documentation.

        References:
            Met Office UM Documentation Paper 080, UM Version 10.8,
            last updated 2014-12-05.

        Args:
            mixing_ratio (numpy.ndarray):
                Array of mixing ratios.
            specific_heat (numpy.ndarray):
                Array of specific heat capacities of moist air (J kg-1 K-1).
            latent_heat (numpy.ndarray):
                Array of latent heats of condensation of water vapour
                (J kg-1).
            temperature (numpy.ndarray):
                Array of temperatures (K).

        Returns:
            numpy.ndarray:
                Array of the enthalpy gradient with respect to temperature.
        """
        numerator = (mixing_ratio * latent_heat ** 2)
        denominator = consts.R_WATER_VAPOUR * temperature ** 2
        return numerator/denominator + specific_heat

    def _calculate_wet_bulb_temperature(
            self, pressure, relative_humidity, temperature):
        """
        Calculate an array of wet bulb temperatures from inputs in
        the correct units.

        A Newton iterator is used to minimise the gradient of enthalpy
        against temperature.

        Args:
            pressure (numpy.ndarray):
                Array of air Pressure (Pa).
            relative_humidity (numpy.ndarray):
                Array of relative humidities (1).
            temperature (numpy.ndarray):
                Array of air temperature (K).

        Returns:
            numpy.ndarray:
                Array of wet bulb temperature (K).

        """
        # Initialise psychrometric variables
        latent_heat = self._calculate_latent_heat(temperature)
        saturation_mixing_ratio = self._calculate_mixing_ratio(
            temperature, pressure)
        mixing_ratio = relative_humidity * saturation_mixing_ratio
        specific_heat = self._calculate_specific_heat(mixing_ratio)
        enthalpy = self._calculate_enthalpy(
            mixing_ratio, specific_heat, latent_heat, temperature)

        # Initialise wet bulb temperature increment
        delta_wbt = 10. * np.broadcast_to(self.precision, temperature.shape)
        delta_wbt_prev = None

        # Iterate to find the wet bulb temperature, using temperature as first
        # guess
        wbt_data = temperature.copy()
        iteration = 0
        while (np.abs(delta_wbt) > self.precision).any():

            enthalpy_new = self._calculate_enthalpy(
                saturation_mixing_ratio, specific_heat, latent_heat, wbt_data)
            enthalpy_gradient = self._calculate_enthalpy_gradient(
                saturation_mixing_ratio, specific_heat, latent_heat, wbt_data)
            delta_wbt = (enthalpy - enthalpy_new) / enthalpy_gradient

            # Increment wet bulb temperature at points which have not converged
            to_update = np.where(np.abs(delta_wbt) > self.precision)
            wbt_data[to_update] = (wbt_data[to_update] + delta_wbt[to_update])

            # If increment is identical to the previous iteration, stop
            if (np.array_equal(delta_wbt, delta_wbt_prev) or
                    iteration > self.maximum_iterations):
                warnings.warn('No further refinement occurring; breaking out '
                              'of Newton iterator and returning result.')
                break

            # Update saturation mixing ratio and iterators
            saturation_mixing_ratio = self._calculate_mixing_ratio(
                wbt_data, pressure)

            delta_wbt_prev = delta_wbt
            iteration += 1

        return wbt_data

    def create_wet_bulb_temperature_cube(
            self, temperature, relative_humidity, pressure):
        """
        Creates a cube of wet bulb temperature values

        Args:
            temperature (iris.cube.Cube):
                Cube of air temperatures.
            relative_humidity (iris.cube.Cube):
                Cube of relative humidities.
            pressure (iris.cube.Cube):
                Cube of air pressures.

        Returns:
            iris.cube.Cube:
                Cube of wet bulb temperature (K).

        """
        temperature.convert_units('K')
        relative_humidity.convert_units(1)
        pressure.convert_units('Pa')
        wbt_data = self._calculate_wet_bulb_temperature(
            pressure.data, relative_humidity.data, temperature.data)

        attributes = generate_mandatory_attributes(
            [temperature, relative_humidity, pressure])
        wbt = create_new_diagnostic_cube(
            'wet_bulb_temperature', 'K', temperature, attributes,
            data=wbt_data)
        return wbt

    def process(self, cubes):
        """
        Call the calculate_wet_bulb_temperature function to calculate wet bulb
        temperatures. This process function splits input cubes over vertical
        levels to mitigate memory issues when trying to operate on multi-level
        data.

        Args:
            cubes (iris.cube.CubeList or list or iris.cube.Cube):
                containing:
                    temperature (iris.cube.Cube):
                        Cube of air temperatures.
                    relative_humidity (iris.cube.Cube):
                        Cube of relative humidities.
                    pressure (iris.cube.Cube):
                        Cube of air pressures.

        Returns:
            iris.cube.Cube:
                Cube of wet bulb temperature (K).
        """
        names_to_extract = ["air_temperature",
                            "relative_humidity",
                            "air_pressure"]
        if len(cubes) != len(names_to_extract):
            raise ValueError(
                f'Expected {len(names_to_extract)} cubes, found {len(cubes)}')

        temperature, relative_humidity, pressure = tuple(
            CubeList(cubes).extract_strict(n) for n in names_to_extract)

        # temperature, relative_humidity, pressure = self._extract_cubes(
        #     cubes, ["air_temperature", "relative_humidity", "air_pressure"])

        slices = self._slice_inputs(temperature, relative_humidity, pressure)

        cubelist = iris.cube.CubeList([])
        for t_slice, rh_slice, p_slice in slices:
            cubelist.append(self.create_wet_bulb_temperature_cube(
                t_slice.copy(), rh_slice.copy(), p_slice.copy()))
        wet_bulb_temperature = cubelist.merge_cube()

        # re-promote any scalar coordinates lost in slice / merge
        wet_bulb_temperature = check_cube_coordinates(
            temperature, wet_bulb_temperature)

        return wet_bulb_temperature


class WetBulbTemperatureIntegral(BasePlugin):
    """Calculate a wet-bulb temperature integral."""

    def __init__(self):
        """Initialise class."""
        self.integration_plugin = Integration("height")

    def process(self, wet_bulb_temperature):
        """
        Calculate the vertical integral of wet bulb temperature from the input
        wet bulb temperatures on height levels.

        Args:
            wet_bulb_temperature (iris.cube.Cube):
                Cube of wet bulb temperatures on height levels.

        Returns:
            wet_bulb_temperature_integral (iris.cube.Cube):
                Cube of wet bulb temperature integral (Kelvin-metres).
        """
        wbt = wet_bulb_temperature.copy()
        wbt.convert_units('degC')
        wbt.coord('height').convert_units('m')
        # Touch the data to ensure it is not lazy
        # otherwise vertical interpolation is slow
        # pylint: disable=pointless-statement
        wbt.data
        wet_bulb_temperature_integral = self.integration_plugin.process(wbt)
        # although the integral is computed over degC the standard unit is
        # 'K m', and these are equivalent
        wet_bulb_temperature_integral.units = Unit('K m')
        return wet_bulb_temperature_integral


class PhaseChangeLevel(BasePlugin):
    """Calculate a continuous field of heights relative to sea level at which
    a phase change of precipitation is expected."""

    def __init__(self, phase_change, grid_point_radius=2):
        """
        Initialise class.

        Args:
            phase_change (str):
                The desired phase change for which the altitude should be
                returned. Options are:

                    snow-sleet - the melting of snow to sleet.
                    sleet-rain - the melting of sleet to rain.

            grid_point_radius (int):
                The radius in grid points used to calculate the maximum
                height of the orography in a neighbourhood as part of this
                calculation.
        """
        phase_changes = {'snow-sleet': {'threshold': 90.,
                                        'name': 'snow_falling'},
                         'sleet-rain': {'threshold': 202.5,
                                        'name': 'rain_falling'}}
        try:
            phase_change_def = phase_changes[phase_change]
        except KeyError:
            msg = ("Unknown phase change '{}' requested.\nAvailable options "
                   "are: {}".format(phase_change,
                                    ', '.join(phase_changes.keys())))
            raise ValueError(msg)

        self.falling_level_threshold = phase_change_def['threshold']
        self.phase_change_name = phase_change_def['name']
        self.missing_data = -300.0
        self.grid_point_radius = grid_point_radius

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<PhaseChangeLevel: falling_level_threshold:{}, '
                  'grid_point_radius: {}>'.format(
                      self.falling_level_threshold,
                      self.grid_point_radius))
        return result

    def find_falling_level(self, wb_int_data, orog_data, height_points):
        """
        Find the phase change level by finding the level of the wet-bulb
        integral data at the required threshold. Wet-bulb integral data
        is only available above ground level and there may be an insufficient
        number of levels in the input data, in which case the required
        threshold may lie outside the Wet-bulb integral data and the value
        at that point will be set to np.nan.

        Args:
            wb_int_data (numpy.ndarray):
                Wet bulb integral data on heights
            orog_data (numpy.ndarray):
                Orographic data
            height_points (numpy.ndarray):
                heights agl

        Returns:
            numpy.ndarray:
                Phase change level data asl.

        """
        # Create cube of heights above sea level for each height in
        # the wet bulb integral cube.
        asl = wb_int_data.copy()
        for i, height in enumerate(height_points):
            asl[i, ::] = orog_data + height

        # Calculate phase change level above sea level by
        # finding the level corresponding to the falling_level_threshold.
        # Interpolate returns an array with height indices
        # for falling_level_threshold so we take the 0 index
        phase_change_level_data = interpolate(
            np.array([self.falling_level_threshold]),
            wb_int_data, asl, axis=0)[0]

        return phase_change_level_data

    def fill_in_high_phase_change_falling_levels(
            self, phase_change_level_data, orog_data, highest_wb_int_data,
            highest_height):
        """
        Fill in any data in the phase change level where the whole wet bulb
        temperature integral is above the the threshold.
        Set these points to the highest height level + orography.

        Args:
            phase_change_level_data (numpy.ndarray):
                Phase change level data (m).
            orog_data (numpy.ndarray):
                Orographic data (m)
            highest_wb_int_data (numpy.ndarray):
                Wet bulb integral data on highest level (K m).
            highest_height (float):
                Highest height at which the integral starts (m).
        """
        points_not_freezing = np.where(
            np.isnan(phase_change_level_data) &
            (highest_wb_int_data > self.falling_level_threshold))
        phase_change_level_data[points_not_freezing] = (
            highest_height + orog_data[points_not_freezing])

    def find_extrapolated_falling_level(self, max_wb_integral, gradient,
                                        intercept, phase_change_level_data,
                                        sea_points):
        r"""
        Find the phase change level below sea level using the linear
        extrapolation of the wet bulb temperature integral and update the
        phase change level array with these values.


        The phase change level is calculated from finding the point where the
        integral of wet bulb temperature crosses the falling level threshold.

        In cases where the wet bulb temperature integral has not reached the
        threshold by the time we reach sea level, we can find a fit to the wet
        bulb temperature profile near the surface, and use this to estimate
        where the phase change level would be below sea level.

        The difference between the wet bulb temperature integral at the
        threshold and the wet bulb integral at the surface is equal to the
        integral of the wet bulb temperature between sea level and
        the negative height corresponding to the phase change level. As we are
        using a simple linear fit, we can integrate this to find an expression
        for the extrapolated phase change level.

        The form of this expression depends on whether the linear fit of wet
        bulb temperature crosses the height axis above or below zero altitude.

        If we have our linear fit of the form:

        .. math::
            {{wet\:bulb\:temperature} = m \times height + c}

        and let :math:`I` be the wet bulb temperature integral we have found
        above sea level.

        If it crosses above zero, then the limits on the integral
        are the phase change level and zero and we find the following
        expression for the phase change level:

        .. math::
            {{phase\:change\:level} = \frac{c \pm \sqrt{
            c^2-2 m (threshold-I)}}{-m}}

        If the linear fit crosses below zero the limits on our integral are
        the phase change level and the point where the linear fit crosses the
        height axis, as only positive wet bulb temperatures count towards the
        integral. In this case our expression for the phase change level is:

        .. math::
            {{phase\:change\:level} = \frac{c \pm \sqrt{
            2 m (I-threshold)}}{-m}}

        Args:
            max_wb_integral (numpy.ndarray):
                The wet bulb temperature integral at sea level.
            gradient (numpy.ndarray):
                The gradient of the line of best fit we are using in the
                extrapolation.
            intercept (numpy.ndarray):
                The intercept of the line of best fit we are using in the
                extrapolation.
            phase_change_level_data (numpy.ndarray):
                The phase change level array with values filled in with phase
                change levels calculated through extrapolation.
            sea_points (numpy.ndarray):
                A boolean array with True where the points are sea points.

        """

        # Make sure we only try to extrapolate points with a valid gradient.
        index = (gradient < 0.0) & sea_points
        gradient = gradient[index]
        intercept = intercept[index]
        max_wb_int = max_wb_integral[index]
        phase_cl = phase_change_level_data[index]

        # For points where -intercept/gradient is greater than zero:
        index2 = (-intercept/gradient >= 0.0)
        inside_sqrt = (
            intercept[index2]**2 - 2*gradient[index2]*(
                self.falling_level_threshold - max_wb_int[index2]))
        phase_cl[index2] = (
            (intercept[index2] - np.sqrt(inside_sqrt))/-gradient[index2])

        # For points where -intercept/gradient is less than zero:
        index2 = (-intercept/gradient < 0.0)
        inside_sqrt = (
            2*gradient[index2]*(
                max_wb_int[index2] - self.falling_level_threshold))
        phase_cl[index2] = (
            (intercept[index2] - np.sqrt(inside_sqrt))/-gradient[index2])
        # Update the phase change level. Clip to ignore extremely negative
        # phase change levels.
        phase_cl = np.clip(phase_cl, -2000, np.inf)
        phase_change_level_data[index] = phase_cl

    @staticmethod
    def linear_wet_bulb_fit(wet_bulb_temperature, heights, sea_points,
                            start_point=0, end_point=5):
        """
        Calculates a linear fit to the wet bulb temperature profile close
        to the surface to use when we extrapolate the wet bulb temperature
        below sea level for sea points.

        We only use a set number of points close to the surface for this fit,
        specified by a start_point and end_point.

        Args:
            wet_bulb_temperature (numpy.ndarray):
                The wet bulb temperature profile at each grid point, with
                height as the leading dimension.
            heights (numpy.ndarray):
                The vertical height levels above orography, matching the
                leading dimension of the wet_bulb_temperature.
            sea_points (numpy.ndarray):
                A boolean array with True where the points are sea points.
            start_point (int):
                The index of the the starting height we want to use in our
                linear fit.
            end_point (int):
                The index of the the end height we want to use in our
                linear fit.

        Returns:
            (tuple): tuple containing:
                **gradient** (numpy.ndarray) - An array, the same shape as a
                2D slice of the wet_bulb_temperature input, containing the
                gradients of the fitted straight line at each point where it
                could be found, filled with zeros elsewhere.

                **intercept** (numpy.ndarray) - An array, the same shape as a
                2D slice of the wet_bulb_temperature input, containing the
                intercepts of the fitted straight line at each point where it
                could be found, filled with zeros elsewhere.

        """
        def fitting_function(wet_bulb_temps):
            """
            A small helper function used to find a linear fit of the
            wet bulb temperature.
            """
            return linregress(
                heights[start_point:end_point],
                wet_bulb_temps[start_point:end_point])
        # Set up empty arrays for gradient and intercept
        gradient = np.zeros(wet_bulb_temperature[0].shape)
        intercept = np.zeros(wet_bulb_temperature[0].shape)
        if np.any(sea_points):
            # Make the 1D sea point array 3D to account for the height axis
            # on the wet bulb temperature array.
            index3d = np.broadcast_to(sea_points, wet_bulb_temperature.shape)
            # Flatten the array to make it more efficient to find a linear fit
            # for every point of interest. We can apply the fitting function
            # along the right axis to apply it to all points in one go.
            wet_bulb_temperature_values = (
                wet_bulb_temperature[index3d].reshape(len(heights), -1))
            gradient_values, intercept_values, _, _, _, = (
                np.apply_along_axis(
                    fitting_function, 0, wet_bulb_temperature_values))
            # Fill in the right gradients and intercepts in the 2D array.
            gradient[sea_points] = gradient_values
            intercept[sea_points] = intercept_values
        return gradient, intercept

    def fill_in_sea_points(
            self, phase_change_level_data, land_sea_data, max_wb_integral,
            wet_bulb_temperature, heights):
        """
        Fill in any sea points where we have not found a phase change level
        by the time we get to sea level, i.e. where the whole wet bulb
        temperature integral is below the threshold.

        This function finds a linear fit to the wet bulb temperature close to
        sea level and uses this to find where an extrapolated wet bulb
        temperature integral would cross the threshold. This results in
        phase change levels below sea level for points where we have applied
        the extrapolation.

        Assumes that height is the first axis in the wet_bulb_integral array.

        Args:
            phase_change_level_data(numpy.ndarray):
                The phase change level array, filled with values for points
                whose wet bulb temperature integral crossed the theshold.
            land_sea_data (numpy.ndarray):
                The binary land-sea mask
            max_wb_integral (numpy.ndarray):
                The wet bulb temperature integral at the final height level
                used in the integration. This has the maximum values for the
                wet bulb temperature integral at any level.
            wet_bulb_temperature (numpy.ndarray):
                The wet bulb temperature profile at each grid point, with
                height as the leading dimension.
            heights (numpy.ndarray):
                The vertical height levels above orography, matching the
                leading dimension of the wet_bulb_temperature.

        """
        sea_points = (
            np.isnan(phase_change_level_data) & (land_sea_data < 1.0) &
            (max_wb_integral < self.falling_level_threshold))
        if np.all(sea_points is False):
            return

        gradient, intercept = self.linear_wet_bulb_fit(wet_bulb_temperature,
                                                       heights, sea_points)

        self.find_extrapolated_falling_level(
            max_wb_integral, gradient, intercept, phase_change_level_data,
            sea_points)

    @staticmethod
    def fill_in_by_horizontal_interpolation(
            phase_change_level_data, max_in_nbhood_orog, orog_data):
        """
        Fill in any remaining unset areas in the phase change level by using
        linear horizontal interpolation across the grid. As phase change levels
        at the highest height levels will be filled in by this point any
        points that still don't have a valid phase change level have the phase
        change level at or below the surface orography.
        This function uses the following steps to help ensure that the filled
        in values are above or below the orography:

        1. Fill in the phase change level for points with no value yet
           set using horizontal interpolation from surrounding set points.
           Only interpolate from surrounding set points at which the phase
           change level is below the maximum orography height in the region
           around the unset point. This helps us avoid spreading very high
           phase change levels across areas where we had missing data.
        2. Fill any gaps that still remain where the linear interpolation has
           not been able to find a value because there is not enough
           data (e.g at the corners of the domain). Use nearest neighbour
           interpolation.
        3. Check whether despite our efforts we have still filled in some
           of the missing points with phase change levels above the orography.
           In these cases set the missing points to the height of orography.

        We then return the filled in array, which hopefully has no more
        missing data.

        Args:
            phase_change_level_data (numpy.ndarray):
                The phase change level array, filled with values for points
                whose wet bulb temperature integral crossed the theshold.
            max_in_nbhood_orog (numpy.ndarray):
                The array containing maximum of the orography field in
                a given radius.
            orog_data(numpy.data):
                The array containing the orography data.
        Returns:
            numpy.ndarray:
                The phase change level array with missing data filled by
                horizontal interpolation.
        """
        # Interpolate linearly across the remaining points
        index = ~np.isnan(phase_change_level_data)
        index_valid_data = (
            phase_change_level_data[index] <= max_in_nbhood_orog[index])
        index[index] = index_valid_data
        phase_cl_filled = phase_change_level_data
        if np.any(index):
            ynum, xnum = phase_change_level_data.shape
            (y_points, x_points) = np.mgrid[0:ynum, 0:xnum]
            values = phase_change_level_data[index]
            # Try to do the horizontal interpolation to fill in any gaps,
            # but if there are not enough points or the points are not arranged
            # in a way that allows the horizontal interpolation, skip
            # and use nearest neighbour intead.
            try:
                phase_change_level_data_updated = griddata(
                    np.where(index), values, (y_points, x_points),
                    method='linear')
            except QhullError:
                phase_change_level_data_updated = phase_change_level_data
            else:
                phase_cl_filled = phase_change_level_data_updated
            # Fill in any remaining missing points using nearest neighbour.
            # This normally only impact points at the corners of the domain,
            # where the linear fit doesn't reach.
            index = ~np.isnan(phase_cl_filled)
            index_valid_data = (
                phase_cl_filled[index] <= max_in_nbhood_orog[index])
            index[index] = index_valid_data
            if np.any(index):
                values = phase_change_level_data_updated[index]
                phase_change_level_data_updated_2 = griddata(
                    np.where(index), values, (y_points, x_points),
                    method='nearest')
                phase_cl_filled = phase_change_level_data_updated_2

        # Set the phase change level at any points that have been filled with
        # phase change levels that are above the orography back to the
        # height of the orography.
        index = (~np.isfinite(phase_change_level_data))
        phase_cl_above_orog = (phase_cl_filled[index] > orog_data[index])
        index[index] = phase_cl_above_orog
        phase_cl_filled[index] = orog_data[index]
        return phase_cl_filled

    def find_max_in_nbhood_orography(self, orography_cube):
        """
        Find the maximum value of the orography in the region around each grid
        point in your orography field by finding the maximum in a neighbourhood
        around that point.

        Args:
            orography_cube (iris.cube.Cube):
                The cube containing a single 2 dimensional array of orography
                data
        Returns:
            iris.cube.Cube:
                The cube containing the maximum in a neighbourhood of the
                orography data.
        """
        radius_in_metres = convert_number_of_grid_cells_into_distance(
            orography_cube, self.grid_point_radius)
        max_in_nbhood_orog = OccurrenceWithinVicinity(
            radius_in_metres).process(orography_cube)
        return max_in_nbhood_orog

    def _calculate_phase_change_level(
            self, wet_bulb_temp, wb_integral, orography, max_nbhood_orog,
            land_sea_data, heights, height_points, highest_height):
        """
        Calculate phase change level and fill in missing points

        Args:
            wet_bulb_temp (numpy.ndarray):
                Wet bulb temperature data
            wb_integral (numpy.ndarray):
                Wet bulb temperature integral
            orography (numpy.ndarray):
                Orography heights
            max_nbhood_orog (numpy.ndarray):
                Maximum orography height in neighbourhood
            land_sea_data (numpy.ndarray):
                Mask of binary land / sea data
            heights (np.ndarray):
                All heights of wet bulb temperature input
            height_points (numpy.ndarray):
                Heights on wet bulb temperature integral slice
            highest_height (float):
                Height of the highest level to which the wet bulb
                temperature has been integrated

        Returns:
            np.ndarray:
                Level at which phase changes

        """
        phase_change_data = self.find_falling_level(
            wb_integral, orography, height_points)

        # Fill in missing data
        self.fill_in_high_phase_change_falling_levels(
            phase_change_data, orography,
            wb_integral.max(axis=0), highest_height)
        self.fill_in_sea_points(
            phase_change_data, land_sea_data,
            wb_integral.max(axis=0), wet_bulb_temp, heights)
        updated_phase_cl = self.fill_in_by_horizontal_interpolation(
            phase_change_data, max_nbhood_orog, orography)
        points = np.where(~np.isfinite(phase_change_data))
        phase_change_data[points] = updated_phase_cl[points]

        # Fill in any remaining points with "missing data" value
        remaining_points = np.where(np.isnan(phase_change_data))
        phase_change_data[remaining_points] = self.missing_data

        return phase_change_data

    def create_phase_change_level_cube(self, wbt, phase_change_level):
        """
        Populate output cube with phase change data

        Args:
            wbt (iris.cube.Cube):
                Wet bulb temperature cube on height levels
            phase_change_level (numpy.ndarray):
                Calculated phase change level in metres

        Returns:
            iris.cube.Cube
        """
        name = 'altitude_of_{}_level'.format(self.phase_change_name)
        attributes = generate_mandatory_attributes([wbt])
        template = next(wbt.slices_over(['height'])).copy()
        template.remove_coord('height')
        return create_new_diagnostic_cube(name, 'm', template, attributes,
                                          data=phase_change_level)

    def process(self, cubes):
        """
        Use the wet bulb temperature integral to find the altitude at which a
        phase change occurs (e.g. snow to sleet). This is achieved by finding
        the height above sea level at which the integral matches an empirical
        threshold that is expected to correspond with the phase change. This
        empirical threshold is the falling_level_threshold. Fill in missing
        data appropriately.

        Args:
        cubes (iris.cube.CubeList or list or iris.cube.Cube):
            containing:
                wet_bulb_temperature (iris.cube.Cube):
                    Cube of wet bulb temperatures on height levels.
                wet_bulb_integral (iris.cube.Cube):
                    Cube of wet bulb temperature integral (Kelvin-metres).
                orog (iris.cube.Cube):
                    Cube of orography (m).
                land_sea_mask (iris.cube.Cube):
                    Cube containing a binary land-sea mask.

        Returns:
            iris.cube.Cube:
                Cube of phase change level above sea level (asl).
        """

        names_to_extract = ["wet_bulb_temperature",
                            "wet_bulb_temperature_integral",
                            "surface_altitude",
                            "land_binary_mask"]
        if len(cubes) != len(names_to_extract):
            raise ValueError(
                f'Expected {len(names_to_extract)} cubes, found {len(cubes)}')

        wet_bulb_temperature, wet_bulb_integral, orog, land_sea_mask = tuple(
            CubeList(cubes).extract_strict(n) for n in names_to_extract)

        wet_bulb_temperature.convert_units('celsius')
        wet_bulb_integral.convert_units('K m')

        # Ensure the wet bulb integral cube's height coordinate is in
        # descending order
        wet_bulb_integral = sort_coord_in_cube(wet_bulb_integral, 'height',
                                               descending=True)

        # Find highest height from height bounds.
        wbt_height_points = wet_bulb_temperature.coord('height').points
        if wet_bulb_integral.coord('height').bounds is None:
            highest_height = wbt_height_points[-1]
        else:
            highest_height = wet_bulb_integral.coord('height').bounds[0][-1]

        # Firstly we need to slice over height, x and y
        x_coord = wet_bulb_integral.coord(axis='x').name()
        y_coord = wet_bulb_integral.coord(axis='y').name()
        orography = next(orog.slices([y_coord, x_coord]))
        land_sea_data = next(land_sea_mask.slices([y_coord, x_coord])).data
        max_nbhood_orog = self.find_max_in_nbhood_orography(orography)

        phase_change = None
        slice_list = ['height', y_coord, x_coord]
        for wb_integral, wet_bulb_temp in zip(
                wet_bulb_integral.slices(slice_list),
                wet_bulb_temperature.slices(slice_list)):
            phase_change_data = self._calculate_phase_change_level(
                wet_bulb_temp.data, wb_integral.data, orography.data,
                max_nbhood_orog.data, land_sea_data, wbt_height_points,
                wb_integral.coord('height').points, highest_height)

            # preserve dimensionality of input cube (in case of scalar or
            # length 1 dimensions)
            if phase_change is None:
                phase_change = phase_change_data
            elif not isinstance(phase_change, list):
                phase_change = [phase_change]
                phase_change.append(phase_change_data)
            else:
                phase_change.append(phase_change_data)

        phase_change_level = self.create_phase_change_level_cube(
            wet_bulb_temperature, np.array(phase_change, dtype=np.float32))

        return phase_change_level
