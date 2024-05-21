# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to contain Psychrometric Calculations."""

import functools
from typing import List, Tuple, Union

import numpy as np
from iris.cube import Cube, CubeList
from numpy import ndarray
from scipy.optimize import newton

import improver.constants as consts
from improver import BasePlugin
from improver.generate_ancillaries.generate_svp_table import (
    SaturatedVapourPressureTable,
)
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.cube_manipulation import sort_coord_in_cube
from improver.utilities.interpolation import interpolate_missing_data
from improver.utilities.mathematical_operations import fast_linear_fit
from improver.utilities.spatial import OccurrenceWithinVicinity

SVP_T_MIN = 183.15
SVP_T_MAX = 338.25
SVP_T_INCREMENT = 0.1


@functools.lru_cache()
def _svp_table() -> ndarray:
    """
    Calculate a saturated vapour pressure (SVP) lookup table.
    The lru_cache decorator caches this table on first call to this function,
    so that the table does not need to be re-calculated if used multiple times.

    A value of SVP for any temperature between T_MIN and T_MAX (inclusive) can be
    obtained by interpolating through the table, as is done in the _svp_from_lookup
    function.

    Returns:
        Array of saturated vapour pressures (Pa).
    """
    svp_data = SaturatedVapourPressureTable(
        t_min=SVP_T_MIN, t_max=SVP_T_MAX, t_increment=SVP_T_INCREMENT
    ).process()
    return svp_data.data


def _svp_from_lookup(temperature: ndarray) -> ndarray:
    """
    Gets value for saturation vapour pressure in a pure water vapour system
    from a pre-calculated lookup table. Interpolates linearly between points in
    the table to the temperatures required.

    Args:
        temperature:
            Array of air temperatures (K).

    Returns:
        Array of saturated vapour pressures (Pa).
    """
    # where temperatures are outside the SVP table range, clip data to
    # within the available range
    t_clipped = np.clip(temperature, SVP_T_MIN, SVP_T_MAX - SVP_T_INCREMENT)

    # interpolate between bracketing values
    table_position = (t_clipped - SVP_T_MIN) / SVP_T_INCREMENT
    table_index = table_position.astype(int)
    interpolation_factor = table_position - table_index
    svp_table_data = _svp_table()
    return (1.0 - interpolation_factor) * svp_table_data[
        table_index
    ] + interpolation_factor * svp_table_data[table_index + 1]


def calculate_svp_in_air(temperature: ndarray, pressure: ndarray) -> ndarray:
    """
    Calculates the saturation vapour pressure in air.  Looks up the saturation
    vapour pressure in a pure water vapour system, and pressure-corrects the
    result to obtain the saturation vapour pressure in air.

    Args:
        temperature:
            Array of air temperatures (K).
        pressure:
            Array of pressure (Pa).

    Returns:
        Saturation vapour pressure in air (Pa).

    References:
        Atmosphere-Ocean Dynamics, Adrian E. Gill, International Geophysics
        Series, Vol. 30; Equation A4.7.
    """
    svp = _svp_from_lookup(temperature)
    temp_C = temperature + consts.ABSOLUTE_ZERO
    correction = 1.0 + 1.0e-8 * pressure * (4.5 + 6.0e-4 * temp_C * temp_C)
    return svp * correction.astype(np.float32)


def dry_adiabatic_temperature(
    initial_temperature: ndarray, initial_pressure: ndarray, final_pressure: ndarray,
) -> ndarray:
    """
    Calculate temperature at final_pressure after adiabatic adjustment of dry air from the
    initial temperature and pressure.

    .. See the documentation for a more detailed discussion of the steps.
    .. include:: extended_documentation/psychrometric_calculations/
       psychrometric_calculations/dry_adiabatic_temperature.rst

    Args:
        initial_temperature:
            Array of initial temperatures (K)
        initial_pressure:
            Array of initial pressures (Pa)
        final_pressure:
            Array of final pressures (Pa)

    Returns:
        Array of final temperatures (K)
    """
    return initial_temperature * (final_pressure / initial_pressure) ** (
        consts.R_DRY_AIR / consts.CP_DRY_AIR
    )


def dry_adiabatic_pressure(
    initial_temperature: ndarray, initial_pressure: ndarray, final_temperature: ndarray,
) -> ndarray:
    """
    Calculate pressure at final_temperature after adiabatic adjustment of dry air from the
    initial temperature and pressure.

    .. See the documentation for a more detailed discussion of the steps.
    .. include:: extended_documentation/psychrometric_calculations/
       psychrometric_calculations/dry_adiabatic_pressure.rst

    Args:
        initial_temperature:
            Array of initial temperatures (K)
        initial_pressure:
            Array of initial pressures (Pa)
        final_temperature:
            Array of final temperatures (K)

    Returns:
        Array of final pressures (Pa)
    """
    return initial_pressure * (final_temperature / initial_temperature) ** (
        consts.CP_DRY_AIR / consts.R_DRY_AIR
    )


def saturated_humidity(temperature: ndarray, pressure: ndarray) -> ndarray:
    """
    Calculate specific humidity mixing ratio of saturated air of given temperature and pressure

    Args:
        temperature:
            Air temperature (K)
        pressure:
            Air pressure (Pa)

    Returns:
        Array of specific humidity values (kg kg-1) representing saturated air

    Method from referenced documentation. Note that EARTH_REPSILON is
    simply given as an unnamed constant in the reference (0.62198).

    References:
        ASHRAE Fundamentals handbook (2005) Equation 22, 24, p6.8
    """
    svp = calculate_svp_in_air(temperature, pressure)
    numerator = consts.EARTH_REPSILON * svp
    denominator = np.maximum(svp, pressure) - ((1.0 - consts.EARTH_REPSILON) * svp)
    return (numerator / denominator).astype(temperature.dtype)


def _calculate_latent_heat(temperature: ndarray) -> ndarray:
    """
    Calculate a temperature adjusted latent heat of condensation for water
    vapour using the relationship employed by the UM.

    Args:
        temperature:
            Array of air temperatures (K).

    Returns:
        Temperature adjusted latent heat of condensation (J kg-1).
    """
    temp_Celsius = temperature + consts.ABSOLUTE_ZERO
    latent_heat = (
        -1.0 * consts.LATENT_HEAT_T_DEPENDENCE * temp_Celsius
        + consts.LH_CONDENSATION_WATER
    )
    return latent_heat


def _latent_heat_release(q1: ndarray, q2: ndarray, temperature: ndarray) -> ndarray:
    """Returns the latent heat released (K) when condensing water vapour from specific humidity
    value q1 to q2, both in kg kg-1 when the temperature is approximately t.
    Returns negative values when initial condition is subsaturated as this method assumes there
    is always liquid water present which can be evaporated.

    Args:
        temperature:
            Array of air temperatures (K). Ideally, the average air temperature between q1 and q2
        q1:
            Specific humidity before latent heat release (kg kg-1)
        q2:
            Specific humidity after latent heat release (kg kg-1)

    Returns:
        Temperature adjustment to apply to account for latent heat release (K).
    """
    return (_calculate_latent_heat(temperature) / consts.CP_DRY_AIR) * (q1 - q2)


def adjust_for_latent_heat(
    temperature_in: ndarray, humidity_in: ndarray, pressure: ndarray
) -> Tuple[ndarray, ndarray]:
    """
    Increases temperature and reduces humidity via latent heat release from condensation until
    values represent 100% relative humidity.

    Subsaturated values will be returned unaltered.

    This method uses the scipy newton solver with a limit of 6 iterations.
    The deepest convection needs more iterations to converge. This is only important
    if we reach the position that all points in an array fail to converge at the same
    pressure level, because the solver raises an exception (although docs say it shouldn't).
    I haven't seen this be true except when the array contains only one point, but increasing
    the maximum number of iterations for small arrays is a very small price to pay for
    stability.


    .. See the documentation for a more detailed discussion of the maths.
    .. include:: extended_documentation/psychrometric_calculations/
       psychrometric_calculations/adjust_for_latent_heat.rst

    Args:
        temperature_in:
            The parcel temperature following a dry adiabatic cooling (K)
        humidity_in:
            The atmosphere specific humidity at the same points (kg kg-1)
        pressure:
            The atmospheric pressure at the same points (Pa)

    Returns:
        tuple of temperature (K) and humidity (kg kg-1) after saturated latent heat release
    """

    def qsat_differential(qs, t, q, p):
        """For a given set of temperature (t), specific humidity (q) and pressure (p),
        and a saturated humidity guess (qs), calculate an adjusted temperature after
        latent heat release and return the difference between the saturated humidity
        at that adjusted temperature and the guess."""
        adj_t = t + _latent_heat_release(q, qs, t)
        return saturated_humidity(adj_t, p) - qs

    humidity = newton(
        qsat_differential,
        humidity_in.copy(),
        args=(temperature_in, humidity_in, pressure),
        tol=1e-6,
        maxiter=6 if humidity_in.size > 100 else 10,
        disp=True,
    ).astype(np.float32)
    temperature = temperature_in + _latent_heat_release(
        humidity_in, humidity, temperature_in
    )
    sub_saturated = np.where(temperature < temperature_in)
    temperature[sub_saturated] = temperature_in[sub_saturated]
    humidity[sub_saturated] = humidity_in[sub_saturated]
    return temperature, humidity


class HumidityMixingRatio(BasePlugin):
    """Returns the humidity mass mixing ratio from temperature, pressure and relative humidity"""

    def __init__(self, model_id_attr: str = None):
        """
        Set up class

        Args:
            model_id_attr:
                Name of model ID attribute to be copied from source cubes to output cube
        """
        self.model_id_attr = model_id_attr
        self.model_id_value = None
        self.mandatory_attributes = None
        self.temperature, self.pressure, self.rel_humidity = None, None, None

    def _make_humidity_cube(self, data: np.ndarray) -> Cube:
        """Puts the data array into a CF-compliant cube"""
        attributes = {}
        if self.model_id_attr:
            attributes[self.model_id_attr] = self.rel_humidity.attributes[
                self.model_id_attr
            ]
        cube = create_new_diagnostic_cube(
            "humidity_mixing_ratio",
            "kg kg-1",
            self.rel_humidity,
            mandatory_attributes=self.mandatory_attributes,
            optional_attributes=attributes,
            data=data,
        )
        return cube

    def process(self, cubes: List[Cube]) -> Cube:
        """
        Calculates the humidity mixing ratio from the inputs.

        Args:
            cubes:
                Cubes, in this order, of temperature (K), pressure (Pa) and relative humidity (1)

        Returns:
            Cube of humidity mixing ratio

        """
        self.mandatory_attributes = generate_mandatory_attributes(cubes)
        self.temperature, self.pressure, self.rel_humidity = cubes
        humidity = (
            saturated_humidity(self.temperature.data, self.pressure.data)
            * self.rel_humidity.data
        )
        return self._make_humidity_cube(humidity)


class PhaseChangeLevel(BasePlugin):
    """Calculate a continuous field of heights relative to sea level at which
    a phase change of precipitation is expected."""

    def __init__(
        self,
        phase_change: str,
        grid_point_radius: int = 2,
        horizontal_interpolation: bool = True,
        model_id_attr: str = None,
    ) -> None:
        """
        Initialise class.

        Args:
            phase_change:
                The desired phase change for which the altitude should be
                returned. Options are:

                    snow-sleet - the melting of snow to sleet.
                    sleet-rain - the melting of sleet to rain.
                    hail-rain - the melting of hail to rain.

            grid_point_radius:
                The radius in grid points used to calculate the maximum
                height of the orography in a neighbourhood to determine points that
                should be excluded from interpolation for being too close to the
                orographic feature where high-resolution models can give highly
                localised results. Zero uses central point only (neighbourhood is disabled).
                One uses central point and one in each direction. Two goes two points etc.
                A grid_point_radius may be specified for data on any projection
                but the effective kernel shape in real space may be irregular.
                Users must be aware of this when choosing whether to use a non-zero
                grid_point_radius with a non-equal areas projection
            horizontal_interpolation:
                If True apply horizontal interpolation to fill in holes in
                the returned phase-change-level that occur because the level
                falls below the orography. If False these areas will be masked.
            model_id_attr (str):
                Name of the attribute used to identify the source model for blending.
        """
        phase_changes = {
            "snow-sleet": {"threshold": 90.0, "name": "snow_falling"},
            "sleet-rain": {"threshold": 202.5, "name": "rain_falling"},
            "hail-rain": {"threshold": 5000, "name": "rain_from_hail_falling"},
        }
        try:
            phase_change_def = phase_changes[phase_change]
        except KeyError:
            msg = (
                "Unknown phase change '{}' requested.\nAvailable options "
                "are: {}".format(phase_change, ", ".join(phase_changes.keys()))
            )
            raise ValueError(msg)

        self.falling_level_threshold = phase_change_def["threshold"]
        self.phase_change_name = phase_change_def["name"]
        self.grid_point_radius = grid_point_radius
        self.horizontal_interpolation = horizontal_interpolation
        self.model_id_attr = model_id_attr

    def find_falling_level(
        self, wb_int_data: ndarray, orog_data: ndarray, height_points: ndarray
    ) -> ndarray:
        """
        Find the phase change level by finding the level of the wet-bulb
        integral data at the required threshold. Wet-bulb integral data
        is only available above ground level and there may be an insufficient
        number of levels in the input data, in which case the required
        threshold may lie outside the Wet-bulb integral data and the value
        at that point will be set to np.nan.

        Args:
            wb_int_data:
                Wet bulb integral data on heights
            orog_data:
                Orographic data
            height_points:
                heights agl

        Returns:
            Phase change level data asl.
        """
        from stratify import interpolate

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
            np.array([self.falling_level_threshold]), wb_int_data, asl, axis=0
        )[0]

        return phase_change_level_data

    def fill_in_high_phase_change_falling_levels(
        self,
        phase_change_level_data: ndarray,
        orog_data: ndarray,
        highest_wb_int_data: ndarray,
        highest_height: float,
    ) -> None:
        """
        Fill in any data in the phase change level where the whole wet bulb
        temperature integral is above the the threshold.
        Set these points to the highest height level + orography.

        Args:
            phase_change_level_data:
                Phase change level data (m).
            orog_data:
                Orographic data (m)
            highest_wb_int_data:
                Wet bulb integral data on highest level (K m).
            highest_height:
                Highest height at which the integral starts (m).
        """
        points_not_freezing = np.where(
            np.isnan(phase_change_level_data)
            & (highest_wb_int_data > self.falling_level_threshold)
        )
        phase_change_level_data[points_not_freezing] = (
            highest_height + orog_data[points_not_freezing]
        )

    def find_extrapolated_falling_level(
        self,
        max_wb_integral: ndarray,
        gradient: ndarray,
        intercept: ndarray,
        phase_change_level_data: ndarray,
        sea_points: ndarray,
    ) -> None:
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
            max_wb_integral:
                The wet bulb temperature integral at sea level.
            gradient:
                The gradient of the line of best fit we are using in the
                extrapolation.
            intercept:
                The intercept of the line of best fit we are using in the
                extrapolation.
            phase_change_level_data:
                The phase change level array with values filled in with phase
                change levels calculated through extrapolation.
            sea_points:
                A boolean array with True where the points are sea points.
        """

        # Make sure we only try to extrapolate points with a valid gradient.
        index = (gradient < 0.0) & sea_points
        gradient = gradient[index]
        intercept = intercept[index]
        max_wb_int = max_wb_integral[index]
        phase_cl = phase_change_level_data[index]

        # For points where -intercept/gradient is greater than zero:
        index2 = -intercept / gradient >= 0.0
        intercept2 = intercept[index2]
        gradient2 = gradient[index2]
        inside_sqrt = intercept2 * intercept2 - 2 * gradient2 * (
            self.falling_level_threshold - max_wb_int[index2]
        )
        phase_cl[index2] = (intercept2 - np.sqrt(inside_sqrt)) / -gradient2

        # For points where -intercept/gradient is less than zero:
        index2 = -intercept / gradient < 0.0
        intercept2 = intercept[index2]
        gradient2 = gradient[index2]
        inside_sqrt = (
            2 * gradient2 * (max_wb_int[index2] - self.falling_level_threshold)
        )
        phase_cl[index2] = (intercept2 - np.sqrt(inside_sqrt)) / -gradient2
        # Update the phase change level. Clip to ignore extremely negative
        # phase change levels.
        phase_cl = np.clip(phase_cl, -2000, np.inf)
        phase_change_level_data[index] = phase_cl

    @staticmethod
    def linear_wet_bulb_fit(
        wet_bulb_temperature: ndarray,
        heights: ndarray,
        sea_points: ndarray,
        start_point: int = 0,
        end_point: int = 5,
    ) -> Tuple[ndarray, ndarray]:
        """
        Calculates a linear fit to the wet bulb temperature profile close
        to the surface to use when we extrapolate the wet bulb temperature
        below sea level for sea points.

        We only use a set number of points close to the surface for this fit,
        specified by a start_point and end_point.

        Args:
            wet_bulb_temperature:
                The wet bulb temperature profile at each grid point, with
                height as the leading dimension.
            heights:
                The vertical height levels above orography, matching the
                leading dimension of the wet_bulb_temperature.
            sea_points:
                A boolean array with True where the points are sea points.
            start_point:
                The index of the the starting height we want to use in our
                linear fit.
            end_point:
                The index of the the end height we want to use in our
                linear fit.

        Returns:
            - An array, the same shape as a
              2D slice of the wet_bulb_temperature input, containing the
              gradients of the fitted straight line at each point where it
              could be found, filled with zeros elsewhere.
            - An array, the same shape as a
              2D slice of the wet_bulb_temperature input, containing the
              intercepts of the fitted straight line at each point where it
              could be found, filled with zeros elsewhere.
        """
        # Set up empty arrays for gradient and intercept
        result_shape = wet_bulb_temperature.shape[1:]
        gradient = np.zeros(result_shape)
        intercept = np.zeros(result_shape)
        if np.any(sea_points):
            # Use only subset of heights.
            wbt = wet_bulb_temperature[start_point:end_point, sea_points]
            hgt = heights[start_point:end_point].reshape(-1, 1)
            gradient_values, intercept_values = fast_linear_fit(hgt, wbt, axis=0)
            gradient[sea_points] = gradient_values
            intercept[sea_points] = intercept_values
        return gradient, intercept

    def fill_in_sea_points(
        self,
        phase_change_level_data: ndarray,
        land_sea_data: ndarray,
        max_wb_integral: ndarray,
        wet_bulb_temperature: ndarray,
        heights: ndarray,
        orography: ndarray,
    ) -> None:
        """
        Fill in any sea points where we have not found a phase change level
        by the time we get to sea level, i.e. where the whole wet bulb
        temperature integral is below the threshold.

        This function finds a linear fit to the wet bulb temperature close to
        sea level and uses this to find where an extrapolated wet bulb
        temperature integral would cross the threshold. This results in
        phase change levels below sea level for points where we have applied
        the extrapolation.

        The linear fit assumes that all sea points have an orography altitude of zero,
        however this is not always the case. The orography altitude is added to phase
        change levels calculated from the linear fit to account for this.

        Assumes that height is the first axis in the wet_bulb_integral array.

        Args:
            phase_change_level_data:
                The phase change level array, filled with values for points
                whose wet bulb temperature integral crossed the theshold.
            land_sea_data:
                The binary land-sea mask
            max_wb_integral:
                The wet bulb temperature integral at the final height level
                used in the integration. This has the maximum values for the
                wet bulb temperature integral at any level.
            wet_bulb_temperature:
                The wet bulb temperature profile at each grid point, with
                height as the leading dimension.
            heights:
                The vertical height levels above orography, matching the
                leading dimension of the wet_bulb_temperature.
            orography:
                Orography heights
        """
        sea_points = (
            np.isnan(phase_change_level_data)
            & (land_sea_data < 1.0)
            & (max_wb_integral < self.falling_level_threshold)
        )
        if np.all(sea_points is False):
            return

        gradient, intercept = self.linear_wet_bulb_fit(
            wet_bulb_temperature, heights, sea_points
        )

        self.find_extrapolated_falling_level(
            max_wb_integral, gradient, intercept, phase_change_level_data, sea_points
        )

        phase_change_level_data[sea_points] += orography[sea_points]

    def find_max_in_nbhood_orography(self, orography_cube: Cube) -> Cube:
        """
        Find the maximum value of the orography in the neighbourhood around
        each grid point. If self.grid_point_radius is zero, the orography is used
        without neighbourhooding.

        Args:
            orography_cube:
                The cube containing a single 2 dimensional array of orography
                data

        Returns:
            The cube containing the maximum in the grid_point_radius neighbourhood
            of the orography data or the orography data itself if the radius is zero
        """
        if self.grid_point_radius >= 1:
            max_in_nbhood_orog = OccurrenceWithinVicinity(
                grid_point_radii=[self.grid_point_radius]
            )(orography_cube)
            return max_in_nbhood_orog
        else:
            return orography_cube.copy()

    def _calculate_phase_change_level(
        self,
        wet_bulb_temp: ndarray,
        wb_integral: ndarray,
        orography: ndarray,
        max_nbhood_orog: ndarray,
        land_sea_data: ndarray,
        heights: ndarray,
        height_points: ndarray,
        highest_height: float,
    ) -> ndarray:
        """
        Calculate phase change level and fill in missing points

        .. See the documentation for a more detailed discussion of the steps.
        .. include:: extended_documentation/psychrometric_calculations/
           psychrometric_calculations/_calculate_phase_change_level.rst

        Args:
            wet_bulb_temp:
                Wet bulb temperature data
            wb_integral:
                Wet bulb temperature integral
            orography:
                Orography heights
            max_nbhood_orog:
                Maximum orography height in neighbourhood (used to determine points that
                can be used for interpolation)
            land_sea_data:
                Mask of binary land / sea data
            heights:
                All heights of wet bulb temperature input
            height_points:
                Heights on wet bulb temperature integral slice
            highest_height:
                Height of the highest level to which the wet bulb
                temperature has been integrated

        Returns:
            Level at which phase changes
        """
        phase_change_data = self.find_falling_level(
            wb_integral, orography, height_points
        )

        # Fill in missing data
        self.fill_in_high_phase_change_falling_levels(
            phase_change_data, orography, wb_integral.max(axis=0), highest_height
        )
        self.fill_in_sea_points(
            phase_change_data,
            land_sea_data,
            wb_integral.max(axis=0),
            wet_bulb_temp,
            heights,
            orography,
        )

        # Any unset points at this stage are set to np.nan; these will be
        # lands points where the phase-change-level is below the orography.
        # These can be filled by optional horizontal interpolation.
        if self.horizontal_interpolation:
            phase_change_data = self._horizontally_interpolate_phase(
                phase_change_data, orography, max_nbhood_orog
            )

        # Mask any points that are still set to np.nan; this should be no
        # points if horizontal interpolation has been used.
        phase_change_data = np.ma.masked_invalid(phase_change_data)

        return phase_change_data

    def _horizontally_interpolate_phase(
        self, phase_change_data: ndarray, orography: ndarray, max_nbhood_orog: ndarray
    ) -> ndarray:
        """
        Fill in missing points via horizontal interpolation.

        Args:
            phase_change_data:
                Level (height) at which the phase changes.
            orography:
                Orography heights
            max_nbhood_orog:
                Maximum orography height in neighbourhood (used to determine points that
                can be used for interpolation)

        Returns:
            Level at which phase changes, with missing data filled in
        """

        with np.errstate(invalid="ignore"):
            max_nbhood_mask = phase_change_data <= max_nbhood_orog
        updated_phase_cl = interpolate_missing_data(
            phase_change_data, limit=orography, valid_points=max_nbhood_mask
        )

        with np.errstate(invalid="ignore"):
            max_nbhood_mask = updated_phase_cl <= max_nbhood_orog
        phase_change_data = interpolate_missing_data(
            updated_phase_cl,
            method="nearest",
            limit=orography,
            valid_points=max_nbhood_mask,
        )

        if np.isnan(phase_change_data).any():
            # This should be rare.
            phase_change_data = interpolate_missing_data(
                phase_change_data, method="nearest", limit=orography,
            )
        return phase_change_data

    def create_phase_change_level_cube(
        self, wbt: Cube, phase_change_level: ndarray
    ) -> Cube:
        """
        Populate output cube with phase change data

        Args:
            wbt:
                Wet bulb temperature cube on height levels
            phase_change_level:
                Calculated phase change level in metres

        Returns:
            Cube with phase change data
        """
        name = "altitude_of_{}_level".format(self.phase_change_name)
        attributes = generate_mandatory_attributes(
            [wbt], model_id_attr=self.model_id_attr
        )
        template = next(wbt.slices_over(["height"])).copy()
        template.remove_coord("height")
        return create_new_diagnostic_cube(
            name, "m", template, attributes, data=phase_change_level
        )

    def process(self, cubes: Union[CubeList, List[Cube]]) -> Cube:
        """
        Use the wet bulb temperature integral to find the altitude at which a
        phase change occurs (e.g. snow to sleet). This is achieved by finding
        the height above sea level at which the integral matches an empirical
        threshold that is expected to correspond with the phase change. This
        empirical threshold is the falling_level_threshold. Fill in missing
        data appropriately.

        Args:
            cubes containing:
                wet_bulb_temperature:
                    Cube of wet bulb temperatures on height levels.
                wet_bulb_integral:
                    Cube of wet bulb temperature integral (Kelvin-metres).
                orog:
                    Cube of orography (m).
                land_sea_mask:
                    Cube containing a binary land-sea mask, with land points
                    set to one and sea points set to zero.

        Returns:
            Cube of phase change level above sea level (asl).

        Raises:
            ValueError: Raise exception if the model_id_attr attribute does not
                match on the input cubes.
        """

        names_to_extract = [
            "wet_bulb_temperature",
            "wet_bulb_temperature_integral",
            "surface_altitude",
            "land_binary_mask",
        ]
        if len(cubes) != len(names_to_extract):
            raise ValueError(
                f"Expected {len(names_to_extract)} cubes, found {len(cubes)}"
            )

        wet_bulb_temperature, wet_bulb_integral, orog, land_sea_mask = tuple(
            CubeList(cubes).extract_cube(n) for n in names_to_extract
        )

        wet_bulb_temperature.convert_units("celsius")
        wet_bulb_integral.convert_units("K m")

        if self.model_id_attr:
            if (
                wet_bulb_temperature.attributes[self.model_id_attr]
                != wet_bulb_integral.attributes[self.model_id_attr]
            ):
                raise ValueError(
                    f"Attribute {self.model_id_attr} does not match on input cubes. "
                    f"{wet_bulb_temperature.attributes[self.model_id_attr]} != "
                    f"{wet_bulb_integral.attributes[self.model_id_attr]}"
                )

        # Ensure the wet bulb integral cube's height coordinate is in
        # descending order
        wet_bulb_integral = sort_coord_in_cube(
            wet_bulb_integral, "height", descending=True
        )

        # Find highest height from height bounds.
        wbt_height_points = wet_bulb_temperature.coord("height").points
        if wet_bulb_integral.coord("height").bounds is None:
            highest_height = wbt_height_points[-1]
        else:
            highest_height = wet_bulb_integral.coord("height").bounds[0][-1]

        # Firstly we need to slice over height, x and y
        x_coord = wet_bulb_integral.coord(axis="x").name()
        y_coord = wet_bulb_integral.coord(axis="y").name()
        orography = next(orog.slices([y_coord, x_coord]))
        land_sea_data = next(land_sea_mask.slices([y_coord, x_coord])).data
        max_nbhood_orog = self.find_max_in_nbhood_orography(orography)

        phase_change = None
        slice_list = ["height", y_coord, x_coord]
        for wb_integral, wet_bulb_temp in zip(
            wet_bulb_integral.slices(slice_list),
            wet_bulb_temperature.slices(slice_list),
        ):
            phase_change_data = self._calculate_phase_change_level(
                wet_bulb_temp.data,
                wb_integral.data,
                orography.data,
                max_nbhood_orog.data,
                land_sea_data,
                wbt_height_points,
                wb_integral.coord("height").points,
                highest_height,
            )

            # preserve dimensionality of input cube (in case of scalar or
            # length 1 dimensions)
            if phase_change is None:
                phase_change = phase_change_data
            else:
                if not isinstance(phase_change, list):
                    phase_change = [phase_change]
                phase_change.append(phase_change_data)

        phase_change_level = self.create_phase_change_level_cube(
            wet_bulb_temperature, np.ma.masked_array(phase_change, dtype=np.float32)
        )

        return phase_change_level
