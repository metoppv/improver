# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
""" Utilites to do with day and night."""

import numpy as np
import math


def solar_declination(day_of_year):
    """
    Calculate the Declination for the day of the year.

    Args:
        day_of_year (int):
            Day of the year

    Returns:
        declination (float):
            Declination in degrees.North-South
    """
    # Declination (degrees):
    declination = -23.5 * math.cos(math.radians(0.9856 * day_of_year + 9.3))
    return declination


def solar_hour_angle(longitudes, day_of_year, utc_hour):
    """
    Calculate the Solar Hour angle for an array of longitudes.

    Args:
        longitudes (float or numpy.array):
            A single Longitude or array of Longitudes
            longitudes needs to be between 180.0 and -180.0
        day_of_year (int):
            Day of the year
        utc_hour (float):
            Hour of the day in UTC

    Returns:
        solar_hour_angle (numpy.array)
            Hour angle in degrees East-West
    """
    thetao = 2*math.pi*day_of_year/365.0
    eqt = (0.000075 + 0.001868 * math.cos(thetao) -
           0.032077 * math.sin(thetao) - 0.014615 * math.cos(2*thetao) -
           0.040849 * math.sin(2*thetao))

    # Longitudinal Correction from the Grenwich Meridian
    lon_correction = 24.0*longitudes/360.0
    # Solar time (hours):
    solar_time = utc_hour + lon_correction + eqt*12/math.pi
    # Hour angle (degrees):
    solar_hour_angle = (solar_time - 12.0) * 15.0

    return solar_hour_angle


def solar_elevation(latitudes, longitudes, day_of_year, utc_hour):
    """
    Calculate the Solar elevation.

    Args:
        latitude (float or numpy.array):
            A single Latitudes or array of Latitudes
            longitudes needs to be between -90.0 and 90.0
        longitude (float or numpy.array):
            A single Longitude or array of Longitudes
            longitudes needs to be between 180.0 and -180.0
        day_of_year (int):
            Day of the year
        utc_hour (float):
            Hour of the day in UTC

    Returns:
        solar_elevation (numpy.array):
            Solar elevation in degrees
    """
    declination = solar_declination(day_of_year)
    decl = math.radians(declination)
    hour_angle = solar_hour_angle(longitudes, day_of_year, utc_hour)
    rad_hours = np.radians(hour_angle)
    lats = np.radians(latitudes)
    # Calculate solar position:

    solar_elevation = (np.arcsin(np.sin(decl) * np.sin(lats) +
                                 np.cos(decl) * np.cos(lats) *
                                 np.cos(rad_hours)))

    solar_elevation = np.degrees(solar_elevation)

    return solar_elevation


def daynight_terminator(longitudes, day_of_year, utc_hour):
    """
    Calculate the Latitude values of the daynight terminator

    Args:
        longitudes (numpy.array):
            Array of longitudes
        day_of_year (int):
            Day of the year
        utc_hour (float):
            Hour of the day in UTC

    Returns:
        latitudes (numpy.array):
            latitudes of the daynight terminator
    """
    declination = solar_declination(day_of_year)
    decl = math.radians(declination)
    hour_angle = solar_hour_angle(longitudes, day_of_year, utc_hour)
    rad_hour = np.radians(hour_angle)
    lats = np.arctan(-np.cos(rad_hour)/np.tan(decl))
    lats = np.degrees(lats)
    return lats


def daynight_mask_slow(cube):
    """
    Calculate the daynight mask for the provided cube

    Args:
        cube (iris.cube.Cube):
            input cube

    Returns:
        daynight_mask (iris.cube.Cube):
            daynight mask cube
    """
    return cube
