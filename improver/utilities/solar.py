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
""" Utilities to find the relative position of the sun."""
from datetime import datetime, timedelta
from typing import Union

import iris
import numpy as np
from iris.cube import Cube
from numpy import ndarray

from improver import BasePlugin
from improver.constants import DAYS_IN_YEAR, HOURS_IN_DAY, MINUTES_IN_HOUR
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.spatial import lat_lon_determine, transform_grid_to_lat_lon


def get_day_of_year(time: datetime) -> int:
    """Get day of the year from given datetime, starting at 0 for 1st Jan.

    Args:
        time:
            Datetime from which to extract day-of-year.

    Returns:
        The day of year corresponding to the specified datetime.
    """
    return time.timetuple().tm_yday - 1


def get_hour_of_day(time: datetime) -> float:
    """Get the hour of day from datetime, with the minute values included as
    a fraction of the hour, and seconds rounded to the nearest minute.

    Where rounding seconds moves the datetime into the following day, the
    hour_of_day returned is 24.0 rather than 0.0 to ensure the hour_of_day is
    consistent with the day_of_year value associated with datetime.

    Args:
        time:
            Datetime from which to extract utc-hour.

    Returns:
        The utc hour corresponding to the specified datetime. Minute values
        are expressed as fraction of an hour.
    """
    # Round times to nearest minute by adding seconds component to times
    rounded_time = time + timedelta(seconds=time.second)
    # Want to avoid the situation where rounding time takes us into the next day.
    if rounded_time.day != time.day:
        return float(HOURS_IN_DAY)
    else:
        return (
            rounded_time.hour * MINUTES_IN_HOUR + rounded_time.minute
        ) / MINUTES_IN_HOUR


def calc_solar_declination(day_of_year: int) -> float:
    """
    Calculate the Declination for the day of the year.

    Calculation equivalent to the calculation defined in
    NOAA Earth System Research Lab Low Accuracy Equations
    https://www.esrl.noaa.gov/gmd/grad/solcalc/sollinks.html

    Args:
        day_of_year:
            Day of the year 0 to 365, 0 = 1st January

    Returns:
        Declination in degrees.North-South
    """
    # Declination (degrees):
    # = -(axial_tilt)*cos(360./orbital_year * day_of_year - solstice_offset)
    if day_of_year < 0 or day_of_year > DAYS_IN_YEAR:
        msg = "Day of the year must be between 0 and 365"
        raise ValueError(msg)
    solar_declination = -23.5 * np.cos(np.radians(0.9856 * day_of_year + 9.3))
    return solar_declination


def calc_solar_time(
    longitudes: Union[float, ndarray],
    day_of_year: int,
    utc_hour: float,
    normalise=False,
) -> Union[float, ndarray]:
    """
    Calculate the Local Solar Time for each element of an array of longitudes.

    Calculation equivalent to the calculation defined in
    NOAA Earth System Research Lab Low Accuracy Equations
    https://www.esrl.noaa.gov/gmd/grad/solcalc/sollinks.html

    Args:
        longitudes:
            A single Longitude or array of Longitudes
            longitudes needs to be between 180.0 and -180.0 degrees
        day_of_year:
            Day of the year 0 to 365, 0 = 1st January
        utc_hour:
            Hour of the day in UTC
        normalise:
            Normalise hour values to be in range 0-24

    Returns:
        Local solar time in hours.

    """
    if day_of_year < 0 or day_of_year > DAYS_IN_YEAR:
        msg = "Day of the year must be between 0 and 365"
        raise ValueError(msg)
    if utc_hour < 0.0 or utc_hour > 24.0:
        msg = "Hour must be between 0 and 24.0"
        raise ValueError(msg)
    thetao = 2 * np.pi * day_of_year / DAYS_IN_YEAR
    eqt = (
        0.000075
        + 0.001868 * np.cos(thetao)
        - 0.032077 * np.sin(thetao)
        - 0.014615 * np.cos(2 * thetao)
        - 0.040849 * np.sin(2 * thetao)
    )

    # Longitudinal Correction from the Grenwich Meridian
    lon_correction = 24.0 * longitudes / 360.0
    # Solar time (hours):
    solar_time = utc_hour + lon_correction + eqt * 12 / np.pi
    # Normalise the hours to the range 0-24
    if normalise:
        solar_time = solar_time % 24

    return solar_time


def calc_solar_hour_angle(
    longitudes: Union[float, ndarray], day_of_year: int, utc_hour: float
) -> Union[float, ndarray]:
    """
    Calculate the Solar Hour angle from the Local Solar Time.

    Args:
        longitudes:
            A single Longitude or array of Longitudes
            longitudes needs to be between 180.0 and -180.0 degrees
        day_of_year:
            Day of the year 0 to 365, 0 = 1st January
        utc_hour:
            Hour of the day in UTC

    Returns:
        Hour angles in degrees East-West.
    """
    # Solar time (hours):
    solar_time = calc_solar_time(longitudes, day_of_year, utc_hour)
    # Hour angle (degrees):
    solar_hour_angle = (solar_time - 12.0) * 15.0

    return solar_hour_angle


def calc_solar_elevation(
    latitudes: Union[float, ndarray],
    longitudes: Union[float, ndarray],
    day_of_year: int,
    utc_hour: float,
    return_sine: bool = False,
) -> Union[float, ndarray]:
    """
    Calculate the Solar elevation.

    Args:
        latitudes:
            A single Latitude or array of Latitudes
            latitudes needs to be between -90.0 and 90.0
        longitudes:
            A single Longitude or array of Longitudes
            longitudes needs to be between 180.0 and -180.0
        day_of_year:
            Day of the year 0 to 365, 0 = 1st January
        utc_hour:
            Hour of the day in UTC in hours
        return_sine:
            If True return sine of solar elevation.
            Default False.

    Returns:
        Solar elevation in degrees for each location.
    """
    if np.min(latitudes) < -90.0 or np.max(latitudes) > 90.0:
        msg = "Latitudes must be between -90.0 and 90.0"
        raise ValueError(msg)
    if day_of_year < 0 or day_of_year > DAYS_IN_YEAR:
        msg = "Day of the year must be between 0 and 365"
        raise ValueError(msg)
    if utc_hour < 0.0 or utc_hour > 24.0:
        msg = "Hour must be between 0 and 24.0"
        raise ValueError(msg)
    declination = calc_solar_declination(day_of_year)
    decl = np.radians(declination)
    hour_angle = calc_solar_hour_angle(longitudes, day_of_year, utc_hour)
    rad_hours = np.radians(hour_angle)
    lats = np.radians(latitudes)
    # Calculate solar position:

    solar_elevation = np.sin(decl) * np.sin(lats) + np.cos(decl) * np.cos(
        lats
    ) * np.cos(rad_hours)
    if not return_sine:
        solar_elevation = np.degrees(np.arcsin(solar_elevation))

    return solar_elevation


def daynight_terminator(
    longitudes: ndarray, day_of_year: int, utc_hour: float
) -> ndarray:
    """
    Calculate the Latitude values of the daynight terminator
    for the given longitudes.

    Args:
        longitudes:
            Array of longitudes.
            longitudes needs to be between 180.0 and -180.0 degrees
        day_of_year:
            Day of the year 0 to 365, 0 = 1st January
        utc_hour:
            Hour of the day in UTC

    Returns:
        latitudes of the daynight terminator
    """
    if day_of_year < 0 or day_of_year > DAYS_IN_YEAR:
        msg = "Day of the year must be between 0 and 365"
        raise ValueError(msg)
    if utc_hour < 0.0 or utc_hour > 24.0:
        msg = "Hour must be between 0 and 24.0"
        raise ValueError(msg)
    declination = calc_solar_declination(day_of_year)
    decl = np.radians(declination)
    hour_angle = calc_solar_hour_angle(longitudes, day_of_year, utc_hour)
    rad_hour = np.radians(hour_angle)
    lats = np.arctan(-np.cos(rad_hour) / np.tan(decl))
    lats = np.degrees(lats)
    return lats


class DayNightMask(BasePlugin):
    """
    Plugin Class to generate a daynight mask for the provided cube
    """

    def __init__(self) -> None:
        """ Initial the DayNightMask Object """
        self.night = 0
        self.day = 1
        self.irregular = False

    def __repr__(self) -> str:
        """Represent the configured plugin instance as a string."""
        result = "<DayNightMask : " "Day = {}, Night = {}>".format(self.day, self.night)
        return result

    def _create_daynight_mask(self, cube: Cube) -> Cube:
        """
        Create blank daynight mask cube

        Args:
            cube:
                cube with the times and coordinates required for mask

        Returns:
            Blank daynight mask cube. The resulting cube will be the
            same shape as the time, y, and x coordinate, other coordinates
            will be ignored although they might appear as attributes
            on the cube as it is extracted from the first slice.
        """
        slice_coords = [cube.coord(axis="y"), cube.coord(axis="x")]

        # To handle non-orthogonal spatial coordinates, i.e. multiple coordinates
        # that share the same dimension, as in a spot-forecast.
        spatial_coords = []
        for dim in set([cube.coord_dims(crd) for crd in slice_coords]):
            spatial_coords.append(cube.coords(dimensions=dim)[0])

        if len(spatial_coords) != len(slice_coords):
            self.irregular = True
            slice_coords = spatial_coords

        if cube.coord("time") in cube.coords(dim_coords=True):
            slice_coords.insert(0, cube.coord("time"))

        template = next(cube.slices(slice_coords))
        demoted_coords = [
            crd
            for crd in cube.coords(dim_coords=True)
            if crd not in template.coords(dim_coords=True)
        ]
        for crd in demoted_coords:
            template.remove_coord(crd)
        attributes = generate_mandatory_attributes([template])
        title_attribute = {"title": "Day-Night mask"}
        data = np.full(template.data.shape, self.night, dtype=np.int32)
        daynight_mask = create_new_diagnostic_cube(
            "day_night_mask",
            1,
            template,
            attributes,
            optional_attributes=title_attribute,
            data=data,
            dtype=np.int32,
        )
        return daynight_mask

    def _daynight_lat_lon_cube(
        self, mask_cube: Cube, day_of_year: int, utc_hour: float
    ) -> Cube:
        """
        Calculate the daynight mask for the provided Lat Lon cube

        Args:
            mask_cube:
                daynight mask cube - data initially set to self.night
            day_of_year:
                day of the year 0 to 365, 0 = 1st January
            utc_hour:
                Hour in UTC

        Returns:
            daynight mask cube - daytime set to self.day
        """
        lons = mask_cube.coord("longitude").points
        lats = mask_cube.coord("latitude").points
        terminator_lats = daynight_terminator(lons, day_of_year, utc_hour)
        if self.irregular:
            lats_on_lon = lats
            terminator_on_lon = terminator_lats
        else:
            lons_zeros = np.zeros_like(lons)
            lats_zeros = np.zeros_like(lats).reshape(len(lats), 1)
            lats_on_lon = lats.reshape(len(lats), 1) + lons_zeros
            terminator_on_lon = lats_zeros + terminator_lats

        dec = calc_solar_declination(day_of_year)
        if dec > 0.0:
            index = np.where(lats_on_lon >= terminator_on_lon)
        else:
            index = np.where(lats_on_lon < terminator_on_lon)
        mask_cube.data[index] = self.day
        return mask_cube

    def process(self, cube: Cube) -> Cube:
        """
        Calculate the daynight mask for the provided cube. Note that only the
        hours and minutes of the dtval variable are used. To ensure consistent
        behaviour with changes of second or subsecond precision, the second
        component is added to the time object. This means that when the hours
        and minutes are used, we have correctly rounded to the nearest minute,
        e.g.::

           dt(2017, 1, 1, 11, 59, 59) -- +59 --> dt(2017, 1, 1, 12, 0, 58)
           dt(2017, 1, 1, 12, 0, 1)   -- +1  --> dt(2017, 1, 1, 12, 0, 2)
           dt(2017, 1, 1, 12, 0, 30)  -- +30 --> dt(2017, 1, 1, 12, 1, 0)

        If the cube's time coordinate has time bounds the mid-point of these
        bounds rather than the time point is used in the calculation. This
        ensures that should a majority of the period described fall into
        daylight hours, the data is not classified as falling at night. This
        is an issue as the time point is always located at the end of the
        time bounds. Without this approach weather symbols representing
        e.g. 20-21Z will be marked as night symbols should the sun set at
        20:59, which is not suitable.

        Args:
            cube:
                input cube

        Returns:
            daynight mask cube, daytime set to self.day
            nighttime set to self.night.
            The resulting cube will be the same shape as
            the time, y, and x coordinate, other coordinates
            will be ignored although they might appear as attributes
            on the cube as it is extracted from the first slice.
        """
        daynight_mask = self._create_daynight_mask(cube)

        modified_masks = iris.cube.CubeList()
        for mask_cube in daynight_mask.slices_over("time"):
            if mask_cube.coord("time").has_bounds():
                lb, ub = cube.coord("time").cell(0).bound
                dtval = lb + (ub - lb) / 2
            else:
                dtval = mask_cube.coord("time").cell(0).point
            day_of_year = get_day_of_year(dtval)
            utc_hour = get_hour_of_day(dtval)
            trg_crs = lat_lon_determine(mask_cube)
            # Grids that are not Lat Lon
            if trg_crs is not None:
                lats, lons = transform_grid_to_lat_lon(mask_cube)
                solar_el = calc_solar_elevation(lats, lons, day_of_year, utc_hour)
                mask_cube.data[np.where(solar_el > 0.0)] = self.day
            else:
                mask_cube = self._daynight_lat_lon_cube(
                    mask_cube, day_of_year, utc_hour
                )
            modified_masks.append(mask_cube)
        return modified_masks.merge_cube()
