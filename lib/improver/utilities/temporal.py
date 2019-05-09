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
"""Provide support utilities for making temporal calculations."""

import re

from datetime import datetime
from datetime import time as dt_time
from datetime import timedelta
from datetime import timezone
import warnings

import numpy as np

import cf_units as unit
from cf_units import Unit

import iris
from iris import Constraint
from iris.time import PartialDateTime
from iris.exceptions import CoordinateNotFoundError

from improver.utilities.cube_manipulation import build_coordinate


def cycletime_to_datetime(cycletime, cycletime_format="%Y%m%dT%H%MZ"):
    """Convert a string representating the cycletime of the
    format YYYYMMDDTHHMMZ into a datetime object.

     Args:
         cycletime (string):
             A cycletime that can be converted into a datetime using the
             cycletime_format supplied.

     Keyword Args:
         cycletime_format (str):
             String containing the desired format for the cycletime.
    Returns:
        datetime:
            A correctly formatted datetime object.
    """
    return datetime.strptime(cycletime, cycletime_format)


def datetime_to_cycletime(adatetime, cycletime_format="%Y%m%dT%H%MZ"):
    """Convert a datetime object into a string representing the cycletime
    of the format YYYYMMDDTHHMMZ.

     Args:
         adatetime (datetime.datetime):
             A datetime that can be converted into a cycletime using the
             cycletime_format supplied.
     Keyword Args:
         cycletime_format (str):
             String containing the desired format for the cycletime.
    Returns:
        str:
            A correctly formatted string.
    """
    return datetime.strftime(adatetime, cycletime_format)


def cycletime_to_number(
        cycletime, cycletime_format="%Y%m%dT%H%MZ",
        time_unit="hours since 1970-01-01 00:00:00",
        calendar="gregorian"):
    """Convert a cycletime of the format YYYYMMDDTHHMMZ into a numeric
    time value.

    Args:
        cycletime (str):
            A cycletime that can be converted into a datetime using the
            cycletime_format supplied.

    Keyword Args:
        cycletime_format (str):
            String containg the appropriate directives to indicate how
            the output datetime should display.
        time_unit (str):
            String representation of the cycletime units.
        calendar (str):
            String describing the calendar used for defining the cycletime.
            The choice of calendar must be supported by cf_units.CALENDARS.

    Returns:
        float:
            A numeric value to represent the datetime using assumed choices
            for the unit of time and the calendar.
    """
    dtval = cycletime_to_datetime(cycletime,
                                  cycletime_format=cycletime_format)
    return unit.date2num(dtval, time_unit, calendar)


def forecast_period_coord(
        cube, force_lead_time_calculation=False, result_units="seconds"):
    """
    Return or calculate the lead time coordinate (forecast_period)
    within a cube, either by reading the forecast_period coordinate,
    or by calculating the difference between the time (points and bounds) and
    the forecast_reference_time. The units of the forecast_period, time and
    forecast_reference_time coordinates are converted, if required. The final
    coordinate will have units of seconds.

    Args:
        cube (Iris.cube.Cube):
            Cube from which the lead times will be determined.

    Keyword Args:
        force_lead_time_calculation (bool):
            Force the lead time to be calculated from the
            forecast_reference_time and the time coordinate, even if
            the forecast_period coordinate exists.
            Default is False.
        result_units (str or cf_units.Unit):
            Desired units for the resulting forecast period coordinate.

    Returns:
        coord (iris.coords.AuxCoord or DimCoord):
            Describing the points and their units for
            'forecast_period'. A DimCoord is returned if the
            forecast_period coord is already present in the cube as a
            DimCoord and this coord does not need changing, otherwise
            it will be an AuxCoord. Units are result_units.
    """
    if cube.coords("forecast_period"):
        fp_type = cube.coord("forecast_period").dtype
    else:
        fp_type = np.int32

    if cube.coords("forecast_period") and not force_lead_time_calculation:
        result_coord = cube.coord("forecast_period").copy()
        try:
            result_coord.convert_units(result_units)
        except ValueError as err:
            msg = "For forecast_period: {}".format(err)
            raise ValueError(msg)

    # Try to return forecast_reference_time - time coordinate.
    elif cube.coords("time") and cube.coords("forecast_reference_time"):
        time_units = cube.coord("time").units
        t_coord = cube.coord("time")
        fr_coord = cube.coord("forecast_reference_time")
        fr_type = fr_coord.dtype
        try:
            fr_coord.convert_units(time_units)
        except ValueError as err:
            msg = "For forecast_reference_time: {}".format(err)
            raise ValueError(msg)
        time_points = np.array(
            [c.point for c in t_coord.cells()])
        forecast_reference_time_points = np.array(
            [c.point for c in fr_coord.cells()])
        required_lead_times = (
            time_points - forecast_reference_time_points)
        # Convert the timedeltas to a total in seconds.
        required_lead_times = np.array(
            [x.total_seconds() for x in required_lead_times]).astype(fr_type)
        if t_coord.bounds is not None:
            time_bounds = np.array(
                [c.bound for c in t_coord.cells()])
            required_lead_bounds = (
                time_bounds - forecast_reference_time_points)
            # Convert the timedeltas to a total in seconds.
            required_lead_bounds = np.array(
                [[b.total_seconds() for b in x]
                 for x in required_lead_bounds]).astype(fr_type)
        else:
            required_lead_bounds = None
        coord_type = iris.coords.AuxCoord
        if cube.coords("forecast_period"):
            if isinstance(
                    cube.coord("forecast_period"), iris.coords.DimCoord):
                coord_type = iris.coords.DimCoord
        result_coord = coord_type(
            required_lead_times,
            standard_name='forecast_period',
            bounds=required_lead_bounds,
            units="seconds")
        result_coord.convert_units(result_units)
        if np.any(result_coord.points < 0):
            msg = ("The values for the time {} and "
                   "forecast_reference_time {} coordinates from the "
                   "input cube have produced negative values for the "
                   "forecast_period. A forecast does not generate "
                   "values in the past.").format(
                       cube.coord("time").points,
                       cube.coord("forecast_reference_time").points)
            warnings.warn(msg)
    else:
        msg = ("The forecast period coordinate is not available within {}."
               "The time coordinate and forecast_reference_time "
               "coordinate were also not available for calculating "
               "the forecast_period.".format(cube))
        raise CoordinateNotFoundError(msg)

    result_coord.points = result_coord.points.astype(fp_type)
    if result_coord.bounds is not None:
        result_coord.bounds = result_coord.bounds.astype(fp_type)

    return result_coord


def iris_time_to_datetime(time_coord):
    """
    Convert iris time to python datetime object. Working in UTC.

    Args:
        time_coord (iris.coord.Coord):
            Iris time coordinate element(s).

    Returns:
        list of datetime.datetime objects
            The time element(s) recast as a python datetime object.
    """
    coord = time_coord.copy()
    coord.convert_units('seconds since 1970-01-01 00:00:00')
    return [datetime.utcfromtimestamp(value) for value in coord.points]


def datetime_to_iris_time(dt_in, time_units="hours"):
    """
    Convert python datetime.datetime into hours, minutes or seconds
    since 1970-01-01 00Z.

    Args:
        dt_in (datetime.datetime object):
            Time to be converted.

        time_units (str):
            Name of time unit. Currently only "hours", "minutes" or
            "seconds" are supported. Alternatively, an origin time can be
            supported, for example "seconds since 1970-01-01 00:00:00",
            however, "since 1970-01-01 00:00:00" will be ignored.

    Returns:
        result (float):
            Time since epoch in the units defined by the time_units
            with default floating point precision.
    """
    if all(time_unit not in time_units for time_unit in
           ["hours", "minutes", "seconds"]):
        msg = ("The time unit must contain 'hours', 'minutes' or 'seconds'. "
               "The time unit was {}".format(time_units))
        raise ValueError(msg)
    result = dt_in.replace(tzinfo=timezone.utc).timestamp()
    if "hours" in time_units:
        result /= 3600.
    elif "minutes" in time_units:
        result /= 60
    elif "seconds" in time_units:
        pass
    return result


def datetime_constraint(time_in, time_max=None):
    """
    Constructs an iris equivalence constraint from a python datetime object.

    Args:
        time_in (datetime.datetime object):
            The time to be used to build an iris constraint.
        time_max (datetime.datetime object):
            Optional max time, which if provided leads to a range constraint
            being returned up to < time_max.

    Returns:
        time_extract (iris.Constraint):
            An iris constraint to be used in extracting data at the given time
            from a cube.
    """
    time_start = PartialDateTime(
        time_in.year, time_in.month, time_in.day, time_in.hour)

    if time_max is None:
        time_extract = Constraint(time=lambda cell: cell.point == time_start)
    else:
        time_limit = PartialDateTime(
            time_max.year, time_max.month, time_max.day, time_max.hour)
        time_extract = Constraint(
            time=lambda cell: time_start <= cell < time_limit)
    return time_extract


def extract_cube_at_time(cubes, time, time_extract):
    """
    Extract a single cube at a given time from a cubelist.

    Args:
        cubes (iris.cube.CubeList):
            CubeList of a given diagnostic over several times.
        time (datetime.datetime object):
            Time at which forecast data is needed.
        time_extract (iris.Constraint):
            Iris constraint for the desired time.

    Returns:
        cube (iris.cube.Cube):
            Cube of data at the desired time.

    Raises:
        ValueError if the desired time is not available within the cubelist.
    """
    try:
        cube_in, = cubes.extract(time_extract)
        return cube_in
    except ValueError:
        msg = ('Forecast time {} not found within data cubes.'.format(
            time.strftime("%Y-%m-%d:%H:%M")))
        warnings.warn(msg)
        return None


def set_utc_offset(longitudes):
    """
    Simplistic timezone setting for unset sites that uses 15 degree bins
    centred on 0 degrees longitude. Used for on the fly site generation
    when no more rigorous source of timeszone information is provided.

    Args:
        longitudes (List):
            List of longitudes.

    Returns:
        utc_offsets (List):
            List of utc_offsets calculated using longitude.
    """
    return np.floor((np.array(longitudes) + 7.5)/15.)


def get_forecast_times(forecast_length, forecast_date=None,
                       forecast_time=None):
    """
    Generate a list of python datetime objects specifying the desired forecast
    times. This list will be created from input specifications if provided.
    Otherwise defaults are to start today at the most recent 6-hourly interval
    (00, 06, 12, 18) and to run out to T+144 hours.

    Args:
        forecast_length (int):
            An integer giving the desired length of the forecast output in
            hours (e.g. 48 for a two day forecast period).
        forecast_date (string (YYYYMMDD)):
            A string of format YYYYMMDD defining the start date for which
            forecasts are required. If unset it defaults to today in UTC.
        forecast_time (int):
            An integer giving the hour on the forecast_date at which to start
            the forecast output; 24hr clock such that 17 = 17Z for example. If
            unset it defaults to the latest 6 hour cycle as a start time.

    Returns:
        forecast_times (list of datetime.datetime objects):
            A list of python datetime.datetime objects that represent the
            times at which diagnostic data should be extracted.

    Raises:
        ValueError : raised if the input date is not in the expected format.
    """
    date_format = re.compile('[0-9]{8}')

    if forecast_date is None:
        start_date = datetime.utcnow().date()
    else:
        if date_format.match(forecast_date) and len(forecast_date) == 8:
            start_date = datetime.strptime(forecast_date,
                                           "%Y%m%d").date()
        else:
            raise ValueError('Date {} is in unexpected format; should be '
                             'YYYYMMDD.'.format(forecast_date))

    if forecast_time is None:
        # If no start hour provided, go back to the nearest multiple of 6
        # hours (e.g. utcnow = 11Z --> 06Z).
        forecast_start_time = datetime.combine(
            start_date, dt_time(divmod(datetime.utcnow().hour, 6)[0]*6))
    else:
        forecast_start_time = datetime.combine(start_date,
                                               dt_time(forecast_time))

    # Generate forecast times. Hourly to T+48, 3 hourly to T+forecast_length.
    forecast_times = [forecast_start_time + timedelta(hours=x) for x in
                      range(min(forecast_length, 49))]
    forecast_times = (forecast_times +
                      [forecast_start_time + timedelta(hours=x) for x in
                       range(51, forecast_length+1, 3)])

    return forecast_times


def unify_forecast_reference_time(cubes, cycletime):
    """Function to unify the forecast_reference_time across the input cubes
    provided. The cycletime specified is used as the forecast_reference_time.
    This function is intended for use in grid blending, where the models
    being blended may not have been run at the same cycle time, but should
    be given the same forecast period weightings.

    Args:
        cubes (iris.cube.CubeList or iris.cube.Cube):
            Cubes that will have their forecast_reference_time unified.
            If a single cube is provided the forecast_reference_time will be
            updated. Any bounds on the forecast_reference_time coord will be
            discarded.
        cycletime (datetime.datetime):
            Datetime for the cycletime that will be used to replace the
            forecast_reference_time on the individual cubes.

    Returns:
        result_cubes (iris.cube.CubeList):
            Cubes that have had their forecast_reference_time unified.

    Raises:
        ValueError: if forecast_reference_time is a dimension coordinate
    """
    if isinstance(cubes, iris.cube.Cube):
        cubes = iris.cube.CubeList([cubes])

    result_cubes = iris.cube.CubeList([])
    for cube in cubes:
        frt_units = cube.coord('forecast_reference_time').units
        frt_type = cube.coord('forecast_reference_time').dtype
        new_frt_units = Unit('seconds since 1970-01-01 00:00:00')
        frt_points = np.round(
            [new_frt_units.date2num(cycletime)]).astype(frt_type)
        frt_coord = build_coordinate(
            frt_points, standard_name="forecast_reference_time", bounds=None,
            template_coord=cube.coord('forecast_reference_time'),
            units=new_frt_units)
        frt_coord.convert_units(frt_units)
        frt_coord.points = frt_coord.points.astype(frt_type)
        cube.remove_coord("forecast_reference_time")
        cube.add_aux_coord(frt_coord, data_dims=None)

        # If a forecast period coordinate already exists on a cube, replace
        # this coordinate, otherwise create a new coordinate.
        fp_units = "seconds"
        if cube.coords("forecast_period"):
            fp_units = cube.coord("forecast_period").units
            cube.remove_coord("forecast_period")
        fp_coord = forecast_period_coord(
            cube, force_lead_time_calculation=True, result_units=fp_units)
        cube.add_aux_coord(fp_coord, data_dims=cube.coord_dims("time"))
        result_cubes.append(cube)
    return result_cubes


def find_latest_cycletime(cubelist):
    """
    Find the latest cycletime from the cubes in a cubelist and convert it into
    a datetime object.

    Args:
        cubelist (iris.cube.CubeList):
            A list of cubes each containing single time step from different
            forecast cycles.

    Returns:
        cycletime (datetime object):
            A datetime object corresponding to the latest forecast reference
            time in the input cubelist.
    """
    # Get cycle time as latest forecast reference time
    if any([cube.coord_dims("forecast_reference_time")
            for cube in cubelist]):
        raise ValueError(
            "Expecting scalar forecast_reference_time for each input "
            "cube - cannot replace a dimension coordinate")

    frt_coord = cubelist[0].coord("forecast_reference_time").copy()
    for cube in cubelist:
        next_coord = cube.coord("forecast_reference_time").copy()
        next_coord.convert_units(frt_coord.units)
        if next_coord.points[0] > frt_coord.points[0]:
            frt_coord = next_coord
    cycletime, = frt_coord.units.num2date(
        frt_coord.points)
    return cycletime


def extract_nearest_time_point(
        cube, dt, time_name="time", allowed_dt_difference=None):
    """Find the nearest time point to the time point provided.

    Args:
        cube (iris.cube.Cube):
            Cube or CubeList that will be extracted from using the supplied
            time_point
        dt (datetime.datetime):
            Datetime representation of a time that will be used within the
            extraction from the cube supplied.
        time_name (str):
            Name of the "time" coordinate that will be extracted. This must be
            "time" or "forecast_reference_time".
        allowed_dt_difference (float or None):
            Defines a limit to the maximum difference between the datetime
            provided and the time points available within the cube. If
            this limit is exceeded, then an error is raised.
            This must be defined in seconds.

    Returns:
        cube (iris.cube.Cube):
            Cube following extraction to return the cube that is nearest
            to the time point supplied.

    Raises:
        ValueError: The requested datetime is not available within the
            allowed difference.
    """
    if time_name not in ["time", "forecast_reference_time"]:
        msg = ("{} is not a valid time_name. "
               "The time_name must be either "
               "'time' or 'forecast_reference_time'.")
        raise ValueError(msg)

    time_point = datetime_to_iris_time(
        dt, time_units=cube.coord(time_name).units.origin)
    time_point_index = (
        cube.coord(time_name).nearest_neighbour_index(time_point))
    nearest_dt, = (
        iris_time_to_datetime(cube.coord(time_name).copy()[time_point_index]))
    if allowed_dt_difference:
        if abs((dt - nearest_dt).total_seconds()) > allowed_dt_difference:
            msg = ("The datetime {} is not available within the input cube "
                   "within the allowed difference {}. "
                   "The nearest datetime available was {}".format(
                       dt, allowed_dt_difference, nearest_dt))
            raise ValueError(msg)
    constr = iris.Constraint(coord_values={time_name: nearest_dt})
    cube = cube.extract(constr)
    return cube
