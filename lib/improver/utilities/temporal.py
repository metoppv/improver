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
"""Provide support utilities for making temporal calculations."""

import cf_units as unit
from datetime import datetime
import warnings

import numpy as np

import iris
from iris.exceptions import CoordinateNotFoundError


def cycletime_to_datetime(cycletime, cycletime_format="%Y%m%dT%H%MZ"):
    """Convert a cycletime of the format YYYYMMDDTHHMMZ into a datetime object.

     Args:
         cycletime (string):
             A cycletime that can be converted into a datetime using the
             cycletime_format supplied.

     Keyword Args:
         cycletime_format (string):
             String containg the appropriate directives to indicate how
             the output datetime should display.

    Returns:
        datetime:
            A correctly formatted datetime object.
    """
    return datetime.strptime(cycletime, cycletime_format)


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
    dt = cycletime_to_datetime(cycletime, cycletime_format=cycletime_format)
    return unit.date2num(dt, time_unit, calendar)


def forecast_period_coord(
        cube,
        force_lead_time_calculation=False):
    """
    Return or calculate the lead time coordinate (forecast_period)
    within a cube, either by reading the forecast_period coordinate,
    or by calculating the difference between the time and the
    forecast_reference_time. If the forecast_period coordinate is
    present, the points are assumed to represent the desired lead times
    with the bounds not being considered. The units of the
    forecast_period, time and forecast_reference_time coordinates are
    converted, if required. The final coordinate will have units of
    seconds.

    Args:
        cube (Iris.cube.Cube):
            Cube from which the lead times will be determined.

    Keyword Args:
        force_lead_time_calculation (bool):
            Force the lead time to be calculated from the
            forecast_reference_time and the time coordinate, even if
            the forecast_period coordinate exists.
            Default is False.

    Returns:
        coord (iris.coords.AuxCoord or DimCoord):
            Describing the points and their units for
            'forecast_period'. A DimCoord is returned if the
            forecast_period coord is already present in the cube as a
            DimCoord and this coord does not need changing, otherwise
            it will be an AuxCoord. Units are seconds.

    """
    result_units = "seconds"
    # Try to return forecast period coordinate in hours.
    if cube.coords("forecast_period") and not force_lead_time_calculation:
        fp_coord = cube.coord("forecast_period").copy()
        try:
            fp_coord.convert_units(result_units)
        except ValueError as err:
            msg = "For forecast_period: {}".format(err)
            raise ValueError(msg)
        return fp_coord

    # Try to return forecast_reference_time - time coordinate.
    if cube.coords("time") and cube.coords("forecast_reference_time"):
        time_units = cube.coord("time").units
        t_coord = cube.coord("time")
        fr_coord = cube.coord("forecast_reference_time")
        try:
            fr_coord.convert_units(time_units)
        except ValueError as err:
            msg = "For forecast_reference_time: {}".format(err)
            raise ValueError(msg)
        with iris.FUTURE.context(cell_datetime_objects=True):
            time_points = np.array(
                [c.point for c in t_coord.cells()])
            forecast_reference_time_points = np.array(
                [c.point for c in fr_coord.cells()])
        required_lead_times = (
            time_points - forecast_reference_time_points)
        # Convert the timedeltas to a total in seconds.
        required_lead_times = np.array(
            [x.total_seconds() for x in required_lead_times])
        coord_type = iris.coords.AuxCoord
        if cube.coords("forecast_period"):
            if isinstance(
                    cube.coord("forecast_period"), iris.coords.DimCoord):
                coord_type = iris.coords.DimCoord
        result_coord = coord_type(
            required_lead_times,
            standard_name='forecast_period',
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
        return result_coord
    msg = ("The forecast period coordinate is not available "
           "within {}."
           "The time coordinate and forecast_reference_time "
           "coordinate were also not available for calculating "
           "the forecast_period.".format(cube))
    raise CoordinateNotFoundError(msg)
