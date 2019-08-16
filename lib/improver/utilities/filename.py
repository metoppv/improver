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
"""Module for generating IMPROVER-compliant file names."""

from iris.exceptions import CoordinateNotFoundError


def generate_file_name(cube, parameter=None, include_period=False):
    """
    From a forecast cube, generate an IMPROVER-suitable file name using the
    correct lead time. Based on existing StaGE functionality. Requires a
    "time" coordinate. If the cube has no "forecast_period" coordinate (for
    example if the input is a radar composite or other observation), this
    function creates a dummy string representing a forecast period of zero.

    The filename generated will be of the format:
    20180806T2300Z-PT0012H00M-lwe_precip_rate.nc.
    If a period is included, the filename will become:
    20180806T2300Z-PT0012H00M-lwe_precip_accumulation-PT03H.nc

    Args:
        cube (iris.cube.Cube):
            Cube containing nowcast data
        parameter (str):
            Optional parameter name to use in the output filename rather than
            taking the name of the cube diagnostic.
        include_period (bool):
            Optional argument to indicate whether a period, accumulation or
            time window identifier should be included within the filename.

    Returns:
        filename (str):
            File base name to which to write

    Raises:
        ValueError: In order to calculate the period, either the
            forecast_period or the time coordinate must have bounds.
        ValueError: The period deduced by the coordinate bounds must be either
            less than 1 hour or in terms of whole hours.

    """

    vtime = (cube.coord('time').units).num2date(cube.coord('time').points)[0]
    validity_time_string = '{:04}{:02}{:02}T{:02}{:02}Z'.format(
        vtime.year, vtime.month, vtime.day, vtime.hour, vtime.minute)

    try:
        forecast_period_coord = cube.coord('forecast_period').copy()
        forecast_period_coord.convert_units('s')
        forecast_period, = forecast_period_coord.points
        forecast_period_hours = int(forecast_period // 3600)
        forecast_period_minutes = int(
            (forecast_period - 3600*forecast_period_hours)) // 60
        forecast_period_string = 'PT{:04}H{:02}M'.format(
            forecast_period_hours, forecast_period_minutes)
    except CoordinateNotFoundError:
        forecast_period_string = 'PT0000H00M'

    if parameter is None:
        parameter = cube.name().replace(' ', '_').lower()
        for char in ["/", "(", ")"]:
            parameter = parameter.replace(char, '')
        parameter = parameter.replace('__', '_')

    period_string = None
    if include_period:
        # If a period should be included within the filename, then check the
        # forecast_period and time coordinates for bounds that can be used
        # to define the period. Depending upon the bounds specified by the
        # coordinates, the format of the period will either be 'PT??M'
        # to represent a period in minutes, where ?? will be replaced by the
        # actual minutes, or 'PT??H' to represent a period in hours, where ??
        # will be replaced by the actual hours.
        coord_units = {"forecast_period": "seconds",
                       "time": "seconds since 1970-01-01 00:00:00"}
        for coord_name in ["forecast_period", "time"]:
            if cube.coords(coord_name):
                coord = cube.coord(coord_name).copy()
                if hasattr(coord, "bounds") and coord.bounds is not None:
                    coord.convert_units(coord_units[coord_name])
                    break
        else:
            msg = ("Neither the forecast_period coordinate nor the time "
                   "coordinate has bounds. Therefore the period required "
                   "for the filename could not be calculated.")
            raise ValueError(msg)

        bounds_diff = coord.bounds[0][1] - coord.bounds[0][0]
        bounds_diff_hours = int(bounds_diff // 3600)
        bounds_diff_minutes = int(
            (bounds_diff - 3600*bounds_diff_hours)) // 60
        if bounds_diff_minutes:
            if bounds_diff_hours:
                msg = ("If the difference between the bounds of the {} "
                       "coordinate should either be less than one hour "
                       "or should in terms of whole hours."
                       "Differences in the hours of {} and in the minutes "
                       "of {} is not supported.".format(
                           coord_name, bounds_diff_hours,
                           bounds_diff_minutes))
                raise ValueError(msg)
            period_string = 'PT{:02}M'.format(forecast_period_minutes)
        else:
            period_string = 'PT{:02}H'.format(forecast_period_hours)

    # Construct filename with or without an additional string to describe the
    # accumulation period or time window of relevance.
    if include_period:
        filename = '{}-{}-{}-{}.nc'.format(
            validity_time_string, forecast_period_string, parameter,
            period_string)
    else:
        filename = '{}-{}-{}.nc'.format(
            validity_time_string, forecast_period_string, parameter)

    return filename
