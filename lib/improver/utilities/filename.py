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
"""Module for generating IMPROVER-compliant file names."""

from iris.exceptions import CoordinateNotFoundError


def generate_file_name(cube, parameter=None):
    """
    From a forecast cube, generate an IMPROVER-suitable file name using the
    correct lead time.  Based on existing StaGE functionality.  Requires a
    "time" coordinate.  If the cube has no "forecast_period" coordinate (for
    example if the input is a radar composite or other observation), this
    function creates a dummy string representing a forecast period of zero.

    Args:
        cube (iris.cube.Cube):
            Cube containing nowcast data

    Kwargs:
        parameter (str):
            Optional parameter name to use

    Returns:
        filename (str):
            File base name to which to write

    Raises:
        iris.exceptions.CoordinateNotFoundError:
            If the input cube has no "time" coordinate
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

    filename = '{}-{}-{}.nc'.format(
        validity_time_string, forecast_period_string, parameter)

    return filename
