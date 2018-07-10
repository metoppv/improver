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
from improver.utilities.temporal import iris_time_to_datetime


def generate_file_name(cube):
    """
    From a forecast cube, generate an IMPROVER-suitable file name using the
    correct lead time.  Requires a "time" coordinate.  If the cube has no
    forecast_period coordinate, creates a dummy lead time string.

    Args:
        cube (iris.cube.Cube):
            Cube containing nowcast data

        lead_time (int):
            Lead time of advection nowcast, in minutes

    Returns:
        filename (str):
            File base name to which to write

    Raises:
        iris.exceptions.CoordinateNotFoundError:
            If the input cube has no "time" coordinate
    """

    cdtime = (cube.coord('time').units).num2date(cube.coord('time').points)[0]
    cycle_time_string = '{}{:02}{:02}T{:02}{:02}Z'.format(
        cdtime.year, cdtime.month, cdtime.day, cdtime.hour, cdtime.minute)

    try:
        lead_time_coord = cube.coord('forecast_period')
        lead_time_coord.convert_units('s')
        lead_time, = lead_time_coord.points
        lead_time_hours = int(lead_time // 3600)
        lead_time_minutes = int((lead_time - 3600*lead_time_hours)) // 60
        lead_time_string = 'PT{:04}H{:02}M'.format(
            lead_time_hours, lead_time_minutes)
    except CoordinateNotFoundError:
        lead_time_string = 'PT0000H00M'

    parameter = cube.name().replace(' ', '_').lower()

    filename = '{}-{}-{}.nc'.format(
        cycle_time_string, lead_time_string, parameter)

    return filename
