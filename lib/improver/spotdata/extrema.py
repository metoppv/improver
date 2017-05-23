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

"""Gridded data extraction for the Improver site specific process chain."""

import numpy as np
from iris.analysis import MAX as IMAX
from iris.analysis import MIN as IMIN
from iris import FUTURE
from improver.spotdata.write_output import WriteOutput

FUTURE.cell_datetime_objects = True


class ExtractExtrema(object):
    '''Extract diagnostic maxima and minima in a given time period.'''

    def __init__(self, method):
        """
        The class is called with the desired method, in this case the period
        over which to calculate the extrema values.

        This all needs to be updated to work properly with local times if that
        is desirable, and to present additional options. And to actually
        function as advertised.

        INCOMPLETE.

        """
        self.method = method

    def process(self, cube):
        '''Call the required method'''
        function = getattr(self, self.method)
        function(cube)

    @staticmethod
    def In24hr(cube):
        '''
        Calculate extrema values for diagnostic in cube over 24 hour period.

        Args:
        -----
        cube  : Cube of diagnostic data.

        Returns:
        --------
        Nil. Writes out cube of extrema data.

        '''
        cube.coord('time').points = cube.coord('time').points.astype(np.int64)

        cube_max = cube.collapsed('time', IMAX)
        cube_min = cube.collapsed('time', IMIN)

        cube_max.long_name = cube_max.name() + '_max'
        cube_min.long_name = cube_min.name() + '_min'
        cube_max.standard_name = None
        cube_min.standard_name = None

        cube.coord('time').points = cube.coord('time').points.astype(np.int32)

        WriteOutput('as_netcdf').process(cube_max)
        WriteOutput('as_netcdf').process(cube_min)

# def local_dates_in_cube(cube):
#     '''
#     Incomplete work on using local date time information.
#
#     OUT OF DATE AND INCOMPLETE.
#     '''
#     from datetime import timedelta as timedelta
#
#     dates_in_cube = unit.num2date(
#         b.coord('time').points, b.coord('time').units.name,
#         b.coord('time').units.calendar)
#
#     start_time = dates_in_cube[0] - timedelta(hours=12)
#     if start_time.hour < 18:
#         start_day = start_time.date()
#     else:
#         start_day = dates_in_cube[0].date()
#
#     end_time = (dates_in_cube[-1] + timedelta(hours=12)).date()
