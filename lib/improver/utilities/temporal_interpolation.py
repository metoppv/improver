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
"""Class for Temporal Interpolation calculations."""

from datetime import timedelta
import warnings
import numpy as np

import iris
from iris.exceptions import CoordinateNotFoundError

from improver.utilities.temporal import iris_time_to_datetime
from improver.utilities.solar import DayNightMask, calc_solar_elevation
from improver.utilities.cube_manipulation import merge_cubes


class TemporalInterpolation(object):

    """
    Interpolate data to intermediate times between the validity times of two
    cubes. This can be used to fill in missing data (e.g. for radar fields) or
    to ensure data is available at the required intervals when model data is
    not available at these times.
    """

    def __init__(self, interval_in_minutes=None, times=None,
                 interpolation_method='linear'):
        """
        Initialise class.

        Keyword Args:
            interval_in_minutes (int):
                Specifies the interval in minutes at which to interpolate
                between the two input cubes. A number of minutes which does not
                divide up the interval equally will raise an exception::
                    e.g. cube_t0 valid at 03Z, cube_t1 valid at 06Z,
                    interval_in_minutes = 60 --> interpolate to 04Z and 05Z.
            times (list or tuple of datetime.datetime objects):
                A list of datetime objects specifying the times to which to
                interpolate.
            interpolation_method (str):
                Method of interpolation to use. Default is linear.
                Only methods in known_interpolation_methods can be used.

        Raises:
            ValueError: If neither interval_in_minutes nor times are set.
            ValueError: If interpolation method not in known list.

        """
        if interval_in_minutes is None and times is None:
            raise ValueError("TemporalInterpolation: One of "
                             "'interval_in_minutes' or 'times' must be set. "
                             "Currently both are none.")

        self.interval_in_minutes = interval_in_minutes
        self.times = times
        known_interpolation_methods = ['linear', 'solar']
        if interpolation_method not in known_interpolation_methods:
            raise ValueError("TemporalInterpolation: Unknown interpolation "
                             "method {}. ".format(interpolation_method))
        self.interpolation_method = interpolation_method

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<TemporalInterpolation: interval_in_minutes: {}, '
                  'times: {}, method: {}>')
        return result.format(self.interval_in_minutes, self.times,
                             self.interpolation_method)

    def construct_time_list(self, initial_time, final_time):
        """
        A function to construct a list of datetime objects formatted
        appropriately for use by iris' interpolation method.

        Args:
            initial_time (datetime.datetime):
                The start of the period over which a time list is to be
                constructed.
            final_time (datetime.datetime).
                The end of the period over which a time list is to be
                constructed.

        Returns:
            list:
                A list containing a tuple that specifies the coordinate and a
                list of points along that coordinate to which to interpolate,
                as required by the iris interpolation method:
                    e.g. [('time', [<datetime object 0>,
                                    <datetime object 1>])]

        Raises:
            ValueError: If list of times provided falls outside the range
                        specified by the initial and final times.
            ValueError: If the interval_in_minutes does not divide the time
                        range up equally.
        """
        time_list = []
        if self.times is not None:
            self.times = sorted(self.times)
            if self.times[0] < initial_time or self.times[-1] > final_time:
                raise ValueError(
                    'List of times falls outside the range given by '
                    'initial_time and final_time. ')
            time_list = self.times
        else:
            if ((final_time - initial_time).seconds %
                    (60 * self.interval_in_minutes) != 0):
                raise ValueError(
                    'interval_in_minutes provided to time_interpolate does not'
                    ' divide into the interval equally.')

            time_entry = initial_time
            while True:
                time_entry = (time_entry +
                              timedelta(minutes=self.interval_in_minutes))
                if time_entry >= final_time:
                    break
                time_list.append(time_entry)

        return [('time', time_list)]

    def solar_interpolate(self, cube, time_list):
        """
        Interpolate solar radiation parameter which are zero if the
        sun is below the horizon.

        Args:
            cube (iris.cube.Cube):
                cube containing diagnostic cube valid at the beginning
                of the period and cube valid at the end of the period
            time_list (list):
                A list containing a tuple that specifies the coordinate and a
                list of points along that coordinate to which to interpolate,
                as required by the iris interpolation method:
                    e.g. [('time', [<datetime object 0>,
                                    <datetime object 1>])]
        Returns:
            interpolated_cubes (iris.cube.CubeList):
                A list of cubes interpolated to the desired times.

        """
        # iris.analysis.Linear() modifies the dtype of time and forecast_period
        # coords so need to revert back
        dtype_time = cube.coord('time').points.dtype
        dtype_fp = cube.coord('forecast_period').points.dtype

        interpolated_cube = (
            cube.interpolate(time_list, iris.analysis.Linear()))
        interpolated_cubes = iris.cube.CubeList()
        daynightplugin = DayNightMask()
        daynight_mask = daynightplugin.process(interpolated_cube)

        for i, single_time in enumerate(interpolated_cube.slices_over('time')):
            index = np.where(daynight_mask.data[i] == daynightplugin.night)
            if len(single_time.shape) > 2:
                single_time.data[::, index[0], index[1]] = 0.0
            else:
                single_time.data[index[0], index[1]] = 0.0
            coord_time = single_time.coord('time')
            coord_time.points = np.around(coord_time.points).astype(dtype_time)
            coord_fp = single_time.coord('forecast_period')
            coord_fp.points = np.around(coord_fp.points).astype(dtype_fp)
            interpolated_cubes.append(single_time)

        return interpolated_cubes

    def process(self, cube_t0, cube_t1):
        """
        Interpolate data to intermediate times between validity times of
        cube_t0 and cube_t1.

        Args:
            cube_t0 (iris.cube.Cube):
                A diagnostic cube valid at the beginning of the period within
                which interpolation is to be permitted.
            cube_t1 (iris.cube.Cube):
                A diagnostic cube valid at the end of the period within which
                interpolation is to be permitted.

        Returns:
            interpolated_cubes (iris.cube.CubeList):
                A list of cubes interpolated to the desired times.

        Raises:
            TypeError: If cube_t0 and cube_t1 are not of type iris.cube.Cube.
            CoordinateNotFoundError: The input cubes contain no time
                                     coordinate.
            ValueError: Cubes contain multiple validity times.
            ValueError: The input cubes are ordered such that the initial time
                        cube has a later validity time than the final cube.
        """
        if (not isinstance(cube_t0, iris.cube.Cube) or
                not isinstance(cube_t1, iris.cube.Cube)):
            raise TypeError('Inputs to TemporalInterpolation are not of type '
                            'iris.cube.Cube')

        try:
            initial_time, = iris_time_to_datetime(cube_t0.coord('time'))
            final_time, = iris_time_to_datetime(cube_t1.coord('time'))
        except CoordinateNotFoundError:
            msg = ('Cube provided to time_interpolate contains no time '
                   'coordinate.')
            raise CoordinateNotFoundError(msg)
        except ValueError:
            msg = ('Cube provided to time_interpolate contains multiple '
                   'validity times, only one expected.')
            raise ValueError(msg)

        if initial_time > final_time:
            raise ValueError('time_interpolate input cubes ordered incorrectly'
                             ', with the final time being before the initial '
                             'time.')

        time_list = self.construct_time_list(initial_time, final_time)
        cubes = iris.cube.CubeList([cube_t0, cube_t1])
        cube = merge_cubes(cubes)

        if self.interpolation_method == 'solar':
            interpolated_cubes = self.solar_interpolate(cube, time_list)
        else:
            interpolated_cube = cube.interpolate(time_list,
                                                 iris.analysis.Linear())

            # iris.analysis.Linear() modifies the dtype of
            # time and forecast_period coords so need to revert back

            dtype_time = cube_t0.coord('time').points.dtype
            dtype_fp = cube_t0.coord('forecast_period').points.dtype

            interpolated_cubes = iris.cube.CubeList()
            for single_time in interpolated_cube.slices_over('time'):
                coord_time = single_time.coord('time')
                coord_time.points = (
                    np.around(coord_time.points).astype(dtype_time))
                coord_fp = single_time.coord('forecast_period')
                coord_fp.points = np.around(coord_fp.points).astype(dtype_fp)
                interpolated_cubes.append(single_time)

        return interpolated_cubes
