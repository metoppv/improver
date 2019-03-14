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

from datetime import datetime, timedelta
import numpy as np

import iris
from iris.exceptions import CoordinateNotFoundError

from improver.utilities.temporal import iris_time_to_datetime
from improver.utilities.solar import DayNightMask, calc_solar_elevation
from improver.utilities.cube_manipulation import merge_cubes
from improver.utilities.spatial import (
    lat_lon_determine, transform_grid_to_lat_lon)


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
        known_interpolation_methods = ['linear', 'solar', 'daynight']
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

    @staticmethod
    def check_cube_coords_dtype(cube, new_cube, dim_coords=None):
        """
        Update the data type of the coords within the new_cube
        to those in the original cube. This can be limited to a list of
        dimension coordinates and associated auxillary coordinates if required.

        Args:
        cube (iris.cube.Cube):
            The original cube that will be used to checked against for the
            required data type of that coordinate in the new_cube.
        new_cube (iris.cube.Cube):
            The cube that must be checked and adjusted using the coordinates
            from the original cube.
        dim_coords (list):
            List of dimension coordinate names. This list will be used to
            limit the correction of the coordinate data type
            to these coordinates and associated auxillary coordinates.
            Default is None.

        Returns:
        new_cube (iris.cube.Cube):
            Modified cube with the relevant coordinates datatypes
            set to the same as on the original cube.

        Raises:
            CoordinateNotFoundError : If original cube does not have the
                coordinate in dim_coords.
            CoordinateNotFoundError : If coordinate in original cube is
                not in new_cube.
        """
        if dim_coords is None:
            coord_list = [coord.name() for coord in cube.coords()]
        else:
            coord_list = []
            for coord_name in dim_coords:
                try:
                    req_dim = cube.coord_dims(coord_name)
                    associated_coords = [crd.name() for crd in cube.coords()
                                         if cube.coord_dims(crd) == req_dim
                                         and crd.name() is not coord_name]
                    coord_list.append(coord_name)
                    for val in associated_coords:
                        coord_list.append(val)
                except CoordinateNotFoundError:
                    msg = ('Original cube does not have the coordinate in'
                           ' dim_coords {}'.format(coord_name))
                    raise CoordinateNotFoundError(msg)
        for coord_name in coord_list:
            dtype_orig = cube.coord(coord_name).points.dtype
            try:
                new_coord = new_cube.coord(coord_name)
                # TODO: Provide safe support for time coordinates that could
                # potentially be in units of "hours since 1970-01-01 00:00:00."
                if new_coord.points.dtype != dtype_orig:
                    units_orig = new_coord.units
                    new_coord.convert_units(
                        "seconds since 1970-01-01 00:00:00")
                    new_coord.points = np.around(new_coord.points)
                    new_coord.points = new_coord.points.astype(dtype_orig)
                    new_coord.convert_units(units_orig)
                if (hasattr(new_coord, "bounds")
                        and new_coord.bounds is not None and
                        new_coord.bounds.dtype != dtype_orig):
                    units_orig = new_coord.units
                    new_coord.convert_units(
                        "seconds since 1970-01-01 00:00:00")
                    new_coord.bounds = np.around(new_coord.bounds)
                    new_coord.bounds = new_coord.bounds.astype(dtype_orig)
                    new_coord.convert_units(units_orig)
            except CoordinateNotFoundError:
                msg = ('new_cube does not have the coordinate in'
                       ' the original cube {}'.format(coord_name))
                raise CoordinateNotFoundError(msg)
        return new_cube

    @staticmethod
    def calc_sin_phi(dtval, lats, lons):
        """
        Calculate sin of solar elevation

        Args:
            dtval (datetime.datetime):
                Date and time.
            lats (np.array):
                Array 2d of latitudes for each point
            lons (np.array):
                Array 2d of longitudes for each point
        Returns:
            sin_phi (np.array):
                Array of sine of solar elevation at each point

        """
        day_of_year = (dtval - datetime(dtval.year, 1, 1)).days
        utc_hour = (dtval.hour * 60.0 + dtval.minute) / 60.0
        sin_phi = calc_solar_elevation(lats, lons, day_of_year,
                                       utc_hour, return_sine=True)
        return sin_phi

    @staticmethod
    def calc_lats_lons(cube):
        """
        Calculate the lats and lons of each point for a given cube.

        Args:
            cube (iris.cube.Cube):
                cube containing x and y axis
        Returns:
            (tuple) : tuple containing:
                **lats** (np.array):
                    2d Array of latitudes for each point.
                **lons** (np.array):
                    2d Array of longitudes for each point.

        """
        trg_crs = lat_lon_determine(cube)
        if trg_crs is not None:
            xycube = next(cube.slices([cube.coord(axis='y'),
                                       cube.coord(axis='x')]))
            lats, lons = transform_grid_to_lat_lon(xycube)
        else:
            lats_row = cube.coord('latitude').points
            lons_col = cube.coord('longitude').points
            lats = np.repeat(lats_row[:, np.newaxis], len(lons_col), axis=1)
            lons = np.repeat(lons_col[np.newaxis, :], len(lats_row), axis=0)
        return lats, lons

    def solar_interpolate(self, cube, interpolated_cube):
        """
        Interpolate solar radiation parameter which are zero if the
        sun is below the horizon.

        Args:
            cube (iris.cube.Cube):
                cube containing diagnostic cube valid at the beginning
                of the period and cube valid at the end of the period
            interpolated_cube (iris.cube.Cube):
                cube containing Linear interpolation of
                cube at interpolation times in time_list.
        Returns:
            interpolated_cubes (iris.cube.CubeList):
                A list of cubes interpolated to the desired times.

        """

        interpolated_cubes = iris.cube.CubeList()
        (lats, lons) = self.calc_lats_lons(cube)
        prev_data = cube[0].data
        next_data = cube[1].data
        dtvals = iris_time_to_datetime(cube.coord('time'))
        # Calculate sine of solar elevation for cube valid at the
        # beginning of the period.
        dtval_prev = dtvals[0]
        sin_phi_prev = self.calc_sin_phi(dtval_prev, lats, lons)
        # Calculate sine of solar elevation for cube valid at the
        # end of the period.
        dtval_next = dtvals[1]
        sin_phi_next = self.calc_sin_phi(dtval_next, lats, lons)
        # Length of time between beginning and end in seconds
        diff_step = (dtval_next - dtval_prev).seconds

        for single_time in interpolated_cube.slices_over('time'):
            # Calculate sine of solar elevation for cube at this
            # interpolated time.
            dtval_interp = iris_time_to_datetime(single_time.coord('time'))[0]
            sin_phi_interp = self.calc_sin_phi(dtval_interp, lats, lons)
            # Length of time between beginning and interpolated time in seconds
            diff_interp = (dtval_interp - dtval_prev).seconds
            # Set all values to 0.0, to be replaced
            # with values calculated through this solar method.
            single_time.data[::] = 0.0
            sun_up = np.where(sin_phi_interp > 0.0)
            # Solar value is calculated only for points where the sun is up
            # and is a weighted combination of the data using the sine of
            # solar elevation and the data in the cube valid
            # at the begining and end.

            # If the cube containing cube valid at the beginning of the period
            # and at the end of the period has more than x and y coordinates
            # the calculation needs to adapted to accommodate this.
            if len(single_time.shape) > 2:
                prevv = (
                    prev_data[::, sun_up[0], sun_up[1]]/sin_phi_prev[sun_up])
                nextv = (
                    next_data[::, sun_up[0], sun_up[1]]/sin_phi_next[sun_up])
                single_time.data[::, sun_up[0], sun_up[1]] = (
                    sin_phi_interp[sun_up] *
                    (prevv + (nextv - prevv) * (diff_interp/diff_step)))
            else:
                prevv = prev_data[sun_up]/sin_phi_prev[sun_up]
                nextv = next_data[sun_up]/sin_phi_next[sun_up]
                single_time.data[sun_up] = (sin_phi_interp[sun_up] *
                                            (prevv + (nextv - prevv)
                                             * (diff_interp/diff_step)))
            # cube with new data added to interpolated_cubes cube List.
            interpolated_cubes.append(single_time)
        return interpolated_cubes

    @staticmethod
    def daynight_interpolate(interpolated_cube):
        """
        Interpolate solar radiation parameter which are zero if the
        sun is below the horizon.

        Args:
            interpoldated_cube (iris.cube.Cube):
                cube containing Linear interpolation of
                cube at interpolation times in time_list.

        Returns:
            interpolated_cubes (iris.cube.CubeList):
                A list of cubes interpolated to the desired times.

        """

        interpolated_cubes = iris.cube.CubeList()
        daynightplugin = DayNightMask()
        daynight_mask = daynightplugin.process(interpolated_cube)

        for i, single_time in enumerate(interpolated_cube.slices_over('time')):
            index = np.where(daynight_mask.data[i] == daynightplugin.night)
            if len(single_time.shape) > 2:
                single_time.data[::, index[0], index[1]] = 0.0
            else:
                single_time.data[index] = 0.0
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

        interpolated_cube = cube.interpolate(time_list,
                                             iris.analysis.Linear())
        self.check_cube_coords_dtype(cube_t0,
                                     interpolated_cube,
                                     dim_coords=['time'])
        interpolated_cubes = iris.cube.CubeList()
        if self.interpolation_method == 'solar':
                interpolated_cubes = self.solar_interpolate(cube,
                                                            interpolated_cube)
        elif self.interpolation_method == 'daynight':
                interpolated_cubes = (
                    self.daynight_interpolate(interpolated_cube))
        else:
            for single_time in interpolated_cube.slices_over('time'):
                interpolated_cubes.append(single_time)

        return interpolated_cubes
