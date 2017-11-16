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
"""Calculate extrema values for diagnostics."""

import numpy as np
import copy
import datetime
from datetime import datetime as dt
import iris
from iris.cube import Cube, CubeList
from iris.coords import DimCoord
from improver.spotdata.common_functions import (iris_time_to_datetime,
                                                datetime_constraint,
                                                dt_to_utc_hours)


class ExtractExtrema(object):
    """Extract diagnostic maxima and minima in a given time period."""

    def __init__(self, period, start_hour=9):
        """
        The class is used to calculate maxima and minima values of a diagnostic
        over the supplied period (in hours), starting from a given hour in the
        24hr clock. All extrema values are calculated in the local time of the
        site being considered.

        Args:
            period (int (units: hours)):
                Period in hours over which to calculate the extrema values,
                e.g. 24 hours for maxima/minima in a whole day.

            start_hour (int (units: hours)):
                Hour in local_time on the 24hr clock at which to start the
                series of periods, e.g. period=12, start_hour=9 --> 09-21,
                21-09, etc. The default hour of 0900 is chosen to align with
                the NCM (national climate message) reporting period.

        """
        self.period = period
        self.start_hour = start_hour

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<ExtractExtrema: period: {}, start_hour: {}>')
        return result.format(self.period, self.start_hour)

    def process(self, cube):
        """
        Calculate extrema values for diagnostic in cube over the period given
        from the start_hour, both set at initialisation.

        Args:
            cube  (iris.cube.Cube):
                Cube of diagnostic data with a utc_offset coordinate.

        Returns:
            period_cubes (iris.cube.CubeList):
                CubeList of diagnostic extrema cubes.

        """
        # Change to 64 bit to avoid the 2038 problem with any time
        # manipulations on units in seconds since the epoch.
        cube.coord('time').points = cube.coord('time').points.astype(np.int64)

        # Adjust times on cube to be local to each site.
        local_tz_cube = make_local_time_cube(cube)

        # Starts at start_hour on first available day, runs until start_hour on
        # final_date.
        start_time, end_time = get_datetime_limits(local_tz_cube.coord('time'),
                                                   self.start_hour)
        num_periods = int(
            np.ceil((end_time - start_time).total_seconds()/3600/self.period))
        starts = [start_time + datetime.timedelta(hours=i*self.period)
                  for i in range(num_periods)]
        ends = [time + datetime.timedelta(hours=self.period)
                for time in starts]

        # Extract extrema values over desired time periods, producing a cube
        # for each period.
        period_cubes = CubeList()
        for period_start, period_end in zip(starts, ends):
            extrema_constraint = datetime_constraint(period_start, period_end)
            with iris.FUTURE.context(cell_datetime_objects=True):
                cube_over_period = local_tz_cube.extract(extrema_constraint)
            if cube_over_period is not None:
                # Ensure time dimension of resulting cube reflects period.
                mid_time = dt_to_utc_hours(period_start +
                                           (period_end - period_start)/2)
                bounds = [dt_to_utc_hours(period_start),
                          dt_to_utc_hours(period_end)]

                extremas = [['max', iris.analysis.MAX],
                            ['min', iris.analysis.MIN]]
                for name, method in extremas:
                    cube_out = cube_over_period.collapsed('time', method)
                    cube_out.long_name = cube_out.name() + '_' + name
                    cube_out.standard_name = None
                    cube_out.coord('time').convert_units(
                        'hours since 1970-01-01 00:00:00')
                    cube_out.coord('time').points = mid_time
                    cube_out.coord('time').bounds = bounds
                    period_cubes.append(cube_out)

        return period_cubes


def make_local_time_cube(cube):
    """
    Construct a cube in which data are arranged along a dimension
    coordinate of local time. This allows for the calculation of maxima/
    minima values over given ranges of local time (e.g. 09Z - 21Z maxima).

    e.g.::

      UTC Coord  :  12   13   14   15   16
      Data: Site 1  300  302  296  294  290 (UTC offset = -2)
            Site 2  280  282  283  280  279 (UTC offset = +2)

    Data redistributed according to UTC offset to sit on local time::

      Local times:  10   11   12   13   14   15   16   17   18
      Data: Site 1  300  302  296  294  290  -    -    -    -
            Site 2  -    -    -    -    280  282  283  280  279

    There will be missing but masked data in locations which are not
    forecast at the given local time, e.g. a SpotData site at a UTC
    location for a 12Z run will have no forecast temperatures on the local
    time axis for times earlier than 12Z.

    Maxima/Minima can then be calculated on local time, as makes sense for
    quantities such as maximum in day.

    New index is obtained as::

      Site_UTC_offset + 12 + UTC_coordinate_index

    +12 ensures that UTC offset -12 sits at an index of 0 etc.

    UTC_coordinate_index is the data's index on the original coordinate
    axis. So imagining a model run with a forecast reference time of
    12Z, a temperature at 14Z UTC will sit at index=2. At a SpotData
    site with a UTC offset of -6, the index will become 8::

      Index: 0  1  2  3  4  5  6  7  8  9  10 11 12
      UTC    12 13 14 15 16 17 18 19 20 21 22 23 00
      Data :       X
      Local: 00 01 02 03 04 05 06 07 08 09 10 11 12
      Data :                         X

    Args:
        cube (iris.cube.Cube):
            A cube of site data resulting from the SpotData extraction process.

    Returns:
        cube (iris.cube.Cube):
            A cube with localised time coordinates.

    """
    # Ensure time coordinate is in hours.
    hour_coordinates = cube.coord('time')
    hour_coordinates.convert_units('hours since 1970-01-01 00:00:00')

    # Calculate local time range which spans earliest time UTC -12 for far west
    # to latest time UTC +14 in far east. Ignores half hour time zones.
    local_time_min = int(hour_coordinates.points[0]-12)
    local_time_max = int(hour_coordinates.points[-1]+14)
    local_times = range(local_time_min, local_time_max+1, 1)

    # Create iris.coord.DimCoord of local times.
    local_time_coord = DimCoord(local_times, standard_name='time',
                                units=hour_coordinates.units)

    # Create empty array to contain extrema data.
    new_data = np.full((len(local_times), cube.data.shape[1]), np.nan)

    # Create ascending indices to help with filling new_data array.
    n_sites = cube.data.shape[1]
    row_index = range(0, n_sites)

    # Loop through times in UTC and displace data in array so that each datum
    # sits at its local time.
    for i_time in range(len(hour_coordinates.points)):
        indices = cube.coord('utc_offset').points.astype(int) + 12 + i_time
        new_data[indices, row_index] = cube.data[i_time]

    # Mask invalid/unset data points.
    new_data = np.ma.masked_invalid(new_data)

    # Return cube on local time.
    metadata_dict = copy.deepcopy(cube.metadata._asdict())
    new_cube = Cube(new_data,
                    dim_coords_and_dims=[(local_time_coord, 0),
                                         (cube.coord('index'), 1)],
                    **metadata_dict)

    forecast_ref_time = cube.coord('forecast_reference_time')
    forecast_ref_time.points = forecast_ref_time.points[0]*len(local_times)
    new_cube.add_aux_coord(forecast_ref_time,
                           cube.coord_dims('forecast_reference_time'))

    # Exclude forecast period as it is somewhat confusing on a local time
    # cube (e.g. may be +2 for a time 8 hours before forecast reference time).
    for coord in cube.aux_coords:
        if (coord.name() != 'forecast_reference_time' and
                coord.name() != 'forecast_period'):
            new_cube.add_aux_coord(coord, cube.coord_dims(coord.name()))

    return new_cube


def get_datetime_limits(time_coord, start_hour):
    """
    Determine the date limits of a time coordinate axis and return time limits
    using a provided hour on that day.

    Args:
        time_coord (iris.coords.DimCoord):
            An iris time coordinate from which to extract the date limits.

        start_hour (int):
            The hour on a 24hr clock at which to set the returned times.

    Returns:
        (tuple) : tuple containing:
            **start_time** (datetime.datetime object):
                First day on a time coordinate, with the time set to the hour
                given by start hour

            **end_time** (datetime.datetime object):
                Last day on a time coordinate, with the time set to the hour
                given by start hour
    """
    dates = iris_time_to_datetime(time_coord)
    start_time = dt.combine(min(dates).date(), datetime.time(start_hour))
    end_time = dt.combine(max(dates).date(), datetime.time(start_hour))
    return start_time, end_time
