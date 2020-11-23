# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
"""General utilities for parsing and extracting cubes at times"""

import warnings
from datetime import datetime, timedelta, timezone

import cf_units
import iris
import numpy as np
from iris import Constraint
from iris.coords import AuxCoord
from iris.time import PartialDateTime

from improver import PostProcessingPlugin
from improver.metadata.constants.time_types import TIME_COORDS
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.cube_checker import spatial_coords_match


def cycletime_to_datetime(cycletime, cycletime_format="%Y%m%dT%H%MZ"):
    """Convert a string representating the cycletime of the
    format YYYYMMDDTHHMMZ into a datetime object.

    Args:
        cycletime (str):
            A cycletime that can be converted into a datetime using the
            cycletime_format supplied.
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
        cycletime_format (str):
            String containing the desired format for the cycletime.
    Returns:
        str:
            A correctly formatted string.
    """
    return datetime.strftime(adatetime, cycletime_format)


def cycletime_to_number(
    cycletime,
    cycletime_format="%Y%m%dT%H%MZ",
    time_unit="hours since 1970-01-01 00:00:00",
    calendar="gregorian",
):
    """Convert a cycletime of the format YYYYMMDDTHHMMZ into a numeric
    time value.

    Args:
        cycletime (str):
            A cycletime that can be converted into a datetime using the
            cycletime_format supplied.
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
    dtval = cycletime_to_datetime(cycletime, cycletime_format=cycletime_format)
    return cf_units.date2num(dtval, time_unit, calendar)


def iris_time_to_datetime(time_coord, point_or_bound="point"):
    """
    Convert iris time to python datetime object. Working in UTC.

    Args:
        time_coord (iris.coords.Coord):
            Iris time coordinate element(s).

    Returns:
        list of datetime.datetime:
            The time element(s) recast as a python datetime object.
    """
    coord = time_coord.copy()
    coord.convert_units("seconds since 1970-01-01 00:00:00")
    if point_or_bound == "point":
        datetime_list = [value.point for value in coord.cells()]
    elif point_or_bound == "bound":
        datetime_list = [value.bound for value in coord.cells()]
    return datetime_list


def datetime_to_iris_time(dt_in):
    """
    Convert python datetime.datetime into seconds since 1970-01-01 00Z.

    Args:
        dt_in (datetime.datetime):
            Time to be converted into seconds since 1970-01-01 00Z.

    Returns:
        float:
            Time since epoch in the seconds as desired dtype.
    """
    result = dt_in.replace(tzinfo=timezone.utc).timestamp()
    return np.int64(result)


def datetime_constraint(time_in, time_max=None):
    """
    Constructs an iris equivalence constraint from a python datetime object.

    Args:
        time_in (datetime.datetime):
            The time to be used to build an iris constraint.
        time_max (datetime.datetime):
            Optional max time, which if provided leads to a range constraint
            being returned up to < time_max.

    Returns:
        iris.Constraint:
            An iris constraint to be used in extracting data at the given time
            from a cube.
    """
    time_start = PartialDateTime(time_in.year, time_in.month, time_in.day, time_in.hour)

    if time_max is None:
        time_extract = Constraint(time=lambda cell: cell.point == time_start)
    else:
        time_limit = PartialDateTime(
            time_max.year, time_max.month, time_max.day, time_max.hour
        )
        time_extract = Constraint(time=lambda cell: time_start <= cell < time_limit)
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
        iris.cube.Cube:
            Cube of data at the desired time.

    Raises:
        ValueError if the desired time is not available within the cubelist.
    """
    try:
        (cube_in,) = cubes.extract(time_extract)
        return cube_in
    except ValueError:
        msg = "Forecast time {} not found within data cubes.".format(
            time.strftime("%Y-%m-%d:%H:%M")
        )
        warnings.warn(msg)
        return None


def extract_nearest_time_point(cube, dt, time_name="time", allowed_dt_difference=0):
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
        allowed_dt_difference (int):
            An int in seconds to define a limit to the maximum difference
            between the datetime provided and the time points available within
            the cube. If this limit is exceeded, then an error is raised.
            This must be defined in seconds.
            Default is 0.

    Returns:
        iris.cube.Cube:
            Cube following extraction to return the cube that is nearest
            to the time point supplied.

    Raises:
        ValueError: The requested datetime is not available within the
            allowed difference.
    """
    if time_name not in ["time", "forecast_reference_time"]:
        msg = (
            "{} is not a valid time_name. "
            "The time_name must be either "
            "'time' or 'forecast_reference_time'."
        )
        raise ValueError(msg)

    time_point = datetime_to_iris_time(dt)
    time_point_index = cube.coord(time_name).nearest_neighbour_index(time_point)
    (nearest_dt,) = iris_time_to_datetime(
        cube.coord(time_name).copy()[time_point_index]
    )
    if abs((dt - nearest_dt).total_seconds()) > allowed_dt_difference:
        msg = (
            "The datetime {} is not available within the input "
            "cube within the allowed difference {} seconds. "
            "The nearest datetime available was {}".format(
                dt, allowed_dt_difference, nearest_dt
            )
        )
        raise ValueError(msg)
    constr = iris.Constraint(coord_values={time_name: nearest_dt})
    cube = cube.extract(constr)
    return cube


class TimezoneExtraction(PostProcessingPlugin):
    """Plugin to extract local time offsets"""

    def __init__(self):
        self.time_coord_standards = TIME_COORDS["time"]
        self.time_points = None
        self.output_cube = None

    def create_output_cube(self, cube, utc_time):
        """
        Constructs the output cube and an array that will be come the time coord

        Args:
            cube (iris.cube.Cube):
                Cube of data to extract timezone-offsets from. Must contain a time
                coord spanning all the timezones.
            utc_time (datetime.datetime):
                The UTC time of the output cube. This will form a scalar "utc" coord on
                the output cube.

        """
        # Import add_coordinate here to avoid circular import
        from improver.synthetic_data.set_up_test_cubes import add_coordinate

        template_cube = cube.slices_over("time").next().copy()
        template_cube.remove_coord("time")
        template_cube.remove_coord("forecast_period")
        self.output_cube = create_new_diagnostic_cube(
            template_cube.name(),
            template_cube.units,
            template_cube,
            generate_mandatory_attributes([template_cube]),
            optional_attributes=template_cube.attributes,
            data=np.ma.masked_all_like(template_cube.data).astype(template_cube.dtype),
        )
        # Spatial coordinates are always the last two on the cube (-2 and -1)
        # Iris doesn't recognise negative indices for these, so at ndim to get the
        # positive equivalent.
        self.time_points = np.ma.masked_all_like(
            cube.slices([n + cube.ndim for n in [-2, -1]]).next().data
        ).astype(self.time_coord_standards.dtype)

        # Create a UTC time coordinate to help with plotting data.
        utc_coord_standards = TIME_COORDS["utc"]
        utc_units = cf_units.Unit(
            utc_coord_standards.units, calendar=utc_coord_standards.calendar,
        )
        self.output_cube = add_coordinate(
            self.output_cube,
            [utc_time],
            "utc",
            coord_units=utc_units,
            dtype=utc_coord_standards.dtype,
            is_datetime=True,
        )
        self.output_cube = iris.util.squeeze(self.output_cube)

    def fill_timezones(self, input_cube, timezone_cube):
        """
        Populates the output cube data with data from input_cube. This is done by
        multiplying the inverse of the timezone_cube.data with the input_cube.data and
        summing along the time axis. Because timezone_cube.data is a mask of 1 and 0,
        inversing it gives 1 where we WANT data and 0 where we don't. Summing these up
        produces the result. The same logic can be used for times.
        Modifies self.output_cube and self.time_points.

        Args:
            input_cube (iris.cube.Cube):
                Cube of data to extract timezone-offsets from. Must contain a time
                coord spanning all the timezones.
            timezone_cube (iris.cube.Cube):
                Cube describing the UTC offset for the local time at each grid location
                Must have the same spatial coords as input_cube.
                Cube will have a UTC_offset coord. Data will be 0 (included) or
                1 (excluded) indicating which points are in each time zone.

        """
        result = input_cube.data * (1 - timezone_cube.data)
        self.output_cube.data = result.sum(axis=0)
        times = input_cube.coord("time").points * (1 - timezone_cube.data)
        self.time_points = times.sum(axis=0)

    def check_input_cube_dims(self, input_cube):
        """Ensures input cube has exactly three dimensions: time, y, x

        Raises:
            ValueError:
                If the input cube does not have exactly the expected three coords."""
        xy_coords = [input_cube.coord(axis=n) for n in "yx"]
        expected_coords = ["time"] + [coord.name() for coord in xy_coords]
        cube_coords = [coord.name() for coord in input_cube.coords(dim_coords=True)]
        if expected_coords != cube_coords:
            raise ValueError(
                f"Expected coords on input_cube: time, y, x ({expected_coords}). Found {cube_coords}"
            )

    def check_input_cube_time(self, input_cube, timezone_cube, output_utc_time):
        """Ensures input cube and timezone_cube cover exactly the right points

        Raises:
            ValueError:
                If the time coord on the input cube does not match the required times.
        """
        input_time_points = [cell.point for cell in input_cube.coord("time").cells()]
        timezone_coord = timezone_cube.coord("UTC_offset")
        timezone_coord.convert_units("seconds")
        output_times = [
            output_utc_time + timedelta(seconds=np.int(offset))
            for offset in timezone_coord.points
        ]
        if input_time_points != output_times:
            raise ValueError(
                f"Time coord on input cube does not match required times. Expected\n"
                + "\n".join([f"{t:%Y%m%dT%H%MZ}" for t in output_times])
                + "\nFound:\n"
                + "\n".join([f"{t:%Y%m%dT%H%MZ}" for t in input_time_points])
            )

    def check_timezones_are_unique(self, timezone_cube):
        """Ensures that each grid point falls into exactly one time zone.

        Raises:
            ValueError:
                If the timezone_cube does not map exactly one time zone to each spatial
                point.
        """
        if not ((1 - timezone_cube.data).sum(axis=0) == 1).all():
            raise ValueError(
                "Timezone cube does not map exactly one time zone to each spatial point"
            )

    def process(self, input_cube, timezone_cube, output_utc_time):
        """
        Calculates timezone-offset data for the specified UTC output times

        Args:
            input_cube (iris.cube.Cube):
                Cube of data to extract timezone-offsets from. Must contain a time
                coord spanning all the timezones.
            timezone_cube (iris.cube.Cube):
                Cube describing the UTC offset for the local time at each grid location
                Must have the same spatial coords as input_cube.
            output_utc_time (datetime.datetime):
                The UTC time of the output cube. This will form a scalar "utc" coord on
                the output cube.

        Returns:
            iris.cube.Cube:
                Output local-time cube. The time coord will span the spatial coords.
                The utc coord will match the output_utc_time_list supplied. All other
                coords and attributes will match those found on input_cube.
        """
        self.check_input_cube_dims(input_cube)
        spatial_coords_match(input_cube, timezone_cube)
        self.check_input_cube_time(input_cube, timezone_cube, output_utc_time)
        self.check_timezones_are_unique(timezone_cube)
        self.create_output_cube(input_cube, output_utc_time)

        self.fill_timezones(input_cube, timezone_cube)

        time_units = cf_units.Unit(
            self.time_coord_standards.units,
            calendar=self.time_coord_standards.calendar,
        )
        time_coord = AuxCoord(self.time_points, standard_name="time", units=time_units,)
        self.output_cube.add_aux_coord(time_coord, (0, 1))

        return self.output_cube
