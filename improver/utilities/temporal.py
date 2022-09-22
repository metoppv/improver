# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
from typing import List, Optional, Union

import cf_units
import iris
import numpy as np
from cftime import DatetimeGregorian
from iris import Constraint
from iris.coords import AuxCoord, Coord
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError
from iris.time import PartialDateTime
from numpy import int64

from improver import PostProcessingPlugin
from improver.metadata.check_datatypes import enforce_dtype
from improver.metadata.constants.time_types import TIME_COORDS
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.cube_checker import spatial_coords_match
from improver.utilities.cube_manipulation import MergeCubes, enforce_coordinate_ordering


def cycletime_to_datetime(
    cycletime: str, cycletime_format: str = "%Y%m%dT%H%MZ"
) -> datetime:
    """Convert a string representating the cycletime of the
    format YYYYMMDDTHHMMZ into a datetime object.

    Args:
        cycletime:
            A cycletime that can be converted into a datetime using the
            cycletime_format supplied.
        cycletime_format:
            String containing the desired format for the cycletime.

    Returns:
        A correctly formatted datetime object.
    """
    return datetime.strptime(cycletime, cycletime_format)


def datetime_to_cycletime(
    adatetime: datetime, cycletime_format: str = "%Y%m%dT%H%MZ"
) -> str:
    """Convert a datetime object into a string representing the cycletime
    of the format YYYYMMDDTHHMMZ.

    Args:
        adatetime:
            A datetime that can be converted into a cycletime using the
            cycletime_format supplied.
        cycletime_format:
            String containing the desired format for the cycletime.

    Returns:
        A correctly formatted string.
    """
    return datetime.strftime(adatetime, cycletime_format)


def cycletime_to_number(
    cycletime: str,
    cycletime_format: str = "%Y%m%dT%H%MZ",
    time_unit: str = "hours since 1970-01-01 00:00:00",
    calendar: str = "gregorian",
) -> float:
    """Convert a cycletime of the format YYYYMMDDTHHMMZ into a numeric
    time value.

    Args:
        cycletime:
            A cycletime that can be converted into a datetime using the
            cycletime_format supplied.
        cycletime_format:
            String containg the appropriate directives to indicate how
            the output datetime should display.
        time_unit:
            String representation of the cycletime units.
        calendar:
            String describing the calendar used for defining the cycletime.
            The choice of calendar must be supported by cf_units.CALENDARS.

    Returns:
        A numeric value to represent the datetime using assumed choices
        for the unit of time and the calendar.
    """
    dtval = cycletime_to_datetime(cycletime, cycletime_format=cycletime_format)
    return cf_units.date2num(dtval, time_unit, calendar)


def iris_time_to_datetime(
    time_coord: Coord, point_or_bound: str = "point"
) -> List[datetime]:
    """
    Convert iris time to python datetime object. Working in UTC.

    Args:
        time_coord:
            Iris time coordinate element(s).

    Returns:
        The time element(s) recast as a python datetime object.
    """
    coord = time_coord.copy()
    coord.convert_units("seconds since 1970-01-01 00:00:00")
    if point_or_bound == "point":
        datetime_list = [value.point for value in coord.cells()]
    elif point_or_bound == "bound":
        datetime_list = [value.bound for value in coord.cells()]
    return datetime_list


def datetime_to_iris_time(dt_in: Union[datetime, DatetimeGregorian]) -> int64:
    """
    Convert python datetime.datetime or cftime.DatetimeGregorian object into
    seconds since 1970-01-01 00Z.

    Args:
        dt_in:
            Time to be converted into seconds since 1970-01-01 00Z.

    Returns:
        Time since epoch in the seconds.
    """
    if isinstance(dt_in, DatetimeGregorian):
        dt_in = datetime(
            dt_in.year, dt_in.month, dt_in.day, dt_in.hour, dt_in.minute, dt_in.second
        )
    result = dt_in.replace(tzinfo=timezone.utc).timestamp()
    return np.int64(result)


def datetime_constraint(
    time_in: datetime, time_max: Optional[datetime] = None
) -> Constraint:
    """
    Constructs an iris equivalence constraint from a python datetime object.

    Args:
        time_in:
            The time to be used to build an iris constraint.
        time_max:
            Optional max time, which if provided leads to a range constraint
            being returned up to < time_max.

    Returns:
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


def extract_cube_at_time(
    cubes: CubeList, time: datetime, time_extract: Constraint
) -> Cube:
    """
    Extract a single cube at a given time from a cubelist.

    Args:
        cubes:
            CubeList of a given diagnostic over several times.
        time:
            Time at which forecast data is needed.
        time_extract:
            Iris constraint for the desired time.

    Returns:
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


def extract_nearest_time_point(
    cube: Cube, dt: datetime, time_name: str = "time", allowed_dt_difference: int = 0
) -> Cube:
    """Find the nearest time point to the time point provided.

    Args:
        cube:
            Cube or CubeList that will be extracted from using the supplied
            time_point
        dt:
            Datetime representation of a time that will be used within the
            extraction from the cube supplied.
        time_name:
            Name of the "time" coordinate that will be extracted. This must be
            "time" or "forecast_reference_time".
        allowed_dt_difference:
            An int in seconds to define a limit to the maximum difference
            between the datetime provided and the time points available within
            the cube. If this limit is exceeded, then an error is raised.
            This must be defined in seconds.
            Default is 0.

    Returns:
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


def relabel_to_period(cube: Cube, period: Optional[int] = None):
    """Add or replace bounds for the forecast period and time coordinates
    on a cube.

    Args:
        cube:
            The cube for a diagnostic that will be modified to represent the
            required period.
        period:
            The period in hours.

    Returns:
        Cube with metadata updated to represent the specified period.
    """
    if period is None:
        msg = (
            "A period must be specified when relabelling a diagnostic "
            "to have a particular period."
        )
        raise ValueError(msg)
    elif period < 1:
        msg = (
            "Only periods of one hour or greater are supported. "
            f"The period supplied was {period} hours."
        )
        raise ValueError(msg)

    for coord in ["forecast_period", "time"]:
        cube.coord(coord).bounds = np.array(
            [cube.coord(coord).points[0] - period * 3600, cube.coord(coord).points[0]],
            dtype=TIME_COORDS[coord].dtype,
        )
    return cube


class TimezoneExtraction(PostProcessingPlugin):
    """Plugin to extract local time offsets"""

    def __init__(self) -> None:
        self.time_coord_standards = TIME_COORDS["time"]
        self.time_points = None
        self.time_bounds = None
        self.time_units = cf_units.Unit(
            self.time_coord_standards.units,
            calendar=self.time_coord_standards.calendar,
        )
        self.timezone_cube = None
        self.output_data = None

    def create_output_cube(self, cube: Cube, local_time: datetime) -> Cube:
        """
        Constructs the output cube

        Args:
            cube:
                Cube of data to extract timezone-offsets from. Must contain a time
                coord spanning all the timezones.
            local_time:
                The "local" time of the output cube as %Y%m%dT%H%MZ. This will form a
                scalar "time_in_local_timezone" coord on the output cube, while the
                "time" coord will be auxillary to the spatial coords and will show the
                UTC time that matches the local_time at each point.

        """
        template_cube = cube.slices_over("time").next().copy()
        template_cube.remove_coord("time")
        template_cube.remove_coord("forecast_period")
        output_cube = create_new_diagnostic_cube(
            template_cube.name(),
            template_cube.units,
            template_cube,
            generate_mandatory_attributes([template_cube]),
            optional_attributes=template_cube.attributes,
            data=self.output_data,
        )

        # Copy cell-methods from template_cube
        [output_cube.add_cell_method(cm) for cm in template_cube.cell_methods]

        # Create a local time coordinate to help with plotting data.
        local_time_coord_standards = TIME_COORDS["time_in_local_timezone"]
        local_time_units = cf_units.Unit(
            local_time_coord_standards.units,
            calendar=local_time_coord_standards.calendar,
        )
        timezone_points = np.array(
            np.round(local_time_units.date2num(local_time)),
            dtype=local_time_coord_standards.dtype,
        )
        output_cube.add_aux_coord(
            AuxCoord(
                timezone_points,
                long_name="time_in_local_timezone",
                units=local_time_units,
            )
        )
        output_cube.add_aux_coord(
            AuxCoord(
                self.time_points,
                bounds=self.time_bounds,
                standard_name="time",
                units=self.time_units,
            ),
            [n + output_cube.ndim for n in [-2, -1]],
        )
        return output_cube

    def _fill_timezones(self, input_cube: Cube) -> None:
        """
        Populates the output cube data with data from input_cube. This is done by
        multiplying the inverse of the timezone_cube.data with the input_cube.data and
        summing along the time axis. Because timezone_cube.data is a mask of 1 and 0,
        inverting it gives 1 where we WANT data and 0 where we don't. Summing these up
        produces the result. The same logic can be used for times.
        Modifies self.output_cube and self.time_points.
        Assumes that input_cube and self.timezone_cube have been arranged so that time
        or UTC_offset are the inner-most coord (dim=-1).

        Args:
            input_cube:
                Cube of data to extract timezone-offsets from. Must contain a time
                coord spanning all the timezones.

        Raises:
            TypeError:
                If combining the timezone_cube and input_cube results in float64 data.
                (Hint: timezone_cube should be int8 and input cube should be float32)
        """
        # Get the output_data
        result = input_cube.data * (1 - self.timezone_cube.data)
        self.output_data = result.sum(axis=-1)

        # Check resulting dtype
        enforce_dtype("multiply", [input_cube, self.timezone_cube], result)

        # Sort out the time points
        input_time_points = input_cube.coord("time").points
        # Add scalar coords to allow broadcast to spatial coords.
        times = input_time_points.reshape((1, 1, len(input_time_points))) * (
            1 - self.timezone_cube.data
        )
        self.time_points = times.sum(axis=-1)

        # Sort out the time bounds (if present)
        if input_cube.coord("time").bounds is not None:
            bounds = []
            # Index 0 = lower bound, index 1 = upper bound
            for index in [0, 1]:
                time_bounds = input_cube.coord("time").bounds[:, index]
                time_bounds = time_bounds.reshape((1, 1, len(time_bounds))) * (
                    1 - self.timezone_cube.data
                )
                bounds.append(time_bounds.sum(axis=-1))

            self.time_bounds = np.stack(bounds, axis=-1)

    def check_input_cube_dims(self, input_cube: Cube) -> None:
        """Ensures input cube has at least three dimensions: time, y, x. Promotes time
        to be the inner-most dimension (dim=-1).

        Raises:
            ValueError:
                If the input cube does not have exactly the expected three coords.
                If the spatial coords on input_cube and timezone_cube do not match.
        """
        time_coord_name = "time"
        expected_coords = [time_coord_name] + [
            input_cube.coord(axis=n).name() for n in "yx"
        ]
        cube_coords = [coord.name() for coord in input_cube.coords(dim_coords=True)]
        if not all(
            [expected_coord in cube_coords for expected_coord in expected_coords]
        ):
            try:
                time_aux_dim = input_cube.coord_dims(time_coord_name)
            except CoordinateNotFoundError as err:
                raise CoordinateNotFoundError(
                    f"Expected coords on input_cube: time, y, x ({expected_coords})."
                    f"Found {cube_coords}"
                ) from err

            temporary_time_dimcoord = input_cube.coord(time_coord_name).copy()
            temporary_time_dimcoord.bounds = None
            time_coord_name = "time_points"
            temporary_time_dimcoord.rename(time_coord_name)
            input_cube.add_aux_coord(temporary_time_dimcoord, time_aux_dim)
            iris.util.promote_aux_coord_to_dim_coord(input_cube, time_coord_name)

        enforce_coordinate_ordering(input_cube, [time_coord_name], anchor_start=False)

        # Remove the temporary name for the anonymous time dimension
        if time_coord_name != "time":
            input_cube.remove_coord(time_coord_name)
        if not spatial_coords_match([input_cube, self.timezone_cube]):
            raise ValueError(
                "Spatial coordinates on input_cube and timezone_cube do not match."
            )

    def check_input_cube_time(self, input_cube: Cube, local_time: datetime) -> bool:
        """Ensures input cube and timezone_cube cover exactly the right points and that
        the time and UTC_offset coords have the same order. If not a warning is raised
        and the plugin will return nothing.

        Time points are compared as these fall at the end of time periods under IMPROVER
        definitions. This means that a partial period, e.g. 15-00, is allowed as long
        as it runs to the end of the intended period. Any period that is curtailed such
        that it doesn't reach the end of intended period, e.g. 00-15 will not be allowed.
        This means that we can update same day forecasts with partial periods, but we
        don't end up with a whole day summary temperature / weather symbol etc. at long
        lead-times that is not really a whole day.

        Returns:
            True if appropriate input data has been provided, False if not.
        Raises:
            ValueError:
                If the time coord on the input cube does not match the required times.
        """
        input_time_points = [cell.point for cell in input_cube.coord("time").cells()]
        # timezone_cube.coord("UTC_offset") is monotonically increasing. It needs to be
        # decreasing so that the required UTC time for local_time will be increasing
        # when it is calculated.
        enforce_coordinate_ordering(
            self.timezone_cube, ["UTC_offset"], anchor_start=False
        )
        self.timezone_cube = self.timezone_cube[:, :, ::-1]

        timezone_coord = self.timezone_cube.coord("UTC_offset")
        timezone_coord.convert_units("seconds")
        output_times = [
            local_time - timedelta(seconds=np.int(offset))
            for offset in timezone_coord.points
        ]
        if input_time_points != output_times:
            if input_cube.coord("time").bounds is not None:
                # When mapping to timezones, the input cube contains data at times that
                # correspond with different timezones. The earliest times correspond to
                # the eastern most timezones, and the latest time to the western most.
                # Incomplete periods in the east are indicative of a same day partial
                # period, i.e. producing a 15-00 instead of 00-00 forecast for Japan.
                # These are desirable as they update the same day forecast.
                # Periods in the west that are shorter than periods in the east are
                # indicative of a data shortfall for representing the whole period at
                # the longest lead-times, i.e. 00-15 instead of 00-00. These should not
                # return output as the period summary would not represent the expected
                #  whole period for these regions.
                bounds = input_cube.coord("time").bounds
                b_diffs = np.diff(bounds)
                if len(b_diffs) > 1 and any(b_diffs[-1] < b_diffs[:-1]):
                    return False

            raise ValueError(
                "Time coord on input cube does not match required times. Expected\n"
                + "\n".join([f"{t:%Y%m%dT%H%MZ}" for t in output_times])
                + "\nFound:\n"
                + "\n".join([f"{t:%Y%m%dT%H%MZ}" for t in input_time_points])
            )
        return True

    def check_timezones_are_unique(self) -> None:
        """Ensures that each grid point falls into exactly one time zone

        Raises:
            ValueError:
                If the timezone_cube does not map exactly one time zone to each spatial
                point.
        """
        if not ((1 - self.timezone_cube.data).sum(axis=0) == 1).all():
            raise ValueError(
                "Timezone cube does not map exactly one time zone to each spatial point"
            )

    def process(
        self,
        input_cubes: Union[CubeList, List[Cube]],
        timezone_cube: Cube,
        local_time: datetime,
    ) -> Cube:
        """
        Calculates timezone-offset data for the specified UTC output times

        Args:
            input_cubes:
                Cube or list of cubes of data to extract timezone-offsets from. Must
                contain a time coord spanning all the timezones.
            timezone_cube:
                Cube describing the UTC offset for the local time at each grid location.
                Must have the same spatial coords as input_cube.
            local_time:
                The "local" time of the output cube. This will form a
                scalar "time_in_local_timezone" coord on the output cube, while the
                "time" coord will be auxillary to the spatial coords and will show the
                UTC time that matches the local_time at each point.

        Returns:
            Output local-time cube. The time coord will span the spatial coords.
            The time_in_local_timezone coord will match the local_time supplied.
            All other coords and attributes will match those found on input_cube.
        """
        if isinstance(input_cubes, iris.cube.Cube):
            input_cube = input_cubes
        else:
            input_cube = MergeCubes()(CubeList(input_cubes))

        self.timezone_cube = timezone_cube.copy()
        self.check_timezones_are_unique()
        if not self.check_input_cube_time(input_cube, local_time):
            return None
        self.check_input_cube_dims(input_cube)

        self._fill_timezones(input_cube)
        output_cube = self.create_output_cube(input_cube, local_time)

        return output_cube
