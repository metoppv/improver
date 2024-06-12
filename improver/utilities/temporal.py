# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""General utilities for parsing and extracting cubes at times"""

import warnings
from datetime import datetime, timezone
from typing import List, Optional, Union

import cf_units
import iris
import numpy as np
from cftime import DatetimeGregorian
from iris import Constraint
from iris.coords import CellMethod, Coord
from iris.cube import Cube, CubeList
from iris.time import PartialDateTime
from numpy import int64

from improver.metadata.constants.time_types import DT_FORMAT, TIME_COORDS


def cycletime_to_datetime(
    cycletime: str, cycletime_format: str = DT_FORMAT
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
    adatetime: datetime, cycletime_format: str = DT_FORMAT
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
    cycletime_format: str = DT_FORMAT,
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


def integrate_time(cube: Cube, new_name: str = None) -> Cube:
    """
    Multiply a frequency or rate cube by the time period given by the
    time bounds over which it is defined to return a count or accumulation.
    The frequency or rate must be defined with time bounds, e.g. an average
    frequency across the period.

    The returned cube has units equivalent to the input cube multiplied by
    seconds.

    Any time related cell methods are removed from the output cube and a new
    "sum" over time cell method is added.

    Args:
        Cube:
            A cube of average frequency or rate within a defined period.
        new_name:
            A new name for the resulting diagnostic.

    Returns:
        The cube with the data multiplied by the period in seconds defined
        by the time time bounds

    Raises:
        ValueError: If the input cube time coordinate does not have time
                    bounds.
    """
    # Ensure cube has a time coordinate with bounds
    if not cube.coord("time").has_bounds():
        raise ValueError(
            "time coordinate must have bounds to apply this time-bounds " "integration"
        )

    # For each grid of data associated with a time, multiply the rate / frequency
    # by the associated time interval to get an accumulation / count over the
    # period.
    integrated_cube = iris.cube.CubeList()
    for cslice in cube.slices_over("time"):
        (multiplier,) = np.diff(cslice.coord("time").cell(0).bound)
        multiplier = multiplier.total_seconds()
        cslice.data *= multiplier
        integrated_cube.append(cslice)

    integrated_cube = integrated_cube.merge_cube()

    # Modify the cube units to reflect the multiplication by time.
    integrated_cube.units *= cf_units.Unit("s")
    if new_name is not None:
        integrated_cube.rename(new_name)

    # Add a suitable cell method to describe what has been done and remove
    # former cell methods associated with the time coordinate which are now
    # out of date.
    new_cell_method = CellMethod("sum", coords=["time"])
    new_cell_methods = [new_cell_method]
    for cm in integrated_cube.cell_methods:
        if not "time" in cm.coord_names:
            new_cell_methods.append(cm)

    integrated_cube.cell_methods = new_cell_methods

    return integrated_cube
