#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to interpolate data between validity times"""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    start_cube: cli.inputcube,
    end_cube: cli.inputcube,
    *,
    interval_in_mins: int = None,
    times: cli.comma_separated_list = None,
    interpolation_method="linear",
    accumulation: bool = False,
    max: bool = False,
    min: bool = False,
):
    """Interpolate data between validity times.

    Interpolate data to intermediate times between the validity times of two
    cubes. This can be used to fill in missing data (e.g. for radar fields)
    or to ensure data is available at the required intervals when model data
    is not available at these times.

    Args:
        start_cube (iris.cube.Cube):
            Cube containing the data at the beginning.
        end_cube (iris.cube.Cube):
            Cube containing the data at the end.
        interval_in_mins (int):
            Specifies the interval in minutes at which to interpolate between
            the two input cubes.
            A number of minutes which does not divide up the interval equally
            will raise an exception.
            If intervals_in_mins is set then times can not be used.
        times (str):
            Specifies the times in the format {YYYYMMDD}T{HHMM}Z
            at which to interpolate between the two input cubes.
            Where {YYYYMMDD} is year, month, day and {HHMM} is hour and minutes
            e.g 20180116T0100Z. More than one time can be provided separated
            by a comma.
            If times are set, interval_in_mins can not be used.
        interpolation_method (str):
            ["linear", "solar", "daynight"]
            Specifies the interpolation method;
            solar interpolates using the solar elevation,
            daynight uses linear interpolation but sets night time points to
            0.0 linear is linear interpolation.
        accumulation:
            Set True if the diagnostic being temporally interpolated is a
            period accumulation. The output will be renormalised to ensure
            that the total across the period constructed from the shorter
            intervals matches the total across the period from the coarser
            intervals. Trends between adjacent input periods will be used
            to provide variation across the interpolated periods.
        max:
            Set True if the diagnostic being temporally interpolated is a
            period maximum. Trends between adjacent input periods will be used
            to provide variation across the interpolated periods where these
            are consistent with the inputs.
        min:
            Set True if the diagnostic being temporally interpolated is a
            period minimum. Trends between adjacent input periods will be used
            to provide variation across the interpolated periods where these
            are consistent with the inputs.
    Returns:
        iris.cube.CubeList:
            A list of cubes interpolated to the desired times. The
            interpolated cubes will always be in chronological order of
            earliest to latest regardless of the order of the input.
    """
    from improver.utilities.cube_manipulation import MergeCubes
    from improver.utilities.temporal import cycletime_to_datetime, iris_time_to_datetime
    from improver.utilities.temporal_interpolation import TemporalInterpolation

    (time_start,) = iris_time_to_datetime(start_cube.coord("time"))
    (time_end,) = iris_time_to_datetime(end_cube.coord("time"))
    if time_end < time_start:
        # swap cubes
        start_cube, end_cube = end_cube, start_cube

    if times is not None:
        times = [cycletime_to_datetime(timestr) for timestr in times]

    result = TemporalInterpolation(
        interval_in_minutes=interval_in_mins,
        times=times,
        interpolation_method=interpolation_method,
        accumulation=accumulation,
        max=max,
        min=min,
    )(start_cube, end_cube)
    return MergeCubes()(result)
