#!/usr/bin/env python
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
"""Script to interpolate data between validity times"""

from improver import cli


@cli.clizefy
@cli.with_output
def process(start_cube: cli.inputcube,
            end_cube: cli.inputcube,
            *,
            interval_in_mins: int = None,
            times: cli.comma_separated_list = None,
            interpolation_method='linear'):
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

    Returns:
        iris.cube.CubeList:
            A list of cubes interpolated to the desired times. The
            interpolated cubes will always be in chronological order of
            earliest to latest regardless of the order of the input.
    """
    from improver.utilities.cube_manipulation import merge_cubes
    from improver.utilities.temporal import (
        cycletime_to_datetime, iris_time_to_datetime)
    from improver.utilities.temporal_interpolation import TemporalInterpolation

    time_start, = iris_time_to_datetime(start_cube.coord('time'))
    time_end, = iris_time_to_datetime(end_cube.coord('time'))
    if time_end < time_start:
        # swap cubes
        start_cube, end_cube = end_cube, start_cube

    if times is not None:
        times = [cycletime_to_datetime(timestr) for timestr in times]

    result = TemporalInterpolation(
        interval_in_minutes=interval_in_mins, times=times,
        interpolation_method=interpolation_method
    ).process(start_cube, end_cube)
    return merge_cubes(result)
