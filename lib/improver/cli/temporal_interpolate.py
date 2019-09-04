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

from improver.argparser import ArgParser
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf
from improver.utilities.temporal import (
    cycletime_to_datetime, iris_time_to_datetime)
from improver.utilities.temporal_interpolation import TemporalInterpolation


def main(argv=None):
    """
    Interpolate data to intermediate times between the validity times of two
    cubes. This can be used to fill in missing data (e.g. for radar fields) or
    to ensure data is available at the required intervals when model data is
    not available at these times.
    """
    parser = ArgParser(
        description='Interpolate data between validity times ')

    parser.add_argument('infiles', metavar='INFILES', nargs=2,
                        help='Files contain the data at the beginning'
                        ' and end of the period (2 files required).')

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--interval_in_mins", metavar="INTERVAL_IN_MINS",
                       default=None, type=int,
                       help="Specifies the interval in minutes"
                       " at which to interpolate "
                       "between the two input cubes."
                       " A number of minutes which does not "
                       "divide up the interval equally will "
                       "raise an exception. If intervals_in_mins "
                       "is set then times can not be set.")

    group.add_argument("--times", metavar="TIMES",
                       default=None, nargs="+", type=str,
                       help="Specifies the times in the format "
                       "{YYYYMMDD}T{HHMM}Z "
                       " at which to interpolate "
                       "between the two input cubes."
                       "Where {YYYYMMDD} is year, month day "
                       "and {HHMM} is hour and minutes e.g "
                       "20180116T0100Z. More than one time"
                       "can be provided separated by a space "
                       "but if times are set interval_in_mins "
                       "can not be set")

    parser.add_argument("--interpolation_method",
                        metavar="INTERPOLATION_METHOD",
                        default="linear",
                        choices=["linear", "solar", "daynight"],
                        help="Specifies the interpolation method; "
                        "solar interpolates using the solar elevation, "
                        "daynight uses linear interpolation but sets"
                        " night time points to 0.0, "
                        "linear is linear interpolation. "
                        "Default is linear.")

    parser.add_argument("--output_files", metavar="OUTPUT_FILES",
                        required=True, nargs="+",
                        help="List of output files."
                        " The interpolated files will always be"
                        " in the chronological order of"
                        " earliest to latest "
                        " regardless of the order of the infiles.")

    args = parser.parse_args(args=argv)

    # Load Cubes
    cube_0 = load_cube(args.infiles[0])
    cube_1 = load_cube(args.infiles[1])

    # Process Cubes
    interpolated_cubes = process(cube_0, cube_1, args.interval_in_mins,
                                 args.times, args.interpolation_method)

    # Save Cubes
    len_files = len(args.output_files)
    len_cubes = len(interpolated_cubes)
    if len_files == len_cubes:
        for i, cube_out in enumerate(interpolated_cubes):
            save_netcdf(cube_out, args.output_files[i])
    else:
        msg = ("Output_files do not match cubes created. "
               "{} files given but {} required.".format(len_files,
                                                        len_cubes))
        raise ValueError(msg)


def process(cube_0, cube_1, interval_in_mins=None, in_times=None,
            interpolation_method='linear'):
    """Module to interpolate data between validity times.

    Interpolate data to intermediate times between the validity times of two
    cubes. This can be used to fill in missing data (e.g. for radar fields)
    or to ensure data is available at the required intervals when model data
    is not available at these times.

    Args:
        cube_0 (iris.cube.Cube):
            Cube containing the data at the beginning.
        cube_1 (iris.cube.Cube):
            Cube containing the data at the end.
        interval_in_mins (int):
            Specifies the interval in minutes at which to interpolate between
            the two input cubes.
            A number of minutes which does not divide up the interval equally
            will raise an exception.
            If intervals_in_mins is set then in_times can not be used.
            Default is None.
        in_times (str):
            Specifies the times in the format {YYYYMMDD}T{HHMM}Z
            at which to interpolate between the two input cubes.
            Where {YYYYMMDD} is year, month, day and {HHMM} is hour and minutes
            e.g 20180116T0100Z. More than one time can be provided separated
            by a space.
            If in_times are set, interval_in_mins can not be used.
            Default is None.
        interpolation_method (str):
            ["linear", "solar", "daynight"]
            Specifies the interpolation method;
            solar interpolates using the solar elevation,
            daynight uses linear interpolation but sets night time points to
            0.0 linear is linear interpolation.
            Default is linear.

    Returns:
        result (iris.cube.Cubelist):
            A list of cubes interpolated to the desired times. The
            interpolated cubes will always be in chronological order of
            earliest to latest regardless of the order of the input.
    """
    time_0, = iris_time_to_datetime(cube_0.coord('time'))
    time_1, = iris_time_to_datetime(cube_1.coord('time'))
    if time_0 < time_1:
        cube_start = cube_0
        cube_end = cube_1
    else:
        cube_start = cube_1
        cube_end = cube_0

    interval = interval_in_mins
    method = interpolation_method
    if in_times is None:
        times = in_times
    else:
        times = []
        for timestr in in_times:
            times.append(cycletime_to_datetime(timestr))

    result = (
        TemporalInterpolation(interval_in_minutes=interval,
                              times=times,
                              interpolation_method=method).process(cube_start,
                                                                   cube_end))
    return result


if __name__ == "__main__":
    main()
