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
"""Script to extrapolate input data given advection velocity fields."""

import iris
from iris import Constraint

from improver.argparser import ArgParser
from improver.nowcasting.pysteps_advection import PystepsExtrapolate
from improver.utilities.cli_utilities import load_json_or_none
from improver.utilities.cube_manipulation import merge_cubes
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf
from improver.wind_calculations.wind_components import ResolveWindComponents


def main(argv=None):
    """Extrapolate data forward in time."""

    parser = ArgParser(
        description="Extrapolate input data to required lead times.")
    parser.add_argument("input_filepath", metavar="INPUT_FILEPATH",
                        type=str, help="Path to input NetCDF file.")
    parser.add_argument("output_filepath", metavar="OUTPUT_FILEPATH",
                        help="The output path for the resulting NetCDF")

    speed = parser.add_argument_group('Advect using files containing speed and'
                                      ' direction')
    speed.add_argument("--advection_speed_filepath", type=str, help="Path"
                       " to input file containing advection speeds,"
                       " usually wind speeds, on multiple pressure levels.")
    speed.add_argument("--advection_direction_filepath", type=str,
                       help="Path to input file containing the directions from"
                       " which advection speeds are coming (180 degrees from"
                       " the direction in which the speed is directed). The"
                       " directions should be on the same grid as the input"
                       " speeds, including the same vertical levels.")
    speed.add_argument("--pressure_level", type=int, default=75000, help="The"
                       " pressure level in Pa to extract from the multi-level"
                       " advection_speed and advection_direction files. The"
                       " velocities at this level are used for advection.")
    parser.add_argument("--orographic_enhancement_filepaths", nargs="+",
                        type=str, default=None, help="List or wildcarded "
                        "file specification to the input orographic "
                        "enhancement files. Orographic enhancement files are "
                        "compulsory for precipitation fields.")
    parser.add_argument("--json_file", metavar="JSON_FILE", default=None,
                        help="Filename for the json file containing "
                        "required changes to the attributes. "
                        "Defaults to None.", type=str)
    parser.add_argument("--max_lead_time", type=int, default=360,
                        help="Maximum lead time required (mins).")
    parser.add_argument("--lead_time_interval", type=int, default=15,
                        help="Interval between required lead times (mins).")
    parser.add_argument("--u_and_v_filepath", type=str, help="Path to u and v"
                        " cubelist.  This cubelist should contains eastward "
                        "and northwards advection velocities. These advection "
                        "velocities will be extracted with the constraint "
                        "'precipitation_advection_[x or y]_velocity'"
                        " for x or y.'")

    args = parser.parse_args(args=argv)

    v_cube = load_cube(args.u_and_v_filepath,
                       "precipitation_advection_y_velocity", allow_none=True)
    u_cube = load_cube(args.u_and_v_filepath,
                       "precipitation_advection_x_velocity", allow_none=True)

    # Load Cubes and JSON
    speed_cube = direction_cube = None

    input_cube = load_cube(args.input_filepath)
    orographic_enhancement_cube = load_cube(
        args.orographic_enhancement_filepaths, allow_none=True)

    s_path, d_path = (args.advection_speed_filepath,
                      args.advection_direction_filepath)
    level_constraint = Constraint(pressure=args.pressure_level)
    if s_path and d_path:
        try:
            speed_cube = load_cube(s_path, constraints=level_constraint)
            direction_cube = load_cube(d_path, constraints=level_constraint)
        except ValueError as err:
            raise ValueError(
                '{} Unable to extract specified pressure level from given '
                'speed and direction files.'.format(err))

    attributes_dict = load_json_or_none(args.json_file)
    # Process Cubes
    result = process(
        input_cube, u_cube, v_cube, speed_cube, direction_cube,
        orographic_enhancement_cube, attributes_dict, args.max_lead_time,
        args.lead_time_interval)

    # Save Cube
    save_netcdf(result, args.output_filepath)


def process(input_cube, u_cube=None, v_cube=None, speed_cube=None,
            direction_cube=None, orographic_enhancement_cube=None,
            attributes_dict=None, max_lead_time=360, lead_time_interval=15):
    """Module  to extrapolate input cubes given advection velocity fields.

    Args:
        input_cube (iris.cube.Cube):
            The input Cube to be processed.
        u_cube (iris.cube.Cube):
            Cube with the velocities in the x direction.
            Must be used with v_cube.
            speed_cube and direction_cube must be None.
        v_cube (iris.cube.Cube):
            Cube with the velocities in the y direction.
            Must be used with u_cube.
            speed_cube and direction_cube must be None.
        speed_cube (iris.cube.Cube):
            Cube containing advection speeds, usually wind speed.
            Must be used with direction_cube.
            u_cube and v_cube must be None.
        direction_cube (iris.cube.Cube):
            Cube from which advection speeds are coming. The directions
            should be on the same grid as the input speeds, including the same
            vertical levels.
            Must be used with speed_cube.
            u_cube and v_cube must be None.
        orographic_enhancement_cube (iris.cube.Cube):
            Cube containing the orographic enhancement fields. May have data
            for multiple times in the cube.
            Default is None.
        attributes_dict (dict):
            Dictionary containing the required changes to the attributes.
            Default is None.
        max_lead_time (int):
            Maximum lead time required (mins).
            Default is 360.
        lead_time_interval (int):
            Interval between required lead times (mins).
            Default is 15.

    Returns:
        iris.cube.CubeList:
            New cubes with updated time and extrapolated data.

    Raises:
        ValueError:
            can either use speed_cube and direction_cube or u_cube and v_cube.
    """
    if (speed_cube or direction_cube) and (u_cube or v_cube):
        raise ValueError('Cannot mix advection component velocities with speed'
                         ' and direction')
    if not (speed_cube and direction_cube) and not (u_cube and v_cube):
        raise ValueError('Either speed and direction or u and v cubes '
                         'are needed.')

    if speed_cube and direction_cube:
        u_cube, v_cube = ResolveWindComponents().process(
            speed_cube, direction_cube)

    # extrapolate input data to required lead times
    forecast_plugin = PystepsExtrapolate(lead_time_interval, max_lead_time)
    forecast_cubes = forecast_plugin.process(input_cube, u_cube, v_cube,
                                             orographic_enhancement_cube,
                                             attributes_dict=attributes_dict)

    return merge_cubes(forecast_cubes)


if __name__ == "__main__":
    main()
