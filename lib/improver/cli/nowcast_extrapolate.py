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

import os

import iris
import numpy as np
from iris import Constraint

from improver.argparser import ArgParser
from improver.nowcasting.accumulation import Accumulation
from improver.nowcasting.forecasting import CreateExtrapolationForecast
from improver.utilities.cli_utilities import load_json_or_none
from improver.utilities.filename import generate_file_name
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf
from improver.wind_calculations.wind_components import ResolveWindComponents


def main(argv=None):
    """Extrapolate data forward in time."""

    parser = ArgParser(
        description="Extrapolate input data to required lead times.")
    parser.add_argument("input_filepath", metavar="INPUT_FILEPATH",
                        type=str, help="Path to input NetCDF file.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--output_dir", metavar="OUTPUT_DIR", type=str,
                       default="", help="Directory to write output files.")
    group.add_argument("--output_filepaths", nargs="+", type=str,
                       help="List of full paths to output nowcast files, in "
                       "order of increasing lead time.")

    optflw = parser.add_argument_group('Advect using files containing the x '
                                       ' and y components of the velocity')
    optflw.add_argument("--eastward_advection_filepath", type=str, help="Path"
                        " to input file containing Eastward advection "
                        "velocities.")
    optflw.add_argument("--northward_advection_filepath", type=str, help="Path"
                        " to input file containing Northward advection "
                        "velocities.")

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
                        "required changes to the metadata. Information "
                        "describing the intended contents of the json file "
                        "is available in "
                        "improver.utilities.cube_metadata.amend_metadata."
                        "Every output cube will have the metadata_dict "
                        "applied. Defaults to None.", type=str)
    parser.add_argument("--max_lead_time", type=int, default=360,
                        help="Maximum lead time required (mins).")
    parser.add_argument("--lead_time_interval", type=int, default=15,
                        help="Interval between required lead times (mins).")

    accumulation_args = parser.add_argument_group(
        'Calculate accumulations from advected fields')
    accumulation_args.add_argument(
        "--accumulation_fidelity", type=int, default=0,
        help="If set, this CLI will additionally return accumulations"
        " calculated from the advected fields. This fidelity specifies the"
        " time interval in minutes between advected fields that is used to"
        " calculate these accumulations. This interval must be a factor of"
        " the lead_time_interval.")
    accumulation_args.add_argument(
        "--accumulation_period", type=int, default=15,
        help="The period over which the accumulation is calculated (mins). "
        "Only full accumulation periods will be computed. At lead times "
        "that are shorter than the accumulation period, no accumulation "
        "output will be produced.")
    accumulation_args.add_argument(
        "--accumulation_units", type=str, default='m',
        help="Desired units in which the accumulations should be expressed,"
        "e.g. mm")

    # Load Cubes
    args = parser.parse_args(args=argv)

    metadata_dict = load_json_or_none(args.json_file)

    upath, vpath = (args.eastward_advection_filepath,
                    args.northward_advection_filepath)
    spath, dpath = (args.advection_speed_filepath,
                    args.advection_direction_filepath)

    # load files and initialise advection plugin
    input_cube = load_cube(args.input_filepath)
    orographic_enhancement_cube = load_cube(
        args.orographic_enhancement_filepaths, allow_none=True)

    speed_cube = direction_cube = ucube = vcube = None
    if (upath and vpath) and not (spath or dpath):
        ucube = load_cube(upath)
        vcube = load_cube(vpath)
    elif (spath and dpath) and not (upath or vpath):
        level_constraint = Constraint(pressure=args.pressure_level)
        try:
            speed_cube = load_cube(spath, constraints=level_constraint)
            direction_cube = load_cube(dpath, constraints=level_constraint)
        except ValueError as err:
            raise ValueError(
                '{} Unable to extract specified pressure level from given '
                'speed and direction files.'.format(err))
    else:
        raise ValueError('Cannot mix advection component velocities with speed'
                         ' and direction')

    # Process Cubes
    accumulation_cubes, forecast_to_return = process(
        input_cube, ucube, vcube, speed_cube, direction_cube,
        orographic_enhancement_cube, metadata_dict, args.max_lead_time,
        args.lead_time_interval, args.accumulation_fidelity,
        args.accumulation_period, args.accumulation_units)

    # Save Cube
    if args.output_filepaths and \
            len(args.output_filepaths) != len(forecast_to_return):
        raise ValueError("Require exactly one output file name for each "
                         "forecast lead time")
    for i, cube in enumerate(forecast_to_return):
        # save to a suitably-named output file
        if args.output_filepaths:
            file_name = args.output_filepaths[i]
        else:
            file_name = os.path.join(
                args.output_dir, generate_file_name(cube))
        save_netcdf(cube, file_name)

    if args.accumulation_fidelity > 0:
        # return accumulation cubes
        for i, cube in enumerate(accumulation_cubes):
            file_name = os.path.join(args.output_dir, generate_file_name(cube))
            save_netcdf(cube, file_name)


def process(input_cube, u_cube, v_cube, speed_cube, direction_cube,
            orographic_enhancement_cube=None, metadata_dict=None,
            max_lead_time=360, lead_time_interval=15, accumulation_fidelity=0,
            accumulation_period=15, accumulation_units='m'):
    """Module  to extrapolate input cubes given advection velocity fields.

    Args:
        input_cube (iris.cube.Cube):
            The input Cube to be processed.
        u_cube (iris.cube.Cube):
            Cube with the velocities in the x direction.
            Must be used with v_cube.
            s_cube and d_cube must be None.
        v_cube (iris.cube.Cube):
            Cube with the velocities in the y direction.
            Must be used with u_cube.
            s_cube and d_cube must be None.
        speed_cube (iris.cube.Cube):
            Cube containing advection speeds, usually wind speed.
            Must be used with d_cube.
            u_cube and v_cube must be None.
        direction_cube (iris.cube.Cube):
            Cube from which advection speeds are coming. The directions
            should be on the same grid as the input speeds, including the same
            vertical levels.
            Must be used with d_cube.
            u_cube and v_cube must be None.
        orographic_enhancement_cube (iris.cube.Cube):
            Cube containing the orographic enhancement fields. May have data
            for multiple times in the cube.
            Default is None.
        metadata_dict (dict):
            Dictionary containing the required changes to the metadata.
            Information describing the intended contents of the dictionary
            is available in improver.utilities.cube_metadata.amend_metadata.
            Every output cube will have the metadata_dict applied.
            Default is None.
        max_lead_time (int):
            Maximum lead time required (mins).
            Default is 360.
        lead_time_interval (int):
            Interval between required lead times (mins).
            Default is 15.
        accumulation_fidelity (int):
            If set, this will additionally return accumulations calculated
            from the advected fields. This fidelity specifies the time
            interval in minutes between advected fields that is used to
            calculate these accumulations. This interval must be a factor of
            the lead_time_interval.
            Default is 0.
        accumulation_period (int):
            The period over which the accumulation is calculated (mins).
            Only full accumulation periods will be computed. At lead times
            that are shorter than the accumulation period, no accumulation
            output will be produced.
        accumulation_units (str):
            Desired units in which the accumulations should be expressed.
            e.g. 'mm'
            Default is 'm'.

    Returns:
        (tuple) tuple containing:
            **accumulation_cubes** (iris.cube.Cubelist):
                A cubelist containing precipitation accumulation cubes where
                the accumulation periods are determined by the
                lead_time_interval.
            **forecast_to_return** (iris.cube.Cubelist):
                New cubes with updated time and extrapolated data.

    Raises:
        ValueError:
            can either use s_cube and d_cube or u_cube and v_cube.
            Therefore: (s and d)⊕(u and v)
        ValueError:
            If accumulation_fidelity is greater than 0 and max_lead_time is not
            cleanly divisible by accumulation_fidelity.
    """

    if (speed_cube and direction_cube) and not (u_cube or v_cube):
        u_cube, v_cube = ResolveWindComponents().process(
            speed_cube, direction_cube)
    elif (u_cube or v_cube) and (speed_cube or direction_cube):
        raise ValueError('Cannot mix advection component velocities with speed'
                         ' and direction')
    # generate list of lead times in minutes
    lead_times = np.arange(0, max_lead_time + 1, lead_time_interval)

    # determine whether accumulations are also to be returned.
    time_interval = lead_time_interval
    if accumulation_fidelity > 0:
        fraction, _ = np.modf(max_lead_time / accumulation_fidelity)
        if fraction != 0:
            msg = ("The specified lead_time_interval ({}) is not cleanly "
                   "divisible by the specified accumulation_fidelity ({}). As "
                   "a result the lead_time_interval cannot be constructed from"
                   " accumulation cubes at this fidelity.")
            raise ValueError(msg.format(lead_time_interval,
                                        accumulation_fidelity))

        time_interval = accumulation_fidelity
        lead_times = np.arange(0, max_lead_time + 1, time_interval)

    lead_time_filter = lead_time_interval // time_interval
    forecast_plugin = CreateExtrapolationForecast(
        input_cube, u_cube, v_cube,
        orographic_enhancement_cube=orographic_enhancement_cube,
        metadata_dict=metadata_dict)

    # extrapolate input data to required lead times
    forecast_cubes = iris.cube.CubeList()
    for i, lead_time in enumerate(lead_times):
        forecast_cubes.append(
            forecast_plugin.extrapolate(leadtime_minutes=lead_time))

    forecast_to_return = forecast_cubes[::lead_time_filter].copy()
    # return rate cubes
    # calculate accumulations if required
    accumulation_cubes = None
    if accumulation_fidelity > 0:
        lead_times = (
            np.arange(lead_time_interval, max_lead_time + 1,
                      lead_time_interval))
        plugin = Accumulation(
            accumulation_units=accumulation_units,
            accumulation_period=accumulation_period * 60,
            forecast_periods=lead_times * 60)
        accumulation_cubes = plugin.process(forecast_cubes)

    return accumulation_cubes, forecast_to_return


if __name__ == "__main__":
    main()
