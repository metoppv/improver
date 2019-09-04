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
"""Script to calculate optical flow advection velocities with option to
extrapolate."""

import os

import iris
import numpy as np

from improver.argparser import ArgParser
from improver.nowcasting.forecasting import CreateExtrapolationForecast
from improver.nowcasting.optical_flow import OpticalFlow
from improver.nowcasting.utilities import ApplyOrographicEnhancement
from improver.utilities.cli_utilities import load_json_or_none
from improver.utilities.filename import generate_file_name
from improver.utilities.load import load_cubelist, load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
    """Calculate optical flow advection velocities and (optionally)
    extrapolate data."""

    parser = ArgParser(
        description="Calculate optical flow components from input fields "
        "and (optionally) extrapolate to required lead times.")

    parser.add_argument("input_filepaths", metavar="INPUT_FILEPATHS",
                        nargs=3, type=str, help="Paths to the input radar "
                        "files. There should be 3 input files at T, T-1 and "
                        "T-2 from which to calculate optical flow velocities. "
                        "The files require a 'time' coordinate on which they "
                        "are sorted, so the order of inputs does not matter.")
    parser.add_argument("--output_dir", metavar="OUTPUT_DIR", type=str,
                        default='', help="Directory to write all output files,"
                        " or only advection velocity components if "
                        "NOWCAST_FILEPATHS is specified.")
    parser.add_argument("--nowcast_filepaths", nargs="+", type=str,
                        default=None, help="Optional list of full paths to "
                        "output nowcast files. Overrides OUTPUT_DIR. Ignored "
                        "unless '--extrapolate' is set.")
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

    # OpticalFlow plugin configurable parameters
    parser.add_argument("--ofc_box_size", type=int, default=30, help="Size of "
                        "square 'box' (in grid squares) within which to solve "
                        "the optical flow equations.")
    parser.add_argument("--smart_smoothing_iterations", type=int, default=100,
                        help="Number of iterations to perform in enforcing "
                        "smoothness constraint for optical flow velocities.")

    # AdvectField options
    parser.add_argument("--extrapolate", action="store_true", default=False,
                        help="Optional flag to advect current data forward to "
                        "specified lead times.")
    parser.add_argument("--max_lead_time", type=int, default=360,
                        help="Maximum lead time required (mins).  Ignored "
                        "unless '--extrapolate' is set.")
    parser.add_argument("--lead_time_interval", type=int, default=15,
                        help="Interval between required lead times (mins). "
                        "Ignored unless '--extrapolate' is set.")

    args = parser.parse_args(args=argv)

    # Load Cubes and JSON.
    metadata_dict = load_json_or_none(args.json_file)
    original_cube_list = load_cubelist(args.input_filepaths)
    oe_cube = load_cube(args.orographic_enhancement_filepaths,
                        allow_none=True)

    # Process
    forecast_cubes, u_and_v_mean = process(
        original_cube_list, oe_cube, metadata_dict, args.ofc_box_size,
        args.smart_smoothing_iterations, args.extrapolate,
        args.max_lead_time, args.lead_time_interval)

    # Save Cubes
    for wind_cube in u_and_v_mean:
        file_name = generate_file_name(wind_cube)
        save_netcdf(wind_cube, os.path.join(args.output_dir, file_name))

    # advect latest input data to the required lead times
    if args.extrapolate:
        if args.nowcast_filepaths:
            if len(args.nowcast_filepaths) != len(forecast_cubes):
                raise ValueError("Require exactly one output file name for "
                                 "each forecast lead time")

        for i, cube in enumerate(forecast_cubes):
            # save to a suitably-named output file
            if args.nowcast_filepaths:
                file_name = args.nowcast_filepaths[i]
            else:
                file_name = os.path.join(
                    args.output_dir, generate_file_name(cube))
            save_netcdf(cube, file_name)


def process(original_cube_list, orographic_enhancement_cube=None,
            metadata_dict=None, ofc_box_size=30,
            smart_smoothing_iterations=100, extrapolate=False,
            max_lead_time=360, lead_time_interval=15):
    """Calculates optical flow and can (optionally) extrapolate data.

    Calculates optical flow components from input fields and (optionally)
    extrapolate to required lead times.

    Args:
        original_cube_list (iris.cube.CubeList):
            Cubelist from which to calculate optical flow velocities.
            The cubes require a 'time' coordinate on which they are sorted,
            so the order of cubes does not matter.
        orographic_enhancement_cube (iris.cube.Cube):
            Cube containing the orographic enhancement fields.
            Default is None.
        metadata_dict (dict):
            Dictionary containing required changes to the metadata.
            Information describing the intended contents of the dictionary is
            available in improver.utilities.cube_metadata.amend_metadata.
            Every output cube will have the metadata_dict applied.
            Default is None.
        ofc_box_size (int):
            Size of square 'box' (in grid spaces) within which to solve
            the optical flow equations.
            Default is 30.
        smart_smoothing_iterations (int):
            Number of iterations to perform in enforcing smoothness constraint
            for optical flow velocities.
            Default is 100.
        extrapolate (bool):
            If True, advects current data forward to specified lead times.
            Default is False.
        max_lead_time (int):
            Maximum lead time required (mins). Ignored unless extrapolate is
            True.
            Default is 360.
        lead_time_interval (int):
            Interval between required lead times (mins). Ignored unless
            extrapolate is True.
            Default is 15.

    Returns:
        (tuple): tuple containing:
            **forecast_cubes** (list<Cube>):
                List of Cubes if extrapolate is True, else None.
            **u_and_v_mean** (list<Cube>):
                List of the umean and vmean cubes.

    Raises:
        ValueError:
            If there is no oe_cube but a cube is called 'precipitation_rate'.

    """
    if orographic_enhancement_cube:
        cube_list = ApplyOrographicEnhancement("subtract").process(
            original_cube_list, orographic_enhancement_cube)
    else:
        cube_list = original_cube_list
        if any("precipitation_rate" in cube.name() for cube in cube_list):
            cube_names = [cube.name() for cube in cube_list]
            msg = ("For precipitation fields, orographic enhancement "
                   "filepaths must be supplied. The names of the cubes "
                   "supplied were: {}".format(cube_names))
            raise ValueError(msg)

    # order input files by validity time
    cube_list.sort(key=lambda x: x.coord("time").points[0])
    time_coord = cube_list[-1].coord("time")
    # calculate optical flow velocities from T-1 to T and T-2 to T-1
    ofc_plugin = OpticalFlow(iterations=smart_smoothing_iterations,
                             metadata_dict=metadata_dict)
    u_cubes = iris.cube.CubeList([])
    v_cubes = iris.cube.CubeList([])
    for older_cube, newer_cube in zip(cube_list[:-1], cube_list[1:]):
        ucube, vcube = ofc_plugin.process(older_cube, newer_cube,
                                          boxsize=ofc_box_size)
        u_cubes.append(ucube)
        v_cubes.append(vcube)

    # average optical flow velocity components
    u_cube = u_cubes.merge_cube()
    u_mean = u_cube.collapsed("time", iris.analysis.MEAN)
    u_mean.coord("time").points = time_coord.points
    u_mean.coord("time").units = time_coord.units

    v_cube = v_cubes.merge_cube()
    v_mean = v_cube.collapsed("time", iris.analysis.MEAN)
    v_mean.coord("time").points = time_coord.points
    v_mean.coord("time").units = time_coord.units

    u_and_v_mean = [u_mean, v_mean]
    forecast_cubes = []
    if extrapolate:
        # generate list of lead times in minutes
        lead_times = np.arange(0, max_lead_time + 1,
                               lead_time_interval)
        forecast_plugin = CreateExtrapolationForecast(
            original_cube_list[-1], u_mean, v_mean,
            orographic_enhancement_cube=orographic_enhancement_cube,
            metadata_dict=metadata_dict)
        # extrapolate input data to required lead times
        for i, lead_time in enumerate(lead_times):
            forecast_cubes.append(forecast_plugin.extrapolate(
                leadtime_minutes=lead_time))

    return forecast_cubes, u_and_v_mean


if __name__ == "__main__":
    main()
