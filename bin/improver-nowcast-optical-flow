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
import json
import iris
import numpy as np

from improver.argparser import ArgParser
from improver.nowcasting.forecasting import CreateExtrapolationForecast
from improver.nowcasting.optical_flow import OpticalFlow
from improver.nowcasting.utilities import ApplyOrographicEnhancement
from improver.utilities.filename import generate_file_name
from improver.utilities.load import load_cube, load_cubelist
from improver.utilities.save import save_netcdf


def main():
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

    args = parser.parse_args()

    # read input data
    original_cube_list = load_cubelist(args.input_filepaths)

    if args.orographic_enhancement_filepaths:
        # Subtract orographic enhancement
        oe_cube = load_cube(args.orographic_enhancement_filepaths)
        cube_list = ApplyOrographicEnhancement("subtract").process(
            original_cube_list, oe_cube)
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

    metadata_dict = None
    if args.json_file:
        # Load JSON file for metadata amendments.
        with open(args.json_file, 'r') as input_file:
            metadata_dict = json.load(input_file)

    # calculate optical flow velocities from T-1 to T and T-2 to T-1
    ofc_plugin = OpticalFlow(iterations=args.smart_smoothing_iterations,
                             metadata_dict=metadata_dict)
    ucubes = iris.cube.CubeList([])
    vcubes = iris.cube.CubeList([])
    for older_cube, newer_cube in zip(cube_list[:-1], cube_list[1:]):
        ucube, vcube = ofc_plugin.process(older_cube, newer_cube,
                                          boxsize=args.ofc_box_size)
        ucubes.append(ucube)
        vcubes.append(vcube)

    # average optical flow velocity components
    ucube = ucubes.merge_cube()
    umean = ucube.collapsed("time", iris.analysis.MEAN)
    umean.coord("time").points = time_coord.points
    umean.coord("time").units = time_coord.units

    vcube = vcubes.merge_cube()
    vmean = vcube.collapsed("time", iris.analysis.MEAN)
    vmean.coord("time").points = time_coord.points
    vmean.coord("time").units = time_coord.units

    # save mean optical flow components as netcdf files
    for wind_cube in [umean, vmean]:
        file_name = generate_file_name(wind_cube)
        save_netcdf(wind_cube, os.path.join(args.output_dir, file_name))

    # advect latest input data to the required lead times
    if args.extrapolate:

        # generate list of lead times in minutes
        lead_times = np.arange(0, args.max_lead_time+1,
                               args.lead_time_interval)

        if args.nowcast_filepaths:
            if len(args.nowcast_filepaths) != len(lead_times):
                raise ValueError("Require exactly one output file name for "
                                 "each forecast lead time")

        forecast_plugin = CreateExtrapolationForecast(
            original_cube_list[-1], umean, vmean,
            orographic_enhancement_cube=oe_cube, metadata_dict=metadata_dict)
        # extrapolate input data to required lead times
        for i, lead_time in enumerate(lead_times):
            forecast_cube = forecast_plugin.extrapolate(
                leadtime_minutes=lead_time)

            # save to a suitably-named output file
            if args.nowcast_filepaths:
                file_name = args.nowcast_filepaths[i]
            else:
                file_name = os.path.join(
                    args.output_dir, generate_file_name(forecast_cube))
            save_netcdf(forecast_cube, file_name)


if __name__ == "__main__":
    main()
