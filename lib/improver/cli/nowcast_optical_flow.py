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

import iris
from iris.cube import CubeList

from improver.argparser import ArgParser
from improver.nowcasting.optical_flow import generate_optical_flow_components
from improver.nowcasting.utilities import ApplyOrographicEnhancement
from improver.utilities.cli_utilities import load_json_or_none
from improver.utilities.load import load_cubelist, load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
    """Calculate optical flow advection velocities"""

    parser = ArgParser(
        description="Calculate optical flow components from input fields.")

    parser.add_argument("input_filepaths", metavar="INPUT_FILEPATHS",
                        nargs=3, type=str, help="Paths to the input radar "
                        "files. There should be 3 input files at T, T-1 and "
                        "T-2 from which to calculate optical flow velocities. "
                        "The files require a 'time' coordinate on which they "
                        "are sorted, so the order of inputs does not matter.")
    parser.add_argument("output_filepath", metavar="OUTPUT_FILEPATH",
                        help="The output path for the resulting NetCDF")

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
                        "required changes to attributes. "
                        "Every output file will have the attributes_dict "
                        "applied. Defaults to None.", type=str)

    # OpticalFlow plugin configurable parameters
    parser.add_argument("--ofc_box_size", type=int, default=30, help="Size of "
                        "square 'box' (in grid squares) within which to solve "
                        "the optical flow equations.")
    parser.add_argument("--smart_smoothing_iterations", type=int, default=100,
                        help="Number of iterations to perform in enforcing "
                        "smoothness constraint for optical flow velocities.")

    args = parser.parse_args(args=argv)

    # Load Cubes and JSON
    attributes_dict = load_json_or_none(args.json_file)
    original_cube_list = load_cubelist(args.input_filepaths)
    oe_cube = load_cube(args.orographic_enhancement_filepaths, allow_none=True)

    # Process
    result = process(
        original_cube_list, oe_cube, attributes_dict, args.ofc_box_size,
        args.smart_smoothing_iterations)

    # Save Cubes
    save_netcdf(result, args.output_filepath)


def process(original_cube_list, orographic_enhancement_cube=None,
            attributes_dict=None, ofc_box_size=30,
            smart_smoothing_iterations=100):
    """Calculate optical flow components from input fields.

    Args:
        original_cube_list (iris.cube.CubeList):
            Cubelist from which to calculate optical flow velocities.
            The cubes require a 'time' coordinate on which they are sorted,
            so the order of cubes does not matter.
        orographic_enhancement_cube (iris.cube.Cube):
            Cube containing the orographic enhancement fields.
            Default is None.
        attributes_dict (dict):
            Dictionary containing required changes to the attributes.
            Every output file will have the attributes_dict applied.
            Default is None.
        ofc_box_size (int):
            Size of square 'box' (in grid spaces) within which to solve
            the optical flow equations.
            Default is 30.
        smart_smoothing_iterations (int):
            Number of iterations to perform in enforcing smoothness constraint
            for optical flow velocities.
            Default is 100.

    Returns:
        iris.cube.CubeList:
            List of the umean and vmean cubes.

    Raises:
        ValueError:
            If there is no oe_cube but a cube is called 'precipitation_rate'.

    """
    # order input files by validity time
    original_cube_list.sort(key=lambda x: x.coord("time").points[0])

    # subtract orographic enhancement
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

    # calculate optical flow velocities from T-1 to T and T-2 to T-1, and
    # average to produce the velocities for use in advection
    u_mean, v_mean = generate_optical_flow_components(
        cube_list, ofc_box_size, smart_smoothing_iterations, attributes_dict)

    return CubeList([u_mean, v_mean])


if __name__ == "__main__":
    main()
