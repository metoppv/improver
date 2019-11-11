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
"""Script to run the UV index plugin."""

from improver.argparser import ArgParser
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf
from improver.uv_index import calculate_uv_index


def main(argv=None):
    """ Calculate the UV index using the data
    in the input cubes."""
    parser = ArgParser(
        description="Calculates the UV index.")
    parser.add_argument("radiation_flux_upward",
                        metavar="RADIATION_FLUX_UPWARD",
                        help="Path to a NetCDF file of radiation flux "
                        "in uv upward at surface.")
    parser.add_argument("radiation_flux_downward",
                        metavar="RADIATION_FLUX_DOWNWARD",
                        help="Path to a NetCDF file of radiation flux "
                        "in uv downward at surface.")
    parser.add_argument("output_filepath", metavar="OUTPUT_FILE",
                        help="The output path for the processed NetCDF")

    args = parser.parse_args(args=argv)

    # Load Cube
    rad_uv_up = load_cube(args.radiation_flux_upward)
    rad_uv_down = load_cube(args.radiation_flux_downward)

    # Process Cube
    result = process(rad_uv_up, rad_uv_down)
    # Save Cube
    save_netcdf(result, args.output_filepath)


def process(rad_uv_up, rad_uv_down):
    """Calculate the UV index using the data in the input cubes.

    Calculate the uv index using radiation flux in UV downward at surface,
    radiation flux UV upwards at surface and a scaling factor. The scaling
    factor is configured by the user.

    Args:
        rad_uv_up (iris.cube.Cube):
            Cube of radiation flux in UV upwards at surface.
        rad_uv_down (iris.cube.Cube):
            Cube of radiation flux in UV downwards at surface.

    Returns:
        iris.cube.Cube:
            Processed Cube.
    """
    result = calculate_uv_index(rad_uv_up, rad_uv_down)
    return result


if __name__ == "__main__":
    main()
