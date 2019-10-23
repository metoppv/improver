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
"""Script to extend a radar mask based on coverage data."""

from improver.argparser import ArgParser
from improver.metadata.check_datatypes import check_cube_not_float64
from improver.nowcasting.utilities import ExtendRadarMask
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
    """Extend radar mask based on coverage data."""
    parser = ArgParser(description="Extend radar mask based on coverage "
                       "data.")
    parser.add_argument("radar_data_filepath", metavar="RADAR_DATA_FILEPATH",
                        type=str, help="Full path to input NetCDF file "
                        "containing the radar variable to remask.")
    parser.add_argument("coverage_filepath", metavar="COVERAGE_FILEPATH",
                        type=str, help="Full path to input NetCDF file "
                        "containing radar coverage data.")
    parser.add_argument("output_filepath", metavar="OUTPUT_FILEPATH",
                        type=str, help="Full path to save remasked radar data "
                        "NetCDF file.")
    parser.add_argument("--fix_float64", action='store_true', default=False,
                        help="Check and fix cube for float64 data. Without "
                             "this option an exception will be raised if "
                             "float64 data is found but no fix applied.")

    args = parser.parse_args(args=argv)

    # Load Cubes
    radar_data = load_cube(args.radar_data_filepath)
    coverage = load_cube(args.coverage_filepath)

    # Process Cube
    remasked_data = process(coverage, radar_data, args.fix_float64)

    # Save Cube
    save_netcdf(remasked_data, args.output_filepath)


def process(coverage, radar_data, fix_float64=False):
    """ Extend radar mask based on coverage data.

    Extends the mask on radar data based on the radar coverage composite.
    Update the mask on the input cube to reflect where coverage is valid.

    Args:
        coverage (iris.cube.Cube):
            Cube containing the radar data to remask
        radar_data (iris.cube.Cube):
            Cube containing the radar coverage data.
        fix_float64 (bool):
            Check and fix cube for float64 data. Without this, an exception
            will be raised if float64 data is found but no fix applied.

    Returns:
        iris.cube.Cube:
            A cube with the remasked radar data.
    """
    # extend mask
    result = ExtendRadarMask().process(radar_data, coverage)
    # Check and fix for float64 data only option:
    check_cube_not_float64(result, fix=fix_float64)
    return result


if __name__ == "__main__":
    main()
