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
"""Script to create wind-gust data."""

from improver.argparser import ArgParser

from improver.wind_calculations.wind_gust_diagnostic import WindGustDiagnostic
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main():
    """Load in arguments for wind-gust diagnostic.
    Wind-gust and Wind-speed data should be supplied along with the required
    percentile value. The wind-gust diagnostic will be the Max of the specified
    percentile data.
    Currently
        * Typical gusts is
          MAX(wind-gust(50th percentile),wind-speed(95th percentile))
        * Extreme gust is
          MAX(wind-gust(95th percentile),wind-speed(100th percentile))
    If no percentile values are supplied the code defaults
    to values for Typical gusts.
    """
    parser = ArgParser(
        description="Calculate revised wind-gust data using a specified "
        "percentile of wind-gust data and a specified percentile "
        "of wind-speed data through the WindGustDiagnostic plugin. "
        "The wind-gust diagnostic will be the Max of the specified "
        "percentile data."
        "Currently Typical gusts is "
        "MAX(wind-gust(50th percentile),wind-speed(95th percentile))"
        "and Extreme gust is "
        "MAX(wind-gust(95th percentile),wind-speed(100th percentile)). "
        "If no percentile values are supplied the code defaults "
        "to values for Typical gusts.")
    parser.add_argument("input_filegust", metavar="INPUT_FILE_GUST",
                        help="A path to an input Wind Gust Percentile"
                        " NetCDF file")
    parser.add_argument("input_filews", metavar="INPUT_FILE_WINDSPEED",
                        help="A path to an input Wind Speed Percentile"
                        " NetCDF file")
    parser.add_argument("output_filepath", metavar="OUTPUT_FILE",
                        help="The output path for the processed NetCDF")
    parser.add_argument("--percentile_gust", metavar="PERCENTILE_GUST",
                        default="50.0",
                        help="Percentile of wind-gust required."
                        " Default=50.0", type=float)
    parser.add_argument("--percentile_ws",
                        metavar="PERCENTILE_WIND_SPEED",
                        default="95.0",
                        help="Percentile of wind-speed required."
                        " Default=95.0", type=float)

    args = parser.parse_args()
    cube_wg = load_cube(args.input_filegust)
    cube_ws = load_cube(args.input_filews)
    result = (
        WindGustDiagnostic(args.percentile_gust,
                           args.percentile_ws).process(cube_wg, cube_ws))
    save_netcdf(result, args.output_filepath)


if __name__ == "__main__":
    main()
