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
"""Script to calculate continuous phase change level."""

from improver.argparser import ArgParser

from improver.psychrometric_calculations.psychrometric_calculations import (
    FallingSnowLevel)
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
    """Load in arguments and get going."""
    parser = ArgParser(
        description="Calculate a continuous phase change level. This is an "
            "altitude at which precipitation is expected to change phase, "
            "e.g. snow to sleet.")
    parser.add_argument("wet_bulb_temperature", metavar="WBT",
                        help="Path to a NetCDF file of wet bulb temperatures "
                        "on height levels.")
    parser.add_argument("wet_bulb_integral", metavar="WBTI",
                        help="Path to a NetCDF file of wet bulb temperature "
                        "integrals calculated vertically downwards to height "
                        "levels.")
    parser.add_argument("orography", metavar="OROGRAPHY",
                        help="Path to a NetCDF file containing "
                        "the orography height in m of the terrain "
                        "over which the continuous phase change level is "
                        "being calculated.")
    parser.add_argument("land_sea_mask", metavar="LAND_SEA_MASK",
                        help="Path to a NetCDF file containing "
                        "the binary land-sea mask for the points "
                        "for which the continuous phase change level is "
                        "being calculated. Land points are set to 1, sea "
                        "points are set to 0.")
    parser.add_argument("output_filepath", metavar="OUTPUT_FILE",
                        help="The output path for the processed NetCDF")
    parser.add_argument("--falling_level_threshold",
                        metavar="FALLING_LEVEL_THRESHOLD",
                        default=90.0, type=float,
                        help=("Cutoff threshold for the wet-bulb integral used"
                              " to calculate the phase change level. This "
                              "threshold indicates the level at which falling "
                              "precipitation is deemed to have melted to have "
                              "changed to a new phase."
                              "The default value is 90.0, an empirically "
                              "derived value for the transition from snow to "
                              "sleet."))
    args = parser.parse_args(args=argv)

    # Load Cubes
    wet_bulb_temperature = load_cube(args.wet_bulb_temperature,
                                     no_lazy_load=True)
    wet_bulb_integral = load_cube(args.wet_bulb_integral, no_lazy_load=True)
    orog = load_cube(args.orography, no_lazy_load=True)
    land_sea = load_cube(args.land_sea_mask, no_lazy_load=True)

    # Process Cube
    result = process(wet_bulb_temperature, wet_bulb_integral, orog, land_sea,
                     args.falling_level_threshold)

    # Save Cube
    save_netcdf(result, args.output_filepath)


def process(wet_bulb_temperature, wet_bulb_integral, orog, land_sea,
            falling_level_threshold=90.0):
    """Module to calculate continuous snow falling level.

    Use wet-bulb temperature integrals to generate a snow falling level.
    This is found by finding the height above sea level corresponding to the
    falling_level_threshold in the integral data.

    Args:
        wet_bulb_temperature (iris.cube.Cube):
            Cube of wet bulb temperatures on height levels.
        wet_bulb_integral (iris.cube.Cube):
            Cube of wet bulb temperature integrals calculated vertically "
            "downwards to height levels.
        orog (iris.cube.Cube):
            Cube of the orography height in m of the terrain over which the
            continuous falling snow level is being calculated.
        land_sea (iris.cube.Cube):
            Cube containing the binary land-sea mask for the points for which
            the continuous falling snow level is being calculated. Land points
            are set to 1, sea points are set to 0.
        precision (float):
            Precision to which the wet-bulb temperature is required: This is
            used by the Newton iteration.
            Default is 0.005.
        falling_level_threshold (float):
            Cutoff threshold for the wet-bulb integral used to calculate the
            falling snow level. This threshold indicates the level at which
            falling snow is deemed to have melted to become rain.
            Default is 90.0.

    Returns:
        iris.cube.Cube:
            Processed Cube of falling snow level above sea level.
    """
    result = FallingSnowLevel(
        falling_level_threshold=falling_level_threshold).process(
        wet_bulb_temperature, wet_bulb_integral, orog, land_sea)
    return result


if __name__ == "__main__":
    main()
