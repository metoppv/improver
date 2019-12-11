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
    PhaseChangeLevel)
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
    """Load in arguments and get going."""
    parser = ArgParser(
        description="Calculate a continuous phase change level. This is an "
        "altitude at which precipitation is expected to change phase, "
        "e.g. snow to sleet.")
    parser.add_argument("phase_change",
                        metavar="PHASE_CHANGE", type=str,
                        help="The desired phase change for which the altitude"
                        "should be returned. Options are: 'snow-sleet', the "
                        "melting of snow to sleet; sleet-rain - the melting of"
                        " sleet to rain.")
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

    args = parser.parse_args(args=argv)

    # Load Cubes
    wet_bulb_temperature = load_cube(args.wet_bulb_temperature,
                                     no_lazy_load=True)
    wet_bulb_integral = load_cube(args.wet_bulb_integral, no_lazy_load=True)
    orog = load_cube(args.orography, no_lazy_load=True)
    land_sea = load_cube(args.land_sea_mask, no_lazy_load=True)

    # Process Cube
    result = process(args.phase_change, wet_bulb_temperature,
                     wet_bulb_integral, orog, land_sea)

    # Save Cube
    save_netcdf(result, args.output_filepath)


def process(phase_change, wet_bulb_temperature, wet_bulb_integral, orog,
            land_sea):
    """Calculate a continuous field of heights relative to sea level
    at which a phase change of precipitation is expected.

    This is achieved by finding the height above sea level at which the
    integral of wet bulb temperature matches an empirical threshold that is
    expected to correspond with the phase change.

    Args:
        phase_change (str):
            The desired phase change for which the altitude should be
            returned. Options are:

                snow-sleet - the melting of snow to sleet.
                sleet-rain - the melting of sleet to rain.
        wet_bulb_temperature (iris.cube.Cube):
            Cube of wet bulb temperatures on height levels.
        wet_bulb_integral (iris.cube.Cube):
            Cube of wet bulb temperature integrals calculated vertically
            downwards to height levels.
        orog (iris.cube.Cube):
            Cube of the orography height in m.
        land_sea (iris.cube.Cube):
            Cube containing the binary land-sea mask. Land points are set to 1,
            sea points are set to 0.

    Returns:
        iris.cube.Cube:
            Processed Cube of phase change altitude relative to sea level.
    """
    result = PhaseChangeLevel(
        phase_change=phase_change).process(
            wet_bulb_temperature, wet_bulb_integral, orog, land_sea)
    return result


if __name__ == "__main__":
    main()
