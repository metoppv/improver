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
"""Script to calculate continuous snow falling level."""

from improver.argparser import ArgParser

from improver.psychrometric_calculations.psychrometric_calculations import (
    FallingSnowLevel)
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
    """Load in arguments and get going."""
    parser = ArgParser(
        description="Calculate the continuous falling snow level ")
    parser.add_argument("temperature", metavar="TEMPERATURE",
                        help="Path to a NetCDF file of air temperatures at"
                        " heights (m) at the points for which the continuous "
                        "falling snow level is being calculated.")
    parser.add_argument("relative_humidity", metavar="RELATIVE_HUMIDITY",
                        help="Path to a NetCDF file of relative_humidities at"
                        " heights (m) at the points for which the continuous "
                        "falling snow level is being calculated.")
    parser.add_argument("pressure", metavar="PRESSURE",
                        help="Path to a NetCDF file of air pressures at"
                        " heights (m) at the points for which the continuous "
                        "falling snow level is being calculated.")
    parser.add_argument("orography", metavar="OROGRAPHY",
                        help="Path to a NetCDF file containing "
                        "the orography height in m of the terrain "
                        "over which the continuous falling snow level is "
                        "being calculated.")
    parser.add_argument("land_sea_mask", metavar="LAND_SEA_MASK",
                        help="Path to a NetCDF file containing "
                        "the binary land-sea mask for the points "
                        "for which the continuous falling snow level is "
                        "being calculated. Land points are set to 1, sea "
                        "points are set to 0.")
    parser.add_argument("output_filepath", metavar="OUTPUT_FILE",
                        help="The output path for the processed NetCDF")
    parser.add_argument("--precision", metavar="NEWTON_PRECISION",
                        default=0.005, type=float,
                        help="Precision to which the wet bulb temperature "
                        "is required: This is used by the Newton iteration "
                        "default value is 0.005")
    parser.add_argument("--falling_level_threshold",
                        metavar="FALLING_LEVEL_THRESHOLD",
                        default=90.0, type=float,
                        help=("Cutoff threshold for the wet-bulb integral used"
                              " to calculate the falling snow level. This "
                              "threshold indicates the level at which falling "
                              "snow is deemed to have melted to become rain. "
                              "The default value is 90.0, an empirically "
                              "derived value."))
    args = parser.parse_args(args=argv)

    # Load Cubes
    temperature = load_cube(args.temperature, no_lazy_load=True)
    relative_humidity = load_cube(args.relative_humidity, no_lazy_load=True)
    pressure = load_cube(args.pressure, no_lazy_load=True)
    orog = load_cube(args.orography, no_lazy_load=True)
    land_sea = load_cube(args.land_sea_mask, no_lazy_load=True)

    # Process Cube
    result = process(temperature, relative_humidity, pressure, orog,
                     land_sea, args.precision, args.falling_level_threshold)

    # Save Cube
    save_netcdf(result, args.output_filepath)


def process(temperature, relative_humidity, pressure, orog, land_sea,
            precision=0.005, falling_level_threshold=90.0):
    """Module to calculate continuous snow falling level.

    Calculate the wet-bulb temperature integral by firstly calculating the
    wet-bulb temperature from the inputs provided and then calculating the
    vertical integral of the wet-bulb temperature.
    Find the falling_snow_level by finding the height above sea level
    corresponding to the falling_level_threshold in the integral data.

    Args:
        temperature (iris.cube.Cube):
            Cube of air temperature at heights (m) at the points for which the
            continuous falling snow level is being calculated.
        relative_humidity (iris.cube.Cube):
            Cube of relative humidities at heights (m) at the points for which
            the continuous falling snow level is being calculated.
        pressure (iris.cube.Cube):
            Cube of air pressure at heights (m) at the points for which the
            continuous falling snow level is being calculated.
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
        result (iris.cube.Cube):
            Processed Cube of falling snow level above sea level.
    """
    result = FallingSnowLevel(
        precision=precision,
        falling_level_threshold=falling_level_threshold).process(
        temperature,
        relative_humidity,
        pressure,
        orog,
        land_sea)
    return result


if __name__ == "__main__":
    main()
