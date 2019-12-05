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
"""Script to calculate wet bulb temperature integrals."""

from improver.argparser import ArgParser

from improver.psychrometric_calculations.psychrometric_calculations import (
    WetBulbTemperatureIntegral)
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
    """Load in arguments and get going."""
    parser = ArgParser(
        description="Calculate the wet bulb temperature integral")
    parser.add_argument("wet_bulb_temperature", metavar="WBT",
                        help="Path to a NetCDF file of wet bulb temperature on"
                        " height levels (m).")
    parser.add_argument("output_filepath", metavar="OUTPUT_FILE",
                        help="The output path for the processed NetCDF")
    args = parser.parse_args(args=argv)

    # Load Cubes
    wet_bulb_temperature = load_cube(args.wet_bulb_temperature,
                                     no_lazy_load=True)

    # Process Cube
    result = process(wet_bulb_temperature)

    # Save Cube
    save_netcdf(result, args.output_filepath)


def process(wet_bulb_temperature):
    """Module to calculate wet bulb temperature integral.

    Calculate the wet-bulb temperature integral using the input wet bulb
    temperature data. The integral will be calculated at the height levels on
    which the wet bulb temperatures are provided.

    Args:
        wet_bulb_temperature (iris.cube.Cube):
            Cube of wet bulb temperatures on height levels.

    Returns:
        iris.cube.Cube:
            Processed Cube of wet bulb integrals.
    """
    result = WetBulbTemperatureIntegral().process(
        wet_bulb_temperature)
    return result


if __name__ == "__main__":
    main()
