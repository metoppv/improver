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
"""CLI to generate wet bulb temperatures from air temperature, relative
   humidity, and pressure data. """

from improver.argparser import ArgParser

from improver.psychrometric_calculations.psychrometric_calculations import (
    WetBulbTemperature)
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
    """Parser to accept input data and an output destination before invoking
    the wet bulb temperature plugin. Also accepted is an optional
    convergence_condition argument that can be used to specify the tolerance of
    the Newton iterator used to calculate the wet bulb temperatures."""

    parser = ArgParser(
        description='Calculate a field of wet bulb temperatures.')
    parser.add_argument('temperature', metavar='TEMPERATURE',
                        help='Path to a NetCDF file of air temperatures at '
                        'the points for which the wet bulb temperatures are '
                        'being calculated.')
    parser.add_argument('relative_humidity', metavar='RELATIVE_HUMIDITY',
                        help='Path to a NetCDF file of relative humidities at '
                        'the points for for which the wet bulb temperatures '
                        'are being calculated.')
    parser.add_argument('pressure', metavar='PRESSURE',
                        help='Path to a NetCDF file of air pressures at the '
                        'points for which the wet bulb temperatures are being'
                        ' calculated.')
    parser.add_argument('output_filepath', metavar='OUTPUT_FILE',
                        help='The output path for the processed NetCDF.')
    parser.add_argument('--convergence_condition',
                        metavar='CONVERGENCE_CONDITION', type=float,
                        default=0.05,
                        help='The convergence condition for the Newton '
                        'iterator in K. When the wet bulb temperature '
                        'stops changing by more than this amount between'
                        ' iterations, the solution is accepted.')

    args = parser.parse_args(args=argv)
    # Load Cubes
    temperature = load_cube(args.temperature)
    relative_humidity = load_cube(args.relative_humidity)
    pressure = load_cube(args.pressure)
    # Process Cube
    result = process(temperature, relative_humidity, pressure,
                     args.convergence_condition)
    # Save Cube
    save_netcdf(result, args.output_filepath)


def process(temperature, relative_humidity, pressure,
            convergence_condition=0.05):
    """Module to generate wet-bulb temperatures.

    Call the calculate_wet_bulb_temperature function to calculate wet-bulb
    temperatures. This process function splits input cubes over vertical levels
    to mitigate memory issues when trying to operate on multi-level data.

    Args:
        temperature (iris.cube.Cube):
            Cube of air temperatures.
        relative_humidity (iris.cube.Cube):
            Cube of relative humidities.
        pressure (iris.cube.Cube):
            Cube of air pressure.
        convergence_condition (float):
            The precision to which the Newton iterator must converge before
            returning wet-bulb temperatures.
            Default is 0.05.

    Returns:
        result (iris.cube.Cube):
            Cube of wet-bulb temperature (K).

    """
    result = (WetBulbTemperature(precision=convergence_condition).
              process(temperature, relative_humidity, pressure))
    return result


if __name__ == "__main__":
    main()
