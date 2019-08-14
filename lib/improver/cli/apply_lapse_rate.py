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
"""Script to apply lapse rates to temperature data."""

from improver.argparser import ArgParser
from improver.lapse_rate import apply_gridded_lapse_rate
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
    """Apply lapse rates to temperature data."""
    parser = ArgParser(description='Apply downscaling temperature adjustment '
                       'using calculated lapse rate.')

    parser.add_argument('temperature_filepath', metavar='TEMPERATURE_FILEPATH',
                        help='Full path to input temperature NetCDF file')
    parser.add_argument('lapse_rate_filepath', metavar='LAPSE_RATE_FILEPATH',
                        help='Full path to input lapse rate NetCDF file')
    parser.add_argument('source_orography', metavar='SOURCE_OROG_FILE',
                        help='Full path to NetCDF file containing the source '
                        'model orography')
    parser.add_argument('target_orography', metavar='TARGET_OROG_FILE',
                        help='Full path to target orography NetCDF file '
                        '(to which temperature will be downscaled)')
    parser.add_argument('output_file', metavar='OUTPUT_FILE', help='File name '
                        'to write lapse rate adjusted temperature data')

    args = parser.parse_args(args=argv)

    # Load cubes
    temperature = load_cube(args.temperature_filepath)
    lapse_rate = load_cube(args.lapse_rate_filepath)
    source_orog = load_cube(args.source_orography)
    target_orog = load_cube(args.target_orography)

    # Process Cubes
    adjusted_temperature = process(temperature, lapse_rate, source_orog,
                                   target_orog)
    # Save to Cube
    save_netcdf(adjusted_temperature, args.output_file)


def process(temperature, lapse_rate, source_orog, target_orog):
    """ Apply downscaling temperature adjustment using calculated lapse rate.

    Args:
        temperature (iris.cube.Cube):
            Input temperature Cube.
        lapse_rate (iris.cube.Cube):
            Lapse rate Cube.
        source_orog (iris.cube.Cube):
            Source model orography.
        target_orog (iris.cube.Cube):
            Target orography to which temperature will be downscaled.

    Returns:
        result (iris.cube.Cube):
            Cube after lapse rate has been applied to temperature data.
    """
    # apply lapse rate to temperature data
    result = apply_gridded_lapse_rate(
        temperature, lapse_rate, source_orog, target_orog)
    return result


if __name__ == "__main__":
    main()
