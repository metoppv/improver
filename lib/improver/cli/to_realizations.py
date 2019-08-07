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
"""ver_stream_processing"""
from improver.argparser import ArgParser
from improver.cli import (percentiles_to_realizations,
                          probabilities_to_realizations, extract)
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
    """Convert Cubes to realizations."""
    parser = ArgParser(
        description='Converts a cube into realizations')
    parser.add_argument(
        'input_filepath', metavar='INPUT_FILE',
        help='A path to an input NetCDF file containing the current forecast '
             'to be processed. The file provided could have the coordinates '
             '"percentile", "threshold" or "height" and "wind speed".')
    parser.add_argument(
        'output_filepath', metavar='OUTPUT_FILE',
        help='The output path for the processed NetCDF.')
    parser.add_argument(
        'no_of_realizations', metavar='NUMBER_OF_REALIZATIONS',
        help='The number of percentiles to be generated. This is equal to '
             'the number of ensemble realizations that will be generated.')
    parser.add_argument(
        '--ecc_bounds_warning', default=False,
        action='store_true',
        help='If True, where calculated percentiles are '
             'outside the ECC bounds range, raise a warning '
             'rather than an exception.')

    args = parser.parse_args(args=argv)

    # Load Cube
    cube = load_cube(args.input_filepath)
    # Process Cube
    result = process(cube, args.no_of_realizations, args.ecc_bounds_warning)
    # Save Cube
    save_netcdf(result, args.output_filepath)


def process(cube, no_of_realizations, ecc_bounds_warning):
    """Converts a cube into a realizations.

    Args:
        cube (iris.cube.Cube):
            A cube to be processed.
        no_of_realizations:
            The number of percentiles to be generated. This is equal to the
            number of ensemble realizations that will be generated.
        ecc_bounds_warning:
            If True, where percentiles exceed the ECC bounds range, raises a
            warning rather than an exception.

    Returns:
        cube (iris.cube.Cube):
            A processed cube.
    """

    if cube.coords('percentile'):
        cube = percentiles_to_realizations.process(
            cube, no_of_percentiles=no_of_realizations,
            rebadging=True, ecc_bounds_warning=ecc_bounds_warning)
    elif cube.coord(var_name='threshold'):
        cube = probabilities_to_realizations.process(
            cube, no_of_realizations=no_of_realizations, rebadging=True,
            ecc_bounds_warning=ecc_bounds_warning)
    elif cube.coords('height') and cube.name('wind_speed'):
        cube = extract.process(cube, 'height=10')

    return cube


if __name__ == '__main__':
    main()
