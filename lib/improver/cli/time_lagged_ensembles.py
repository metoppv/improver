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
"""Script to run time-lagged ensembles."""

import warnings

import iris

from improver.argparser import ArgParser
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf
from improver.utilities.time_lagging import GenerateTimeLaggedEnsemble


def main(argv=None):
    """Load in the arguments and ensure they are set correctly. Then run
    the time-lagged ensembles on the input cubes.
    """
    parser = ArgParser(
        description='This combines the realizations from different forecast '
                    'cycles into one cube. It does this by taking an input '
                    'CubeList containing forecasts from different cycles and '
                    'merges them into a single cube, removing any metadata '
                    'that does not match.')
    parser.add_argument('input_filenames', metavar='INPUT_FILENAMES',
                        nargs="+", type=str,
                        help='Paths to input NetCDF files for the time-lagged '
                        'ensemble to combine the realizations.')
    parser.add_argument('output_file', metavar='OUTPUT_FILE',
                        help='The output file for the processed NetCDF.')
    args = parser.parse_args(args=argv)

    # Load the cubes
    cubes = iris.cube.CubeList([])
    for filename in args.input_filenames:
        new_cube = load_cube(filename)
        cubes.append(new_cube)

    # Process Cube
    result = process(cubes)

    # Save Cube
    save_netcdf(result, args.output_file)


def process(cubes):
    """Module to run time-lagged ensembles.

    This combines the realization from different forecast cycles into one cube.
    It does this by taking an input Cubelist containing forecasts from
    different cycles and merges them into a single cube, removing any
    metadata that does not match.

    Args:
        cubes (iris.cube.Cubelist):
            CubeList for the time-lagged ensemble to combine the realizations.

    Returns:
        result (iris.cube.Cube):
            Merged Cube.

    Raises:
        ValueError:
            If cubes have mismatched validity times.
    """

    # Warns if a single file is input
    if len(cubes) == 1:
        warnings.warn('Only a single cube input, so time lagging will have '
                      'no effect.')
        return cubes[0]
    # Raises an error if the validity times do not match
    else:
        for i, this_cube in enumerate(cubes):
            for later_cube in cubes[i+1:]:
                if this_cube.coord('time') == later_cube.coord('time'):
                    continue
                else:
                    msg = ("Cubes with mismatched validity times are not "
                           "compatible.")
                    raise ValueError(msg)
        return GenerateTimeLaggedEnsemble().process(cubes)


if __name__ == "__main__":
    main()
