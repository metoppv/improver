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
"""Script to calculate mean wind direction from ensemble realizations."""

from improver.argparser import ArgParser
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf
from improver.wind_calculations.wind_direction import WindDirection


def main(argv=None):
    """Load in arguments to calculate mean wind direction from ensemble
       realizations."""

    cli_specific_arguments = [(['--backup_method'],
                               {'dest': 'backup_method',
                                'default': 'neighbourhood',
                                'choices': ['neighbourhood',
                                            'first_realization'],
                                'help': ('Backup method to use if '
                                         'there is low confidence in'
                                         ' the wind_direction. '
                                         'Options are first_realization'
                                         ' or neighbourhood, '
                                         'first_realization should only '
                                         'be used with global lat-lon data. '
                                         'Default is neighbourhood.')})]

    cli_definition = {'central_arguments': ('input_file', 'output_file'),
                      'specific_arguments': cli_specific_arguments,
                      'description': ('Run wind direction to calculate mean'
                                      ' wind direction from '
                                      'ensemble realizations')}

    args = ArgParser(**cli_definition).parse_args(args=argv)

    # Load Cube
    wind_direction = load_cube(args.input_filepath)

    # Returns 3 cubes - r_vals and confidence_measure cubes currently
    # only contain experimental data to be used for further research.
    # Process Cube
    cube_mean_wdir, _, _ = process(wind_direction, args.backup_method)

    # Save Cube
    save_netcdf(cube_mean_wdir, args.output_filepath)


def process(wind_direction, backup_method):
    """Calculates mean wind direction from ensemble realization.

    Create a cube containing the wind direction averaged over the ensemble
    realizations.

    Args:
        wind_direction (iris.cube.Cube):
            Cube containing the wind direction from multiple ensemble
            realizations.
        backup_method (str):
            Backup method to use if the complex numbers approach has low
            confidence.
            "first_realization" uses the value of realization zero.
            "neighbourhood" (default) recalculates using the complex numbers
            approach with additional realization extracted from neighbouring
            grid points from all available realizations.

    Returns (tuple of 3 Cubes):
        cube_mean_wdir (iris.cube.Cube):
            Cube containing the wind direction averaged from the ensemble
            realizations.
        cube_r_vals (numpy.ndarray):
            3D array - Radius taken from average complex wind direction angle.
        cube_confidence_measure (numpy.ndarray):
            3D array - The average distance from mean normalised - used as a
            confidence value.
    """
    # Returns 3 cubes - r_vals and confidence_measure cubes currently
    # only contain experimental data to be used for further research.
    result = (
        WindDirection(backup_method=backup_method).process(wind_direction))
    return result


if __name__ == "__main__":
    main()
