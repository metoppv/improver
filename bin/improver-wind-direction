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

from improver.wind_calculations.wind_direction import WindDirection
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main():
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

    args = ArgParser(**cli_definition).parse_args()

    wind_direction = load_cube(args.input_filepath)

    # Returns 3 cubes - r_vals and confidence_measure cubes currently
    # only contain experimental data to be used for further research.
    bmethod = args.backup_method
    cube_mean_wdir, cube_r_vals, cube_confidence_measure = (
        WindDirection(backup_method=bmethod).process(wind_direction))

    save_netcdf(cube_mean_wdir, args.output_filepath)


if __name__ == "__main__":
    main()
