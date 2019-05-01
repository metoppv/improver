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
"""Script to calculate gradient of input field in x and y direction."""

import os

import iris

from improver.argparser import ArgParser
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf
from improver.utilities.spatial import DifferenceBetweenAdjacentGridSquares


def main():
    """Load in arguments to calculate the gradient between adjacent grid cells
       and save the output gradient fields."""

    cli_specific_arguments = [(['--force'],
                               {'dest': 'force',
                                'default': False,
                                'action': 'store_true',
                                'help': ('If True, ancillaries will be '
                                         'generated even if doing so will '
                                         'overwrite existing files.')})]

    cli_definition = {'central_arguments': ('input_file', 'output_file'),
                      'specific_arguments': cli_specific_arguments,
                      'description': ('Read the input field, and calculate '
                                      'the gradient in x and y directions.')}

    args = ArgParser(**cli_definition).parse_args()

    # Check if improver ancillary already exists.
    if not os.path.exists(args.output_filepath) or args.force:
        input_field = load_cube(args.input_filepath)
        gradients = DifferenceBetweenAdjacentGridSquares().process(input_field)
        gradients = iris.cube.CubeList([gradients[0], gradients[1]])
        save_netcdf(gradients, args.output_filepath)
    else:
        print(args.output_filepath)
        msg = 'File already exists here: {}'.format(args.output_filepath)
        raise IOError(msg)


if __name__ == "__main__":
    main()
