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
"""
Script to translate meta-data relating to the grid_id attribute from StaGE
version 1.1.0 to StaGE version 1.2.0.
"""

import os

from improver.argparser import ArgParser
from improver.utilities.cube_metadata import update_stage_v110_metadata
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
    """
    Translate meta-data relating to the grid_id attribute from StaGE version
    1.1.0 to StaGE version 1.2.0.
    """

    cli_definition = {'central_arguments': ['input_file', 'output_file'],
                      'specific_arguments': [],
                      'description': ('Translates meta-data relating to the '
                                      'grid_id attribute from StaGE version '
                                      '1.1.0 to StaGE version 1.2.0. '
                                      'Files that have no "grid_id" attribute '
                                      'are not recognised as v1.1.0 and are '
                                      'not changed. Has no effect if '
                                      'input_file and output_file are the '
                                      'same and contain a cube with non '
                                      'v1.1.0 meta-data')}

    args = ArgParser(**cli_definition).parse_args(args=argv)
    # Load Cube
    cube = load_cube(args.input_filepath, no_lazy_load=True)
    # Process Cube
    cube_changed = process(cube)

    # Save Cube
    # Create normalised file paths to make them comparable
    in_file_norm = os.path.normpath(args.input_filepath)
    out_file_norm = os.path.normpath(args.output_filepath)
    if cube_changed or in_file_norm != out_file_norm:
        save_netcdf(cube, args.output_filepath)


def process(cube):
    """Update grid_id meta-data for StaGE.

    Translates meta-data relating to the grid_id attribute from StaGE
    version 1.1.0 to StaGE version 1.2.0.
    Files that have no "grid_id" attribute are not recognised as v1.1.0 and
    are not changed. Has no effect if input_file and output_file are the
    same and contain a cube with non v1.1.0 meta-data.

    Args:
        cube (iris.cube.Cube):
            Cube to be changed.

    Returns:
        result (iris.cube.Cube):
            Processed Cube.

    """
    result = update_stage_v110_metadata(cube)
    return result


if __name__ == "__main__":
    main()
