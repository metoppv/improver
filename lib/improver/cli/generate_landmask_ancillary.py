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
"""Script to run landmask ancillary generation."""

import os

from improver.argparser import ArgParser
from improver.generate_ancillaries.generate_ancillary import (
    CorrectLandSeaMask)
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
    """Load in arguments and get going."""
    parser = ArgParser(
        description=('Read the input landmask, and correct '
                     'to boolean values.'))
    parser.add_argument('--force', dest='force', default=False,
                        action='store_true',
                        help=('If True, ancillaries will be generated '
                              'even if doing so will overwrite existing '
                              'files.'))
    parser.add_argument('input_filepath_standard',
                        metavar='INPUT_FILE_STANDARD',
                        help='A path to an input NetCDF file to be processed')
    parser.add_argument('output_filepath', metavar='OUTPUT_FILE',
                        help='The output path for the processed NetCDF')
    args = parser.parse_args(args=argv)

    # Check if improver ancillary already exists.
    if not os.path.exists(args.output_filepath) or args.force:
        # Load Cube
        landmask = load_cube(args.input_filepath_standard)

        # Process Cube
        land_binary_mask = process(landmask)

        # Save Cube
        save_netcdf(land_binary_mask, args.output_filepath)
    else:
        print('File already exists here: ', args.output_filepath)


def process(landmask):
    """Runs landmask ancillary generation.

    Read in the interpolated landmask and round
    values < 0.5 to False
    values >= 0.5 to True.

    Args:
        landmask (iris.cube.Cube):
            Cube to process

    Returns:
        (iris.cube.Cube):
            A cube landmask of boolean values.
    """
    return CorrectLandSeaMask().process(landmask)


if __name__ == "__main__":
    main()
