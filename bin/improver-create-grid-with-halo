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
"""Script to generate an ancillary "grid_with_halo" file."""

import numpy as np
import iris

from improver.argparser import ArgParser
from improver.utilities.pad_spatial import create_cube_with_halo
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main():
    """Generate target grid with a halo around the source file grid."""

    parser = ArgParser(description='Generate grid with halo from a source '
                       'domain input file. The grid is populated with zeroes.')
    parser.add_argument('input_file', metavar='INPUT_FILE', help="NetCDF file "
                        "containing data on a source grid.")
    parser.add_argument('output_file', metavar='OUTPUT_FILE', help="NetCDF "
                        "file defining the target grid with additional halo.")
    parser.add_argument('--halo_radius', metavar='HALO_RADIUS', default=162000,
                        type=float, help="Size of halo (in m) with which to "
                        "pad the input grid.  Default is 162 000 m.")
    args = parser.parse_args()

    cube = load_cube(args.input_file)
    halo_cube = create_cube_with_halo(cube, args.halo_radius)
    save_netcdf(halo_cube, args.output_file)


if __name__ == '__main__':
    main()
