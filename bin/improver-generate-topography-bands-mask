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
"""Script to run topographic bands mask generation."""

from improver.argparser import ArgParser
import os
import json

from improver.generate_ancillaries.generate_ancillary import (
    GenerateOrographyBandAncils)
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf

# The following dictionary defines the orography altitude bands in metres
# above/below sea level for which masks are required.

THRESHOLDS_DICT = {'bounds': [[-500., 50.], [50., 100.], [100., 150.],
                              [150., 200.], [200., 250.], [250., 300.],
                              [300., 400.], [400., 500.], [500., 650.],
                              [650., 800.], [800., 950.], [950., 6000.]],
                   'units': 'm'}


def main():
    """Load in arguments and get going."""
    parser = ArgParser(
        description=('Reads input orography and landmask fields. Creates a '
                     'series of masks, where each mask excludes data below or'
                     ' equal to the lower threshold, and excludes data above '
                     'the upper threshold.'))
    parser.add_argument('input_filepath_standard_orography',
                        metavar='INPUT_FILE_STANDARD_OROGRAPHY',
                        help=('A path to an input NetCDF orography file to '
                              'be processed'))
    parser.add_argument('output_filepath', metavar='OUTPUT_FILE',
                        help='The output path for the processed NetCDF.')
    parser.add_argument('--input_filepath_landmask', metavar='INPUT_FILE_LAND',
                        help=('A path to an input NetCDF land mask file to be '
                              'processed. If provided, sea points will be '
                              'set to zero in every band. If '
                              'no land mask is provided, sea points will be '
                              'included in the appropriate topographic band.'))
    parser.add_argument('--force', dest='force', default=False,
                        action='store_true',
                        help=('If keyword is set (i.e. True), ancillaries '
                              'will be generated even if doing so will '
                              'overwrite existing files'))
    parser.add_argument('--thresholds_filepath',
                        metavar='THRESHOLDS_FILEPATH',
                        default=None,
                        help=("The path to a json file which can be used "
                              "to set the number and size of topographic "
                              "bounds. If unset a default bounds dictionary"
                              " will be used."
                              "The dictionary has the following form: "
                              "{'bounds': [[-500., 50.], [50., 100.], "
                              "[100., 150.],[150., 200.], [200., 250.], "
                              "[250., 300.], [300., 400.], [400., 500.], "
                              "[500., 650.],[650., 800.], [800., 950.], "
                              "[950., 6000.]], 'units': 'm'}"))
    args = parser.parse_args()

    if args.thresholds_filepath:
        with open(args.thresholds_filepath, 'r') as filehandle:
            thresholds_dict = json.loads(filehandle.read())
    else:
        thresholds_dict = THRESHOLDS_DICT

    if not os.path.exists(args.output_filepath) or args.force:
        orography = load_cube(args.input_filepath_standard_orography)
        orography = next(orography.slices([orography.coord(axis='y'),
                                           orography.coord(axis='x')]))

        landmask = None
        if args.input_filepath_landmask:
            try:
                landmask = load_cube(args.input_filepath_landmask)
            except IOError as err:
                msg = ("Loading land mask has been unsuccessful: {}. "
                       "This may be because the land mask could not be "
                       "located in {}; run "
                       'improver-generate-landmask-ancillary first.').format(
                           err, args.input_filepath_landmask)
                raise IOError(msg)

            landmask = next(landmask.slices([landmask.coord(axis='y'),
                                             landmask.coord(axis='x')]))

        result = GenerateOrographyBandAncils().process(
            orography, thresholds_dict, landmask=landmask)
        result = result.concatenate_cube()
        save_netcdf(result, args.output_filepath)
    else:
        print('File already exists here: ', args.output_filepath)


if __name__ == "__main__":
    main()
