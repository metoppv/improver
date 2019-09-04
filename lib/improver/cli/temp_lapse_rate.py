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
"""Script to calculate temperature lapse rates for given temperature and
orogrophy datasets."""

import numpy as np

from improver.argparser import ArgParser
from improver.constants import DALR, U_DALR
from improver.lapse_rate import LapseRate
from improver.utilities.cube_metadata import amend_metadata
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
    """Calculate temperature lapse rates."""
    parser = ArgParser(
        description='Calculate temperature lapse rates in units of K m-1 '
                    'over a given orography grid. ')
    parser.add_argument('temperature_filepath',
                        metavar='INPUT_TEMPERATURE_FILE',
                        help='A path to an input NetCDF temperature file to'
                             'be processed. ')
    parser.add_argument('--orography_filepath',
                        metavar='INPUT_OROGRAPHY_FILE',
                        help='A path to an input NetCDF orography file. ')
    parser.add_argument('--land_sea_mask_filepath',
                        metavar='LAND_SEA_MASK_FILE',
                        help='A path to an input NetCDF land/sea mask file. ')
    parser.add_argument('output_filepath', metavar='OUTPUT_FILE',
                        help='The output path for the processed temperature '
                             'lapse rates NetCDF. ')
    parser.add_argument('--max_height_diff',
                        metavar='MAX_HEIGHT_DIFF', type=float, default=35,
                        help='Maximum allowable height difference between the '
                             'central point and points in the neighbourhood '
                             'over which the lapse rate will be calculated '
                             '(metres).')
    parser.add_argument('--nbhood_radius',
                        metavar='NBHOOD_RADIUS', type=int, default=7,
                        help='Radius of neighbourhood around each point. '
                             'The neighbourhood will be a square array with '
                             'side length 2*nbhood_radius + 1.')
    parser.add_argument('--max_lapse_rate',
                        metavar='MAX_LAPSE_RATE', type=float, default=-3*DALR,
                        help='Maximum lapse rate allowed which must be '
                             'provided in units of K m-1. Default is -3*DALR')
    parser.add_argument('--min_lapse_rate',
                        metavar='MIN_LAPSE_RATE', type=float, default=DALR,
                        help='Minimum lapse rate allowed which must be '
                             'provided in units of K m-1. Default is the DALR')
    parser.add_argument('--return_dalr', action='store_true', default=False,
                        help='Flag to return a cube containing the dry '
                             'adiabatic lapse rate rather than calculating '
                             'the true lapse rate.')

    args = parser.parse_args(args=argv)

    # Load Cubes
    temperature_cube = load_cube(args.temperature_filepath)
    orography_cube = None
    land_sea_mask_cube = None
    if not args.return_dalr:
        orography_cube = load_cube(args.orography_filepath)
        land_sea_mask_cube = load_cube(args.land_sea_mask_filepath)

    # Process Cube
    result = process(temperature_cube, orography_cube, land_sea_mask_cube,
                     args.max_height_diff, args.nbhood_radius,
                     args.max_lapse_rate, args.min_lapse_rate,
                     args.return_dalr)

    # Save Cube
    save_netcdf(result, args.output_filepath)


def process(temperature_cube, orography_cube, land_sea_mask_cube,
            max_height_diff=35, nbhood_radius=7, max_lapse_rate=3*DALR,
            min_lapse_rate=DALR, return_dalr=False):
    """Calculate temperature lapse rates in units of K m-1 over orography grid.

    Args:
        temperature_cube (iris.cube.Cube):
            A cube of air temperature to be processed (K).
        orography_cube (iris.cube.Cube):
             A Cube containing orography data (metres).
        land_sea_mask_cube (iris.cube.Cube):
            A cube containing a binary land-sea mask.
            True for land-points.
            False for sea.
        max_height_diff (float):
            Maximum allowable height difference between the central point and
            points in the neighbourhood over which the lapse rate will be
            calculated.
            Default is 35.
        nbhood_radius (int):
            Radius of neighbourhood around each point. The neighbourhood
            will be a square array with side length 2*nbhood_radius + 1.
            The default value of 7 is from the reference paper.
        max_lapse_rate (float):
            Maximum lapse rate allowed.
            Default is 3*improver.constants.DALR.
        min_lapse_rate (float):
            Minimum lapse rate allowed.
            Default is improver.constants.DALR.
        return_dalr (bool):
            If True, returns a cube containing the dry adiabatic lapse rate
            rather than calculating the true lapse rate.

    Returns:
        result (iris.cube.Cube):
            Cube containing lapse rate (K m-1)

    Raises:
        ValueError:
            If minimum lapse rate is greater than maximum.
        ValueError:
            If Maximum height difference is less than zero.
        ValueError:
            If neighbourhood radius is less than zero.

    """
    if min_lapse_rate > max_lapse_rate:
        msg = 'Minimum lapse rate specified is greater than the maximum.'
        raise ValueError(msg)

    if max_height_diff < 0:
        msg = 'Maximum height difference specified is less than zero.'
        raise ValueError(msg)

    if nbhood_radius < 0:
        msg = 'Neighbourhood radius specified is less than zero.'
        raise ValueError(msg)

    if return_dalr:
        result = temperature_cube.copy(
            data=np.full_like(temperature_cube.data, U_DALR.points[0]))
        result.rename('air_temperature_lapse_rate')
        result.units = U_DALR.units
    else:
        result = LapseRate(
            max_height_diff=max_height_diff,
            nbhood_radius=nbhood_radius,
            max_lapse_rate=max_lapse_rate,
            min_lapse_rate=min_lapse_rate).process(temperature_cube,
                                                   orography_cube,
                                                   land_sea_mask_cube)
    attributes = {"title": "delete", "source": "delete",
                  "history": "delete", "um_version": "delete"}
    result = amend_metadata(result, attributes=attributes)
    return result


if __name__ == "__main__":
    main()
