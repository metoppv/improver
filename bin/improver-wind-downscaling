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
"""Script to run wind downscaling."""

from improver.argparser import ArgParser
import warnings

import numpy as np
import iris
from iris.exceptions import CoordinateNotFoundError

from improver.wind_calculations import wind_downscaling
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf
from improver.utilities.cube_extraction import apply_extraction


def main():
    """Load in arguments and get going."""
    parser = ArgParser(
        description='Run wind downscaling to apply roughness correction and'
                    ' height correction to wind fields (as described in'
                    ' Howard and Clark [2007]). All inputs must be on the same'
                    ' standard grid')
    parser.add_argument('wind_speed_filepath', metavar='WIND_SPEED_FILE',
                        help='Location of the wind speed on standard grid'
                             ' file. Any units can be supplied.')
    parser.add_argument('silhouette_roughness_filepath', metavar='AOS_FILE',
                        help='Location of model silhouette roughness file. '
                             'Units of field: dimensionless')
    parser.add_argument('sigma_filepath', metavar='SIGMA_FILE',
                        help='Location of standard deviation of model '
                             'orography height file. Units of field: m')
    parser.add_argument('target_orog_filepath',
                        metavar='TARGET_OROGRAPHY_FILE',
                        help='Location of target orography file to downscale'
                             ' fields to.'
                             'Units of field: m')
    parser.add_argument('standard_orog_filepath',
                        metavar='STANDARD_OROGRAPHY_FILE',
                        help='Location of orography on standard grid file '
                             '(interpolated model orography.'
                             ' Units of field: m')
    parser.add_argument('model_resolution', metavar='MODEL_RESOLUTION',
                        help='Original resolution of model orography (before'
                             ' interpolation to standard grid).'
                             ' Units of field: m')
    parser.add_argument('output_filepath', metavar='OUTPUT_FILE',
                        help='The output path for the processed NetCDF')
    parser.add_argument('--output_height_level', metavar='OUTPUT_HEIGHT_LEVEL',
                        default=None,
                        help='If only a single height level is desired as '
                        'output from wind-downscaling, this option can be '
                        'used to select the height level. If no units are '
                        'provided with the --output_height_level_units '
                        'option, metres are assumed.')
    parser.add_argument('--output_height_level_units',
                        metavar='OUTPUT_HEIGHT_LEVEL_UNITS', default='m',
                        help='If a single height level is selected as output '
                        'using the --output_height_level option, this '
                        'additional argument may be used to specify the units '
                        'of the value entered to select the level. e.g. hPa')
    parser.add_argument('--height_levels_filepath',
                        metavar='HEIGHT_LEVELS_FILE',
                        help='Location of file containing height levels '
                             'coincident with wind speed field.')
    parser.add_argument('--veg_roughness_filepath',
                        metavar='VEGETATIVE_ROUGHNESS_LENGTH_FILE',
                        help='Location of vegetative roughness length file.'
                             ' Units of field: m')
    args = parser.parse_args()

    if args.output_height_level_units and not args.output_height_level:
        warnings.warn('--output_height_level_units has been set but no '
                      'associated height level has been provided. These units '
                      'will have no effect.')

    wind_speed = load_cube(args.wind_speed_filepath)
    silhouette_roughness_filepath = load_cube(
        args.silhouette_roughness_filepath)
    sigma = load_cube(args.sigma_filepath)
    target_orog = load_cube(args.target_orog_filepath)
    standard_orog = load_cube(args.standard_orog_filepath)
    if args.height_levels_filepath:
        height_levels = load_cube(args.height_levels_filepath)
    else:
        height_levels = None
    if args.veg_roughness_filepath:
        veg_roughness_cube = load_cube(args.veg_roughness_filepath)
    else:
        veg_roughness_cube = None
    try:
        wind_speed_iterator = wind_speed.slices_over('realization')
    except CoordinateNotFoundError:
        wind_speed_iterator = [wind_speed]
    wind_speed_list = iris.cube.CubeList()
    for wind_speed_slice in wind_speed_iterator:
        result = (
            wind_downscaling.RoughnessCorrection(
                silhouette_roughness_filepath, sigma, target_orog,
                standard_orog, float(args.model_resolution),
                z0_cube=veg_roughness_cube,
                height_levels_cube=height_levels).process(wind_speed_slice))
        wind_speed_list.append(result)

    # Temporary fix for chunking problems when merging cubes
    max_npoints = max([np.prod(cube.data.shape) for cube in wind_speed_list])
    while iris._lazy_data._MAX_CHUNK_SIZE < max_npoints:
        iris._lazy_data._MAX_CHUNK_SIZE *= 2

    wind_speed = wind_speed_list.merge_cube()
    non_dim_coords = [x.name() for x in wind_speed.coords(dim_coords=False)]
    if 'realization' in non_dim_coords:
        wind_speed = iris.util.new_axis(wind_speed, 'realization')

    if args.output_height_level:
        constraints = {'height': float(args.output_height_level)}
        units = {'height': args.output_height_level_units}
        single_level = apply_extraction(
            wind_speed, iris.Constraint(**constraints), units)
        if not single_level:
            raise ValueError(
                'Requested height level not found, no cube '
                'returned. Available height levels are:\n'
                '{0:}\nin units of {1:}'.format(
                    wind_speed.coord('height').points,
                    wind_speed.coord('height').units))
        wind_speed = single_level

    save_netcdf(wind_speed, args.output_filepath)


if __name__ == "__main__":
    main()
