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

"""Script to run weighted blending."""

import warnings

import numpy as np
from cf_units import Unit
import json

from improver.argparser import ArgParser
from improver.utilities.load import load_cubelist
from improver.utilities.save import save_netcdf

"""
from improver.utilities.spatial import (
    check_if_grid_is_equal_area, convert_distance_into_number_of_grid_cells)

from improver.blending.calculate_weights_and_blend import (
    calculate_blending_weights)
from improver.blending.spatial_weights import (
    SpatiallyVaryingWeightsFromMask)
from improver.blending.weighted_blend import (
    MergeCubesForWeightedBlending, conform_metadata,
    WeightedBlendAcrossWholeDimension)
"""
from improver.blending.calculate_weights_and_blend import WeightAndBlend


def main(argv=None):
    """Load in arguments and ensure they are set correctly.
       Then load in the data to blend and calculate default weights
       using the method chosen before carrying out the blending."""
    parser = ArgParser(
        description='Calculate the default weights to apply in weighted '
        'blending plugins using the ChooseDefaultWeightsLinear or '
        'ChooseDefaultWeightsNonLinear plugins. Then apply these '
        'weights to the dataset using the BasicWeightedAverage plugin.'
        ' Required for ChooseDefaultWeightsLinear: y0val and ynval.'
        ' Required for ChooseDefaultWeightsNonLinear: cval.'
        ' Required for ChooseWeightsLinear with dict: wts_dict.')

    parser.add_argument('--wts_calc_method',
                        metavar='WEIGHTS_CALCULATION_METHOD',
                        choices=['linear', 'nonlinear', 'dict'],
                        default='linear', help='Method to use to calculate '
                        'weights used in blending. "linear" (default): '
                        'calculate linearly varying blending weights. '
                        '"nonlinear": calculate blending weights that decrease'
                        ' exponentially with increasing blending coordinate. '
                        '"dict": calculate weights using a dictionary passed '
                        'in as a command line argument.')

    parser.add_argument('coordinate', type=str,
                        metavar='COORDINATE_TO_AVERAGE_OVER',
                        help='The coordinate over which the blending '
                             'will be applied.')
    parser.add_argument('--coordinate_unit', metavar='UNIT_STRING',
                        default='hours since 1970-01-01 00:00:00',
                        help='Units for blending coordinate. Default= '
                             'hours since 1970-01-01 00:00:00')
    parser.add_argument('--cycletime', metavar='CYCLETIME', type=str,
                        help='The forecast reference time to be used after '
                        'blending has been applied, in the format '
                        'YYYYMMDDTHHMMZ. If not provided, the blended file '
                        'will take the latest available forecast reference '
                        'time from the input cubes supplied.')
    parser.add_argument('--model_id_attr', metavar='MODEL_ID_ATTR', type=str,
                        default="mosg__model_configuration",
                        help='The name of the netCDF file attribute to be '
                             'used to identify the source model for '
                             'multi-model blends. Default assumes Met Office '
                             'model metadata. Must be present on all input '
                             'files if blending over models.')
    parser.add_argument('--spatial_weights_from_mask',
                        action='store_true', default=False,
                        help='If set this option will result in the generation'
                             ' of spatially varying weights based on the'
                             ' masks of the data we are blending. The'
                             ' one dimensional weights are first calculated '
                             ' using the chosen weights calculation method,'
                             ' but the weights will then be adjusted spatially'
                             ' based on where there is masked data in the data'
                             ' we are blending. The spatial weights are'
                             ' calculated using the'
                             ' SpatiallyVaryingWeightsFromMask plugin.')
    parser.add_argument('weighting_mode', metavar='WEIGHTED_BLEND_MODE',
                        choices=['weighted_mean', 'weighted_maximum'],
                        help='The method used in the weighted blend. '
                             '"weighted_mean": calculate a normal weighted'
                             ' mean across the coordinate. '
                             '"weighted_maximum": multiplies the values in the'
                             ' coordinate by the weights, and then takes the'
                             ' maximum.')

    parser.add_argument('input_filepaths', metavar='INPUT_FILES',
                        nargs="+",
                        help='Paths to input files to be blended.')
    parser.add_argument('output_filepath', metavar='OUTPUT_FILE',
                        help='The output path for the processed NetCDF.')

    spatial = parser.add_argument_group(
        'Spatial weights from mask options',
        'Options for calculating the spatial weights using the '
        'SpatiallyVaryingWeightsFromMask plugin.')
    spatial.add_argument('--fuzzy_length', metavar='FUZZY_LENGTH', type=float,
                         default=20000,
                         help='When calculating spatially varying weights we'
                              ' can smooth the weights so that areas close to'
                              ' areas that are masked have lower weights than'
                              ' those further away. This fuzzy length controls'
                              ' the scale over which the weights are smoothed.'
                              ' The fuzzy length is in terms of m, the'
                              ' default is 20km. This distance is then'
                              ' converted into a number of grid squares,'
                              ' which does not have to be an integer. Assumes'
                              ' the grid spacing is the same in the x and y'
                              ' directions, and raises an error if this is not'
                              ' true. See SpatiallyVaryingWeightsFromMask for'
                              ' more detail.')

    linear = parser.add_argument_group('linear weights options',
                                       'Options for the linear weights '
                                       'calculation in '
                                       'ChooseDefaultWeightsLinear')
    linear.add_argument('--y0val', metavar='LINEAR_STARTING_POINT', type=float,
                        help='The relative value of the weighting start point '
                        '(lowest value of blend coord) for choosing default '
                        'linear weights. This must be a positive float or 0.')
    linear.add_argument('--ynval', metavar='LINEAR_END_POINT',
                        type=float, help='The relative value of the weighting '
                        'end point (highest value of blend coord) for choosing'
                        ' default linear weights. This must be a positive '
                        'float or 0.  Note that if blending over forecast '
                        'reference time, ynval >= y0val would normally be '
                        'expected (to give greater weight to the more recent '
                        'forecast).')

    nonlinear = parser.add_argument_group('nonlinear weights options',
                                          'Options for the non-linear '
                                          'weights calculation in '
                                          'ChooseDefaultWeightsNonLinear')
    nonlinear.add_argument('--cval', metavar='NON_LINEAR_FACTOR', type=float,
                           help='Factor used to determine how skewed the '
                                'non linear weights will be. '
                                'A value of 1 implies equal weighting. If not '
                                'set, a default value of cval=0.85 is set.')

    wts_dict = parser.add_argument_group('dict weights options',
                                         'Options for linear weights to be '
                                         'calculated based on parameters '
                                         'read from a json file dict')
    wts_dict.add_argument('--wts_dict', metavar='WEIGHTS_DICTIONARY',
                          help='Path to json file containing dictionary from '
                          'which to calculate blending weights. Dictionary '
                          'format is as specified in the improver.blending.'
                          'weights.ChooseWeightsLinear plugin.')
    wts_dict.add_argument('--weighting_coord', metavar='WEIGHTING_COORD',
                          default='forecast_period', help='Name of '
                          'coordinate over which linear weights should be '
                          'scaled. This coordinate must be available in the '
                          'weights dictionary.')

    args = parser.parse_args(args=argv)

    # if the linear weights method is called with non-linear args or vice
    # versa, exit with error
    if (args.wts_calc_method == "linear") and args.cval:
        parser.wrong_args_error('cval', 'linear')
    if ((args.wts_calc_method == "nonlinear") and np.any([args.y0val,
                                                          args.ynval])):
        parser.wrong_args_error('y0val, ynval', 'non-linear')
    if (args.wts_calc_method == "dict") and not args.wts_dict:
        parser.error('Dictionary is required if --wts_calc_method="dict"')

    # load cubes to be blended
    cubelist = load_cubelist(args.input_filepaths)

    if args.wts_calc_method == "dict":
        with open(args.wts_dict, 'r') as wts:
            weights_dict = json.load(wts)
    else:
        weights_dict = None

    plugin = WeightAndBlend(
        args.coordinate, args.wts_calc_method,
        blend_coord_unit=args.coordinate_unit,
        weighting_coord=args.weighting_coord, wts_dict=weights_dict,
        y0val=args.y0val, ynval=args.ynval, cval=args.cval)
    result = plugin.process(
        cubelist, args.weighting_mode, cycletime=args.cycletime,
        model_id_attr=args.model_id_attr,
        spatial_weights=args.spatial_weights_from_mask,
        fuzzy_length=args.fuzzy_length)

    save_netcdf(result, args.output_filepath)


if __name__ == "__main__":
    main()
