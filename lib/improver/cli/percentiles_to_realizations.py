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

"""Script to run Ensemble Copula Coupling processing."""

from improver.argparser import ArgParser
from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    RebadgePercentilesAsRealizations, ResamplePercentiles, EnsembleReordering)
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
    """
    Load in the arguments and apply the requested variant of Ensemble
    Copula Coupling for converting percentile data to realizations.
    """
    parser = ArgParser(
        description='Convert a dataset containing '
                    'probabilities into one containing '
                    'ensemble realizations using Ensemble Copula Coupling.')

    # General options:
    parser.add_argument('input_filepath', metavar='INPUT_FILE',
                        help='A path to an input NetCDF file to be processed.'
                             ' Must contain a percentile dimension.')
    parser.add_argument('output_filepath', metavar='OUTPUT_FILE',
                        help='The output path for the processed NetCDF.')
    parser.add_argument('--no_of_percentiles', default=None, type=int,
                        metavar='NUMBER_OF_PERCENTILES',
                        help='The number of percentiles to be generated. '
                             'This is also equal to the number of ensemble '
                             'realizations that will be generated.')
    parser.add_argument('--sampling_method', default='quantile',
                        const='quantile', nargs='?',
                        choices=['quantile', 'random'],
                        metavar='PERCENTILE_SAMPLING_METHOD',
                        help='Method to be used for generating the list of '
                             'percentiles with forecasts generated at each '
                             'percentile. The options are "quantile" and '
                             '"random". "quantile" is the default option. '
                             'The "quantile" option produces equally spaced '
                             'percentiles which is the preferred '
                             'option for full Ensemble Copula Coupling with '
                             'reordering enabled.')
    parser.add_argument(
        '--ecc_bounds_warning', default=False, action='store_true',
        help='If True, where percentiles (calculated as an intermediate '
             'output before realizations) exceed the ECC bounds range, raise '
             'a warning rather than an exception.')

    # Different use cases:
    # (We can either reorder OR rebadge)
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--reordering', default=False, action='store_true',
                       help='The option used to create ensemble realizations '
                       'from percentiles by reordering the input '
                       'percentiles based on the order of the '
                       'raw ensemble forecast.')
    group.add_argument('--rebadging', default=False, action='store_true',
                       help='The option used to create ensemble realizations '
                       'from percentiles by rebadging the input '
                       'percentiles.')

    # If reordering, can do so either based on original realizations,
    # or randomly.
    reordering = parser.add_argument_group(
        'Reordering options', 'Options for reordering the input percentiles '
        'using the raw ensemble forecast as required to create ensemble '
        'realizations.')
    reordering.add_argument('--raw_forecast_filepath',
                            metavar='RAW_FORECAST_FILE',
                            help='A path to an raw forecast NetCDF file to be '
                            'processed. This option is compulsory, if the '
                            'reordering option is selected.')
    reordering.add_argument('--random_ordering', default=False,
                            action='store_true',
                            help='Decide whether or not to use random '
                            'ordering within the ensemble reordering step.')
    reordering.add_argument(
        '--random_seed', default=None,
        help='Option to specify a value for the random seed for testing '
             'purposes, otherwise, the default random seed behaviour is '
             'utilised. The random seed is used in the generation of the '
             'random numbers used for either the random_ordering option to '
             'order the input percentiles randomly, rather than use the '
             'ordering from the raw ensemble, or for splitting tied values '
             'within the raw ensemble, so that the values from the input '
             'percentiles can be ordered to match the raw ensemble.')

    rebadging = parser.add_argument_group(
        'Rebadging options', 'Options for rebadging the input percentiles '
        'as ensemble realizations.')
    rebadging.add_argument('--realization_numbers', default=None,
                           metavar='REALIZATION_NUMBERS', nargs="+",
                           help='A list of ensemble realization numbers to '
                                'use when rebadging the percentiles '
                                'into realizations.')

    args = parser.parse_args(args=argv)

    # CLI argument checking:
    # Can only do one of reordering or rebadging: if options are passed which
    # correspond to the opposite method, raise an exception.
    # Note: Shouldn't need to check that both/none are set, since they are
    # defined as mandatory, but mutually exclusive, options.
    if args.reordering:
        if args.realization_numbers is not None:
            parser.wrong_args_error('realization_numbers', 'reordering')
    if args.rebadging:
        if ((args.raw_forecast_filepath is not None) or
                (args.random_ordering is not False)):
            parser.wrong_args_error(
                'raw_forecast_filepath, random_ordering', 'rebadging')

    # Convert the string of realization_numbers to a list of ints.
    realization_numbers = None
    if args.rebadging:
        if args.realization_numbers is not None:
            realization_numbers = (
                [int(num) for num in args.realization_numbers])

    cube = load_cube(args.input_filepath)
    raw_forecast = load_cube(args.raw_forecast_filepath, allow_none=True)

    # Process Cube
    result_cube = process(cube, raw_forecast, args.no_of_percentiles,
                          args.sampling_method, args.ecc_bounds_warning,
                          args.reordering, args.rebadging,
                          args.random_ordering, args.random_seed,
                          realization_numbers)

    # Save Cube
    save_netcdf(result_cube, args.output_filepath)


def process(cube, raw_forecast=None, no_of_percentiles=None,
            sampling_method='quantile', ecc_bounds_warning=False,
            reordering=False, rebadging=False, random_ordering=False,
            random_seed=None, realization_numbers=None):
    """Runs Ensemble Copula Coupling processing.

    Converts a dataset containing percentiles into one containing ensemble
    realizations using Ensemble Coupla Coupling.

    Args:
        cube (iris.cube.Cube):
            Cube expected to contain a percentiles coordinate.
        raw_forecast (iris.cube.Cube):
            Cube of raw (not post processed) weather data.
            This option is compulsory, if the reordering option is selected.
        no_of_percentiles (int):
            The number of percentiles to be generated. This is also equal to
            the number of ensemble realizations that will be generated.
            Default is None.
        sampling_method (str):
            Method to be used for generating the list of percentiles with
            forecasts generated at each percentile. The options are "quantile"
            and "random".
            The quantile option produces equally spaced percentiles which is
            the preferred option for full ensemble couple coupling with
            reordering enabled.
            Default is 'quantile'.
        ecc_bounds_warning (bool):
            If True where percentiles (calculated as an intermediate output
            before realization) exceed the ECC bounds range, raises a
            warning rather than an exception.
            Default is False.
        reordering (bool):
            The option used to create ensemble realizations from percentiles
            by reordering the input percentiles based on the order of the
            raw ensemble forecast.
            Default is False.
        rebadging (bool):
            The option used to create ensemble realizations from percentiles
            by rebadging the input percentiles.
            Default is False.
        random_ordering (bool):
            If random_ordering is True, the post-processed forecasts are
            reordered randomly, rather than using the ordering of the
            raw ensemble.
            Default is False.
        random_seed (int):
            Option to specify a value for the random seed for testing purposes,
            otherwise, the default random seed behaviour is utilised.
            The random seed is used in the generation of the random numbers
            used for either the random_ordering option to order the input
            percentiles randomly, rather than use the ordering from the
            raw ensemble, or for splitting tied values within the raw ensemble
            so that the values from the input percentiles can be ordered to
            match the raw ensemble.
            Default is None.
        realization_numbers (list of ints):
            A list of ensemble realization numbers to use when rebadging the
            percentiles into realizations.
            Default is None.

    Returns:
        result (iris.cube.Cube):
            The processed Cube.
    """
    if reordering:
        if realization_numbers is not None:
            raise TypeError('realization_numbers cannot be used with '
                            'reordering.')
    if rebadging:
        if raw_forecast is not None:
            raise TypeError('rebadging cannot be used with raw_forecast.')
    if rebadging:
        if random_ordering is not False:
            raise TypeError('rebadging cannot be used with random_ordering.')

    result = ResamplePercentiles(
        ecc_bounds_warning=ecc_bounds_warning).process(
        cube, no_of_percentiles=no_of_percentiles,
        sampling=sampling_method)
    if reordering:
        result = EnsembleReordering().process(
            result, raw_forecast, random_ordering=random_ordering,
            random_seed=random_seed)
    elif rebadging:
        result = RebadgePercentilesAsRealizations().process(
            result, ensemble_realization_numbers=realization_numbers)
    return result


if __name__ == '__main__':
    main()
