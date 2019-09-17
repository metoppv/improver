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
"""Script to estimate coefficients for Ensemble Model Output
Statistics (EMOS), otherwise known as Non-homogeneous Gaussian
Regression (NGR)."""

import warnings

import numpy as np

from improver.argparser import ArgParser
from improver.ensemble_calibration.ensemble_calibration import (
    EstimateCoefficientsForEnsembleCalibration)
from improver.ensemble_calibration.ensemble_calibration_utilities import (
    SplitHistoricForecastAndTruth)
from improver.utilities.load import load_cube, load_cubelist
from improver.utilities.cli_utilities import (
    load_json_or_none)
from improver.utilities.save import save_netcdf


def main(argv=None):
    """Load in arguments for estimating coefficients for Ensemble Model Output
       Statistics (EMOS), otherwise known as Non-homogeneous Gaussian
       Regression (NGR). 2 sources of input data must be provided: historical
       forecasts and historical truth data (to use in calibration). The
       estimated coefficients are written to a netCDF file.
    """
    parser = ArgParser(
        description='Estimate coefficients for Ensemble Model Output '
                    'Statistics (EMOS), otherwise known as Non-homogeneous '
                    'Gaussian Regression (NGR). There are two methods for '
                    'inputting data into this CLI, either by providing the '
                    'historic forecasts and truth separately, or by providing '
                    'a combined list of historic forecasts and truths along '
                    'with historic_forecast_identifier and truth_identifier '
                    'arguments to provide metadata that distinguishes between '
                    'them.')
    parser.add_argument('distribution', metavar='DISTRIBUTION',
                        choices=['gaussian', 'truncated_gaussian'],
                        help='The distribution that will be used for '
                             'calibration. This will be dependent upon the '
                             'input phenomenon. This has to be supported by '
                             'the minimisation functions in '
                             'ContinuousRankedProbabilityScoreMinimisers.')
    parser.add_argument('cycletime', metavar='CYCLETIME', type=str,
                        help='This denotes the cycle at which forecasts '
                             'will be calibrated using the calculated '
                             'EMOS coefficients. The validity time in the '
                             'output coefficients cube will be calculated '
                             'relative to this cycletime. '
                             'This cycletime is in the format '
                             'YYYYMMDDTHHMMZ.')

    # Historic forecast and truth filepaths
    parser.add_argument(
        '--historic_filepath', metavar='HISTORIC_FILEPATH', nargs='+',
        help='Paths to the input NetCDF files containing the '
             'historic forecast(s) used for calibration. '
             'This must be supplied with an associated truth filepath. '
             'Specification of either the combined_filepath, '
             'historic_forecast_identifier or the truth_identifier is '
             'invalid with this argument.')
    parser.add_argument(
        '--truth_filepath', metavar='TRUTH_FILEPATH', nargs='+',
        help='Paths to the input NetCDF files containing the '
             'historic truth analyses used for calibration. '
             'This must be supplied with an associated historic filepath. '
             'Specification of either the combined_filepath, '
             'historic_forecast_identifier or the truth_identifier is '
             'invalid with this argument.')

    # Input filepaths
    parser.add_argument(
        '--combined_filepath', metavar='COMBINED_FILEPATH', nargs='+',
        help='Paths to the input NetCDF files containing '
             'both the historic forecast(s) and truth '
             'analyses used for calibration. '
             'This must be supplied with both the '
             'historic_forecast_identifier and the truth_identifier. '
             'Specification of either the historic_filepath or the '
             'truth_filepath is invalid with this argument.')
    parser.add_argument(
        "--historic_forecast_identifier",
        metavar='HISTORIC_FORECAST_IDENTIFIER',
        help='The path to a json file containing metadata '
             'information that defines the historic forecast. '
             'This must be supplied with both the combined_filepath and the '
             'truth_identifier. Specification of either the historic_filepath'
             'or the truth_filepath is invalid with this argument. '
             'The intended contents is described in improver.'
             'ensemble_calibration.ensemble_calibration_utilities.'
             'SplitHistoricForecastAndTruth.')
    parser.add_argument(
        "--truth_identifier", metavar='TRUTH_IDENTIFIER',
        help='The path to a json file containing metadata '
             'information that defines the truth.'
             'This must be supplied with both the combined_filepath and the '
             'historic_forecast_identifier. Specification of either the '
             'historic_filepath or the truth_filepath is invalid with this '
             'argument. The intended contents is described in improver.'
             'ensemble_calibration.ensemble_calibration_utilities.'
             'SplitHistoricForecastAndTruth.')

    # Output filepath
    parser.add_argument('output_filepath', metavar='OUTPUT_FILEPATH',
                        help='The output path for the processed NetCDF')
    # Optional arguments.
    parser.add_argument('--units', metavar='UNITS',
                        help='The units that calibration should be undertaken '
                             'in. The historical forecast and truth will be '
                             'converted as required.')
    parser.add_argument('--predictor_of_mean', metavar='PREDICTOR_OF_MEAN',
                        choices=['mean', 'realizations'], default='mean',
                        help='String to specify the predictor used to '
                             'calibrate the forecast mean. Currently the '
                             'ensemble mean ("mean") and the ensemble '
                             'realizations ("realizations") are supported as '
                             'options. Default: "mean".')
    parser.add_argument('--max_iterations', metavar='MAX_ITERATIONS',
                        type=np.int32, default=1000,
                        help='The maximum number of iterations allowed '
                             'until the minimisation has converged to a '
                             'stable solution. If the maximum number '
                             'of iterations is reached, but the '
                             'minimisation has not yet converged to a '
                             'stable solution, then the available solution '
                             'is used anyway, and a warning is raised.'
                             'This may be modified for testing purposes '
                             'but otherwise kept fixed. If the '
                             'predictor_of_mean is "realizations", '
                             'then the number of iterations may require '
                             'increasing, as there will be more coefficients '
                             'to solve for.')
    args = parser.parse_args(args=argv)

    # Load Cubes
    historic_forecast = load_cube(args.historic_filepath, allow_none=True)
    truth = load_cube(args.truth_filepath, allow_none=True)

    combined = (load_cubelist(args.combined_filepath)
                if args.combined_filepath else None)
    historic_forecast_dict = (
        load_json_or_none(args.historic_forecast_identifier))
    truth_dict = load_json_or_none(args.truth_identifier)

    # Process Cube
    coefficients = process(historic_forecast, truth, combined,
                           historic_forecast_dict, truth_dict,
                           args.distribution, args.cycletime, args.units,
                           args.predictor_of_mean, args.max_iterations)
    # Save Cube
    # Check whether a coefficients cube has been created. If the historic
    # forecasts and truths provided did not match in validity time, then
    # no coefficients would have been calculated.
    if coefficients:
        save_netcdf(coefficients, args.output_filepath)


def process(historic_forecast, truth, combined, historic_forecast_dict,
            truth_dict, distribution, cycletime, units=None,
            predictor_of_mean='mean', max_iterations=1000):
    """Module for estimate coefficients for Ensemble Model Output Statistics.

    Loads in arguments for estimating coefficients for Ensemble Model
    Output Statistics (EMOS), otherwise known as Non-homogeneous Gaussian
    Regression (NGR). Two sources of input data must be provided: historical
    forecasts and historical truth data (to use in calibration).
    The estimated coefficients are output as a cube.

    Args:
        historic_forecast (iris.cube.Cube):
            The cube containing the historical forecasts used for calibration.
        truth (iris.cube.Cube):
            The cube containing the truth used for calibration.
        combined (iris.cube.CubeList):
            A cubelist containing a combination of historic forecasts and
            associated truths.
        historic_forecast_dict (dict):
            Dictionary specifying the metadata that defines the historic
            forecast. For example:
            ::

                {
                    "attributes": {
                        "mosg__model_configuration": "uk_ens"
                    }
                }
        truth_dict (dict):
            Dictionary specifying the metadata that defines the truth.
            For example:
            ::

                {
                    "attributes": {
                        "mosg__model_configuration": "uk_det"
                    }
                }
        distribution (str):
            The distribution that will be used for calibration. This will be
            dependant upon the input phenomenon.
        cycletime (str):
            This denotes the cycle at which forecasts will be calibrated using
            the calculated EMOS coefficients. The validity time in the output
            coefficients cube will be calculated relative to this cycletime.
            This cycletime is in the format YYYYMMDDTHHMMZ.
        units (str):
            The units that calibration should be undertaken in. The historical
            forecast and truth will be converted as required.
            Default is None.
        predictor_of_mean (str):
            String to specify the input to calculate the calibrated mean.
            Currently the ensemble mean ("mean") and the ensemble realizations
            ("realizations") are supported as the predictors.
            Default is 'mean'.
        max_iterations (int):
            The maximum number of iterations allowed until the minimisation has
            converged to a stable solution. If the maximum number of iterations
            is reached but the minimisation has not yet converged to a stable
            solution, then the available solution is used anyway, and a warning
            is raised.
            If the predictor_of_mean is "realizations", then the number of
            iterations may require increasing, as there will be more
            coefficients to solve.
            Default is 1000.

    Returns:
        result (iris.cube.Cube or None):
            Cube containing the coefficients estimated using EMOS. The cube
            contains a coefficient_index dimension coordinate and a
            coefficient_name auxiliary coordinate. If no historic forecasts or
            truths are found then None is returned.

    Raises:
        ValueError: If the historic forecast and truth inputs are specified,
            then the combined input, historic forecast dictionary and truth
            dictionary should not be specified.
        ValueError: If one of the historic forecast or truth inputs are
            specified, then they should both be specified.
        ValueError: All of the combined_filepath, historic_forecast_identifier
            and truth_identifier arguments should be specified if one of the
            arguments are specified.

    Warns:
        UserWarning: The metadata to identify the desired historic forecast or
            truth has found nothing matching the metadata information supplied.

    """
    # The logic for the if statements below is:
    # 1. Check whether either the historic_forecast or the truth exists.
    # 2. Check that both the historic forecast and the truth exists, otherwise,
    #    raise an error.
    # 3. Check that none of the combined, historic forecast dictionary or
    #    truth dictionary inputs have been provided, as these arguments are
    #    invalid, if the historic forecast and truth inputs have been provided.
    if any([historic_forecast, truth]):
        if all([historic_forecast, truth]):
            if any([combined, historic_forecast_dict, truth_dict]):
                msg = ("If the historic_filepath and truth_filepath arguments "
                       "are specified then none of the the combined_filepath, "
                       "historic_forecast_identifier and truth_identifier "
                       "arguments should be specified.")
                raise ValueError(msg)
        else:
            msg = ("Both the historic_filepath and truth_filepath arguments "
                   "should be specified if one of these arguments are "
                   "specified.")
            raise ValueError(msg)

    # This if block follows the logic:
    # 1. Check whether any of the combined, historic forecast dictionary or
    #    truth dictionary inputs have been provided.
    # 2. If not all of these inputs have been provided then raise an error,
    #    as all of these inputs are required to separate the combined input
    #    into the historic forecasts and truths.
    if any([combined, historic_forecast_dict, truth_dict]):
        if not all([combined, historic_forecast_dict, truth_dict]):
            msg = ("All of the combined_filepath, "
                   "historic_forecast_identifier and truth_identifier "
                   "arguments should be specified if one of the arguments are "
                   "specified.")
            raise ValueError(msg)

    try:
        if combined is not None:
            historic_forecast, truth = SplitHistoricForecastAndTruth(
                historic_forecast_dict, truth_dict).process(combined)
    except ValueError as err:
        # This error arises if the metadata to identify the desired historic
        # forecast or truth has found nothing matching the metadata
        # information supplied.
        if str(err).startswith("The metadata to identify the desired"):
            warnings.warn(str(err))
            result = None
        else:
            raise
    else:
        result = EstimateCoefficientsForEnsembleCalibration(
            distribution, cycletime, desired_units=units,
            predictor_of_mean_flag=predictor_of_mean,
            max_iterations=max_iterations).process(historic_forecast, truth)

    return result


if __name__ == "__main__":
    main()
