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

import numpy as np
import warnings

from improver.argparser import ArgParser
from improver.ensemble_calibration.ensemble_calibration import (
    EstimateCoefficientsForEnsembleCalibration)
from improver.ensemble_calibration.ensemble_calibration_utilities import (
    SplitHistoricForecastAndTruth)
from improver.utilities.load import load_cubelist
from improver.utilities.cli_utilities import (
    load_cube_or_none, load_json_or_none)
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
                    'Gaussian Regression (NGR)')
    parser.add_argument('distribution', metavar='DISTRIBUTION',
                        choices=['gaussian', 'truncated gaussian'],
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
    # Filepaths for historic and truth data.
    parser.add_argument('--historic_filepath', metavar='HISTORIC_FILEPATH',
                        help='A path to an input NetCDF file containing the '
                             'historic forecast(s) used for calibration.')
    parser.add_argument('--truth_filepath', metavar='TRUTH_FILEPATH',
                        help='A path to an input NetCDF file containing the '
                             'historic truth analyses used for calibration.')
    # Input filepaths
    parser.add_argument('--combined_filepath', metavar='COMBINED_FILEPATH',
                        help='The path to the input NetCDF files containing '
                             'both the historic forecast(s) and truth '
                             'analyses used for calibration.')
    parser.add_argument("--historic_forecast_identifier",
                        metavar='HISTORIC_FORECAST_IDENTIFIER',
                        help='The path to a json file containing metadata '
                             'information that defines the historic forecast.')
    parser.add_argument("--truth_identifier", metavar='TRUTH_IDENTIFIER',
                        help='The path to a json file containing metadata '
                             'information that defines the truth.')
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
    historic_forecast = load_cube_or_none(args.historic_filepath)
    truth = load_cube_or_none(args.truth_filepath)

    combined = (load_cubelist(args.combined_filepath)
                if args.combined_filepath else None)
    historic_forecast_dict = (
        load_json_or_none(args.historic_forecast_identifier))
    truth_dict = load_json_or_none(args.truth_identifier)

    if not any([historic_forecast, truth, combined]):
        msg = ("In order to calculate the EMOS coefficients then either "
               "the historic_filepath {} and the truth_filepath {} "
               "should be specified, or the combined_filepaths {} should be "
               "specified alongside the historic_forecast_identifier {} and "
               "truth_identifier {}. In this case the arguments provided "
               "were not adequate.".format(
                    args.historic_filepath, args.truth_filepath,
                    args.combined_filepath, args.historic_forecast_identifier,
                    args.truth_identifier))
        warnings.warn(msg)
        return

    # Process Cube
    coefficients = process(historic_forecast, truth, combined,
                           historic_forecast_dict, truth_dict,
                           args.distribution, args.cycletime, args.units,
                           args.predictor_of_mean, args.max_iterations)
    # Save Cube
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
                {
                    "attributes": {
                        "mosg__model_configuration": "uk_ens"
                    }
                }
        truth_dict (dict):
            Dictionary specifying the metadata that defines the truth.
            For example:
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

    Keyword Args:
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
        result (iris.cube.Cube):
            Cube containing the coefficients estimated using EMOS. The cube
            contains a coefficient_index dimension coordinate and a
            coefficient_name auxiliary coordinate.
    """
    if combined is not None:
        historic_forecast, truth = SplitHistoricForecastAndTruth(
            historic_forecast_dict, truth_dict).process(combined)

    result = EstimateCoefficientsForEnsembleCalibration(
        distribution, cycletime, desired_units=units,
        predictor_of_mean_flag=predictor_of_mean,
        max_iterations=max_iterations).process(historic_forecast, truth)

    return result


if __name__ == "__main__":
    main()
