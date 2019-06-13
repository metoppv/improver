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

from improver.argparser import ArgParser
from improver.ensemble_calibration.ensemble_calibration import (
    EstimateCoefficientsForEnsembleCalibration)
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
    """Load in arguments for estimating coefficients for Ensemble Model Output
       Statistics (EMOS), otherwise known as Non-homogeneous Gaussian
       Regression (NGR). 2 sources of input data must be provided: historical
       forecasts and historical truth data (to use in calibration). The
       estimated coefficients are written to a netCDF file.
    """
    parser = ArgParser(
        description='Estimate coefficients for for Ensemble Model Output '
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
    parser.add_argument('historic_filepath', metavar='HISTORIC_FILEPATH',
                        help='A path to an input NetCDF file containing the '
                             'historic forecast(s) used for calibration.')
    parser.add_argument('truth_filepath', metavar='TRUTH_FILEPATH',
                        help='A path to an input NetCDF file containing the '
                             'historic truth analyses used for calibration.')
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
                             'is used anyway, and a warning is raised. '
                             'If the predictor_of_mean is "realizations", '
                             'then the number of iterations may require '
                             'increasing, as there will be more coefficients '
                             'to solve for.')

    args = parser.parse_args(args=argv)

    historic_forecast = load_cube(args.historic_filepath)
    truth = load_cube(args.truth_filepath)

    # Estimate coefficients using Ensemble Model Output Statistics (EMOS).
    estcoeffs = EstimateCoefficientsForEnsembleCalibration(
        args.distribution, args.cycletime, desired_units=args.units,
        predictor_of_mean_flag=args.predictor_of_mean,
        max_iterations=args.max_iterations)
    coefficients = (
        estcoeffs.estimate_coefficients_for_ngr(historic_forecast, truth))

    save_netcdf(coefficients, args.output_filepath)


if __name__ == "__main__":
    main()
