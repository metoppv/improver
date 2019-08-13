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
"""Script to run ensemble calibration."""

import numpy as np

from iris.exceptions import CoordinateNotFoundError

from improver.argparser import ArgParser
from improver.ensemble_calibration.ensemble_calibration import (
    EnsembleCalibration)
from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    EnsembleReordering,
    GeneratePercentilesFromMeanAndVariance,
    GeneratePercentilesFromProbabilities,
    GenerateProbabilitiesFromMeanAndVariance,
    RebadgePercentilesAsRealizations,
    ResamplePercentiles)
from improver.utilities.cube_checker import find_percentile_coordinate
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
    """Do ensemble calibration using the EnsembleCalibration plugin.
    """
    parser = ArgParser(
        description='Apply the requested ensemble calibration method using '
        'the current forecast (to be calibrated) in the form of '
        'realizations, probabilities, or percentiles, historical '
        'forecasts in the form of realizations and historical truth data '
        '(to use in calibration). The mean and variance output from the '
        'EnsembleCalibration plugin can be written to an output file '
        'if required. If the current forecast is supplied in the form of '
        'probabilities or percentiles, these are converted to realizations '
        'prior to calibration. After calibration, the mean and variance '
        'computed in the calibration are converted to match the format of the '
        'current forecast i.e. if realizations are input, realizations '
        'are output, if probabilities are input, probabilities are output, '
        'and if percentiles are input, percentiles are output.'
        'If realizations are input, realizations are regenerated using '
        'Ensemble Copula Coupling.')
    # Arguments for EnsembleCalibration
    parser.add_argument(
        'units', metavar='UNITS_TO_CALIBRATE_IN',
        help='The unit that calibration should be undertaken in. The current '
             'forecast, historical forecast and truth will be converted as '
             'required.')
    parser.add_argument(
        'distribution', metavar='DISTRIBUTION',
        choices=['gaussian', 'truncated gaussian'],
        help='The distribution that will be used for calibration. This will '
             'be dependent upon the input phenomenon. This has to be '
             'supported by the minimisation functions in '
             'ContinuousRankedProbabilityScoreMinimisers.')
    # Filepaths for current, historic and truth data.
    parser.add_argument(
        'input_filepath', metavar='INPUT_FILE',
        help='A path to an input NetCDF file containing the current forecast '
             'to be processed. The file provided could be in the form of '
             'realizations, probabilities or percentiles.')
    parser.add_argument(
        'historic_filepath', metavar='HISTORIC_DATA_FILE',
        help='A path to an input NetCDF file containing the historic '
             'forecast(s) used for calibration. The file provided must be in '
             'the form of realizations.')
    parser.add_argument(
        'truth_filepath', metavar='TRUTH_DATA_FILE',
        help='A path to an input NetCDF file containing the historic truth '
             'analyses used for calibration.')
    parser.add_argument(
        'output_filepath', metavar='OUTPUT_FILE',
        help='The output path for the processed NetCDF')
    # Optional arguments.
    parser.add_argument(
        '--predictor_of_mean', metavar='CALIBRATE_MEAN_FLAG',
        choices=['mean', 'realizations'], default='mean',
        help='String to specify the input to calculate the calibrated mean. '
             'Currently the ensemble mean ("mean") and the ensemble '
             'realizations ("realizations") are supported as the predictors. '
             'Default: "mean".')
    parser.add_argument(
        '--save_mean', metavar='MEAN_FILE',
        default=False,
        help='Option to save the mean output from EnsembleCalibration plugin. '
             'If used, a path to save the output to must be provided.')
    parser.add_argument(
        '--save_variance', metavar='VARIANCE_FILE',
        default=False,
        help='Option to save the variance output from EnsembleCalibration '
             'plugin. If used, a path to save the output to must be provided.')
    parser.add_argument(
        '--num_realizations', metavar='NUMBER_OF_REALIZATIONS',
        default=None, type=np.int32,
        help='Optional argument to specify the number of '
             'ensemble realizations to produce. '
             'If the current forecast is input as probabilities or '
             'percentiles then this argument is used to create the requested '
             'number of realizations. In addition, this argument is used to '
             'construct the requested number of realizations from the mean '
             'and variance output after applying the EMOS coefficients.'
             'Default will be the number of realizations in the raw input '
             'file, if realizations are provided as input, otherwise if the '
             'input format is probabilities or percentiles, then an error '
             'will be raised if no value is provided.')
    parser.add_argument(
        '--random_ordering', default=False,
        action='store_true',
        help='Option to reorder the post-processed forecasts randomly. If not '
             'set, the ordering of the raw ensemble is used. This option is '
             'only valid when the input format is realizations.')
    parser.add_argument(
        '--random_seed', metavar='RANDOM_SEED', default=None,
        help='Option to specify a value for the random seed for testing '
             'purposes, otherwise, the default random seed behaviour is '
             'utilised. The random seed is used in the generation of the '
             'random numbers used for either the random_ordering option to '
             'order the input percentiles randomly, rather than use the '
             'ordering from the raw ensemble, or for splitting tied values '
             'within the raw ensemble, so that the values from the input '
             'percentiles can be ordered to match the raw ensemble.')
    parser.add_argument(
        '--ecc_bounds_warning', default=False,
        action='store_true',
        help='If True, where the percentiles exceed the ECC bounds range, '
             'raise a warning rather than an exception. This occurs when the '
             'current forecast is in the form of probabilities and is '
             'converted to percentiles, as part of converting the input '
             'probabilities into realizations.')
    parser.add_argument(
        '--max_iterations', metavar='MAX_ITERATIONS',
        type=np.int32, default=1000,
        help='The maximum number of iterations allowed until the minimisation '
             'has converged to a stable solution. If the maximum number of '
             'iterations is reached, but the minimisation has not yet '
             'converged to a stable solution, then the available solution is '
             'used anyway, and a warning is raised. This may be modified for '
             'testing purposes but otherwise kept fixed. If the '
             'predictor_of_mean is "realizations", then the number of '
             'iterations may require increasing, as there will be more '
             'coefficients to solve for.')
    args = parser.parse_args(args=argv)

    # Load cubes
    current_forecast = load_cube(args.input_filepath)
    historic_forecast = load_cube(args.historic_filepath)
    truth = load_cube(args.truth_filepath)
    # Process Cubes
    result, forecast_predictor, forecast_variance = process(
        current_forecast, historic_forecast, truth, args.units,
        args.distribution, args.predictor_of_mean, args.num_realizations,
        args.random_ordering, args.random_seed, args.ecc_bounds_warning,
        args.max_iterations)
    # Save Cube
    save_netcdf(result, args.output_filepath)

    # If required, save the mean and variance.
    if args.save_mean:
        save_netcdf(forecast_predictor, args.save_mean)
    if args.save_variance:
        save_netcdf(forecast_variance, args.save_variance)


def process(current_forecast, historic_forecast, truth, units, distribution,
            predictor_of_mean='mean', num_realizations=None,
            random_ordering=False, random_seed=None, ecc_bounds_warning=False,
            max_iterations=1000):
    """Module for ensemble calibration using the EnsembleCalibration plugin.

    Apply the requested ensemble calibration method using the current
    forecast (to be calibrated) in the form of realizations, probabilities,
    or percentiles. Historical forecasts in the form of realizations and
    historical truth data (to use in calibration).
    If the current forecast is supplied in the form of probabilities or
    percentiles, these are converted to realizations prior to calibration.
    After calibration, the mean and variance computed in the calibration
    are converted to match the format of the current forecast. i.e
    If realizations are input, realizations are output.
    If probabilities are input, probabilities are output.
    Also if realizations are input, realizations are regenerated using
    Ensemble Coupla Coupling.

    Args:
        current_forecast (iris.cube.Cube):
            A Cube containing the current forecast to be calibrated.
        historic_forecast (iris.cube.Cube):
            A cube containing historic forecasts to be used for calibration.
        truth (iris.cube.Cube):
            A Cube containing the historic truth data for calibration.
        units (str):
            The unit that calibration should be undertaken in. The current
            forecast, historical forecast and truth will be converted as
            required.
        distribution (str):
            The distribution that will be used for calibration. This will be
            dependent upon the input phenomenon. This has to be supported by
            the minimisation function in
            ContinuousRankedProbabilityScoreMinimisers.

    Keyword Args:
        predictor_of_mean (str):
            String to specify the input to calculate the calibrated mean.
            Currently the ensemble mean "mean" and the ensemble realizations
            "realizations" are supported as the predictors.
            Default is 'mean'.
        num_realizations (numpy.int32):
            Optional argument to specify the number of ensemble realizations
            to produce. If the current forecast is input as probabilities or
            percentiles then this argument is used to create the requested
            number of realizations. In addition, this argument is used to
            construct the requested number of realizations from the mean and
            variance output after applying the EMOS coefficients.
        random_ordering (bool):
            Optional argument to reorder the post-processed forecasts
            randomly. If not set, the ordering of the raw ensemble is used.
            This option is only valid when the input format is realizations.
        random_seed (int):
            If random_seed is an integer, the integer value is used for the
            random_seed. The random seed is used in the generation of the
            random numbers used for either the random_ordering option to order
            the input percentiles randomly, rather than use the ordering from
            the raw ensemble, or for splitting tied values within the raw
            ensemble, so that the values from the input percentiles can be
            ordered to match the raw ensemble.
            If the random_seed is None, no random seed is set, so the random
            values generated are not reproducible.
        ecc_bounds_warning (bool):
            If True, where the percentiles exceed the ECC bounds range, raises
            a warning rather than an exception. This occurs when the current
            forecast is in the form of probabilities and is converted to
            percentiles, as part of converting the input probabilities into
            realizations.
        max_iterations (int):
            The maximum number of iterations allowed until the minimisation has
            converged to a stable solution. If the maximum number of iterations
            is reached, but the minimisation has not yet converged to a stable
            solution, the available solution is used anyway and a warning
            is raised. If the predictor_of_mean is "realizations", then the
            number of iterations may require increasing, as there will be more
            coefficients to solve for.

    Returns:
        (tuple): tuple containing:
            **result** (iris.cube.Cube):
                A cube after being processed.
            **forecast_predictor** (iris.cube.Cube):
                The mean output from EnsembleCalibration.
            **forecast_variance** (iris.cube.Cube):
                The variance output from EnsembleCalibration.

    Raises:
        ValueError:
            If forecast type is 'percentiles' or 'probabilities' while no
            num_realizations are given.
    """
    original_current_forecast = current_forecast.copy()
    try:
        find_percentile_coordinate(current_forecast)
        input_forecast_type = "percentiles"
    except CoordinateNotFoundError:
        input_forecast_type = "realizations"

    if current_forecast.name().startswith("probability_of"):
        input_forecast_type = "probabilities"
        # If probabilities, convert to percentiles.
        conversion_plugin = GeneratePercentilesFromProbabilities(
            ecc_bounds_warning=ecc_bounds_warning)
    elif input_forecast_type == "percentiles":
        # If percentiles, resample percentiles so that the percentiles are
        # evenly spaced.
        conversion_plugin = ResamplePercentiles(
            ecc_bounds_warning=ecc_bounds_warning)

    # If percentiles, resample percentiles and then rebadge.
    # If probabilities, generate percentiles and then rebadge.
    if input_forecast_type in ["percentiles", "probabilities"]:
        if not num_realizations:
            raise ValueError(
                "The current forecast has been provided as {0}. "
                "These {0} need to be converted to realizations "
                "for ensemble calibration. The args.num_realizations "
                "argument is used to define the number of realizations "
                "to construct from the input {0}, so if the "
                "current forecast is provided as {0} then "
                "args.num_realizations must be defined.".format(
                    input_forecast_type))
        current_forecast = conversion_plugin.process(
            current_forecast, no_of_percentiles=num_realizations)
        current_forecast = (
            RebadgePercentilesAsRealizations().process(current_forecast))

    # Default number of ensemble realizations is the number in
    # the raw forecast.
    # If None or 0 it takes from the length of current_forecast.
    if not num_realizations:
        num_realizations = len(
            current_forecast.coord('realization').points)

    # Ensemble-Calibration to calculate the mean and variance.
    forecast_predictor, forecast_variance = EnsembleCalibration(
        distribution, units,
        predictor_of_mean_flag=predictor_of_mean,
        max_iterations=max_iterations).process(
        current_forecast, historic_forecast, truth)

    # If input forecast is probabilities, convert output into probabilities.
    # If input forecast is percentiles, convert output into percentiles.
    # If input forecast is realizations, convert output into realizations.
    if input_forecast_type == "probabilities":
        result = GenerateProbabilitiesFromMeanAndVariance().process(
            forecast_predictor, forecast_variance, original_current_forecast)
    elif input_forecast_type == "percentiles":
        perc_coord = find_percentile_coordinate(original_current_forecast)
        result = GeneratePercentilesFromMeanAndVariance().process(
            forecast_predictor, forecast_variance,
            percentiles=perc_coord.points)
    elif input_forecast_type == "realizations":
        # Ensemble Copula Coupling to generate realizations
        # from mean and variance.
        percentiles = GeneratePercentilesFromMeanAndVariance().process(
            forecast_predictor, forecast_variance,
            no_of_percentiles=num_realizations)
        result = EnsembleReordering().process(
            percentiles, current_forecast,
            random_ordering=random_ordering,
            random_seed=random_seed)
    return result, forecast_predictor, forecast_variance


if __name__ == "__main__":
    main()
