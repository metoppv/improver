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
"""Script to apply coefficients for Ensemble Model Output
Statistics (EMOS), otherwise known as Non-homogeneous Gaussian
Regression (NGR)."""

import warnings

import numpy as np
from iris.exceptions import CoordinateNotFoundError

from improver.argparser import ArgParser
from improver.ensemble_calibration.ensemble_calibration import (
    ApplyCoefficientsFromEnsembleCalibration)
from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    ConvertLocationAndScaleParametersToPercentiles,
    ConvertLocationAndScaleParametersToProbabilities,
    ConvertProbabilitiesToPercentiles,
    EnsembleReordering,
    RebadgePercentilesAsRealizations,
    ResamplePercentiles)
from improver.metadata.probabilistic import find_percentile_coordinate
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
    """Load in arguments for applying coefficients for Ensemble Model Output
       Statistics (EMOS), otherwise known as Non-homogeneous Gaussian
       Regression (NGR). The coefficients are applied to the forecast
       that is supplied, so as to calibrate the forecast. The calibrated
       forecast is written to a netCDF file. If no coefficients are supplied
       the input forecast is returned unchanged.
    """
    parser = ArgParser(
        description='Apply coefficients for Ensemble Model Output '
                    'Statistics (EMOS), otherwise known as Non-homogeneous '
                    'Gaussian Regression (NGR). The supported input formats '
                    'are realizations, probabilities and percentiles. '
                    'The forecast will be converted to realizations before '
                    'applying the coefficients and then converted back to '
                    'match the input format.')
    # Filepaths for the forecast, EMOS coefficients and the output.
    parser.add_argument(
        'forecast_filepath', metavar='FORECAST_FILEPATH',
        help='A path to an input NetCDF file containing the forecast to be '
             'calibrated. The input format could be either realizations, '
             'probabilities or percentiles.')
    parser.add_argument(
        'coefficients_filepath',
        metavar='COEFFICIENTS_FILEPATH', nargs='?',
        help='(Optional) A path to an input NetCDF file containing the '
             'coefficients used for calibration. If this file is not '
             'provided the input forecast is returned unchanged.')
    parser.add_argument(
        'output_filepath', metavar='OUTPUT_FILEPATH',
        help='The output path for the processed NetCDF')
    parser.add_argument(
        'distribution', metavar="DISTRIBUTION",
        help="The distribution for constructing realizations, percentiles or "
             "probabilities. This should typically match the distribution "
             "used for minimising the Continuous Ranked Probability Score "
             "when estimating the EMOS coefficients. The distributions "
             "available are those supported by scipy.stats. Currently "
             "tested distributions are: 'norm', 'truncnorm'.")
    # Optional arguments.
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
        '--predictor', metavar='PREDICTOR',
        choices=['mean', 'realizations'], default='mean',
        help='String to specify the form of the predictor used to calculate '
             'the location parameter when estimating the EMOS coefficients. '
             'Currently the ensemble mean ("mean") and the ensemble '
             'realizations ("realizations") are supported as options. '
             'Default: "mean".')
    parser.add_argument(
        '--landsea_mask', metavar="LANDSEA_MASK", default=None,
        help="The netCDF file containing a land-sea mask on the same domain "
             "as the forecast that is to be calibrated. Land points are "
             "specified by ones and sea points are specified by zeros. "
             "Supplying this file will enable land-only calibration, in which "
             "sea points are returned without the application of calibration.")
    parser.add_argument(
        '--shape_parameters', metavar="SHAPE_PARAMETERS", nargs="*",
        default=None, type=float,
        help="The shape parameters required for defining the distribution "
             "specified by the distribution argument. The shape parameters "
             "should either be a number or 'inf' or '-inf' to represent "
             "infinity. Further details about appropriate shape parameters "
             "are available in scipy.stats. For the truncated normal "
             "distribution with a lower bound of zero, as available when "
             "estimating EMOS coefficients, the appropriate shape parameters "
             "are 0 and inf.")

    args = parser.parse_args(args=argv)

    # Load Cubes
    current_forecast = load_cube(args.forecast_filepath)
    coeffs = load_cube(args.coefficients_filepath, allow_none=True)
    landsea_mask = load_cube(args.landsea_mask, allow_none=True)
    # Process Cube
    result = process(current_forecast, coeffs, landsea_mask,
                     args.distribution, args.num_realizations,
                     args.random_ordering, args.random_seed,
                     args.ecc_bounds_warning, args.predictor,
                     args.shape_parameters)
    # Save Cube
    save_netcdf(result, args.output_filepath)


def process(current_forecast, coeffs, landsea_mask, distribution,
            num_realizations=None, random_ordering=False, random_seed=None,
            ecc_bounds_warning=False, predictor='mean', shape_parameters=None):

    """Applying coefficients for Ensemble Model Output Statistics.

    Load in arguments for applying coefficients for Ensemble Model Output
    Statistics (EMOS), otherwise known as Non-homogeneous Gaussian
    Regression (NGR). The coefficients are applied to the forecast
    that is supplied, so as to calibrate the forecast. The calibrated
    forecast is written to a cube. If no coefficients are provided the input
    forecast is returned unchanged.

    Args:
        current_forecast (iris.cube.Cube):
            A Cube containing the forecast to be calibrated. The input format
            could be either realizations, probabilities or percentiles.
        coeffs (iris.cube.Cube or None):
            A cube containing the coefficients used for calibration or None.
            If none then then current_forecast is returned unchanged.
        landsea_mask (iris.cube.Cube or None):
            A cube containing the land-sea mask on the same domain as the
            forecast that is to be calibrated. Land points are "
            "specified by ones and sea points are specified by zeros. "
            "If not None this argument will enable land-only calibration, in "
            "which sea points are returned without the application of "
            "calibration."
        distribution (str):
            The distribution for constructing realizations, percentiles or
            probabilities. This should typically match the distribution used
            for minimising the Continuous Ranked Probability Score when
            estimating the EMOS coefficients. The distributions available are
            those supported by :data:`scipy.stats`.
        num_realizations (numpy.int32):
            Optional argument to specify the number of ensemble realizations
            to produce. If the current forecast is input as probabilities or
            percentiles then this argument is used to create the requested
            number of realizations. In addition, this argument is used to
            construct the requested number of realizations from the mean and
            variance output after applying the EMOS coefficients.
            Default is None.
        random_ordering (bool):
            Option to reorder the post-processed forecasts randomly. If not
            set, the ordering of the raw ensemble is used. This option is
            only valid when the input format is realizations.
            Default is False.
        random_seed (int):
            Option to specify a value for the random seed for testing
            purposes, otherwise the default random seen behaviour is utilised.
            The random seed is used in the generation of the random numbers
            used for either the random_ordering option to order the input
            percentiles randomly, rather than use the ordering from the raw
            ensemble, or for splitting tied values within the raw ensemble,
            so that the values from the input percentiles can be ordered to
            match the raw ensemble.
            Default is None.
        ecc_bounds_warning (bool):
            If True, where the percentiles exceed the ECC bounds range,
            raises a warning rather than an exception. This occurs when the
            current forecasts is in the form of probabilities and is
            converted to percentiles, as part of converting the input
            probabilities into realizations.
            Default is False.
        predictor (str):
            String to specify the form of the predictor used to calculate
            the location parameter when estimating the EMOS coefficients.
            Currently the ensemble mean ("mean") and the ensemble
            realizations ("realizations") are supported as the predictors.
            Default is "mean".
        shape_parameters (float or str):
            The shape parameters required for defining the distribution
            specified by the distribution argument. The shape parameters
            should either be a number or 'inf' or '-inf' to represent
            infinity. Further details about appropriate shape parameters
            are available in scipy.stats. For the truncated normal
            distribution with a lower bound of zero, as available when
            estimating EMOS coefficients, the appropriate shape parameters
            are 0 and inf.

    Returns:
        iris.cube.Cube:
            The calibrated forecast cube.

    Raises:
        ValueError:
            If the current forecast is a coefficients cube.
        ValueError:
            If the coefficients cube does not have the right name of
            "emos_coefficients".
        ValueError:
            If the forecast type is 'percentiles' or 'probabilities' while no
            num_realizations are given.

    """
    if coeffs is None:
        msg = ("There are no coefficients provided for calibration. The "
               "uncalibrated forecast will be returned.")
        warnings.warn(msg)
        return current_forecast

    elif coeffs.name() != 'emos_coefficients':
        msg = ("The current coefficients cube does not have the "
               "name 'emos_coefficients'")
        raise ValueError(msg)

    if current_forecast.name() == 'emos_coefficients':
        msg = "The current forecast cube has the name 'emos_coefficients'"
        raise ValueError(msg)

    original_current_forecast = current_forecast.copy()
    try:
        find_percentile_coordinate(current_forecast)
        input_forecast_type = "percentiles"
    except CoordinateNotFoundError:
        input_forecast_type = "realizations"

    if current_forecast.name().startswith("probability_of"):
        input_forecast_type = "probabilities"
        # If probabilities, convert to percentiles.
        conversion_plugin = ConvertProbabilitiesToPercentiles(
            ecc_bounds_warning=ecc_bounds_warning)
    elif input_forecast_type == "percentiles":
        # If percentiles, resample percentiles so that the percentiles are
        # evenly spaced.
        conversion_plugin = ResamplePercentiles(
            ecc_bounds_warning=ecc_bounds_warning)

    # If percentiles, re-sample percentiles and then re-badge.
    # If probabilities, generate percentiles and then re-badge.
    if input_forecast_type in ["percentiles", "probabilities"]:
        if not num_realizations:
            raise ValueError(
                "The current forecast has been provided as {0}. "
                "These {0} need to be converted to realizations "
                "for ensemble calibration. The num_realizations "
                "argument is used to define the number of realizations "
                "to construct from the input {0}, so if the "
                "current forecast is provided as {0} then "
                "num_realizations must be defined.".format(
                    input_forecast_type))
        current_forecast = conversion_plugin.process(
            current_forecast, no_of_percentiles=num_realizations)
        current_forecast = (
            RebadgePercentilesAsRealizations().process(current_forecast))

    # Default number of ensemble realizations is the number in
    # the raw forecast.
    if not num_realizations:
        num_realizations = len(
            current_forecast.coord('realization').points)

    # Apply coefficients as part of Ensemble Model Output Statistics (EMOS).
    ac = ApplyCoefficientsFromEnsembleCalibration(predictor=predictor)
    calibrated_predictor, calibrated_variance = ac.process(
        current_forecast, coeffs, landsea_mask=landsea_mask)
    print("Applied coefficients")
    # If input forecast is probabilities, convert output into probabilities.
    # If input forecast is percentiles, convert output into percentiles.
    # If input forecast is realizations, convert output into realizations.
    if input_forecast_type == "probabilities":
        result = ConvertLocationAndScaleParametersToProbabilities(
            distribution=distribution,
            shape_parameters=shape_parameters).process(
            calibrated_predictor, calibrated_variance,
            original_current_forecast)
    elif input_forecast_type == "percentiles":
        perc_coord = find_percentile_coordinate(original_current_forecast)
        result = ConvertLocationAndScaleParametersToPercentiles(
            distribution=distribution,
            shape_parameters=shape_parameters).process(
            calibrated_predictor, calibrated_variance,
            original_current_forecast, percentiles=perc_coord.points)
    elif input_forecast_type == "realizations":
        # Ensemble Copula Coupling to generate realizations
        # from the location and scale parameter.
        percentiles = ConvertLocationAndScaleParametersToPercentiles(
            distribution=distribution,
            shape_parameters=shape_parameters).process(
            calibrated_predictor, calibrated_variance,
            original_current_forecast, no_of_percentiles=num_realizations)
        result = EnsembleReordering().process(
            percentiles, current_forecast,
            random_ordering=random_ordering, random_seed=random_seed)
    return result


if __name__ == "__main__":
    main()
