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

from improver import cli


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube,
            coefficients: cli.inputcube = None,
            land_sea_mask: cli.inputcube = None,
            *,
            distribution,
            realizations_count: int = None,
            randomise=False,
            random_seed: int = None,
            ignore_ecc_bounds=False,
            predictor='mean',
            shape_parameters: cli.comma_separated_list = None):
    """Applying coefficients for Ensemble Model Output Statistics.

    Load in arguments for applying coefficients for Ensemble Model Output
    Statistics (EMOS), otherwise known as Non-homogeneous Gaussian
    Regression (NGR). The coefficients are applied to the forecast
    that is supplied, so as to calibrate the forecast. The calibrated
    forecast is written to a cube. If no coefficients are provided the input
    forecast is returned unchanged.

    Args:
        cube (iris.cube.Cube):
            A Cube containing the forecast to be calibrated. The input format
            could be either realizations, probabilities or percentiles.
        coefficients (iris.cube.Cube):
            A cube containing the coefficients used for calibration or None.
            If none then then input is returned unchanged.
        land_sea_mask (iris.cube.Cube):
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
        realizations_count (int):
            Option to specify the number of ensemble realizations that will be
            created from probabilities or percentiles for input into EMOS.
        randomise (bool):
            Option to reorder the post-processed forecasts randomly. If not
            set, the ordering of the raw ensemble is used. This option is
            only valid when the input format is realizations.
        random_seed (int):
            Option to specify a value for the random seed for testing
            purposes, otherwise the default random seen behaviour is utilised.
            The random seed is used in the generation of the random numbers
            used for either the randomise option to order the input
            percentiles randomly, rather than use the ordering from the raw
            ensemble, or for splitting tied values within the raw ensemble,
            so that the values from the input percentiles can be ordered to
            match the raw ensemble.
        ignore_ecc_bounds (bool):
            If True, where the percentiles exceed the ECC bounds range,
            raises a warning rather than an exception. This occurs when the
            current forecasts is in the form of probabilities and is
            converted to percentiles, as part of converting the input
            probabilities into realizations.
        predictor (str):
            String to specify the form of the predictor used to calculate
            the location parameter when estimating the EMOS coefficients.
            Currently the ensemble mean ("mean") and the ensemble
            realizations ("realizations") are supported as the predictors.
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
            If the forecast type is 'percentiles' or 'probabilities' and the
            realizations_count argument is not provided.
    """
    import warnings

    import numpy as np
    from iris.exceptions import CoordinateNotFoundError

    from improver.calibration.ensemble_calibration import (
        ApplyCoefficientsFromEnsembleCalibration)
    from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
        EnsembleReordering,
        ConvertLocationAndScaleParametersToPercentiles,
        ConvertLocationAndScaleParametersToProbabilities,
        ConvertProbabilitiesToPercentiles,
        RebadgePercentilesAsRealizations,
        ResamplePercentiles)
    from improver.calibration.utilities import merge_land_and_sea
    from improver.metadata.probabilistic import find_percentile_coordinate

    current_forecast = cube

    if current_forecast.name() in ['emos_coefficients', 'land_binary_mask']:
        msg = "The current forecast cube has the name {}"
        raise ValueError(msg.format(current_forecast.name()))

    if coefficients is None:
        msg = ("There are no coefficients provided for calibration. The "
               "uncalibrated forecast will be returned.")
        warnings.warn(msg)
        return current_forecast

    if coefficients.name() != 'emos_coefficients':
        msg = ("The current coefficients cube does not have the "
               "name 'emos_coefficients'")
        raise ValueError(msg)

    if land_sea_mask and land_sea_mask.name() != 'land_binary_mask':
        msg = ("The land_sea_mask cube does not have the "
               "name 'land_binary_mask'")
        raise ValueError(msg)

    original_current_forecast = current_forecast.copy()
    try:
        find_percentile_coordinate(current_forecast)
        input_forecast_type = "percentiles"
    except CoordinateNotFoundError:
        input_forecast_type = "realizations"

    if current_forecast.name().startswith("probability_of"):
        input_forecast_type = "probabilities"
        conversion_plugin = ConvertProbabilitiesToPercentiles(
            ecc_bounds_warning=ignore_ecc_bounds)
    elif input_forecast_type == "percentiles":
        # Initialise plugin to resample percentiles so that the percentiles are
        # evenly spaced.
        conversion_plugin = ResamplePercentiles(
            ecc_bounds_warning=ignore_ecc_bounds)

    if input_forecast_type in ["percentiles", "probabilities"]:
        if not realizations_count:
            raise ValueError(
                "The current forecast has been provided as {0}. "
                "These {0} need to be converted to realizations "
                "for ensemble calibration. The realizations_count "
                "argument is used to define the number of realizations "
                "to construct from the input {0}, so if the "
                "current forecast is provided as {0} then "
                "realizations_count must be defined.".format(
                    input_forecast_type))
        current_forecast = conversion_plugin.process(
            current_forecast, no_of_percentiles=realizations_count)
        current_forecast = (
            RebadgePercentilesAsRealizations().process(current_forecast))

    # Apply coefficients as part of Ensemble Model Output Statistics (EMOS).
    ac = ApplyCoefficientsFromEnsembleCalibration(predictor=predictor)
    location_parameter, scale_parameter = ac.process(
        current_forecast, coefficients, landsea_mask=land_sea_mask)

    if shape_parameters:
        shape_parameters = [np.float32(x) for x in shape_parameters]

    # Convert the output forecast type (i.e. realizations, percentiles,
    # probabilities) to match the input forecast type.
    if input_forecast_type == "probabilities":
        result = ConvertLocationAndScaleParametersToProbabilities(
            distribution=distribution,
            shape_parameters=shape_parameters).process(
            location_parameter, scale_parameter, original_current_forecast)
    elif input_forecast_type == "percentiles":
        perc_coord = find_percentile_coordinate(original_current_forecast)
        result = ConvertLocationAndScaleParametersToPercentiles(
            distribution=distribution,
            shape_parameters=shape_parameters).process(
            location_parameter, scale_parameter, original_current_forecast,
            percentiles=perc_coord.points)
    elif input_forecast_type == "realizations":
        # Ensemble Copula Coupling to generate realizations
        # from the location and scale parameter.
        no_of_percentiles = len(current_forecast.coord('realization').points)
        percentiles = ConvertLocationAndScaleParametersToPercentiles(
            distribution=distribution,
            shape_parameters=shape_parameters).process(
            location_parameter, scale_parameter, original_current_forecast,
            no_of_percentiles=no_of_percentiles)
        result = EnsembleReordering().process(
            percentiles, current_forecast,
            random_ordering=randomise, random_seed=random_seed)
    if land_sea_mask:
        # Fill in masked sea points with uncalibrated data.
        merge_land_and_sea(result, original_current_forecast)
    return result
