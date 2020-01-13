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
            predictor_of_mean='mean',
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
            Optional argument to specify the number of ensemble realizations
            to produce. If the current forecast is input as probabilities or
            percentiles then this argument is used to create the requested
            number of realizations. In addition, this argument is used to
            construct the requested number of realizations from the mean and
            variance output after applying the EMOS coefficients.
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
        predictor_of_mean (str):
            String to specify the predictor used to calibrate the forecast
            mean. Currently the ensemble mean "mean" as the ensemble
            realization "realization" are supported as options.
        shape_parameters ():
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
            realizations_count are given.

    """
    import warnings

    import numpy as np
    from iris.exceptions import CoordinateNotFoundError

    from improver.ensemble_calibration.ensemble_calibration import (
        ApplyCoefficientsFromEnsembleCalibration)
    from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
        EnsembleReordering,
        GeneratePercentilesFromMeanAndVariance,
        GeneratePercentilesFromProbabilities,
        GenerateProbabilitiesFromMeanAndVariance,
        RebadgePercentilesAsRealizations,
        ResamplePercentiles)
    from improver.metadata.probabilistic import find_percentile_coordinate

    current_forecast = cube

    if coefficients is None:
        msg = ("There are no coefficients provided for calibration. The "
               "uncalibrated forecast will be returned.")
        warnings.warn(msg)
        return current_forecast

    elif coefficients.name() != 'emos_coefficients':
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
        conversion_plugin = GeneratePercentilesFromProbabilities(
            ecc_bounds_warning=ignore_ecc_bounds)
    elif input_forecast_type == "percentiles":
        # If percentiles, resample percentiles so that the percentiles are
        # evenly spaced.
        conversion_plugin = ResamplePercentiles(
            ecc_bounds_warning=ignore_ecc_bounds)

    # If percentiles, re-sample percentiles and then re-badge.
    # If probabilities, generate percentiles and then re-badge.
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

    # Default number of ensemble realizations is the number in
    # the raw forecast.
    if not realizations_count:
        realizations_count = len(
            current_forecast.coord('realization').points)

    # Apply coefficients as part of Ensemble Model Output Statistics (EMOS).
    ac = ApplyCoefficientsFromEnsembleCalibration(
        predictor_of_mean_flag=predictor_of_mean)
    calibrated_predictor, calibrated_variance = ac.process(
        current_forecast, coefficients, landsea_mask=land_sea_mask)

    if shape_parameters:
        shape_parameters = [np.float32(x) for x in shape_parameters]

    # If input forecast is probabilities, convert output into probabilities.
    # If input forecast is percentiles, convert output into percentiles.
    # If input forecast is realizations, convert output into realizations.
    if input_forecast_type == "probabilities":
        result = GenerateProbabilitiesFromMeanAndVariance(
            distribution=distribution,
            shape_parameters=shape_parameters).process(
            calibrated_predictor, calibrated_variance,
            original_current_forecast)
    elif input_forecast_type == "percentiles":
        perc_coord = find_percentile_coordinate(original_current_forecast)
        result = GeneratePercentilesFromMeanAndVariance(
            distribution=distribution,
            shape_parameters=shape_parameters).process(
            calibrated_predictor, calibrated_variance,
            percentiles=perc_coord.points)
    elif input_forecast_type == "realizations":
        # Ensemble Copula Coupling to generate realizations
        # from mean and variance.
        percentiles = GeneratePercentilesFromMeanAndVariance(
            distribution=distribution,
            shape_parameters=shape_parameters).process(
            calibrated_predictor, calibrated_variance,
            no_of_percentiles=realizations_count)
        result = EnsembleReordering().process(
            percentiles, current_forecast,
            random_ordering=randomise, random_seed=random_seed)
    return result
