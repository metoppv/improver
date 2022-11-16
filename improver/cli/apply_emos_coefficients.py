#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
def process(
    *cubes: cli.inputcubelist,
    validity_times: cli.comma_separated_list = None,
    realizations_count: int = None,
    randomise=False,
    random_seed: int = None,
    ignore_ecc_bounds=False,
    tolerate_time_mismatch=False,
    predictor="mean",
    land_sea_mask_name: str = None,
    percentiles: cli.comma_separated_list = None,
):
    """Applying coefficients for Ensemble Model Output Statistics.

    Load in arguments for applying coefficients for Ensemble Model Output
    Statistics (EMOS), otherwise known as Non-homogeneous Gaussian
    Regression (NGR). The coefficients are applied to the forecast
    that is supplied, so as to calibrate the forecast. The calibrated
    forecast is written to a cube. If no coefficients are provided the input
    forecast is returned unchanged.

    Args:
        input_cubes (iris.cube.CubeList):
            A list of cubes containing:
            - A Cube containing the forecast to be calibrated. The input format
            could be either realizations, probabilities or percentiles.
            - A cubelist containing the coefficients used for calibration or None.
            If none then the input, or probability template if provided,
            is returned unchanged.
            - Optionally, cubes representing static additional predictors.
            These static additional predictors are expected not to have a
            time coordinate.
            - Optionally, a cube containing the land-sea mask on the same domain
            as the forecast that is to be calibrated. Land points are
            specified by ones and sea points are specified by zeros.
            The presence of a land-sea mask will enable land-only calibration, in
            which sea points are returned without the application of
            calibration. If a land-sea mask is provided, the land_sea_mask_name
            must also be provided, in order to identify the land-sea mask.
            - Optionally, a cube containing a probability forecast that will be
            used as a template when generating probability output when the input
            format of the forecast cube is not probabilities i.e. realizations
            or percentiles. If no coefficients are provided and a probability
            template is provided, the probability template forecast will be
            returned as the uncalibrated probability forecast.
        validity_times (List[str]):
            Times at which the forecast must be valid. This must be provided
            as a four digit string (HHMM) where the first two digits represent the hour
            and the last two digits represent the minutes e.g. 0300 or 0315. If the
            forecast provided is at a different validity time then no coefficients
            will be applied.
        realizations_count (int):
            Option to specify the number of ensemble realizations that will be
            created from probabilities or percentiles when applying the EMOS
            coefficients.
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
        tolerate_time_mismatch (bool):
            If True, tolerate a mismatch in validity time and forecast period
            for coefficients vs forecasts. Use with caution!
        predictor (str):
            String to specify the form of the predictor used to calculate
            the location parameter when estimating the EMOS coefficients.
            Currently the ensemble mean ("mean") and the ensemble
            realizations ("realizations") are supported as the predictors.
        land_sea_mask_name (str):
            Name of the land-sea mask cube. This must be provided if a
            land-sea mask is provided within the list of input cubes, in
            order to identify the land-sea mask. Providing the land-sea mask
            ensures that only land points will be calibrated.
        percentiles (List[float]):
            The set of percentiles used to create the calibrated forecast.

    Returns:
        iris.cube.Cube:
            The calibrated forecast cube.

    """
    import warnings

    import numpy as np

    from improver.calibration import (
        add_warning_comment,
        split_forecasts_and_coeffs,
        validity_time_check,
    )
    from improver.calibration.ensemble_calibration import ApplyEMOS
    from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
        ResamplePercentiles,
    )

    (
        forecast,
        coefficients,
        additional_predictors,
        land_sea_mask,
        prob_template,
    ) = split_forecasts_and_coeffs(cubes, land_sea_mask_name)

    if validity_times is not None and not validity_time_check(forecast, validity_times):
        if percentiles:
            # Ensure that a consistent set of percentiles are returned,
            # regardless of whether EMOS is successfully applied.
            percentiles = [np.float32(p) for p in percentiles]
            forecast = ResamplePercentiles(ecc_bounds_warning=ignore_ecc_bounds)(
                forecast, percentiles=percentiles
            )
        elif prob_template:
            forecast = prob_template
        forecast = add_warning_comment(forecast)
        return forecast

    if coefficients is None:
        if prob_template:
            msg = (
                "There are no coefficients provided for calibration. As a "
                "probability template has been provided with the aim of "
                "creating a calibrated probability forecast, the probability "
                "template will be returned as the uncalibrated probability "
                "forecast."
            )
            warnings.warn(msg)
            prob_template = add_warning_comment(prob_template)
            return prob_template

        if percentiles:
            # Ensure that a consistent set of percentiles are returned,
            # regardless of whether EMOS is successfully applied.
            percentiles = [np.float32(p) for p in percentiles]
            forecast = ResamplePercentiles(ecc_bounds_warning=ignore_ecc_bounds)(
                forecast, percentiles=percentiles
            )

        msg = (
            "There are no coefficients provided for calibration. The "
            "uncalibrated forecast will be returned."
        )
        warnings.warn(msg)

        forecast = add_warning_comment(forecast)
        return forecast

    calibration_plugin = ApplyEMOS(percentiles=percentiles)
    result = calibration_plugin(
        forecast,
        coefficients,
        additional_fields=additional_predictors,
        land_sea_mask=land_sea_mask,
        prob_template=prob_template,
        realizations_count=realizations_count,
        ignore_ecc_bounds=ignore_ecc_bounds,
        tolerate_time_mismatch=tolerate_time_mismatch,
        predictor=predictor,
        randomise=randomise,
        random_seed=random_seed,
    )
    return result
