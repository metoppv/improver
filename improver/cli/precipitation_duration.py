#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to calculate the fraction of a long period in which precipitation
reaches a given classification."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcubelist,
    min_accumulation_per_hour: cli.comma_separated_list,
    critical_rate: cli.comma_separated_list,
    target_period: float,
    percentiles: cli.comma_separated_list,
    accumulation_diagnostic: str = "probability_of_lwe_thickness_of_precipitation_amount_above_threshold",
    rate_diagnostic: str = "probability_of_lwe_precipitation_rate_above_threshold",
    model_id_attr: str = None,
):
    r"""Classifies periods of precipitation intensity using both the maximum
    precipitation rate in the period and the accumulation in the period. These
    classified periods are then used to determine what fraction of a
    constructed longer period would be classified as such.

    Args:
        cubes (iris.cube.CubeList):
            Cubes covering the expected period that include cubes of:
                max_precip_rate: Maximum precipitation rate in a period.
                precip_accumulation: Precipitation accumulation in a period.
                The periods of these two diagnostics must be the same.
        min_accumulation_per_hour:
            The minimum accumulation per hour in the period, or a list
            of several, used to classify the period. The accumulation is
            used in conjuction with the critical rate. Units of mm.
        critical_rate:
            A rate threshold, or list of rate thresholds, which if the
            maximum rate in the period is in excess of contributes to
            classifying the period. Units of mm/hr.
        target_period:
            The period in hours that the final diagnostic represents.
            This should be equivalent to the period covered by the inputs.
            Specifying this explicitly here is entirely for the purpose of
            checking that the returned diagnostic represents the period
            that is expected. Without this a missing input file could
            lead to a suddenly different overall period.
        percentiles (list):
            Definition of percentiles at which to calculate data.
        accumulation_diagnostic:
            The expected diagnostic name for the accumulation in period
            diagnostic. Used to extract the cubes from the inputs.
        rate_diagnostic:
            The expected diagnostic name for the maximum rate in period
            diagnostic. Used to extract the cubes from the inputs.
        model_id_attr:
            Name of the attribute used to identify the source model for
            blending.

    Returns:
        result (iris.cube.Cube):
            Returns a cube with the combined data.
    """
    from improver.precipitation.precipitation_duration import PrecipitationDuration

    return PrecipitationDuration(
        min_accumulation_per_hour,
        critical_rate,
        target_period,
        percentiles,
        accumulation_diagnostic=accumulation_diagnostic,
        rate_diagnostic=rate_diagnostic,
        model_id_attr=model_id_attr,
    )(*cubes)
