#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to apply ensemble copula coupling."""

from improver import cli
from improver.metadata.probabilistic import (
    is_percentile,
    is_probability,
)
from typing import Optional


@cli.clizefy
@cli.with_output
def process(
    post_processed_forecast: cli.inputcube,
    raw_ensemble_forecast: cli.inputcube,
    tie_break: str = "random",
    random_seed: int = 0,
):
    """
    Given a post-processed forecast in either threshold or percentile format,
    produce a set of calibrated evenly-spaced percentiles having the same size as
    the original ensemble, then use the raw ensemble to order the percentiles at
    each point to produce a calibrated ensemble with realistic spatial structure.

    Args:
        post_processed_forecast (iris.cube.Cube):
            The calibrated forecast.
        raw_ensemble_forecast (iris.cube.Cube):
            The raw ensemble. Must have same dimensions as the post-processed forecast,
            except that the percentile/threshold dimension is replaced with realization.
        tie_break:
            The method of tie breaking to use when the first ordering method contains ties.
            The available methods are "random", to tie-break randomly, and "realization",
            to tie-break by assigning values to the highest numbered realizations first.
        random_seed:
            Used when the tie breaking method is "random". If random_seed is an integer,
            the integer value is used for the random seed. If random_seed is None, no
            random seed is set, so the random values generated are not reproducible.
    Returns:
        iris.cube.Cube:
            Calibrated ensemble forecast, with same dimensions as raw_ensemble_forecast.
    """
    from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
        RebadgePercentilesAsRealizations,
        ResamplePercentiles,
        ConvertProbabilitiesToPercentiles,
        EnsembleReordering,
    )

    num_realizations = len(raw_ensemble_forecast.coord("realization").points)
    if is_probability(post_processed_forecast):
        plugin = ConvertProbabilitiesToPercentiles()
        percentile_cube = plugin.process(post_processed_forecast, num_realizations)
    elif is_percentile(post_processed_forecast):
        plugin = ResamplePercentiles()
        percentile_cube = plugin.process(
            post_processed_forecast, no_of_percentiles=num_realizations
        )
    else:
        raise ValueError(
            "Post-processed forecast must be either a thresholded probability forecast or a percentile forecast."
        )

    plugin = EnsembleReordering()
    return plugin.process(
        percentile_cube,
        raw_ensemble_forecast,
        random_ordering=False,
        random_seed=random_seed,
        tie_break=tie_break,
    )
