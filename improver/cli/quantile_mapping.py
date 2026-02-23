#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to apply quantile mapping"""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube,
    reference_attribute: str,
    preservation_threshold: float = None,
    method: str = "step",
):
    """Adjust forecast values to match the statistical distribution of reference
    data.

    Unlike grid-point methods that match values at each location, this approach uses
    all data across the spatial domain to build the statistical distributions. This is
    particularly useful when forecasts have been smoothed and you want to restore
    realistic variation in the values while preserving the spatial patterns.

    Args:
        cubes:
            A list of cubes containing the forecasts and reference data to be
            used for calibration. They must have the same cube name and will be
            separated based on the reference attribute.
        reference_attribute:
            An attribute and its value in the format of "attribute=value",
            which must be present on cubes to identify them as reference data.
            The remaining cubes will be treated as forecast data.
        reference_cube:
            The reference data that define what the "correct" distribution
            should look like.
        forecast_cube:
            The forecast data you want to correct (e.g. smoothed model output).
        preservation_threshold:
            Optional threshold value below which (exclusive) the forecast values
            are not adjusted. Useful for variables like precipitation where you
            may want to preserve small/zero values.
        method:
            Choose from two methods of converting forecast values into quantiles
            before mapping them onto the reference distribution: 'step' and
            'continuous'. These methods differ in three ways:
            1. How quantiles are assigned to ranked data ('plotting positions').
            - 'step' uses rank/number of points (i/n), which corresponds to the
            formal ECDF definition and treats the largest value as the 1.0
            quantile (100th percentile).
            - 'continuous' uses midpoint plotting positions ((i-0.5)/n), which
            place values in the centre of their rank intervals and avoids
            probabilities of exactly 0 or 1.
            2. How probabilities are mapped back to values.
            - 'step' uses flooring, so each probability maps to the nearest
            lower observed value in the reference distribution, creating the
            step-function mapping.
            - 'continuous' uses interpolation, creating a smoother mapping where
            small changes in probability lead to small changes in value.
            3. How repeated values are treated.
            - 'step' assigns the same quantile to repeated values, so they all
            map to the same value in the reference distribution (creating flat
            steps in the mapping).
            - 'continuous' assigns different quantiles to repeated values,
            spreading them evenly across their range, so they can map to
            different values in the reference distribution.

    Returns:
        Calibrated forecast cube with quantiles mapped to the reference
        distribution.

    Raises:
        ValueError:
            If reference and forecast cubes have incompatible units.
    """
    from improver.calibration import split_forecasts_and_truth
    from improver.calibration.quantile_mapping import QuantileMapping

    forecast_cube, reference_cube, _ = split_forecasts_and_truth(
        cubes, reference_attribute
    )
    plugin = QuantileMapping(
        preservation_threshold=preservation_threshold, method=method
    )
    return plugin.process(
        reference_cube,
        forecast_cube,
    )
