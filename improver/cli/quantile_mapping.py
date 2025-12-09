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
    reference_cube: cli.inputcube,
    forecast_cube: cli.inputcube,
    *,
    mapping_method: str = "floor",
    preservation_threshold: float = None,
    forecast_to_calibrate: cli.inputcube = None,
):
    """Adjust forecast values to match the statistical distribution of reference
    data.

    Unlike grid-point methods that match values at each location, this approach uses
    all data across the spatial domain to build the statistical distributions. This is
    particularly useful when forecasts have been smoothed and you want to restore
    realistic variation in the values while preserving the spatial patterns.

    Args:
        reference_cube:
            The reference data that define what the "correct" distribution
            should look like.
        forecast_cube:
            The forecast data you want to correct (e.g. smoothed model output).
        forecast_to_calibrate:
            Optional different forecast values to transform using the learned
            mapping. If not provided, the forecast_cube data itself will be
            corrected.
        mapping_method:
            Method for inverse CDF calculation. Either "floor" (discrete steps,
            faster) or "interp" (linear interpolation, slower but continuous).
        preservation_threshold:
            Optional threshold value below which (exclusive) the forecast values
            are not adjusted. Useful for variables like precipitation where you
            may want to preserve small/zero values.

    Returns:
        Calibrated forecast cube with quantiles mapped to the reference
        distribution or forecast_to_calibrate data adjusted with the same learned
        mapping.

    Raises:
        ValueError: If reference and forecast cubes have incompatible units.
    """
    from improver.calibration.quantile_mapping import QuantileMapping

    plugin = QuantileMapping(preservation_threshold=preservation_threshold)
    return plugin.process(
        reference_cube,
        forecast_cube,
        forecast_to_calibrate=forecast_to_calibrate,
        mapping_method=mapping_method,
    )
