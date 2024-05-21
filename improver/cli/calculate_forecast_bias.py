#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to calculate the bias values from the specified set of reference forecasts."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube, truth_attribute: str):
    """Calculate forecast bias from the specified set of historical forecasts and truth
    values.

    The historical forecasts are expected to be representative single-valued forecasts
    (eg. control or ensemble mean forecast).

    The bias values are evaluated point-by-point and the associated bias cube
    will retain the same spatial dimensions as the input cubes. By using a
    point-by-point approach, the bias-correction enables a form of statistical
    downscaling where coherent biases exist between a coarse forecast dataset and
    finer truth dataset.

    Where multiple forecasts values are provided, the value returned is the mean value
    over the set of forecast/truth pairs.

    Args:
        cubes (list of iris.cube.Cube):
            A list of cubes containing the historical forecasts and corresponding
            truths used for calibration. The cubes must include the same diagnostic
            name in their names. The cubes will be distinguished using the user
            specified truth attribute.
        truth_attribute (str):
            An attribute and its value in the format of "attribute=value",
            which must be present on truth cubes.

    Returns:
        iris.cube.Cube:
            Cube containing forecast bias values evaluated over the specified set
            of historical forecasts.
    """
    from improver.calibration import split_forecasts_and_truth
    from improver.calibration.simple_bias_correction import CalculateForecastBias

    historical_forecast, historical_truth, _ = split_forecasts_and_truth(
        cubes, truth_attribute
    )
    plugin = CalculateForecastBias()
    return plugin(historical_forecast, historical_truth)
