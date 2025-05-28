#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to run the threshold interpolation plugin."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    forecast_at_thresholds: cli.inputcube,
    *,
    threshold_values: cli.comma_separated_list = None,
    threshold_config: cli.inputjson = None,
    threshold_units: str = None,
):
    """
    Use this CLI to modify the probability thresholds in an existing probability
    forecast cube by linearly interpolating between the existing thresholds.

    Args:
        forecast_at_thresholds:
            Cube expected to contain a threshold coordinate.
        threshold_values:
            The desired output thresholds, either as a list of float values or a 
            single float value.
        threshold_config:
            Threshold configuration containing threshold values. It should contain
            either a list of float values or a dictionary of strings that can be
            interpreted as floats with the structure: "THRESHOLD_VALUE": "None".
            The latter follows the format expected in improver/cli/threshold.py,
            however fuzzy bounds will be ignored here.
            Repeated thresholds with different bounds are ignored; only the
            last duplicate will be used.
            Threshold_values and threshold_config are mutually exclusive
            arguments, defining both will lead to an exception.
        threshold_units:
            Units of the threshold values. If not provided the units are
            assumed to be the same as those of the input cube.

    Returns:
        Cube with forecast values at the desired set of thresholds.
        The threshold coordinate is always the zeroth dimension.
    """
    from improver.utilities.threshold_interpolation import ThresholdInterpolation

    result = ThresholdInterpolation(
        threshold_values, threshold_config, threshold_units
    )(forecast_at_thresholds)

    return result
