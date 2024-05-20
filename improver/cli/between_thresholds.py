#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to calculate probabilities of occurrence between thresholds."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube, *, threshold_ranges: cli.inputjson, threshold_units=None
):
    """
    Calculate the probabilities of occurrence between thresholds

    Args:
        cube (iris.cube.Cube):
            Cube containing input probabilities above or below threshold
        threshold_ranges (list):
            List of 2-item iterables specifying thresholds between which
            probabilities should be calculated
        threshold_units (str):
            Units in which the thresholds are specified.  If None, defaults
            to the units of the threshold coordinate on the input cube.

    Returns:
        iris.cube.Cube:
            Cube containing probability of occurrences between the thresholds
            specified
    """
    from improver.between_thresholds import OccurrenceBetweenThresholds
    from improver.metadata.probabilistic import find_threshold_coordinate

    if threshold_units is None:
        threshold_units = str(find_threshold_coordinate(cube).units)

    plugin = OccurrenceBetweenThresholds(threshold_ranges, threshold_units)
    return plugin(cube)
