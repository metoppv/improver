#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to calculate sleet probability."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(snow: cli.inputcube, rain: cli.inputcube):
    """Calculate sleet probability.

    Calculates the sleet probability using the
    calculate_sleet_probability plugin.

    Args:
        snow (iris.cube.Cube):
            An iris Cube of the probability of snow.
        rain (iris.cube.Cube):
            An iris Cube of the probability of rain.

    Returns:
        iris.cube.Cube:
            Returns a cube with the probability of sleet.
    """

    from improver.precipitation_type.calculate_sleet_prob import (
        calculate_sleet_probability,
    )

    result = calculate_sleet_probability(snow, rain)
    return result
