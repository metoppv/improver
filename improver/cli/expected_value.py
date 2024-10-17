#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to calculate expected value of probability distribution."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube):
    """Calculate expected value from probabilistic data.

    Args:
        cube (iris.cube.Cube):
            Cube with realization, threshold or percentile coordinate.

    Returns:
        iris.cube.Cube:

    """
    from improver.expected_value import ExpectedValue

    output = ExpectedValue()(cube)
    return output
