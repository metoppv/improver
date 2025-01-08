#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to run time-lagged ensembles."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube):
    """Module to time-lag ensembles.

    Combines the realization from different forecast cycles into one cube.
    Takes an input CubeList containing forecasts from different cycles and
    merges them into a single cube.

    Args:
        cubes (list of iris.cube.Cube):
            List of individual ensemble cubes

    Returns:
        iris.cube.Cube:
            Merged cube

    Raises:
        ValueError: If ensembles have mismatched validity times
    """
    from improver.utilities.time_lagging import GenerateTimeLaggedEnsemble

    return GenerateTimeLaggedEnsemble()(cubes)
