#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
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
    import warnings

    from improver.utilities.time_lagging import GenerateTimeLaggedEnsemble

    if len(cubes) == 1:
        warnings.warn("Only a single cube input, so time lagging will have no effect.")
        return cubes[0]

    # raise error if validity times are not all equal
    time_coords = [cube.coord("time") for cube in cubes]
    time_coords_match = [coord == time_coords[0] for coord in time_coords]
    if not all(time_coords_match):
        raise ValueError("Cubes with mismatched validity times are not compatible.")

    return GenerateTimeLaggedEnsemble()(cubes)
