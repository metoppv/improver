#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

"""Script to run weighted blending to collapse realization and forecast_reference_time
coords using equal weights."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube, cycletime: str = None,
):
    """Runs equal-weighted blending for a specific scenario.

    Calculates an equal-weighted blend of input cube data across the realization and
    forecast_reference_time coordinates.

    Args:
        cubes (iris.cube.CubeList):
            Cubelist of cubes to be blended.
        cycletime (str):
            The forecast reference time to be used after blending has been
            applied, in the format YYYYMMDDTHHMMZ. If not provided, the
            blended file takes the latest available forecast reference time
            from the input datasets.

    Returns:
        iris.cube.Cube:
            Merged and blended Cube.
    """
    from iris.cube import CubeList

    from improver.blending.calculate_weights_and_blend import WeightAndBlend
    from improver.utilities.cube_manipulation import collapse_realizations

    cubelist = CubeList()
    for cube in cubes:
        cubelist.append(collapse_realizations(cube))

    plugin = WeightAndBlend("forecast_reference_time", "linear", y0val=0.5, ynval=0.5,)
    cube = plugin(cubelist, cycletime=cycletime,)
    return cube
