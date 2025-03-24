# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Interface to snow_fraction."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube, model_id_attr: str = None):
    """
    Calculates a snow-fraction field from fields of snow and rain (rate or
    accumulation). Where no precipitation is present, the data are filled in from
    the nearest precipitating point.

    snow_fraction = snow / (snow + rain)

    Args:
        cubes (iris.cube.CubeList or list):
            Contains cubes of rain and snow, both must be either rates or accumulations.
        model_id_attr (str):
            Name of the attribute used to identify the source model for
            blending.

    Returns:
        iris.cube.Cube:
            A single cube containing the snow-fraction data.

    """
    from iris.cube import CubeList

    from improver.precipitation.snow_fraction import SnowFraction

    return SnowFraction(model_id_attr=model_id_attr)(CubeList(cubes))
