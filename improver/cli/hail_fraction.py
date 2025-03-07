# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to calculate a hail fraction."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcubelist, model_id_attr: str = None):
    """
    Calculates the fraction of precipitation that is forecast to fall as hail.

    Args:
        cubes (iris.cube.CubeList or list):
            Contains cubes of the maximum vertical updraught, hail size,
            cloud condensation level temperature, convective cloud top temperature,
            altitude of hail to rain falling level and the altitude of the orography.
        model_id_attr (str):
            Name of the attribute used to identify the source model for
            blending.

    Returns:
        iris.cube.Cube:
            A single cube containing the hail fraction.

    """
    from improver.precipitation.hail_fraction import HailFraction

    return HailFraction(model_id_attr=model_id_attr)(*cubes)
