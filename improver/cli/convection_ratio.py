#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to calculate the ratio of convective precipitation to total precipitation."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube, model_id_attr: str = None):
    """ Calculate the convection ratio from convective and dynamic (stratiform)
    precipitation rate components.

    Calculates the convective ratio as:

        ratio = convective_rate / (convective_rate + dynamic_rate)

    Args:
        cubes (iris.cube.CubeList):
            Cubes of "lwe_convective_precipitation_rate" and "lwe_stratiform_precipitation_rate"
            in units that can be converted to "m s-1"
        model_id_attr (str):
            Name of the attribute used to identify the source model for
            blending.

    Returns:
        iris.cube.Cube:
            A cube of convection_ratio of the same dimensions as the input cubes.

    """
    from improver.precipitation_type.convection import ConvectionRatioFromComponents

    if len(cubes) != 2:
        raise IOError(f"Expected 2 input cubes, received {len(cubes)}")
    return ConvectionRatioFromComponents()(cubes, model_id_attr=model_id_attr)
