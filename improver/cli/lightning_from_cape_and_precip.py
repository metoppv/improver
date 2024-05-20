#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to apply latitude-dependent thresholding to a parameter dataset."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube, model_id_attr: str = None,
):
    """
    Apply latitude-dependent thresholds to CAPE and precipitation rate to derive a
    probability-of-lightning cube.
    Does not collapse a realization coordinate.

    Args:
        cubes (list of iris.cube.Cube):
            A cube to be processed.
        model_id_attr (str):
            Name of the attribute used to identify the source model for
            blending.

    Returns:
        iris.cube.Cube:
            Cube of probabilities of lightning relative to a zero rate thresholds
    """
    from iris.cube import CubeList

    from improver.lightning import LightningFromCapePrecip

    result = LightningFromCapePrecip()(CubeList(cubes), model_id_attr=model_id_attr)

    return result
