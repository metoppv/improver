#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to create lightning probabilities from multi-parameter datasets."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube, model_id_attr: str = None):
    """
    From the supplied following cubes:
    Convective Available Potential Energy (CAPE in J/kg),
    Lifted Index (liftind in K),
    Precipitable Water (pwat in kg m-2 or mm. This is used as mm in the regression equations),
    Convective Inhibition (CIN in J/kg),
    3-hour Accumulated Precipitation (apcp in kg m-2 or millimetres),
    calculate a probability of lightning cube using relationships developed using regression
    statistics.

    The cubes for CAPE, lifted index, precipitable water, and CIN must be valid for the beginning
    of the 3-hr accumulated precipitation window.

    Does not collapse a realization coordinate.

    Args:
        cubes (list of iris.cube.Cube):
            Cubes to be processed.
        model_id_attr (str):
            Name of the attribute used to identify the source model for
            blending.

    Returns:
        iris.cube.Cube:
            Cube of probabilities of lightning
    """
    from iris.cube import CubeList

    from improver.lightning import LightningMultivariateProbability_USAF2024

    result = LightningMultivariateProbability_USAF2024()(
        CubeList(cubes), model_id_attr=model_id_attr
    )

    return result
