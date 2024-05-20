#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to calculate a probability of precipitation being showery if present."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube,
    cloud_threshold: float,
    convection_threshold: float,
    model_id_attr: str = None,
):
    """
    Create a shower condition diagnostic that provides the probability that
    precipitation, if present, should be classified as showery. This shower
    condition is created from cloud area fraction and convective ratio fields.

    Args:
        cubes (iris.cube.CubeList):
            Cubes of cloud area fraction and convective ratio that are used
            to calculate a proxy probability that conditions are suitable for
            showery precipitation.
        cloud_threshold (float):
            The cloud area fraction value at which to threshold the input cube.
            This sets the amount of cloud coverage below which conditions are
            likely to be considered showery (i.e. precipitation between sunny
            spells).
        convection_threshold (float):
            The convective ratio value above which, despite the cloud conditions
            not suggesting showers, the precipitation is so clearly derived
            from convection that it should be classified as showery.
        model_id_attr (str):
            Name of the attribute used to identify the source model for
            blending (optional).

    Returns:
        iris.cube.Cube:
            Probability of any precipitation, if present, being classified as
            showery.
    """
    from iris.cube import CubeList

    from improver.precipitation_type.shower_condition_probability import (
        ShowerConditionProbability,
    )

    return ShowerConditionProbability(
        cloud_threshold=cloud_threshold,
        convection_threshold=convection_threshold,
        model_id_attr=model_id_attr,
    )(CubeList(cubes))
