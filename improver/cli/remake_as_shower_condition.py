#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to modify a suitable shower condition proxy diagnostic into a shower
condition cube."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube):
    """
    Modify the name and threshold coordinate of another diagnostic to create
    a shower condition cube. Such a cube provides the probability that any
    precipitation, should it be present, should be classified as showery. Only
    suitable proxies for identifying showery conditions should be modified in
    this way. By modifying cubes in this way it is possible to blend different
    proxies from different models as though they are equivalent diagnostics.
    The user must be satisfied that the proxies are suitable for blending.

    Args:
        cube (iris.cube.Cube):
            A cube containing the diagnostic that is a proxy for showery
            conditions, e.g. cloud texture.

    Returns:
        iris.cube.Cube:
            Probability of any precipitation, if present, being classified as
            showery.
    """
    from improver.precipitation_type.utilities import make_shower_condition_cube

    return make_shower_condition_cube(cube)
