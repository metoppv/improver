#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to clip the input cube's data to be between the specified values"""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube, *, min_value: float = None, max_value: float = None,
):
    """Clip the data in the input cube such that any data above max_value is set equal to
     max_value and any data below min_value is set equal to min_value.

    Args:
        cube (iris.cube.Cube):
            A Cube whose data will be clipped. This can be a cube of spot or gridded data.
        max_value (float):
            If specified any data in cube that is above max_value will be set equal to
            max_value.
        min_value (float):
            If specified any data in cube that is below min_value will be set equal to
            min_value.
    Returns:
        iris.cube.Cube:
            A cube with the same metadata as the input cube but with the data clipped such
            that any data above max_value is set equal to max_value and any data below
            min_value is set equal to min_value.
    """
    from numpy import clip

    cube.data = clip(cube.data, min_value, max_value)

    return cube
