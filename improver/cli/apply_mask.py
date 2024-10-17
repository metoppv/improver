#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to apply provided mask to cube data."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube, mask_name: str, invert_mask: bool = "False"):
    """
    Applies provided mask to cube data. The mask_name is used to extract the mask cube
    from the input cubelist. The other cube in the cubelist is then masked using the
    mask data. If invert_mask is True, the mask will be inverted before it is applied.

    Args:
        cubes (iris.cube.CubeList):
            A list of iris cubes that should contain exactly two cubes: a mask to be applied
            and a cube to apply the mask to. The cubes should have the same dimensions.
        mask_name (str):
            The name of the cube containing the mask data. This should match with exactly one
            of the cubes in the input cubelist.
        invert_mask (bool):
            Use to select whether the mask should be inverted before being applied to the data.

    Returns:
        A cube with the mask applied to the data. The metadata will exactly match the input cube.
    """
    from improver.utilities.mask import apply_mask

    return apply_mask(*cubes, mask_name=mask_name, invert_mask=invert_mask)
