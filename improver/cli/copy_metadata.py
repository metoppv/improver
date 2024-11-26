#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to copy metadata from template_cube to cube"""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube,
    template_cube: cli.inputcube,
    *,
    attributes: cli.comma_separated_list = [],
    aux_coord: cli.comma_separated_list = [],
):
    """
    Copy attribute values from template_cube to cube, overwriting any existing values.

    Args:
        cube (iris.cube.Cube):
            Source cube to be updated.
        template_cube (iris.cube.Cube):
            Source cube to get attribute values from.
        attributes (list):
            List of names of attributes to copy. If any are not present on template_cube, a
            KeyError will be raised.
        aux_coord (list):
            List of names of auxilary coordinates to copy. If any are not present on
            template_cube, a KeyError will be raised. If the aux_coord is already present
            in the cube, it will be overwritten.

    Returns:
        iris.cube.Cube
    """
    from improver.utilities.copy_metadata import CopyMetadata

    plugin = CopyMetadata(attributes, aux_coord)
    return plugin(cube, template_cube)
