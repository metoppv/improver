#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to copy attributes from template_cube to cube"""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube,
    template_cube: cli.inputcube,
    *,
    attributes: cli.comma_separated_list,
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

    Returns:
        iris.cube.Cube
    """
    from improver.utilities.copy_attributes import CopyAttributes

    plugin = CopyAttributes(attributes)
    return plugin(cube, template_cube)
