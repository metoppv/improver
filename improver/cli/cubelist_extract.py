#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to extract a cube from a cubelist"""

from improver import cli


@cli.clizefy
@cli.with_output
def process(cubes: cli.inputcubelist, *, name: str):
    """Extract a single cube from a cubelist whose name matches
    the provided name.

    Args:
        cubes (iris.cube.CubeList):
            A cubelist containing exactly one cube with the provided name to
            be extracted.
        name (str):
            The name of the cube to be extracted.
    Returns:
        iris.cube.Cube:
            The extracted cube whose name matches the provided name.

    """
    from improver.utilities.cube_extraction import cubelist_extract

    return cubelist_extract(cubes, name)
