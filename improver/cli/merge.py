#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to merge the input files into a single file."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube):
    """Merge multiple files together

    Args:
        cubes (list of iris.cube.Cube):
            Cubes to be merged.

    Returns:
        iris.cube.Cube:
            A merged cube.
    """
    from improver.utilities.cube_manipulation import MergeCubes
    return MergeCubes()(*cubes)
