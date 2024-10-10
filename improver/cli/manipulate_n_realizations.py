#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Extend/reduce realization dimension in cube."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube, n_realizations: int,
):
    """Extend or reduce the number of realizations in a cube.

    If more realizations are requested than are in the input cube, then the ensemble
    realizations are recycled. If fewer realizations are requested than are in the input
    cube, then only the first n ensemble realizations are used.

    Args:
        cube (iris.cube.Cube):
            A cube to be processed.
        n_realizations (int):
            The number of realizations in the output cube

    Returns:
        iris.cube.Cube:
            The processed cube.
    """
    from improver.utilities.cube_manipulation import manipulate_realization_dimension

    output = manipulate_realization_dimension(cube, n_realizations)

    return output
