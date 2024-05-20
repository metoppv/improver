#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to generate an ancillary "grid_with_halo" file."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube, *, halo_radius: float = 162000.0):
    """Generate a zeroed grid with halo from a source cube.

    Create a template cube defining a new grid by adding a fixed width halo on
    all sides to the input cube grid. The cube contains no meaningful data.

    Args:
        cube (iris.cube.Cube):
            Contains data on the source grid.
        halo_radius (float):
            Radius in metres of which to pad the input grid.

    Returns:
        iris.cube.Cube:
            The processed cube defining the halo-padded grid (data set to 0)
    """
    from improver.utilities.pad_spatial import create_cube_with_halo

    result = create_cube_with_halo(cube, halo_radius)
    return result
