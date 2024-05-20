#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to collapse the realizations dimension of a cube."""


from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube, *, method: str = "mean", new_name: str = None,
):
    """Collapse the realization dimension of a cube.

    Args:
        cube (iris.cube.Cube):
            Cube to be collapsed.
        method (str):
            One of "sum", "mean", "median", "std_dev", "min", "max".
        new_name (str):
            New name for output cube; if None use iris default.

    Returns:
        iris.cube.Cube:
            Collapsed cube. Dimensions are the same as input cube,
            without realization dimension.

    Raises:
        ValueError: if realization is not a dimension coordinate.
    """

    from improver.utilities.cube_manipulation import collapse_realizations

    if not (cube.coords("realization", dim_coords=True)):
        raise ValueError("realization must be a dimension coordinate.")

    collapsed_cube = collapse_realizations(cube, method=method)
    if new_name:
        collapsed_cube.rename(new_name)

    return collapsed_cube
