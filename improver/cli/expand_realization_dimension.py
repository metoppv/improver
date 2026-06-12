#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to expand the realization dimension of a cube."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube, *, n_realizations_required: int):
    """Expand the realization dimension of a cube.

    Args:
        cube:
            Cube to be expanded.
        n_realizations_required:
            Number of realizations required in the expanded cube.

    Returns:
        iris.cube.Cube
            Expanded cube. Dimensions are the same as input cube, with the realization
            dimension expanded to the specified size.
    """
    from improver.utilities.expand_realization_dimension import (
        ExpandRealizationDimension,
    )

    plugin = ExpandRealizationDimension(n_realizations_required=n_realizations_required)

    return plugin(cube)
