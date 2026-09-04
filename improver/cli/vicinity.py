#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to apply vicinity neighbourhoods to data."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube,
    land_mask: cli.inputcube = None,
    *,
    vicinity: cli.comma_separated_list = None,
    operator: str = "max",
    new_name: str = None,
    apply_cell_method: bool = True,
):
    """Module to apply vicinity processing to data.

    Calculate the operator value within a vicinity radius about each point
    in each x-y slice of the input cube, with default being the maximum.

    If working with thresholded data, using this CLI to calculate a
    neighbourhood maximum ensemble probability, the vicinity process must
    be applied prior to averaging across the ensemble, i.e. collapsing the
    realization coordinate. A typical chain might look like:

      threshold --> vicinity --> realization collapse

    Note that the threshold CLI can be used to perform this series of steps
    in a single call without intermediate output.

    Users should ensure they do not inadvertently apply vicinity processing
    twice, once within the threshold CLI and then again using this CLI.

    As a further note, when applying the mean operator to thresholded data,
    the vicinity CLI will produce output equivalent to the nhbood CLI
    (disregarding the boundary), but less efficiently (particularly when
    applied to masked or datasets containing NaNs). For such cases, it is
    recommended that one use the nbhood CLI instead.

    Args:
        cube (iris.cube.Cube):
            A cube containing data to which a vicinity is to be applied.
        land_mask (iris.cube.Cube):
            Binary land-sea mask data. True for land-points, False for sea.
            Restricts in-vicinity processing to only include points of a
            like mask value.
        vicinity (list of float / int):
            List of distances in metres used to define the vicinities within
            which to search for an occurrence. Each vicinity provided will
            lead to a different gridded field.
        operator (str):
            Operation to apply over vicinity. Options are one of: ["max", "mean", "min", "std"]
            with "max" being the default.
        new_name (str):
            Name to assign to the resultant cube after calculating the vicinity
            values for the specified operator. Where no value is provided, the
            cube will retain the same name as the input cube.
        apply_cell_method (bool):
            If True, a cell method is added to the output cube to describe the
            vicinity operation.
    Returns:
        iris.cube.Cube:
            Cube with the vicinity processed data.
    """
    from improver.utilities.spatial import OccurrenceWithinVicinity

    return OccurrenceWithinVicinity(
        radii=vicinity,
        land_mask_cube=land_mask,
        operator=operator,
        apply_cell_method=apply_cell_method,
    ).process(cube, new_name=new_name)
