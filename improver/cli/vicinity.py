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
<<<<<<< HEAD
    land_mask: cli.inputcube = None,
    *,
    vicinity: cli.comma_separated_list = None,
=======
    vicinity: cli.comma_separated_list = None,
    *,
>>>>>>> Fixes for pre-commit checks.
    operator: str = "max",
    new_name: str = None,
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
    Returns:
        iris.cube.Cube:
            Cube with the vicinity processed data.
    """
    from improver.utilities.spatial import OccurrenceWithinVicinity

<<<<<<< HEAD
    vicinity_cube = OccurrenceWithinVicinity(radii=vicinity, land_mask_cube=land_mask, operator=operator).process(cube)
=======
    vicinity_cube = OccurrenceWithinVicinity(radii=vicinity, operator=operator).process(
        cube
    )
>>>>>>> Fixes for pre-commit checks.

    if new_name is not None:
        vicinity_cube.rename(new_name)

    return vicinity_cube
