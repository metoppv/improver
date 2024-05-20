#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to apply vicinity neighbourhoods to data."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube, vicinity: cli.comma_separated_list = None):
    """Module to apply vicinity processing to data.

    Calculate the maximum value within a vicinity radius about each point
    in each x-y slice of the input cube.

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
        vicinity (list of float / int):
            List of distances in metres used to define the vicinities within
            which to search for an occurrence. Each vicinity provided will
            lead to a different gridded field.

    Returns:
        iris.cube.Cube:
            Cube with the vicinity processed data.
    """
    from improver.utilities.spatial import OccurrenceWithinVicinity

    vicinity = [float(x) for x in vicinity]
    return OccurrenceWithinVicinity(radii=vicinity).process(cube)
