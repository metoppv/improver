#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to extend a radar mask based on coverage data."""
from improver import cli


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube, coverage: cli.inputcube):
    """ Extend radar mask based on coverage data.

    Extends the mask on radar data based on the radar coverage composite.
    Update the mask on the input cube to reflect where coverage is valid.

    Args:
        cube (iris.cube.Cube):
            Cube containing the radar data to remask.
        coverage (iris.cube.Cube):
            Cube containing the radar coverage data.

    Returns:
        iris.cube.Cube:
            A cube with the remasked radar data.
    """
    from improver.nowcasting.utilities import ExtendRadarMask

    # extend mask
    result = ExtendRadarMask()(cube, coverage)
    return result
