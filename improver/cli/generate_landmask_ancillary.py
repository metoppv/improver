#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to run land_sea_mask ancillary generation."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(land_sea_mask: cli.inputcube):
    """Generate a land_sea_mask ancillary.

    Reads in the interpolated land_sea_mask and rounds
    values < 0.5 to False
    values >= 0.5 to True.

    Args:
        land_sea_mask (iris.cube.Cube):
            Cube to process.

    Returns:
        iris.cube.Cube:
            A land_sea_mask of boolean values.
    """
    from improver.generate_ancillaries.generate_ancillary import CorrectLandSeaMask

    return CorrectLandSeaMask()(land_sea_mask)
