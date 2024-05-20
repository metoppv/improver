# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to run GenerateSolarTime ancillary generation."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    target_grid: cli.inputcube, *, time: cli.inputdatetime, new_title: str = None
):
    """Generate a cube containing local solar time, evaluated on the target grid for
    specified time. Local solar time is used as an input to the RainForests calibration
    for rainfall.

    Args:
        target_grid (iris.cube.Cube):
            A cube with the desired grid.
        time (str):
            A datetime specified in the format YYYYMMDDTHHMMZ at which to calculate the
            local solar time.
        new_title:
            New title for the output cube attributes. If None, this attribute is left out
            since it has no prescribed standard.

    Returns:
        iris.cube.Cube:
            A cube containing local solar time.
    """
    from improver.generate_ancillaries.generate_derived_solar_fields import (
        GenerateSolarTime,
    )

    return GenerateSolarTime()(target_grid, time, new_title=new_title)
