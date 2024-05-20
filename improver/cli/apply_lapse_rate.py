#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to apply lapse rates to temperature data."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    temperature: cli.inputcube,
    lapse_rate: cli.inputcube,
    source_orography: cli.inputcube,
    target_orography: cli.inputcube,
):
    """Apply downscaling temperature adjustment using calculated lapse rate.

    Args:
        temperature (iris.cube.Cube):
            Input temperature cube.
        lapse_rate (iris.cube.Cube):
            Lapse rate cube.
        source_orography (iris.cube.Cube):
            Source model orography.
        target_orography (iris.cube.Cube):
            Target orography to which temperature will be downscaled.

    Returns:
        iris.cube.Cube:
            Temperature cube after lapse rate adjustment has been applied.
    """
    from improver.lapse_rate import ApplyGriddedLapseRate

    # apply lapse rate to temperature data
    result = ApplyGriddedLapseRate()(
        temperature, lapse_rate, source_orography, target_orography
    )
    return result
