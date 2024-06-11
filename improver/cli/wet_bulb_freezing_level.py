#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to extract wet-bulb freezing level from wet-bulb temperature on height levels"""

from improver import cli


@cli.clizefy
@cli.with_output
def process(wet_bulb_temperature: cli.inputcube):
    """Module to generate wet-bulb freezing level.

    The height level at which the wet-bulb temperature first drops below 273.15K
    (0 degrees Celsius) is extracted from the wet-bulb temperature cube starting from
    the ground and ascending through height levels.

    In grid squares where the temperature never goes below 273.15K the highest
    height level on the cube is returned. In grid squares where the temperature
    starts below 273.15K the lowest height on the cube is returned.

    Args:
        wet_bulb_temperature (iris.cube.Cube):
            Cube of wet-bulb air temperatures over multiple height levels.

    Returns:
        iris.cube.Cube:
            Cube of wet-bulb freezing level.

    """
    from improver.psychrometric_calculations.wet_bulb_temperature import (
        MetaWetBulbFreezingLevel,
    )

    return MetaWetBulbFreezingLevel()(wet_bulb_temperature)
