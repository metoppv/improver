#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
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
    from improver.utilities.cube_extraction import ExtractLevel

    wet_bulb_freezing_level = ExtractLevel(
        positive_correlation=False, value_of_level=273.15
    )(wet_bulb_temperature)
    wet_bulb_freezing_level.rename("wet_bulb_freezing_level_altitude")

    return wet_bulb_freezing_level
