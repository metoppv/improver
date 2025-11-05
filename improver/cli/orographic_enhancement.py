#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to calculate orographic enhancement."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    temperature: cli.inputcube,
    humidity: cli.inputcube,
    pressure: cli.inputcube,
    wind_speed: cli.inputcube,
    wind_direction: cli.inputcube,
    orography: cli.inputcube,
    *,
    boundary_height: float = 1000.0,
    boundary_height_units="m",
):
    """Calculate orographic enhancement

    Uses the ResolveWindComponents() and OrographicEnhancement() plugins.
    Outputs data on the high resolution orography grid.

    Args:
        temperature (iris.cube.Cube):
             Cube containing temperature at top of boundary layer.
        humidity (iris.cube.Cube):
            Cube containing relative humidity at top of boundary layer.
        pressure (iris.cube.Cube):
            Cube containing pressure at top of boundary layer.
        wind_speed (iris.cube.Cube):
            Cube containing wind speed values.
        wind_direction (iris.cube.Cube):
            Cube containing wind direction values relative to true north.
        orography (iris.cube.Cube):
            Cube containing height of orography above sea level on high
            resolution (1 km) UKPP domain grid.
        boundary_height (float):
            Model height level to extract variables for calculating orographic
            enhancement, as proxy for the boundary layer.
        boundary_height_units (str):
            Units of the boundary height specified for extracting model levels.

    Returns:
        iris.cube.Cube:
            Precipitation enhancement due to orography on the high resolution
            input orography grid.
    """
    from improver.orographic_enhancement import MetaOrographicEnhancement

    return MetaOrographicEnhancement(boundary_height, boundary_height_units)(
        temperature,
        humidity,
        pressure,
        wind_speed,
        wind_direction,
        orography,
    )
