#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Convert wind speed and direction into individual velocity components."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(wind_speed: cli.inputcube, wind_direction: cli.inputcube):
    """Converts speed and direction into individual velocity components.

    Args:
        wind_speed (iris.cube.Cube):
            A cube of wind speed.
        wind_direction (iris.cube.Cube):
            A cube of wind from direction.

    Returns:
        iris.cube.Cubelist:
            A cubelist of the speed and direction as U and V cubes.
    """
    from iris.cube import CubeList

    from improver.wind_calculations.wind_components import ResolveWindComponents

    if not (wind_speed and wind_direction):
        raise TypeError("Neither wind_speed or wind_direction can be none")

    u_cube, v_cube = ResolveWindComponents()(wind_speed, wind_direction)
    return CubeList([u_cube, v_cube])
