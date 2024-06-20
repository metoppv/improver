#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to generate wet bulb temperatures from air temperature, relative
   humidity, and pressure data. """

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube, convergence_condition=0.05, model_id_attr: str = None
):
    """Module to generate wet-bulb temperatures.

    Call the calculate_wet_bulb_temperature function to calculate wet-bulb
    temperatures. This process function splits input cubes over vertical levels
    to mitigate memory issues when trying to operate on multi-level data.

    Args:
        cubes (iris.cube.CubeList or list or iris.cube.Cube):
            containing:
                temperature (iris.cube.Cube):
                    Cube of air temperatures, where these may be on multiple
                    height levels.
                relative_humidity (iris.cube.Cube):
                    Cube of relative humidities, where these may be on multiple
                    height levels.
                pressure (iris.cube.Cube):
                    Cube of air pressure, where these may be on multiple height
                    levels.
        convergence_condition (float):
            The precision in Kelvin to which the Newton iterator must converge
            before returning wet-bulb temperatures.
        model_id_attr (str):
            Name of the attribute used to identify the source model for blending.

    Returns:
        iris.cube.Cube:
            Cube of wet-bulb temperature (K).

    """
    from improver.psychrometric_calculations.wet_bulb_temperature import (
        WetBulbTemperature,
    )

    return WetBulbTemperature(
        precision=convergence_condition, model_id_attr=model_id_attr
    )(*cubes)
