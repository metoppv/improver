#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to generate cloud condensation level from near-surface temperature,
pressure and humidity data."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube, model_id_attr: str = None):
    """Module to generate cloud condensation level.

    Calls the HumidityMixingRatio plugin to calculate humidity mixing ratio from relative humidity.

    Calls the CloudCondensationLevel plugin to calculate cloud condensation level.

    Args:
        cubes (iris.cube.CubeList or list of iris.cube.Cube):
            containing near-surface values, in any order, of:
                temperature (iris.cube.Cube):
                    Cube of air_temperature (K).
                pressure (iris.cube.Cube):
                    Cube of surface_air_pressure (Pa).
                humidity (iris.cube.Cube):
                    Cube of relative_humidity (1).
        model_id_attr (str):
            Name of the attribute used to identify the source model for blending.

    Returns:
        tuple:
            iris.cube.Cube:
                Cube of temperature at cloud condensation level (K)
            iris.cube.Cube:
                Cube of pressure at cloud condensation level (Pa)

    """
    from iris.cube import CubeList

    from improver.psychrometric_calculations.cloud_condensation_level import (
        CloudCondensationLevel,
    )
    from improver.psychrometric_calculations.psychrometric_calculations import (
        HumidityMixingRatio,
    )
    from improver.utilities.flatten import flatten

    cubes = flatten(cubes)
    (temperature, pressure, humidity,) = CubeList(cubes).extract(
        ["air_temperature", "surface_air_pressure", "relative_humidity"]
    )

    humidity_plugin = HumidityMixingRatio(model_id_attr=model_id_attr)
    humidity = humidity_plugin([temperature, pressure, humidity])

    return CloudCondensationLevel(model_id_attr=model_id_attr)(
        [humidity_plugin.temperature, humidity_plugin.pressure, humidity]
    )
