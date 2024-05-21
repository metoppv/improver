#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to generate hail size."""
from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcubelist, model_id_attr: str = None):
    """Module to calculate the size of hail stones from the
    cloud condensation level (ccl) temperature and pressure, temperature
    on pressure levels data, wet bulb freezing altitude above sea level and orography.

    Args:
        cubes (iris.cube.CubeList or list of iris.cube.Cube):
            containing:
                temperature (iris.cube.Cube):
                    Cube of temperature on pressure levels
                wet_bulb_freezing_level_altitude (iris.cube.Cube)
                    Cube of the height of the wet bulb freezing level
                ccl (iris.cube.CubeList)
                    Cube list containing 2 cubes: air temperature at ccl
                    and air pressure at ccl
                orography (iris.cube.Cube):
                    Cube of the orography height.
        model_id_attr (str):
            Name of the attribute used to identify the source model for blending.

    Returns:
        iris.cube.Cube:
            Cube of diameter_of_hail (m).
    """

    from iris.cube import CubeList

    from improver.psychrometric_calculations.hail_size import HailSize
    from improver.utilities.flatten import flatten

    cubes = flatten(cubes)
    (temperature, ccl_pressure, ccl_temperature, wet_bulb_zero, orography,) = CubeList(
        cubes
    ).extract(
        [
            "air_temperature",
            "air_pressure_at_condensation_level",
            "air_temperature_at_condensation_level",
            "wet_bulb_freezing_level_altitude",
            "surface_altitude",
        ]
    )
    return HailSize(model_id_attr=model_id_attr)(
        ccl_temperature, ccl_pressure, temperature, wet_bulb_zero, orography,
    )
