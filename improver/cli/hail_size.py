#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
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
    from improver.psychrometric_calculations.hail_size import HailSize

    return HailSize(model_id_attr=model_id_attr)(*cubes)
