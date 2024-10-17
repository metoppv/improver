#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to calculate wet bulb temperature integral."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(wet_bulb_temperature: cli.inputcube, *, model_id_attr: str = None):
    """Module to calculate wet bulb temperature integral.

    Calculate the wet-bulb temperature integral using the input wet bulb
    temperature data. The integral will be calculated at the height levels on
    which the wet bulb temperatures are provided.

    Args:
        wet_bulb_temperature (iris.cube.Cube):
            Cube of wet bulb temperatures on height levels.
        model_id_attr (str):
            Name of the attribute used to identify the source model for blending.

    Returns:
        iris.cube.Cube:
            Processed Cube of wet bulb integrals.
    """
    from improver.psychrometric_calculations.wet_bulb_temperature import (
        WetBulbTemperatureIntegral,
    )

    return WetBulbTemperatureIntegral(model_id_attr=model_id_attr)(wet_bulb_temperature)
