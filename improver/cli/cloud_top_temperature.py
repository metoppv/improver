#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to generate the convective cloud top temperature from CCL and temperature profile data."""
from improver import cli


@cli.clizefy
@cli.with_output
def process(
    ccl_cubes: cli.inputcubelist,
    temperature: cli.inputcube,
    *,
    model_id_attr: str = None,
):
    """Module to calculate the convective cloud top temperature from the
    cloud condensation level temperature and pressure, and temperature
    on pressure levels data.
    The temperature is that of the parcel after saturated ascent at the last pressure level
    where the parcel is buoyant.
    If the cloud top temperature is less than 4K colder than the cloud condensation level,
    the cloud top temperature is masked.

    Args:
        ccl_cubes (iris.cube.CubeList or list of iris.cube.Cube):
            Cubes of air_temperature and air_pressure at cloud_condensation_level
        temperature (iris.cube.Cube):
            Cube of temperature_at_pressure_levels
        model_id_attr (str):
            Name of the attribute used to identify the source model for blending.

    Returns:
        iris.cube.Cube:
            Cube of cloud_top_temperature (K).

    """

    from improver.psychrometric_calculations.cloud_top_temperature import (
        CloudTopTemperature,
    )

    return CloudTopTemperature(model_id_attr=model_id_attr)(*ccl_cubes, temperature)
