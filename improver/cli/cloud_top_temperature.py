#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""CLI to generate the convective cloud top temperature from CCL and temperature profile data."""
from improver import cli

# Creates the value_converter that clize needs.
input_ccl = cli.create_constrained_inputcubelist_converter(
    "air_temperature_at_condensation_level", "air_pressure_at_condensation_level"
)


@cli.clizefy
@cli.with_output
def process(
    ccl_cubes: input_ccl, temperature: cli.inputcube, *, model_id_attr: str = None
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
