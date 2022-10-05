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
"""CLI to generate hail size."""
from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcubelist, model_id_attr: str = None):
    """Module to calculate the size of hail stones from the
    cloud condensation level (ccl) temperature and pressure, temperature
    on pressure levels data, relative humidity on pressure levels, wet
    bulb freezing altitude above sea level and orography.

    Args:
        cubes (iris.cube.CubeList or list of iris.cube.Cube):
            containing:
                temperature (iris.cube.Cube):
                    Cube of temperature on pressure levels
                relative_humidity (iris.cube.Cube)
                    Cube of relative humidity on pressure levels
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
    (
        temperature,
        relative_humidity,
        ccl_pressure,
        ccl_temperature,
        wet_bulb_zero,
        orography
    ) = CubeList(cubes).extract(
        [
            "air_temperature",
            "relative_humidity",
            "air_pressure_at_condensation_level",
            "air_temperature_at_condensation_level",
            "wet_bulb_freezing_level_altitude",
            "surface_altitude"
        ]
    )
    return HailSize(model_id_attr=model_id_attr)(
        ccl_temperature, ccl_pressure, temperature, relative_humidity, wet_bulb_zero,orography
    )
