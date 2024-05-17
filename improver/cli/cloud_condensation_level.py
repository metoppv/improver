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
    from improver.psychrometric_calculations.cloud_condensation_level import (
        MetaPluginCloudCondensationLevel,
    )

    return MetaPluginCloudCondensationLevel(model_id_attr=model_id_attr)(*cubes)