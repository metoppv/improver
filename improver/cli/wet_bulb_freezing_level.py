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
"""CLI to extract wet-bulb freezing level from wet-bulb temperature on height levels"""

from improver import cli


@cli.clizefy
@cli.with_output
def process(wet_bulb_temperature: cli.inputcube):
    """Module to generate wet-bulb freezing level.

    The height level at which the wet-bulb temperature first drops below 273.15K
    (0 degrees Celsius) is extracted from the wet-bulb temperature cube.

    In grid squares where the temperature never goes below 273.15K the highest
    height level on the cube is returned. In grid square where the temperature
    starts below 273.15K the lowest height on the cube is returned.

    Args:
        wet_bulb_temperature (iris.cube.Cube):
            Cube of wet-bulb air temperatures over multiple height levels.

    Returns:
        iris.cube.Cube:
            Cube of wet-bulb freezing level.

    """
    from improver.utilities.cube_extraction import ExtractLevel

    wet_bulb_freezing_level = ExtractLevel(
        positive_correlation=False, value_of_level=273.15
    )(wet_bulb_temperature)
    wet_bulb_freezing_level.rename("wet_bulb_freezing_level")

    return wet_bulb_freezing_level
