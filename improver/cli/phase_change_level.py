#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
"""Script to calculate continuous phase change level."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube,
            phase_change,
            horizontal_interpolation=True):
    """Height of precipitation phase change relative to sea level.

    Calculated as a continuous 2D field by finding the height above sea level
    at which the integral of wet bulb temperature matches an empirical
    threshold that is expected to correspond with the phase change.

    Args:
        cubes (iris.cube.CubeList or list of iris.cube.Cube):
            containing:
                wet_bulb_temperature (iris.cube.Cube):
                    Cube of wet bulb temperatures on height levels.
                wet_bulb_integral (iris.cube.Cube):
                    Cube of wet bulb temperature integrals calculated
                    vertically downwards to height levels.
                orography (iris.cube.Cube):
                    Cube of the orography height in m.
                land_sea_mask (iris.cube.Cube):
                    Cube containing the binary land-sea mask. Land points are
                    set to 1, sea points are set to 0.
        phase_change (str):
            The desired phase change for which the altitude should be
            returned. Options are:

                snow-sleet - the melting of snow to sleet.
                sleet-rain - the melting of sleet to rain.

        horizontal_interpolation (bool):
            If True apply horizontal interpolation to fill in holes in
            the returned phase-change-level that occur because the level
            falls below the orography. If False these areas will be masked.

    Returns:
        iris.cube.Cube:
            Processed Cube of phase change altitude relative to sea level.
    """
    from improver.psychrometric_calculations.psychrometric_calculations \
        import PhaseChangeLevel

    plugin = PhaseChangeLevel(
        phase_change=phase_change,
        horizontal_interpolation=horizontal_interpolation)
    result = plugin(cubes)
    return result
