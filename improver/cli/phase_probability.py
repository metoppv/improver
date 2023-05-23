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
"""Interface to precip_phase_probability."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube, radius: float = None):
    """
    Converts a phase-change-level cube into the
    probability of a specific precipitation phase being found at the surface.

    Args:
        cubes (iris.cube.CubeList or list):
            Contains cubes of the altitude of the phase-change level (this
            can be snow->sleet, hail->rain or sleet->rain) and the altitude of the
            orography. The name of the phase-change level cube must be
            either "altitude_of_snow_falling_level", "altitude_of_rain_from_hail_falling_level or
            "altitude_of_rain_falling_level". The name of the orography
            cube must be "surface_altitude".
        radius (float or None):
            Neighbourhood radius from which the 20th or 80th percentile is found (m).
            Restrictions in the neighbourhooding code limit the use of a
            neighbourhood to data on equal area projections.
            If set to None percentiles are not generated, and the input
            altitudes are used directly as the phase discriminators.

    """
    from iris.cube import CubeList

    from improver.psychrometric_calculations.precip_phase_probability import (
        PrecipPhaseProbability,
    )

    return PrecipPhaseProbability(radius=radius)(CubeList(cubes))
