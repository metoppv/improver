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
def process(*cubes: cli.inputcube):
    """
    Converts a phase-change-level cube into the probability of a specific
    precipitation phase being found at the surface or at a site's altitude.

    If provided with a phase-change level cube without a percentile coordinate,
    the phase-change level is compared directly to the orographic / site height
    for each grid cell / site. A binary probability of the phase is returned for
    each grid cell / site.

    If the phase-change level cube has a percentile coordinate specific
    percentiles will be used for the snow, rain or hail falling-levels. If a
    snow-falling-level diagnostic is provided, the 80th percentile altitude will
    be used. If a rain-falling-level or hail-falling-level diagnostic is provided,
    the 20th percentile altitude will be used.

    Args:
        cubes (iris.cube.CubeList or list):
            Contains cubes of the altitude of the phase-change level (this
            can be snow->sleet, hail->rain or sleet->rain), and the altitude of the
            orography at grid cells, or the altitude of sites at which the phase
            probability should be returned.

            The name of the phase-change level cube must be either:

             - "altitude_of_snow_falling_level"
             - "altitude_of_rain_from_hail_falling_level"
             - "altitude_of_rain_falling_level"

            If the phase-change level cube contains percentiles, these must include
            the 80th percentile for the snow-falling-level, and the 20th percentile
            for any other phase.

            The name of the orography cube must be "surface_altitude".
            The name of the site ancillary most be "grid_neighbours".
    """
    from iris.cube import CubeList

    from improver.psychrometric_calculations.precip_phase_probability import (
        PrecipPhaseProbability,
    )

    return PrecipPhaseProbability()(CubeList(cubes))
