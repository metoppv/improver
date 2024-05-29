# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
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

    return PrecipPhaseProbability()(*cubes)
