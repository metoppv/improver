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
"""CLI to estimate a scaling factor to account for a correction linked to the
difference in altitude between the grid point and the site location."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    forecast: cli.inputcube,
    truth: cli.inputcube,
    neighbour_cube: cli.inputcube,
    *,
    forecast_period: int,
    dz_lower_bound: float = None,
    dz_upper_bound: float = None,
    land_constraint: bool = False,
    similar_altitude: bool = False,
    site_id_coord: str = "wmo_id",
):
    """Estimate a scaling factor to account for a correction linked to the difference
    in altitude between the grid point and the site location. Note that the output
    will have the same sites as provided by the neighbour cube.

    Args:
        forecast (iris.cube.Cube):
            Historical percentile forecasts. A 50th percentile forecast is required.
        truth (iris.cube.Cube):
            Truths that are expected to match the validity times of the forecasts.
        neighbour_cube (iris.cube.Cube):
            The neighbour cube is a cube of spot-data neighbours and
            the spot site information.
        forecast_period (int):
            The forecast period in hours that is considered representative of the
            input forecasts. This is required as the input forecasts could contain
            multiple forecast periods.
        dz_lower_bound (float):
            The lowest acceptable value for the difference in altitude
            between the grid point and the site. Sites with a lower
            (or more negative) difference in altitude will be excluded.
            If the altitude difference is calculated as site altitude minus the
            grid point altitude, the lower bound therefore indicates how far below
            the orographic surface (i.e. within unresolved valleys) the correction
            should be applied. Defaults to None.
        dz_upper_bound (float):
            The highest acceptable value for the difference in altitude
            between the grid point and the site. Sites with a larger
            positive difference in altitude will be excluded.
            If the altitude difference is calculated as site altitude minus the
            grid point altitude, the upper bound therefore indicates how far above
            the orographic surface (i.e. on unresolved hills or mountains) the
            correction should be applied. Defaults to None.
        land_constraint (bool):
            Use to select the nearest-with-land-constraint neighbour-selection
            method from the neighbour_cube. This means that the grid points
            should be land points except for sites where none were found within
            the search radius when the neighbour cube was created. May be used
            with similar_altitude.
        similar_altitude (bool):
            Use to select the nearest-with-height-constraint neighbour-selection
            method from the neighbour_cube. These are grid points that were found
            to be the closest in altitude to the spot site within the search radius
            defined when the neighbour cube was created. May be used with
            land_constraint.
        site_id_coord (str):
            The name of the site ID coordinate. This defaults to 'wmo_id'.

    Returns:
        iris.cube.Cube:
            Cube containing a scaling factor computed using the difference
            in altitude between the grid point and the site location.
    """

    from improver.calibration.dz_rescaling import EstimateDzRescaling

    plugin = EstimateDzRescaling(
        forecast_period=forecast_period,
        dz_lower_bound=dz_lower_bound,
        dz_upper_bound=dz_upper_bound,
        land_constraint=land_constraint,
        similar_altitude=similar_altitude,
        site_id_coord=site_id_coord,
    )
    return plugin(forecast, truth, neighbour_cube)
