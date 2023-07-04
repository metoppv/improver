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
"""Script to apply height adjustments to spot data."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    spot_cube: cli.inputcube,
    neighbour: cli.inputcube,
    *,
    land_constraint: bool = False,
    similar_altitude: bool = False,
):
    """Apply height adjustment to account for the difference between site altitude and
    grid square orography. The spot forecast contains information representative of the
    associated grid point. This needs to be adjusted to reflect the true site altitude.

    Args:
        spot_cube (iris.cube.Cube):
            A cube of spot forecasts. If this is a cube of probabilities
            then the units of the threshold coordinate must be convertible to
            metres as this is expected to represent a vertical coordinate.
            If this is a cube of percentiles or realizations then the
            units of the cube must be convertible to metres as the cube is
            expected to represent a vertical profile.
        neighbour (iris.cube.Cube):
            A cube containing information about spot-data neighbours and
            the spot site information.
        land_constraint (bool):
            Use to select the nearest-with-land-constraint neighbour-selection
            method from the neighbour_cube. This means that the grid points
            should be land points except for sites where none were found within
            the search radius when the neighbour cube was created. May be used
            with similar_altitude.
        similar_altitude (bool):
            Use to select the nearest-with-height-constraint
            neighbour-selection method from the neighbour_cube. These are grid
            points that were found to be the closest in altitude to the spot
            site within the search radius defined when the neighbour cube was
            created. May be used with land_constraint.

    Returns:
        iris.cube.Cube:
            A cube of spot data values with the same metadata as spot_cube but with data
            adjusted to be relative to site height rather than orography grid square
            height
    """
    from improver.spotdata.height_adjustment import SpotHeightAdjustment
    from improver.spotdata.neighbour_finding import NeighbourSelection

    neighbour_selection_method = NeighbourSelection(
        land_constraint=land_constraint, minimum_dz=similar_altitude
    ).neighbour_finding_method_name()

    result = SpotHeightAdjustment(neighbour_selection_method)(spot_cube, neighbour)
    return result
