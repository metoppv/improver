#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
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
    from improver.spotdata.utilities import neighbour_finding_method_name

    neighbour_selection_method = neighbour_finding_method_name(
        land_constraint=land_constraint, minimum_dz=similar_altitude
    )

    result = SpotHeightAdjustment(neighbour_selection_method)(spot_cube, neighbour)
    return result
