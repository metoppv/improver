#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to calculate continuous phase change level."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube_nolazy,
    phase_change,
    grid_point_radius=2,
    horizontal_interpolation=True,
    model_id_attr: str = None,
):
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
                hail-rain - the melting of hail to rain.
        grid_point_radius (int):
            The radius in grid points used to calculate the maximum
            height of the orography in a neighbourhood to determine points that
            should be excluded from interpolation for being too close to the
            orographic feature where high-resolution models can give highly
            localised results. Zero uses central point only (neighbourhood is disabled).
            One uses central point and one in each direction. Two goes two points etc.

        horizontal_interpolation (bool):
            If True apply horizontal interpolation to fill in holes in
            the returned phase-change-level that occur because the level
            falls below the orography. If False these areas will be masked.
        model_id_attr (str):
            Name of the attribute used to identify the source model for blending.

    Returns:
        iris.cube.Cube:
            Processed Cube of phase change altitude relative to sea level.
    """
    from improver.psychrometric_calculations.psychrometric_calculations import (
        PhaseChangeLevel,
    )

    plugin = PhaseChangeLevel(
        phase_change=phase_change,
        grid_point_radius=grid_point_radius,
        horizontal_interpolation=horizontal_interpolation,
        model_id_attr=model_id_attr,
    )
    result = plugin(cubes)
    return result
