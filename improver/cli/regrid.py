#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to regrid a NetCDF file"""

from improver import cli
from improver.utilities.spatial import RTOL_GRID_SPACING_DEFAULT


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube,
    target_grid: cli.inputcube,
    land_sea_mask: cli.inputcube = None,
    *,
    regrid_mode="bilinear",
    extrapolation_mode="nanmask",
    land_sea_mask_vicinity: float = 25000,
    rtol_grid_spacing: float = RTOL_GRID_SPACING_DEFAULT,
    regridded_title: str = None,
):
    """Regrids source cube data onto a target grid. Optional land-sea awareness.

    Args:
        cube (iris.cube.Cube):
            Source cube to be regridded.
        target_grid (iris.cube.Cube):
            Cube defining the spatial grid onto which to regrid the source data.
            If also using land_sea_mask-aware regridding then this must be
            land_binary_mask data.
        land_sea_mask (iris.cube.Cube):
            Cube describing the land_binary_mask on the source grid if land-sea
            aware regridding is required, with land points set to one and sea points
            set to zero. This can be a larger domain than cube, so long as cube is
            a strict cut-out (not a reprojection) of the land_sea_mask domain.
        regrid_mode (str):
            Selects which regridding techniques to use. Default uses
            iris.analysis.Linear(); "nearest" uses iris.analysis.Nearest();
            "nearest-with-mask" uses Nearest() with land-sea awareness.
            "nearest-2": new/fast version without using Iris
            "nearest-with-mask-2": new super-fast version without using Iris
            "bilinear-with-mask": bilinear option with land-sea mask considered
        extrapolation_mode (str):
            Mode to use for extrapolating data into regions beyond the limits
            of the input cube domain. Refer to online documentation for
            iris.analysis.
            Modes are -
            extrapolate - extrapolated points will take their values from the
            nearest source point
            nan - extrapolated points will be set to NaN
            error - a ValueError will be raised notifying an attempt to
            extrapolate
            mask - extrapolated points will always be masked, even if
            the source data is not a MaskedArray
            nanmask - if the source data is a MaskedArray extrapolated points
            will be masked; otherwise they will be set to NaN
        land_sea_mask_vicinity (float):
            Radius of vicinity to search for a coastline, in metres.
        rtol_grid_spacing (float):
            Relative tolerance to use when calculating grid spacing. Only used
            with the following regrid modes: "nearest-2",
            "nearest-with-mask-2", "bilinear-2", "bilinear-with-mask-2".
        regridded_title (str):
            New "title" attribute to be set if the field is being regridded
            (since "title" may contain grid information). If None, a default
            value is used.

    Returns:
        iris.cube.Cube:
            Processed cube.

    Raises:
        ValueError:
            If source land_sea_mask is supplied but regrid mode is not
            "nearest-with-mask".
        ValueError:
            If regrid_mode is "nearest-with-mask" but no source land_sea_mask
            is provided (from plugin).
    """
    from improver.regrid.landsea import RegridLandSea

    if land_sea_mask:
        if regrid_mode not in (
            "nearest-with-mask",
            "nearest-with-mask-2",
            "bilinear-with-mask-2",
        ):
            msg = (
                "Land-mask file supplied without appropriate regrid-mode. "
                "Use --regrid-mode nearest-with-mask."
            )
            raise ValueError(msg)
        _ = cube.data
        _ = target_grid.data
        _ = land_sea_mask.data

    return RegridLandSea(
        regrid_mode=regrid_mode,
        extrapolation_mode=extrapolation_mode,
        landmask=land_sea_mask,
        landmask_vicinity=land_sea_mask_vicinity,
        rtol_grid_spacing=rtol_grid_spacing,
    )(cube, target_grid, regridded_title=regridded_title)
