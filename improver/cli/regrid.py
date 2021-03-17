#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""Script to regrid a NetCDF file"""

from improver import cli


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
            set to zero.
        regrid_mode (str):
            Selects which regridding techniques to use. Default uses
            iris.analysis.Linear(); "nearest" uses iris.analysis.Nearest();
            "nearest-with-mask" uses Nearest() with land-sea awareness.
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
    from improver.standardise import RegridLandSea

    if land_sea_mask and "nearest-with-mask" not in regrid_mode:
        msg = (
            "Land-mask file supplied without appropriate regrid-mode. "
            "Use --regrid-mode nearest-with-mask."
        )
        raise ValueError(msg)

    return RegridLandSea(
        regrid_mode=regrid_mode,
        extrapolation_mode=extrapolation_mode,
        landmask=land_sea_mask,
        landmask_vicinity=land_sea_mask_vicinity,
    )(cube, target_grid, regridded_title=regridded_title)
