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
"""Script to standardise a NetCDF file by one or more of regridding, updating
meta-data and demoting float64 data to float32"""

from improver import cli


@cli.clizefy
@cli.with_output
def process(source_data: cli.inputcube,
            target_grid: cli.inputcube = None,
            source_landmask: cli.inputcube = None,
            *,
            regrid_mode='bilinear',
            extrapolation_mode='nanmask',
            landmask_vicinity: float = 25000,
            regridded_title: str = None,
            new_metadata: cli.inputjson = None,
            coords_to_remove: cli.comma_separated_list = None,
            new_name: str = None,
            new_units: str = None,
            fix_float64=False):
    """Standardises a cube by one or more of regridding, updating meta-data etc

    Standardise a source cube. Available options are regridding (bi-linear or
    nearest-neighbour, optionally with land-mask awareness), renaming,
    converting units, updating attributes and / or converting float64 data to
    float32.

    Args:
        source_data (iris.cube.Cube):
            Source cube to be standardised
        target_grid (iris.cube.Cube):
            If specified, then regridding of the source against the target
            grid is enabled. If also using landmask-aware regridding then this
            must be land_binary_mask data.
        source_landmask (iris.cube.Cube):
            A cube describing the land_binary_mask on the source-grid if
            coastline-aware regridding is required.
        regrid_mode (str):
            Selects which regridding techniques to use. Default uses
            iris.analysis.Linear(); "nearest" uses Nearest() (for less
            continuous fields, e.g precipitation); "nearest-with-mask"
            ensures that target data are sources from points with the same
            mask value (for coast-line-dependant variables like temperature).
        extrapolation_mode (str):
            Mode to use for extrapolating data into regions beyond the limits
            of the source_data domain. Refer to online documentation for
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
        landmask_vicinity (float):
            Radius of vicinity to search for a coastline, in metres.
        regridded_title (str):
            New "title" attribute to be set if the field is being regridded
            (since "title" may contain grid information). If None, a default
            value is used.
        new_metadata (dict):
            Dictionary containing required changes that will be applied to
            the attributes.
        coords_to_remove (list):
            List of names of scalar coordinates to remove.
        new_name (str):
            Name of output cube.
        new_units (str):
            Units to convert to.
        fix_float64 (bool):
            If True, checks and fixes cube for float64 data. Without this
            option an exception will be raised if float64 data is found but no
            fix applied.

    Returns:
        iris.cube.Cube:
            Processed cube.

    Raises:
        ValueError:
            If source landmask is supplied but regrid mode is not
            "nearest-with-mask".
        ValueError:
            If regrid_mode is "nearest-with-mask" but no source landmask is
            provided (from plugin).
    """
    from improver.standardise import StandardiseGridAndMetadata

    if (source_landmask and
            "nearest-with-mask" not in regrid_mode):
        msg = ("Land-mask file supplied without appropriate regrid-mode. "
               "Use --regrid-mode nearest-with-mask.")
        raise ValueError(msg)

    plugin = StandardiseGridAndMetadata(
        regrid_mode=regrid_mode, extrapolation_mode=extrapolation_mode,
        landmask=source_landmask, landmask_vicinity=landmask_vicinity)
    output_data = plugin.process(
        source_data, target_grid, new_name=new_name, new_units=new_units,
        regridded_title=regridded_title, coords_to_remove=coords_to_remove,
        attributes_dict=new_metadata, fix_float64=fix_float64)

    return output_data
