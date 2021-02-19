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
"""Script to extract a spatial subset of input file data."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube,
    *,
    grid_spec: cli.inputjson = None,
    site_list: cli.comma_separated_list = None,
):
    """
    Extract a thinned spatial cutout or subset of sites from a data file. Supports
    extraction from a spot file based on a list of sites, or grid subsetting using
    a dictionary specification of the following form:

    {"projection_x_coordinate": {"min": -100000, "max": 150000, "thin": 5},
     "projection_y_coordinate": {"min": -100000, "max": 200000, "thin": 5},
     "latitude": {"min": 45, "max": 52, "thin": 2},
     "longitude": {"min": -2, "max": 6, "thin": 2}}

    Args:
        cube (iris.cube.Cube):
            Input dataset
        grid_spec (Dict[str, Dict[str, int]]):
            Dictionary containing bounding grid points and an integer "thinning
            factor" for each of UK and global grid, to create cutouts.  Eg a
            "thinning factor" of 10 would mean every 10th point being taken for
            the cutout.  The expected dictionary has keys that are spatial coordinate
            names, with values that are dictionaries with "min", "max" and "thin" keys.
            The dictionary MUST contain entries for the spatial coordinates on the
            input cube, and MAY contain additional entries (which will be ignored).
        site_list (list):
            List of WMO site IDs to extract.  These IDs must match the type and format
            of the "wmo_id" coordinate on the input spot cube.

    Returns:
        iris.cube.Cube:
            Subset of input cube as specified by input constraints
    """
    from improver.utilities.cube_extraction import subset_data

    return subset_data(cube, grid_spec=grid_spec, site_list=site_list)
