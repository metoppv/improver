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
"""Script to standardise a NetCDF file by updating metadata and demoting
float64 data to float32"""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube,
    *,
    attributes_config: cli.inputjson = None,
    coords_to_remove: cli.comma_separated_list = None,
    new_name: str = None,
    new_units: str = None,
):
    """
    Standardise a source cube. Available options are renaming, converting units,
    updating attributes and removing named scalar coordinates. Remaining scalar
    coordinates are collapsed, CellMethod("point": "time") is discarded, and data
    are cast to IMPROVER standard datatypes and units.

    Deprecated behaviour:
    Translates metadata relating to the grid_id attribute from StaGE
    version 1.1.0 to StaGE version 1.2.0. Cubes that have no "grid_id"
    attribute are not recognised as v1.1.0 and are not changed.

    Args:
        cube (iris.cube.Cube):
            Source cube to be standardised
        attributes_config (dict):
            Dictionary containing required changes that will be applied to
            the attributes.
        coords_to_remove (list):
            List of names of scalar coordinates to remove.
        new_name (str):
            Name of output cube.
        new_units (str):
            Units to convert to.

    Returns:
        iris.cube.Cube
    """
    from improver.metadata.amend import update_stage_v110_metadata
    from improver.standardise import StandardiseMetadata

    # update_stage_v110_metadata is deprecated. Please ensure metadata is
    # StaGE version 1.2.0 compatible.
    update_stage_v110_metadata(cube)

    return StandardiseMetadata()(
        cube,
        new_name=new_name,
        new_units=new_units,
        coords_to_remove=coords_to_remove,
        attributes_dict=attributes_config,
    )
