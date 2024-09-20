#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
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
    coord_modification: cli.inputjson = None,
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
        coord_modification (dict):
            A dictionary allowing the direct modification of scalar coordinate
            values in the original units of the coordinate. To be used with
            extreme caution. For example this might be: {"height": 1.5} to set
            the height coordinate to have a value of 1.5m (assuming original
            units of m). Type is inferred, so providing a value of 2 will result
            in an integer type, whilst a value of 2.0 will result in a float
            type (where this is not modified by type enforcement).
        new_name (str):
            Name of output cube.
        new_units (str):
            Units to convert to.

    Returns:
        iris.cube.Cube
    """
    from improver.standardise import StandardiseMetadata

    return StandardiseMetadata(
        new_name=new_name,
        new_units=new_units,
        coords_to_remove=coords_to_remove,
        coord_modification=coord_modification,
        attributes_dict=attributes_config,
    )(cube)
