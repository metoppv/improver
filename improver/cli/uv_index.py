#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to run the UV index plugin."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(uv_flux_down: cli.inputcube, *, model_id_attr: str = None):
    """Calculate the UV index using the data in the input cubes.

    Calculate the uv index using the radiation flux in UV downward at surface.

    Args:
        uv_flux_down (iris.cube.Cube):
            Cube of radiation flux in UV downwards at surface.
        model_id_attr (str):
            Name of the attribute used to identify the source model for
            blending.

    Returns:
        iris.cube.Cube:
            Processed Cube.
    """
    from improver.uv_index import calculate_uv_index

    result = calculate_uv_index(uv_flux_down, model_id_attr=model_id_attr)
    return result
