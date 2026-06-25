#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to run the UV index plugin."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    uv_flux_down: cli.inputcube, *, scale_factor: float = 3.6, model_id_attr: str = None
):
    """Calculate the UV index using the data in the input cubes.

    Calculate the uv index using the radiation flux in UV downward at surface.

    Args:
        uv_flux_down (iris.cube.Cube):
            Cube of radiation flux in UV downwards at surface.
        scale_factor:
            The uv scale factor. Default is 3.6 (m2 W-1). This factor has
            been empirically derived and should not be
            changed except if there are scientific reasons to
            do so. For more information see section 2.1.1 of the paper
            referenced below.
        model_id_attr (str):
            Name of the attribute used to identify the source model for
            blending.

    Returns:
        iris.cube.Cube:
            Processed Cube.

    References:
        Turner, E.C, Manners, J. Morcrette, C. J, O'Hagan, J. B,
        & Smedley, A.R.D. (2017): Toward a New UV Index Diagnostic
        in the Met Office's Forecast Model. Journal of Advances in
        Modeling Earth Systems 9, 2654-2671.
    """
    from improver.uv_index import calculate_uv_index

    result = calculate_uv_index(
        uv_flux_down, scale_factor=scale_factor, model_id_attr=model_id_attr
    )
    return result
