#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to create Turbulence Indices above 1500 meters from multi-parameter datasets."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube, model_id_attr: str = None):
    """
    From the supplied set of cubes at two, presumable adjacent, pressure levels, calculate the
    Turbulence Index above 1500 m based on Ellrod 1997.
    Values are typically small on the order of 1e-7 and are in units of 1/second^2 (i.e., s-2).
    The returned Cube will have a long name beginning with "TurbulenceIndexAbove1500m" and
    concatenated with a string representing the pressure level of the calculations in millibars.
        E.g., name="TurbulenceIndexAbove1500m550mb"
    The calculations are performed on the greater pressure level (lowest altitude) provided.

    Args:
        cubes (list of iris.cube.Cube):
            Cubes to be processed.
        model_id_attr (str):
            Name of the attribute used to identify the source model for
            blending.

    Returns:
        iris.cube.Cube:
            Cube of Turbulence Index calculated at greatest provided pressure level in units of 1/second^2.
    """
    from iris.cube import CubeList

    from improver.turbulence import TurbulenceIndexAbove1500m_USAF

    result = TurbulenceIndexAbove1500m_USAF()(CubeList(cubes), model_id_attr=model_id_attr)

    return result
