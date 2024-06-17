#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to generate maximum vertical updraught from CAPE and max precip rate data."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube, model_id_attr: str = None):
    """Module to generate maximum vertical updraught.

    Call the VerticalUpdraught plugin to calculate maximium vertical updraught.

    Args:
        cubes (iris.cube.CubeList or list of iris.cube.Cube):
            containing:
                cape (iris.cube.Cube):
                    Cube of convective_available_potential_energy, valid at the start
                    of the precip time window.
                precip (iris.cube.Cube):
                    Cube of lwe_precipitation_rate_max, over a time window.
        model_id_attr (str):
            Name of the attribute used to identify the source model for blending.

    Returns:
        iris.cube.Cube:
            Cube of vertical updraught (m s-1).

    """
    from improver.wind_calculations.vertical_updraught import VerticalUpdraught

    return VerticalUpdraught(model_id_attr=model_id_attr)(cubes)
