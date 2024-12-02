#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to calculate the virtual temperature from temperature and humidity mixing ratio."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcubelist):
    """Calculate the virtual temperature from temperature and humidity mixing ratio.

    Args:
        cubes (iris.cube.CubeList or list of iris.cube.Cube):
            containing:
                temperature (iris.cube.Cube):
                    Cube of temperature.
                humidity_mixing_ratio (iris.cube.Cube):
                    Cube of humidity mixing ratio.

    Returns:
        iris.cube.Cube:
            Cube of virtual_temperature.
    """
    from improver.virtual_temperature import VirtualTemperature

    return VirtualTemperature()(*cubes)
