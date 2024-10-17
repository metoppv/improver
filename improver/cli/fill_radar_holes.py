#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to fill in small holes in the radar composite"""
from improver import cli


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube):
    """ Fill in small "no data" holes in the radar composite

    Args:
        cube (iris.cube.Cube):
            Masked radar composite

    Returns:
        iris.cube.Cube
    """
    from improver.nowcasting.utilities import FillRadarHoles

    result = FillRadarHoles()(cube)
    return result
