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
"""Script to run topographic bands mask generation."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    orography: cli.inputcube,
    land_sea_mask: cli.inputcube = None,
    *,
    bands_config: cli.inputjson = None,
):
    """Runs topographic bands mask generation.

    Reads orography and land_sea_mask fields of a cube. Creates a series of
    masks, where each mask excludes data below or equal to the lower threshold
    and excludes data above the upper threshold.

    Args:
        orography (iris.cube.Cube):
            The orography on a standard grid.
        land_sea_mask (iris.cube.Cube):
            The land mask on standard grid, with land points set to one and
            sea points set to zero. If provided sea points will be set
            to zero in every band. If no land mask is provided, sea points will
            be included in the appropriate topographic band.
        bands_config (dict):
            Definition of orography bands required.
            The expected format of the dictionary is e.g
            {'bounds':[[0, 50], [50, 200]], 'units': 'm'}
            The default dictionary has the following form:
            {'bounds': [[-500., 50.], [50., 100.],
            [100., 150.],[150., 200.], [200., 250.],
            [250., 300.], [300., 400.], [400., 500.],
            [500., 650.],[650., 800.], [800., 950.],
            [950., 6000.]], 'units': 'm'}

    Returns:
        iris.cube.Cube:
            list of orographic band mask cube.

    """
    from improver.generate_ancillaries.generate_ancillary import (
        THRESHOLDS_DICT,
        GenerateOrographyBandAncils,
    )

    if bands_config is None:
        bands_config = THRESHOLDS_DICT

    if land_sea_mask:
        land_sea_mask = next(
            land_sea_mask.slices(
                [land_sea_mask.coord(axis="y"), land_sea_mask.coord(axis="x")]
            )
        )

    orography = next(
        orography.slices([orography.coord(axis="y"), orography.coord(axis="x")])
    )

    result = GenerateOrographyBandAncils()(
        orography, bands_config, landmask=land_sea_mask
    )
    result = result.concatenate_cube()
    return result
