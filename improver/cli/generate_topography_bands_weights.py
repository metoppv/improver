#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to run topographic bands weights generation."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    orography: cli.inputcube,
    land_sea_mask: cli.inputcube = None,
    *,
    bands_config: cli.inputjson = None,
):
    """Runs topographic weights generation.

    Reads the orography and land_sea_mask fields of a cube. Creates a series of
    topographic zone weights to indicate where an orography point sits within
    the defined topographic bands. If the orography point is in the centre of
    a topographic band, then a single band will have a weight 1.0.
    If the orography point is at the edge of a topographic band, then the
    upper band will have a 0.5 weight whilst the lower band will also have a
    0.5 weight. Otherwise the weight will vary linearly between the centre of
    a topographic band and the edge.

    Args:
        orography (iris.cube.Cube):
            The orography on a standard grid.
        land_sea_mask (iris.cube.Cube):
            Land mask on a standard grid, with land points set to one and
            sea points set to zero. If provided, sea points will be
            masked and set to the default fill value. If no land mask is
            provided, weights will be generated for sea points as well as land
            in the appropriate topographic band.
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
            Cube containing the weights depending upon where the orography
            point is within the topographical zones.
    """
    from improver.generate_ancillaries.generate_ancillary import THRESHOLDS_DICT
    from improver.generate_ancillaries.generate_topographic_zone_weights import (
        GenerateTopographicZoneWeights,
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

    result = GenerateTopographicZoneWeights()(
        orography, bands_config, landmask=land_sea_mask
    )
    return result
