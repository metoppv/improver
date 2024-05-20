#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to set night values to zero for UV index."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube):
    """Sets night values to zero for UV index.

    Args:
        cube (iris.cube.Cube):
            Cube that will have night values set to zero.  This should contain
            either diagnostic values or probabilities of UV index above threshold.

    Returns:
        iris.cube.Cube:
            Input cube with all night values set to zero.

    Raises:
        ValueError: If input cube is suspicious, within reason.  Note that this is
            a general check: the CLI expects a cube of UV index or probability of
            UV index above thresold, and will raise an error if given a probability
            below threshold, but will not recognise a completely inappropriate cube
            (eg temperature in Kelvin).  Therefore this CLI should be used with care.
    """
    import numpy as np

    from improver.metadata.probabilistic import is_probability
    from improver.utilities.solar import DayNightMask

    if is_probability(cube):
        if "above_threshold" not in cube.name():
            raise ValueError(f"{cube.name()} unsuitable for night masking")

    mask = DayNightMask()(cube).data
    # Broadcast mask to shape of input cube to account for additional dimensions.
    mask = np.broadcast_to(mask, cube.shape)
    # setting night values to zero.
    cube.data = np.where(mask == DayNightMask().night, 0, cube.data)
    return cube
