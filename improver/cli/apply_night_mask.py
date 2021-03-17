#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
