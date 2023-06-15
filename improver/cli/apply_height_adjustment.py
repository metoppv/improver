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
"""Script to apply height adjustments to spot data."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    spot_cube: cli.inputcube, neighbour: cli.inputcube, neighbour_selection_method: str ="nearest"
):
    """Apply height adjustment to account for the difference between site altitude and
    grid square orography.

    Args:
        spot_cube (iris.cube.Cube):
            A cube of spot data values.
        neighbour (iris.cube.Cube):
            A cube containing information about spot-data neighbours and
            the spot site information.
        neighbour_selection_method (str):
            The neighbour cube may contain one or several sets of grid
            coordinates that match a spot site. These are determined by
            the neighbour finding method employed. This keyword is used to
            extract the desired set of coordinates from the neighbour cube.

    Returns:
        iris.cube.Cube:
            A cube of spot data values with the same metadata as spot_cube but with data
            adjusted to be relative to site height rather than orography grid square
            height
    """
    from improver.spotdata.height_adjustment import SpotHeightAdjustment

    result = SpotHeightAdjustment(neighbour_selection_method)(spot_cube, neighbour)
    return result
