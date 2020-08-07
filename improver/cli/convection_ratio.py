#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
"""Script to calculate the ratio of convective precipitation to total precipitation."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube,):
    """ Calculate the convection ratio from convective and dynamic precipitation rate
    components.

    Calculates the convective ratio as:

        ratio = convective_rate / (convective_rate + dynamic_rate)

    Then calculates the mean ratio across realizations.

    Args:
        cubes (iris.cube.CubeList):
            Cubes of "convective_precipitation_rate" and "dynamic_precipitation_rate"
            in units that can be converted to "m s-1"

    Returns:
        iris.cube.Cube:
            A single cube of convection_ratio.
    """
    from improver.blending.calculate_weights_and_blend import WeightAndBlend
    from improver.convection import ConvectionRatioFromComponents
    from iris.coords import CellMethod

    if len(cubes) != 2:
        raise IOError(f"Expected 2 input cubes, received {len(cubes)}")
    convection_ratio = ConvectionRatioFromComponents()(cubes)
    mean_convection_ratio = WeightAndBlend(
        "realization", "linear", y0val=1.0, ynval=1.0
    )(convection_ratio)
    mean_convection_ratio.add_cell_method(CellMethod("mean", "realization"))
    return mean_convection_ratio
