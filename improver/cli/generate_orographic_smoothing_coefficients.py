#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
"""CLI for generating orographic smoothing_coefficients."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(orography: cli.inputcube,
            *,
            min_smoothing_coefficient: float = 0.0,
            max_smoothing_coefficient: float = 1.0,
            coefficient: float = 1.0,
            power: float = 1.0):
    """Generate smoothing coefficients for recursive filtering based on
    orography gradients.

    A smoothing coefficient determines how much "value" of a cell
    undergoing filtering is comprised of the current value at that cell and
    how much comes from the adjacent cell preceding it in the direction in
    which filtering is being applied. A larger smoothing_coefficient results in
    a more significant proportion of a cell's new value coming from its
    neighbouring cell.

    The smoothing coefficients are calculated from the orography gradient using
    a simple equation and the user defined values for coefficient and power:

    smoothing_coefficient = coefficient * gradient**power

    The resulting values are scaled between min_smoothing_coefficient and
    max_smoothing_coefficient to give the desired range of
    smoothing_coefficients. These can be provided in reverse (i.e. min > max)
    to invert the smoothing coefficients in relation to the orographic
    gradient, providing smoothing coefficients that are largest where the
    orography gradient is shallowest.

    Args:
        orography (iris.cube.Cube):
            A 2D field of orography on the grid for which
            smoothing_coefficients are to be generated.
        min_smoothing_coefficient (float):
            The minimum value of smoothing_coefficient.
        max_smoothing_coefficient (float):
            The maximum value of smoothing_coefficient.
        coefficient (float):
            The coefficient for the smoothing_coefficient equation.
        power (float):
            The power for the smoothing_coefficient equation.

    Returns:
        iris.cube.CubeList:
            Processed CubeList containing smoothing_coefficients_x and
            smoothing_coefficients_y cubes.
    """
    from improver.utilities.ancillary_creation import (
        OrographicSmoothingCoefficients)
    return OrographicSmoothingCoefficients(
        min_smoothing_coefficient, max_smoothing_coefficient, coefficient,
        power).process(orography)
