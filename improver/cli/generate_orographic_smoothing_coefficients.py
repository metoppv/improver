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
"""CLI for generating orographic smoothing_coefficients."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    orography: cli.inputcube,
    mask: cli.inputcube = None,
    *,
    min_gradient_smoothing_coefficient: float = 0.5,
    max_gradient_smoothing_coefficient: float = 0.0,
    power: float = 1.0,
    use_mask_boundary: bool = False,
    invert_mask: bool = False,
):
    """Generate smoothing coefficients for recursive filtering based on
    orography gradients.

    A smoothing coefficient determines how much "value" of a cell
    undergoing filtering is comprised of the current value at that cell and
    how much comes from the adjacent cell preceding it in the direction in
    which filtering is being applied. A larger smoothing_coefficient results in
    a more significant proportion of a cell's new value coming from its
    neighbouring cell.

    The smoothing coefficients are calculated from the orography gradient using
    a simple equation with the user defined value for the power:

    smoothing_coefficient = gradient**power

    The resulting values are scaled between min_gradient_smoothing_coefficient
    and max_gradient_smoothing_coefficient to give the desired range of
    smoothing_coefficients. These limiting values must be greater than or equal to
    zero and less than or equal to 0.5.

    Note that the smoothing coefficients are returned on a grid that is one cell
    smaller in the given dimension than the input orography, i.e. the smoothing
    coefficients in the x-direction are returned on a grid that is one cell
    smaller in x than the input. This is because the coefficients are used in
    both forward and backward passes of the recursive filter, so they need to be
    symmetric between cells in the original grid to help ensure conservation.

    Args:
        orography (iris.cube.Cube):
            A 2D field of orography on the grid for which
            smoothing_coefficients are to be generated.
        mask (iris.cube.Cube):
            A mask that defines where the smoothing coefficients should
            be zeroed. Further options below determine how this mask is used.
        min_gradient_smoothing_coefficient (float):
            The value of recursive filter smoothing_coefficient to be used
            where the orography gradient is a minimum. Generally this number
            will be larger than the max_gradient_smoothing_coefficient as
            quantities are likely to be smoothed more across flat terrain.
        max_gradient_smoothing_coefficient (float):
            The value of recursive filter smoothing_coefficient to be used
            where the orography gradient is a maximum. Generally this number
            will be smaller than the min_gradient_smoothing_coefficient as
            quantities are likely to be smoothed less across complex terrain.
        power (float):
            The power for the smoothing_coefficient equation.
        use_mask_boundary (bool):
            A mask can be provided to define a region in which smoothing
            coefficients are set to zero, i.e. no smoothing. If this
            option is set to True then rather than the whole masked region
            being set to zero, only the smoothing coefficient cells that mark
            the transition from masked to unmasked will be set to zero. The
            primary purpose for this is to prevent smoothing across land-sea
            boundaries.
        invert_mask (bool):
            By default, if a mask is provided and use_mask_boundary is False,
            all the smoothing coefficients corresponding to a mask value of 1
            will be zeroed. Setting invert_mask to True reverses this behaviour
            such that mask values of 0 set the points to be zeroed in the
            smoothing coefficients. If use_mask_boundary is True this option
            has no effect.
    Returns:
        iris.cube.CubeList:
            Processed CubeList containing smoothing_coefficients_x and
            smoothing_coefficients_y cubes.
    """
    from improver.generate_ancillaries.generate_orographic_smoothing_coefficients import (
        OrographicSmoothingCoefficients,
    )

    plugin = OrographicSmoothingCoefficients(
        min_gradient_smoothing_coefficient,
        max_gradient_smoothing_coefficient,
        power,
        use_mask_boundary,
        invert_mask,
    )

    return plugin(orography, mask)
