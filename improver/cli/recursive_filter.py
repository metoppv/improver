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
"""Module to apply a recursive filter to neighbourhooded data."""

from improver import cli

input_smoothing_coefficients = cli.create_constrained_inputcubelist_converter(
    'smoothing_coefficient_x', 'smoothing_coefficient_y')


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube,
            smoothing_coefficients: input_smoothing_coefficients,
            mask: cli.inputcube = None,
            *,
            iterations: int = 1,
            remask=False):
    """Module to apply a recursive filter to neighbourhooded data.

    Run a recursive filter to convert a square neighbourhood into a
    Gaussian-like kernel or smooth over short distances. The filter uses a
    smoothing_coefficient (between 0 and 1) to control what proportion of the
    probability is passed onto the next grid-square in the x and y directions.
    The smoothing_coefficient can be set on a grid square by grid-square basis
    for the x and y directions separately (using two arrays of
    smoothing_coefficients of the same dimensionality as the domain).

    Args:
        cube (iris.cube.Cube):
            Cube to be processed.
        smoothing_coefficients (iris.cube.CubeList):
            CubeList describing the smoothing_coefficients to be used in the x
            and y directions.
        mask (iris.cube.Cube):
            Cube to mask the processed cube.
        iterations (int):
            Number of times to apply the filter. (Typically < 3)
            Number of iterations should be 2 or less, higher values have been
            shown to lead to poorer conservation.
        remask (bool):
            Re-apply mask to recursively filtered output.

    Returns:
        iris.cube.Cube:
            The processed Cube.
    """
    from improver.nbhood.recursive_filter import RecursiveFilter

    smoothing_coefficients_x_cube, smoothing_coefficients_y_cube = (
        smoothing_coefficients)
    plugin = RecursiveFilter(iterations=iterations, re_mask=remask)
    return plugin(
        cube,
        smoothing_coefficients_x=smoothing_coefficients_x_cube,
        smoothing_coefficients_y=smoothing_coefficients_y_cube,
        mask_cube=mask)
