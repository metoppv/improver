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


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube,
            *,
            mask_cube: cli.inputcube = None,
            alphas_x_cube: cli.inputcube = None,
            alphas_y_cube: cli.inputcube = None,
            alpha_x: float = None,
            alpha_y: float = None,
            iterations: int = 1,
            re_mask = False):
    """Module to apply a recursive filter to neighbourhooded data.

    Run a recursive filter to convert a square neighbourhood into a
    Gaussian-like kernel or smooth over short distances. The filter uses an
    alpha parameter (0 alpha < 1) to control what proportion of the
    probability is passed onto the next grid-square in the x and y directions.
    The alpha parameter can be set on a grid square by grid-square basis for
    the x and y directions separately (using two arrays of alpha parameters
    of the same dimensionality as the domain).
    Alternatively a single alpha value can be set for each of the x and y
    direction and a float for the y direction and vice versa.

    Args:
        cube (iris.cube.Cube):
            Cube to be processed.
        mask_cube (iris.cube.Cube):
            Cube to mask the processed cube.
            Default is None.
        alphas_x_cube (iris.cube.Cube):
            Cube describing the alpha factors to be used for smoothing in the
            x direction.
            Default is None.
        alphas_y_cube (iris.cube.Cube):
            Cube describing the alpha factors to be used for smoothing in the
            y direction.
            Default is None.
        alpha_x (float):
            A single alpha factor (0 < alpha_x < 1) to be applied to every
            grid square in the x direction.
            Default is None.
        alpha_y (float):
            A single alpha factor (0 < alpha_y < 1) to be applied to every grid
            square in the y direction.
            Default is None.
        iterations (int):
            Number of times to apply the filter. (Typically < 3)
            Number of iterations should be 2 or less, higher values have been
            shown to lead to poorer conservation.
            Default is 1 (one).
        re_mask (bool):
            Re-apply mask to recursively filtered output.
            Default is False.

    Returns:
        iris.cube.Cube:
            The processed Cube.
    """
    from improver.nbhood.recursive_filter import RecursiveFilter
    result = RecursiveFilter(
        alpha_x=alpha_x, alpha_y=alpha_y,
        iterations=iterations, re_mask=re_mask).process(
        cube, alphas_x=alphas_x_cube, alphas_y=alphas_y_cube,
        mask_cube=mask_cube)
    return result
