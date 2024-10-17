#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to apply a recursive filter to neighbourhooded data."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube,
    smoothing_coefficients: cli.inputcubelist,
    *,
    iterations: int = 1,
):
    """Module to apply a recursive filter to neighbourhooded data.

    Run a recursive filter to convert a square neighbourhood into a
    Gaussian-like kernel or smooth over short distances. The filter uses a
    smoothing_coefficient (between 0 and 1) to control what proportion of the
    probability is passed onto the next grid-square in the x and y directions.

    Each iteration of the recursive filter applies the smoothing coefficients
    forwards and backwards in both the x and y directions. Applying 1-10
    iterations of the filter is typical. Each iteration further smooths the
    data, meaning the user must make a judgement regarding the number of
    iterations to apply that preserves real detail whilst removing artefacts
    in their data.

    The IMPROVER plugin actually limits the maximum smoothing coefficient to
    a value of 0.5. Above this the smoothing is considered to be too great.
    The smoothing_coefficient can be set on a grid square by grid-square basis
    for the x and y directions separately (using two arrays of
    smoothing_coefficients of the same dimensionality as the domain).

    Args:
        cube (iris.cube.Cube):
            Cube to be processed.
        smoothing_coefficients (iris.cube.CubeList):
            CubeList describing the smoothing_coefficients to be used in the x
            and y directions.
        iterations (int):
            Number of times to apply the filter.

    Returns:
        iris.cube.Cube:
            The processed Cube.
    """
    from improver.nbhood.recursive_filter import RecursiveFilter

    plugin = RecursiveFilter(iterations=iterations)
    return plugin(cube, smoothing_coefficients=smoothing_coefficients)
