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

from improver.argparser import ArgParser

from improver.nbhood.recursive_filter import RecursiveFilter
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
    """Load in arguments and get going."""
    parser = ArgParser(
        description="Run a recursive filter to convert a square neighbourhood "
        "into a Gaussian-like kernel or smooth over short "
        "distances. The filter uses an alpha parameter (0 < alpha < 1) to "
        "control what proportion of the probability is passed onto the next "
        "grid-square in the x and y directions. The alpha parameter can be "
        "set on a grid-square by grid-square basis for the x and y directions "
        "separately (using two arrays of alpha parameters of the same "
        "dimensionality as the domain). Alternatively a single alpha value "
        "can be set for each of the x and y directions. These methods can be "
        "mixed, e.g. an array for the x direction and a float for the y "
        "direction and vice versa.")
    parser.add_argument("input_filepath", metavar="INPUT_FILE",
                        help="A path to an input NetCDF file to be processed")
    parser.add_argument("output_filepath", metavar="OUTPUT_FILE",
                        help="The output path for the processed NetCDF")
    parser.add_argument("--input_filepath_alphas_x", metavar="ALPHAS_X_FILE",
                        help="A path to a NetCDF file describing the alpha "
                        "factors to be used for smoothing in the x "
                        "direction")
    parser.add_argument("--input_filepath_alphas_y", metavar="ALPHAS_Y_FILE",
                        help="A path to a NetCDF file describing the alpha "
                        "factors to be used for smoothing in the y "
                        "direction")
    parser.add_argument("--alpha_x", metavar="ALPHA_X",
                        default=None, type=float,
                        help="A single alpha factor (0 < alpha_x < 1) to be "
                        "applied to every grid square in the x "
                        "direction.")
    parser.add_argument("--alpha_y", metavar="ALPHA_Y",
                        default=None, type=float,
                        help="A single alpha factor (0 < alpha_y < 1) to be "
                        "applied to every grid square in the y "
                        "direction.")
    parser.add_argument("--iterations", metavar="ITERATIONS",
                        default=1, type=int,
                        help="Number of times to apply the filter, default=1 "
                        "(typically < 5)")
    parser.add_argument('--input_mask_filepath', metavar='INPUT_MASK_FILE',
                        help='A path to an input mask NetCDF file to be '
                        'used to mask the input file.')
    parser.add_argument("--re_mask", action='store_true', default=False,
                        help="Re-apply mask to recursively filtered output.")

    args = parser.parse_args(args=argv)

    # Load Cubes.
    cube = load_cube(args.input_filepath)
    mask_cube = load_cube(args.input_mask_filepath, allow_none=True)
    alphas_x_cube = load_cube(args.input_filepath_alphas_x, allow_none=True)
    alphas_y_cube = load_cube(args.input_filepath_alphas_y, allow_none=True)
    # Process Cube
    result = process(cube, mask_cube, alphas_x_cube, alphas_y_cube,
                     args.alpha_x, args.alpha_y, args.iterations, args.re_mask)
    # Save Cube
    save_netcdf(result, args.output_filepath)


def process(cube, mask_cube=None, alphas_x_cube=None, alphas_y_cube=None,
            alpha_x=None, alpha_y=None, iterations=1, re_mask=False):
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
            Number of times to apply the filter. (Typically < 5)
            Default is 1 (one).
        re_mask (bool):
            Re-apply mask to recursively filtered output.
            Default is False.

    Returns:
        result (iris.cube.Cube):
            The processed Cube.
    """
    result = RecursiveFilter(
        alpha_x=alpha_x, alpha_y=alpha_y,
        iterations=iterations, re_mask=re_mask).process(
        cube, alphas_x=alphas_x_cube, alphas_y=alphas_y_cube,
        mask_cube=mask_cube)
    return result


if __name__ == "__main__":
    main()
