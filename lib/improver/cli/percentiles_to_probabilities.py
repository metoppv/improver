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
"""Script to collapse cube coordinates and calculate percentiled data."""

from improver.argparser import ArgParser

from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf
from improver.utilities.statistical_operations import \
    ProbabilitiesFromPercentiles2D


def main(argv=None):
    r"""
    Load arguments and run ProbabilitiesFromPercentiles plugin.

    Plugin generates probabilities at a fixed threshold (height) from a set of
    (height) percentiles.

    Example:

        Snow-fall level::

            Reference field: Percentiled snow fall level (m ASL)
            Other field: Orography (m ASL)

            300m ----------------- 30th Percentile snow fall level
            200m ----_------------ 20th Percentile snow fall level
            100m ---/-\----------- 10th Percentile snow fall level
            000m --/---\----------  0th Percentile snow fall level
            ______/     \_________ Orogaphy

        The orography heights are compared against the heights that correspond
        with percentile values to find the band in which they fall, then
        interpolated linearly to obtain a probability of snow level at / below
        the ground surface.
    """
    parser = ArgParser(
        description="Calculate probability from a percentiled field at a "
        "2D threshold level.  Eg for 2D percentile levels at different "
        "heights, calculate probability that height is at ground level, where"
        " the threshold file contains a 2D topography field.")
    parser.add_argument("percentiles_filepath", metavar="PERCENTILES_FILE",
                        help="A path to an input NetCDF file containing a "
                        "percentiled field")
    parser.add_argument("threshold_filepath", metavar="THRESHOLD_FILE",
                        help="A path to an input NetCDF file containing a "
                        "threshold value at which probabilities should be "
                        "calculated.")
    parser.add_argument("output_filepath", metavar="OUTPUT_FILE",
                        help="The output path for the processed NetCDF")
    parser.add_argument("output_diagnostic_name",
                        metavar="OUTPUT_DIAGNOSTIC_NAME", type=str,
                        help="Name for data in output file e.g. "
                        "probability_of_snow_falling_level_below_ground_level")
    args = parser.parse_args(args=argv)

    # Load Cubes
    threshold_cube = load_cube(args.threshold_filepath)
    percentiles_cube = load_cube(args.percentiles_filepath)

    # Process Cubes
    probability_cube = process(percentiles_cube, threshold_cube,
                               args.output_diagnostic_name)

    # Save Cubes
    save_netcdf(probability_cube, args.output_filepath)


def process(percentiles_cube, threshold_cube, output_diagnostic_name):
    """Calculates probability from a percentiled field.

    Plugin generates probabilities at a fixed threshold (height) from a set
    of (height) percentiles.

    Args:
        percentiles_cube (iris.cube.Cube):
            The percentiled field from which probabilities will be obtained
            using the input cube.
            This cube should contain a percentiles dimension, with fields of
            values that correspond to these percentiles. The cube passed to
            the process method will contain values of the same diagnostic.
        threshold_cube (iris.cube.Cube):
            A cube of values that effectively behave as thresholds, for which
            it is desired to obtain probability values from a percentiled
            reference cube.
        output_diagnostic_name (str):
            The name of the cube being created, e.g
            'probability_of_snow_falling_level_below_ground_level'

    Returns:
        probability_cube (iris.cube.Cube):
            A cube of probabilities obtained by interpolating between
            percentile values at the "threshold" level.
    """
    result = ProbabilitiesFromPercentiles2D(percentiles_cube,
                                            output_diagnostic_name)
    probability_cube = result.process(threshold_cube)
    return probability_cube


if __name__ == "__main__":
    main()
