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
"""Script to calculate probabilities of occurrence between thresholds."""

import json

from improver.argparser import ArgParser
from improver.between_thresholds import OccurrenceBetweenThresholds
from improver.metadata.probabilistic import find_threshold_coordinate
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def main(argv=None):
    """Load arguments"""
    parser = ArgParser(description='Calculates probabilities of occurrence '
                       'between thresholds')
    parser.add_argument('input_file', metavar='INPUT_FILE',
                        help='Path to NetCDF file containing probabilities '
                        'above or below thresholds')
    parser.add_argument('output_file', metavar='OUTPUT_FILE',
                        help='Path to NetCDF file to write probabilities of '
                        'occurrence between thresholds')
    parser.add_argument('threshold_ranges', metavar='THRESHOLD_RANGES',
                        help='Path to json file specifying threshold ranges')
    parser.add_argument('--threshold_units', metavar='THRESHOLD_UNITS',
                        type=str, default=None,
                        help='Units in which thresholds are specified')
    args = parser.parse_args(args=argv)

    # Load Cube and json
    cube = load_cube(args.input_file)
    with open(args.threshold_ranges) as input_file:
        # read list of thresholds from json file
        threshold_ranges = json.load(input_file)

    # Process Cube
    result = process(
        cube, threshold_ranges, threshold_units=args.threshold_units)

    # Save Cube
    save_netcdf(result, args.output_file)


def process(cube, threshold_ranges, threshold_units=None):
    """
    Calculates probabilities of occurrence between thresholds

    Args:
        cube (iris.cube.Cube):
            Cube containing input probabilities above or below threshold
        threshold_ranges (list):
            List of 2-item iterables specifying thresholds between which
            probabilities should be calculated
        threshold_units (str):
            Units in which the thresholds are specified.  If None, defaults
            to the units of the threshold coordinate on the input cube.

    Returns:
        (iris.cube.Cube):
            Cube containing probability of occurrences between the thresholds
            specified
    """
    if threshold_units is None:
        threshold_units = str(find_threshold_coordinate(cube).units)

    plugin = OccurrenceBetweenThresholds(threshold_ranges, threshold_units)
    return plugin.process(cube)


if __name__ == "__main__":
    main()
