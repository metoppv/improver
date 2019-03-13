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
"""Script to apply thresholding to a parameter dataset."""

from improver.argparser import ArgParser

import numpy as np
import json
import cf_units
import warnings

from improver.threshold import BasicThreshold
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf
from improver.utilities.cube_metadata import in_vicinity_name_format
from improver.utilities.spatial import OccurrenceWithinVicinity
from improver.blending.weights import ChooseDefaultWeightsLinear
from improver.blending.weighted_blend import WeightedBlendAcrossWholeDimension


def main():
    """Load in arguments and get going."""
    parser = ArgParser(
        description="Calculate the threshold truth value of input data "
        "relative to the provided threshold value. By default data are "
        "tested to be above the thresholds, though the --below_threshold "
        "flag enables testing below thresholds. A fuzzy factor or fuzzy "
        "bounds may be provided to capture data that is close to the "
        "threshold.")
    parser.add_argument("input_filepath", metavar="INPUT_FILE",
                        help="A path to an input NetCDF file to be processed")
    parser.add_argument("output_filepath", metavar="OUTPUT_FILE",
                        help="The output path for the processed NetCDF")
    parser.add_argument("threshold_values", metavar="THRESHOLD_VALUES",
                        nargs="*", type=float,
                        help="Threshold value or values about which to "
                        "calculate the truth values; e.g. 270 300. "
                        "Must be omitted if --threshold_config is used.")
    parser.add_argument("--threshold_config", metavar="THRESHOLD_CONFIG",
                        type=str,
                        help="Threshold configuration JSON file containing "
                        "thresholds and (optionally) fuzzy bounds. Best used "
                        "in combination  with --threshold_units. "
                        "It should contain a dictionary of strings that can "
                        "be interpreted as floats with the structure: "
                        " \"THRESHOLD_VALUE\": [LOWER_BOUND, UPPER_BOUND] "
                        "e.g: {\"280.0\": [278.0, 282.0], "
                        "\"290.0\": [288.0, 292.0]}, or with structure "
                        " \"THRESHOLD_VALUE\": \"None\" (no fuzzy bounds). "
                        "Repeated thresholds with different bounds are not "
                        "handled well. Only the last duplicate will be used.")
    parser.add_argument("--threshold_units", metavar="THRESHOLD_UNITS",
                        default=None, type=str,
                        help="Units of the threshold values. If not provided "
                        "the units are assumed to be the same as those of the "
                        "input dataset. Specifying the units here will allow "
                        "a suitable conversion to match the input units if "
                        "possible.")
    parser.add_argument("--below_threshold", default=False,
                        action='store_true',
                        help="By default truth values of 1 are returned for "
                        "data ABOVE the threshold value(s). Using this flag "
                        "changes this behaviour to return 1 for data below "
                        "the threshold values.")
    parser.add_argument("--fuzzy_factor", metavar="FUZZY_FACTOR",
                        default=None, type=float,
                        help="A decimal fraction defining the factor about "
                        "the threshold value(s) which should be treated as "
                        "fuzzy. Data which fail a test against the hard "
                        "threshold value may return a fractional truth value "
                        "if they fall within this fuzzy factor region. Fuzzy "
                        "factor must be in the range 0-1, with higher values "
                        "indicating a narrower fuzzy factor region / sharper "
                        "threshold. NB A fuzzy factor cannot be used with a "
                        "zero threshold or a threshold_config file.")
    parser.add_argument("--collapse-coord", type=str,
                        metavar="COLLAPSE-COORD", default="None",
                        help="An optional ability to set which coordinate "
                        "we want to collapse over. The default is set "
                        "to None.")
    parser.add_argument("--vicinity", type=float, default=None, help="If set,"
                        " distance in metres used to define the vicinity "
                        "within which to search for an occurrence.")

    args = parser.parse_args()

    # Deal with mutual-exclusions that ArgumentParser can't handle:
    if args.threshold_values and args.threshold_config:
        raise parser.error("--threshold_config option is not compatible "
                           "with THRESHOLD_VALUES list.")
    if args.fuzzy_factor and args.threshold_config:
        raise parser.error("--threshold_config option is not compatible "
                           "with --fuzzy_factor option.")

    cube = load_cube(args.input_filepath)

    if args.threshold_config:
        try:
            # Read in threshold configuration from JSON file.
            with open(args.threshold_config, 'r') as input_file:
                thresholds_from_file = json.load(input_file)
            thresholds = []
            fuzzy_bounds = []
            is_fuzzy = True
            for key in thresholds_from_file.keys():
                thresholds.append(float(key))
                if is_fuzzy:
                    # If the first threshold has no bounds, fuzzy_bounds is
                    # set to None and subsequent bounds checks are skipped
                    if thresholds_from_file[key] == "None":
                        is_fuzzy = False
                        fuzzy_bounds = None
                    else:
                        fuzzy_bounds.append(tuple(thresholds_from_file[key]))
        except ValueError as err:
            # Extend error message with hint for common JSON error.
            raise type(err)(err + " in JSON file {}. \nHINT: Try "
                            "adding a zero after the decimal point.".format(
                                args.threshold_config))
        except Exception as err:
            # Extend any errors with message about WHERE this occurred.
            raise type(err)(err + " in JSON file {}".format(
                args.threshold_config))
    else:
        thresholds = args.threshold_values
        fuzzy_bounds = None

    result_no_collapse_coord = BasicThreshold(
        thresholds, fuzzy_factor=args.fuzzy_factor,
        fuzzy_bounds=fuzzy_bounds, threshold_units=args.threshold_units,
        below_thresh_ok=args.below_threshold).process(cube)

    if args.vicinity is not None:
        # smooth thresholded occurrences over local vicinity
        result_no_collapse_coord = OccurrenceWithinVicinity(
            args.vicinity).process(result_no_collapse_coord)

        new_cube_name = in_vicinity_name_format(
            result_no_collapse_coord.name())

        result_no_collapse_coord.rename(new_cube_name)

    if args.collapse_coord == "None":
        save_netcdf(result_no_collapse_coord, args.output_filepath)
    else:
        # Raise warning if result_no_collapse_coord is masked array
        if np.ma.isMaskedArray(result_no_collapse_coord.data):
            warnings.warn("Collapse-coord option not fully tested with "
                          "masked data.")
        """ This is where we fix values for y0val, slope and weighting_mode.
            In this case they are fixed to the values required for realization
            collapse. This can be changed if other functionality needs to be
            implemented."""
        weights = ChooseDefaultWeightsLinear(y0val=1.0, slope=0.0).process(
            result_no_collapse_coord, args.collapse_coord)

        BlendingPlugin = WeightedBlendAcrossWholeDimension(
            args.collapse_coord, weighting_mode='weighted_mean')
        result_collapse_coord = BlendingPlugin.process(
            result_no_collapse_coord, weights)

        save_netcdf(result_collapse_coord, args.output_filepath)


if __name__ == "__main__":
    main()
