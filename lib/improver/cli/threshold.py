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

import json
import warnings

import numpy as np

from improver.argparser import ArgParser
from improver.blending.calculate_weights_and_blend import WeightAndBlend
from improver.threshold import BasicThreshold
from improver.utilities.cube_metadata import in_vicinity_name_format
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf
from improver.utilities.spatial import OccurrenceWithinVicinity


def main(argv=None):
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

    args = parser.parse_args(args=argv)

    # Deal with mutual-exclusions that ArgumentParser can't handle:
    if args.threshold_values and args.threshold_config:
        raise parser.error("--threshold_config option is not compatible "
                           "with THRESHOLD_VALUES list.")
    if args.fuzzy_factor and args.threshold_config:
        raise parser.error("--threshold_config option is not compatible "
                           "with --fuzzy_factor option.")

    # Load Cube
    cube = load_cube(args.input_filepath)
    threshold_dict = None
    if args.threshold_config:
        try:
            # Read in threshold configuration from JSON file.
            with open(args.threshold_config, 'r') as input_file:
                threshold_dict = json.load(input_file)
        except ValueError as err:
            # Extend error message with hint for common JSON error.
            raise type(err)("{} in JSON file {}. \nHINT: Try "
                            "adding a zero after the decimal point.".
                            format(err, args.threshold_config))
        except Exception as err:
            # Extend any errors with message about WHERE this occurred.
            raise type(err)("{} in JSON file {}".format(
                err, args.threshold_config))

    # Process Cube
    result = process(cube, args.threshold_values, threshold_dict,
                     args.threshold_units,
                     args.below_threshold, args.fuzzy_factor,
                     args.collapse_coord, args.vicinity)
    # Save Cube
    save_netcdf(result, args.output_filepath)


def process(cube, threshold_values=None, threshold_dict=None,
            threshold_units=None, below_threshold=False, fuzzy_factor=None,
            collapse_coord="None", vicinity=None):
    """Module to apply thresholding to a parameter dataset.

    Calculate the threshold truth values of input data relative to the
    provided threshold value. By default data are tested to be above the
    threshold, though the below_threshold boolean enables testing below
    thresholds.
    A fuzzy factor or fuzzy bounds may be provided to capture data that is
    close to the threshold.

    Args:
        cube (iris.cube.Cube):
             A cube to be processed.
        threshold_values (float):
            Threshold value or values about which to calculate the truth
            values; e.g. 270 300. Must be omitted if 'threshold_config'
            is used.
            Default is None.
        threshold_dict (dict):
            Threshold configuration containing threshold values and
            (optionally) fuzzy bounds. Best used in combination with
            'threshold_units' It should contain a dictionary of strings that
            can be interpreted as floats with the structure:
            "THRESHOLD_VALUE": [LOWER_BOUND, UPPER_BOUND]
            e.g: {"280.0": [278.0, 282.0], "290.0": [288.0, 292.0]},
            or with structure
            "THRESHOLD_VALUE": "None" (no fuzzy bounds).
            Repeated thresholds with different bounds are not
            handled well. Only the last duplicate will be used.
            Default is None.
        threshold_units (str):
            Units of the threshold values. If not provided the units are
            assumed to be the same as those of the input cube. Specifying
            the units here will allow a suitable conversion to match
            the input units if possible.
        below_threshold (bool):
            By default truth values of 1 are returned for data ABOVE the
            threshold value(s). Using this boolean changes this behaviours
            to return 1 for data below the threshold values.
        fuzzy_factor (float):
            A decimal fraction defining the factor about the threshold value(s)
            which should be treated as fuzzy. Data which fail a test against
            the hard threshold value may return a fractional truth value if
            they fall within this fuzzy factor region.
            Fuzzy factor must be in the range 0-1, with higher values
            indicating a narrower fuzzy factor region / sharper threshold.
            N.B. A fuzzy factor cannot be used with a zero threshold or a
            threshold_dict.
        collapse_coord (str):
            An optional ability to set which coordinate we want to collapse
            over. The default is set to None.
        vicinity (float):
            If True, distance in metres used to define the vicinity within
            which to search for an occurrence.

    Returns:
        result (iris.cube.Cube):
            processed Cube.

    Raises:
        RuntimeError:
            If threshold_dict and threshold_values are both used.

     Warns:
        warning:
            If collapsing coordinates with a masked array.

    """
    if threshold_dict and threshold_values:
        raise RuntimeError('threshold_dict cannot be used '
                           'with threshold_values')
    if threshold_dict:
        try:
            thresholds = []
            fuzzy_bounds = []
            is_fuzzy = True
            for key in threshold_dict.keys():
                thresholds.append(float(key))
                if is_fuzzy:
                    # If the first threshold has no bounds, fuzzy_bounds is
                    # set to None and subsequent bounds checks are skipped
                    if threshold_dict[key] == "None":
                        is_fuzzy = False
                        fuzzy_bounds = None
                    else:
                        fuzzy_bounds.append(tuple(threshold_dict[key]))
        except ValueError as err:
            # Extend error message with hint for common JSON error.
            raise type(err)(
                "{} in threshold dictionary file. \nHINT: Try adding a zero "
                "after the decimal point.".format(err))
        except Exception as err:
            # Extend any errors with message about WHERE this occurred.
            raise type(err)("{} in dictionary file.".format(err))
    else:
        thresholds = threshold_values
        fuzzy_bounds = None

    result_no_collapse_coord = BasicThreshold(
        thresholds, fuzzy_factor=fuzzy_factor,
        fuzzy_bounds=fuzzy_bounds, threshold_units=threshold_units,
        below_thresh_ok=below_threshold).process(cube)

    if vicinity is not None:
        # smooth thresholded occurrences over local vicinity
        result_no_collapse_coord = OccurrenceWithinVicinity(
            vicinity).process(result_no_collapse_coord)
        new_cube_name = in_vicinity_name_format(
            result_no_collapse_coord.name())
        result_no_collapse_coord.rename(new_cube_name)

    if collapse_coord == "None":
        result = result_no_collapse_coord
    else:
        # Raise warning if result_no_collapse_coord is masked array
        if np.ma.isMaskedArray(result_no_collapse_coord.data):
            warnings.warn("Collapse-coord option not fully tested with "
                          "masked data.")
        # Take a weighted mean across realizations with equal weights
        plugin = WeightAndBlend(collapse_coord, "linear",
                                y0val=1.0, ynval=1.0)
        result_collapse_coord = plugin.process(result_no_collapse_coord)
        result = result_collapse_coord
    return result


if __name__ == "__main__":
    main()
