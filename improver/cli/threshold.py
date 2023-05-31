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
"""Script to apply thresholding to a parameter dataset."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube,
    land_sea_mask: cli.inputcube = None,
    *,
    threshold_values: cli.comma_separated_list = None,
    threshold_config: cli.inputjson = None,
    threshold_units: str = None,
    comparison_operator=">",
    fuzzy_factor: float = None,
    collapse_coord: str = None,
    vicinity: cli.comma_separated_list = None,
    fill_masked: float = None,
):
    """Module to apply thresholding to a parameter dataset.

    Calculate the threshold truth values of input data relative to the
    provided threshold value. A fuzzy factor or fuzzy bounds may be provided
    to smooth probabilities where values are close to the threshold.

    Args:
        cube (iris.cube.Cube):
            A cube to be processed.
        threshold_values (list of float):
            Threshold value or values about which to calculate the truth
            values; e.g. 270,300. Must be omitted if 'threshold_config'
            is used.
        threshold_config (dict):
            Threshold configuration containing threshold values and
            (optionally) fuzzy bounds. Best used in combination with
            'threshold_units' It should contain a dictionary of strings that
            can be interpreted as floats with the structure:
            "THRESHOLD_VALUE": [LOWER_BOUND, UPPER_BOUND]
            e.g: {"280.0": [278.0, 282.0], "290.0": [288.0, 292.0]},
            or with structure "THRESHOLD_VALUE": "None" (no fuzzy bounds).
            Repeated thresholds with different bounds are ignored; only the
            last duplicate will be used.
        threshold_units (str):
            Units of the threshold values. If not provided the units are
            assumed to be the same as those of the input cube. Specifying
            the units here will allow a suitable conversion to match
            the input units if possible.
        comparison_operator (str):
            Indicates the comparison_operator to use with the threshold.
            e.g. 'ge' or '>=' to evaluate data >= threshold or '<' to
            evaluate data < threshold. When using fuzzy thresholds, there is
            no difference between < and <= or > and >=.
            Options: > >= < <= gt ge lt le.
        fuzzy_factor (float of None):
            A decimal fraction defining the factor about the threshold value(s)
            which should be treated as fuzzy. Data which fail a test against
            the hard threshold value may return a fractional truth value if
            they fall within this fuzzy factor region.
            Fuzzy factor must be in the range 0-1, with higher values
            indicating a narrower fuzzy factor region / sharper threshold.
            A fuzzy factor cannot be used with a zero threshold or a
            threshold_config file.
        collapse_coord (str):
            An optional ability to set which coordinate we want to collapse
            over.
        vicinity (list of float / int):
            List of distances in metres used to define the vicinities within
            which to search for an occurrence. Each vicinity provided will
            lead to a different gridded field.
        land_sea_mask (Cube):
            Binary land-sea mask data. True for land-points, False for sea.
            Restricts in-vicinity processing to only include points of a
            like mask value.
        fill_masked (float):
            If provided all masked points in cube will be replaced with the
            provided value before thresholding.

    Returns:
        iris.cube.Cube:
            Cube of probabilities relative to the given thresholds

    Raises:
        ValueError: If threshold_config and threshold_values are both set
        ValueError: If threshold_config is used for fuzzy thresholding
    """
    from improver.threshold import BasicThreshold
    from improver.utilities.cube_manipulation import collapse_realizations
    from improver.utilities.spatial import OccurrenceWithinVicinity

    if threshold_config and threshold_values:
        raise ValueError(
            "--threshold-config and --threshold-values are mutually exclusive "
            "- please set one or the other, not both"
        )
    if threshold_config and fuzzy_factor:
        raise ValueError("--threshold-config cannot be used for fuzzy thresholding")
    if threshold_config:
        thresholds = []
        fuzzy_bounds = []
        for key in threshold_config.keys():
            # Ensure thresholds are float64 to avoid rounding errors during
            # possible unit conversion.
            thresholds.append(float(key))
            # If the first threshold has no bounds, fuzzy_bounds is
            # set to None and subsequent bounds checks are skipped
            if threshold_config[key] == "None":
                fuzzy_bounds = None
                continue
            fuzzy_bounds.append(tuple(threshold_config[key]))
    else:
        # Ensure thresholds are float64 to avoid rounding errors during possible
        # unit conversion.
        thresholds = [float(x) for x in threshold_values]
        fuzzy_bounds = None

    each_threshold_func_list = []

    if vicinity is not None:
        vicinity = [float(x) for x in vicinity]
        # smooth thresholded occurrences over local vicinity
        each_threshold_func_list.append(
            OccurrenceWithinVicinity(radii=vicinity, land_mask_cube=land_sea_mask)
        )
    elif land_sea_mask:
        raise ValueError("Cannot apply land-mask cube without in-vicinity processing")

    if collapse_coord == "realization":
        # TODO change collapse_coord argument to boolean "collapse_realizations"
        # (requires suite change)
        each_threshold_func_list.append(collapse_realizations)
    elif collapse_coord is not None:
        raise ValueError("Cannot collapse over non-realization coordinate")

    if fill_masked is not None:
        fill_masked = float(fill_masked)

    return BasicThreshold(
        thresholds,
        fuzzy_factor=fuzzy_factor,
        fuzzy_bounds=fuzzy_bounds,
        threshold_units=threshold_units,
        comparison_operator=comparison_operator,
        each_threshold_func=each_threshold_func_list,
        fill_masked=fill_masked,
    )(cube)
