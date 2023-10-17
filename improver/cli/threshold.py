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
        land_sea_mask (Cube):
            Binary land-sea mask data. True for land-points, False for sea.
            Restricts in-vicinity processing to only include points of a
            like mask value.
        threshold_values (list of float):
            Threshold value or values (e.g. 270K, 300K) to use when calculating
            the probability of the input relative to the threshold value(s).
            These are provided as a comma separated list, e.g. 270,300
            The units of these values, e.g. K in the example can be defined
            using the threshold_units argument or are otherwise assumed to
            match the units of the diagnostic being thresholded.
            threshold_values and and threshold_config are mutually exclusive
            arguments, defining both will lead to an exception.
        threshold_config (dict):
            Threshold configuration containing threshold values and
            (optionally) fuzzy bounds. Best used in combination with
            'threshold_units'. It should contain a dictionary of strings that
            can be interpreted as floats with the structure:
            "THRESHOLD_VALUE": [LOWER_BOUND, UPPER_BOUND]
            e.g: {"280.0": [278.0, 282.0], "290.0": [288.0, 292.0]},
            or with structure "THRESHOLD_VALUE": "None" (no fuzzy bounds).
            Repeated thresholds with different bounds are ignored; only the
            last duplicate will be used.
            threshold_values and and threshold_config are mutually exclusive
            arguments, defining both will lead to an exception.
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
            over. The only supported options are "realization" or "percentile".
            If "percentile" is requested, the percentile coordinate will be
            rebadged as a realization coordinate prior to collapse. The percentile
            coordinate needs to be evenly spaced around the 50th percentile
            to allow successful conversion from percentiles to realizations and
            subsequent collapsing over the realization coordinate.
        vicinity (list of float / int):
            List of distances in metres used to define the vicinities within
            which to search for an occurrence. Each vicinity provided will
            lead to a different gridded field.
        fill_masked (float):
            If provided all masked points in cube will be replaced with the
            provided value before thresholding.

    Returns:
        iris.cube.Cube:
            Cube of probabilities relative to the given thresholds
    """
    from improver.threshold import Threshold

    return Threshold(
        threshold_values=threshold_values,
        threshold_config=threshold_config,
        fuzzy_factor=fuzzy_factor,
        threshold_units=threshold_units,
        comparison_operator=comparison_operator,
        collapse_coord=collapse_coord,
        vicinity=vicinity,
        fill_masked=fill_masked,
    )(cube, land_sea_mask)
