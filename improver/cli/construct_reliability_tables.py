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
"""CLI to construct reliability tables for use in reliability calibration."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube,
            truth_attribute,
            n_probability_bins: int = 5,
            single_value_limits: bool = True):
    """Populate reliability tables for use in reliability calibration.

    Loads historical forecasts and gridded truths that are compared to build
    reliability tables. Reliability tables are returned as a cube with a
    leading threshold dimension that matches that of the forecast probability
    cubes and the thresholded truth.

    Args:
        cubes (list of iris.cube.Cube):
            A list of cubes containing the historical probability forecasts and
            corresponding truths used for calibration. These cubes must include
            the same diagnostic name in their names, and must both have
            equivalent threshold coordinates. The cubes will be distinguished
            using the user provided truth attribute.
        truth_attribute (str):
            An attribute and its value in the format of "attribute=value",
            which must be present on truth cubes.
        n_probability_bins (int):
            The total number of probability bins required in the reliability
            tables. If single value limits are turned on, these are included in
            this total. If using single_value_limits this value must be at
            least 3.
        single_value_limits (bool):
            Mandates that the extrema bins (0 and 1) should be single valued,
            with a small precision tolerance of 1.0E-6, e.g. 0 to 1.0E-6 for
            the lowest bin, and 1 - 1.0E-6 to 1 for the highest bin.

    Returns:
        iris.cube.Cube:
            Reliability tables for the forecast diagnostic with a leading
            threshold coordinate.
    """
    from improver.calibration import split_forecasts_and_truth
    from improver.calibration.reliability_calibration import (
        ConstructReliabilityCalibrationTables)

    forecast, truth, _ = split_forecasts_and_truth(cubes, truth_attribute)

    return ConstructReliabilityCalibrationTables(
        n_probability_bins=n_probability_bins,
        single_value_limits=single_value_limits).process(forecast, truth)
