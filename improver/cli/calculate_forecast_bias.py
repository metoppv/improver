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
"""CLI to calculate the bias values from the specified set of reference forecasts."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube, truth_attribute: str):
    """Calculate forecast bias from the specified set of historical forecasts and truth
    values.

    The historical forecasts are expected to be representative single-valued forecasts
    (eg. control or ensemble mean forecast).

    The bias values are evaluated point-by-point and the associated bias cube
    will retain the same spatial dimensions as the input cubes. By using a
    point-by-point approach, the bias-correction enables a form of statistical
    downscaling where coherent biases exist between a coarse forecast dataset and
    finer truth dataset.

    Where multiple forecasts values are provided, the value returned is the mean value
    over the set of forecast/truth pairs.

    Args:
        cubes (list of iris.cube.Cube):
            A list of cubes containing the historical forecasts and corresponding
            truths used for calibration. The cubes must include the same diagnostic
            name in their names. The cubes will be distinguished using the user
            specified truth attribute.
        truth_attribute (str):
            An attribute and its value in the format of "attribute=value",
            which must be present on truth cubes.

    Returns:
        iris.cube.Cube:
            Cube containing forecast bias values evaluated over the specified set
            of historical forecasts.
    """
    from improver.calibration import split_forecasts_and_truth
    from improver.calibration.simple_bias_correction import CalculateForecastBias

    historical_forecast, historical_truth, _ = split_forecasts_and_truth(
        cubes, truth_attribute
    )
    plugin = CalculateForecastBias()
    return plugin(historical_forecast, historical_truth)
