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
"""CLI to apply simple bias correction to ensemble members based on bias from the
reference forecast dataset."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    forecast_cube: cli.inputcube, *bias_cubes: cli.inputcube, lower_bound: float = None
):
    """Apply simple bias correction to ensemble members based on the bias from the
    reference forecast dataset.

    The bias cube can either be passed in as a series of bias values for individual
    forecasts (from which the mean value is evaluated), or as a single bias value
    evaluated over a series of reference forecasts.

    A lower bound can be set to ensure that corrected values are physically sensible
    post-bias correction.

    Args:
        forecast_cube (iris.cube.Cube):
            Cube containing the forecast to apply bias correction to.
        bias_cubes (iris.cube.Cube or list of iris.cube.Cube):
            A cube or list of cubes containing forecast bias data over the a specified
            set of forecast reference times. If a list of cubes is passed in, each cube
            should represent the forecast error for a single forecast reference time; the
            mean value will then be evaluated over the forecast_reference_time coordinate.
        lower_bound (float):
            Specifies a lower bound below which values will be remapped to.

    Returns:
        iris.cube.Cube:
            Forecast cube with bias correction applied on a per member basis.
    """
    import iris

    from improver.calibration.simple_bias_correction import ApplyBiasCorrection

    # Check whether bias data supplied, if not then return unadjusted input cube.
    if not bias_cubes:
        return forecast_cube
    else:
        bias_cubes = iris.cube.CubeList(bias_cubes)
        plugin = ApplyBiasCorrection()
        return plugin.process(forecast_cube, bias_cubes, lower_bound)
