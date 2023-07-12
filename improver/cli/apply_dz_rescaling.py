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
"""CLI to apply a scaling factor to account for a correction linked to the
difference in altitude between the grid point and the site location."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    forecast: cli.inputcube,
    scaling_factor: cli.inputcube,
    *,
    site_id_coord: str = "wmo_id",
):
    """Apply a scaling factor to account for a correction linked to the difference
    in altitude between the grid point and the site location.

    Args:
        forecast (iris.cube.Cube):
            Percentile forecasts.
        rescaling_cube (iris.cube.Cube):
            Multiplicative scaling factor to adjust the percentile forecasts.
            This cube is expected to contain multiple values for the forecast_period
            and forecast_reference_time_hour coordinates. The most appropriate
            forecast period and forecast reference_time_hour pair within the
            rescaling cube are chosen using the forecast reference time hour from
            the forecast and the nearest forecast period that is greater than or
            equal to the forecast period of the forecast. This cube is generated
            using the estimate_dz_rescaling CLI.
        site_id_coord (str):
            The name of the site ID coordinate. This defaults to 'wmo_id'.

    Returns:
        iris.cube.Cube:
            Percentile forecasts that have been rescaled to account for a difference
            in altitude between the grid point and the site location.
    """

    from improver.calibration.dz_rescaling import ApplyDzRescaling

    return ApplyDzRescaling(site_id_coord=site_id_coord)(forecast, scaling_factor)
