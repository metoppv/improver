# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2022 Met Office.
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
"""Script to run GenerateClearSkySolarRadiation ancillary generation."""
from datetime import datetime, timezone

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    target_grid: cli.inputcube,
    time: str,
    accumulation_period: int,
    *,
    temporal_spacing: int = 30,
    altitude: cli.inputcube = 0.0,
    linke_turbidity_climatology: cli.inputcube = 3.0,
):
    """Generate clearsky solar radiation data. The clearsky solar radiation is evaluated on the
    target grid for specified time and accumulation period. The clearsky solar radiation data is
    used as an input to the RainForests calibration for rainfall.

    Args:
        target_grid:
            A cube with the desired grid.
        time:
            A datetime specified in the format YYYYMMDDTHHMMZ at which to calculate the
            accumulated clearsky solar radiation.
        accumulation_period:
            The period over which the accumulation is calculated, specified in hours.
        temporal_spacing:
            The spacing between irradiance values used in the evaluation of accumulated
            solar radiation, specified in minutes.
        altitude:
            Altitude data to use in the evaluation of clearsky solar irradiance, which is
            intergated to give the accumulated solar radiation.
        linke_turbidity_climatology:
            Linke turbidity climatology data used in the evaluation of solar irradiance.
            Linke turbidity is a dimensionless value that accounts for relative atmospheric
            scattering of radiation due to aerosols and water vapour.
            The linke turbidity climatology must contain a time dimension that represents
            the day-of-year, from which the associated climatological linke turbidity values
            can be interpolated to for the specified time.

    Returns:
        iris.cube.Cube:
            A cube containing clearsky solar radiation accumulated over the specified
            period, on the same spatial grid as target_grid.
    """
    from improver.generate_ancillaries.generate_derived_solar_fields import (
        GenerateClearskySolarRadiation,
    )

    time = datetime.strptime(time, "%Y%m%dT%H%MZ").replace(tzinfo=timezone.utc)

    return GenerateClearskySolarRadiation()(
        target_grid,
        time,
        accumulation_period,
        temporal_spacing,
        altitude=altitude,
        linke_turbidity_climatology=linke_turbidity_climatology,
    )
