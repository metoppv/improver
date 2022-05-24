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
"""Script to run GenerateClearSkySolarRadiation ancillary generation."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    target_grid: cli.inputcube,
    surface_altitude: cli.inputcube = None,
    linke_turbidity: cli.inputcube = None,
    *,
    time: cli.inputdatetime,
    accumulation_period: int,
    temporal_spacing: int = 30,
):
    """Generate a cube containing clearsky solar radiation data, evaluated on the target grid
    for the specified time and accumulation period. Accumulated clearsky solar radiation is used
    as an input to the RainForests calibration for rainfall.

    Args:
        target_grid (iris.cube.Cube):
            A cube containing the desired spatial grid.
        surface_altitude (iris.cube.Cube):
            Surface altitude data, specified in metres, used in the evaluation of the clearsky
            solar irradiance values. If not provided, a cube with constant value 0.0 m is used,
            created from target_grid.
        linke_turbidity (iris.cube.Cube):
            Linke turbidity data used in the evaluation of the clearsky solar irradiance
            values. Linke turbidity is a dimensionless quantity that accounts for the
            atmospheric scattering of radiation due to aerosols and water vapour, relative
            to a dry atmosphere. If not provided, a cube with constant value 3.0 is used,
            created from target_grid.
        time (str):
            A datetime specified in the format YYYYMMDDTHHMMZ at which to evaluate the
            accumulated clearsky solar radiation. This time is taken to be the end of
            the accumulation period.
        accumulation_period (int):
            The number of hours over which the solar radiation accumulation is defined.
        temporal_spacing (int):
            The time stepping, specified in mins, used in the integration of solar irradiance
            to produce the accumulated solar radiation.

    Returns:
        iris.cube.Cube:
            A cube containing accumulated clearsky solar radiation.
    """
    from improver.generate_ancillaries.generate_derived_solar_fields import (
        GenerateClearskySolarRadiation,
    )

    return GenerateClearskySolarRadiation()(
        target_grid,
        time,
        accumulation_period,
        surface_altitude=surface_altitude,
        linke_turbidity=linke_turbidity,
        temporal_spacing=temporal_spacing,
    )
