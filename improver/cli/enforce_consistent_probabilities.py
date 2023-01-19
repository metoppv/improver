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
"""CLI to enforce consistent probabilities between two forecasts."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcubelist, ref_name: str = None):
    """Module to enforce consistent probabilities between two forecast
    cubes by lowering the probabilities in the forecast cube to be less than or
    equal to the reference forecast.

    Args:
        cubes (iris.cube.CubeList or list of iris.cube.Cube):
            containing:
                forecast_cube (iris.cube.Cube):
                    Cube of probabilities
                ref_forecast (iris.cube.Cube)
                    Cube of probabilities used as the upper cap for
                    forecast_cube probabilities. It be the same shape as
                    forecast_cube but have a different name.

        ref_name (str):
            Name of ref_forecast cube

    Returns:
        iris.cube.Cube:
            Cube with identical metadata to forecast_cube but with
            lowered probabilities to be less than or equal to the
            reference forecast
    """
    from iris.cube import CubeList

    from improver.calibration.reliability_calibration import (
        EnforceConsistentProbabilities,
    )
    from improver.utilities.flatten import flatten

    cubes = flatten(cubes)

    if len(cubes) != 2:
        raise ValueError(
            f"Exactly two cubes should be provided but received {len(cubes)}"
        )

    ref_forecast = CubeList(cubes).extract_cube(ref_name)
    cubes.remove(ref_forecast)

    forecast_cube = cubes[0]

    plugin = EnforceConsistentProbabilities()

    return plugin(forecast_cube, ref_forecast)
