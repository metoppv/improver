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
"""CLI to enforce consistent percentiles between two forecasts."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcubelist,
    ref_name: str = None,
    min_percentage_exceedance: float = 0.0,
):
    """Enforce consistent percentiles between two forecasts by assuming that the
    reference forecast is equal to, or provides a lower bound, for the percentiles
    within the accompanying forecast.

    Args:
        cubes (iris.cube.CubeList or list of iris.cube.Cube):
            containing:
                ref_forecast (iris.cube.Cube)
                    Cube with percentiles that are equal to, or provide a lower bound,
                    for the accompanying forecast. The reference forecast is expected
                    to be the same shape as forecast_cube but have a different name.
                forecast_cube (iris.cube.Cube):
                    Cube with percentiles that will have consistency enforced.

        ref_name (str):
            Name of ref_forecast cube.
        min_percentage_exceedance (float):
            The minimum percentage by which the reference forecast must be exceeded by
            the accompanying forecast. The reference forecast therefore provides a
            lower bound for the accompanying forecast with this forecast limited to
            a minimum of the reference forecast plus the min_percentage_exceedance
            multiplied by the reference forecast. The percentage is expected as a
            value between 0 and 100.

    Returns:
        iris.cube.Cube:
            Cube with identical metadata to forecast_cube but with
            percentiles that are equal to or exceed the reference forecast.
    """
    from iris.cube import CubeList

    from improver.utilities.enforce_consistency import EnforceConsistentPercentiles
    from improver.utilities.flatten import flatten

    cubes = flatten(cubes)

    if len(cubes) != 2:
        raise ValueError(
            f"Exactly two cubes should be provided but received {len(cubes)}"
        )

    ref_forecast_cube = CubeList(cubes).extract_cube(ref_name)
    cubes.remove(ref_forecast_cube)

    (forecast_cube,) = cubes

    plugin = EnforceConsistentPercentiles(
        min_percentage_exceedance=min_percentage_exceedance
    )

    return plugin(ref_forecast_cube, forecast_cube)
