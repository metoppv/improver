
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
"""CLI for SnowSplitter"""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube, output_is_rain: bool):
    """
    Separates the snow/rain contribution from precipitation rate/accumulation.

    Whether the output is a rate or accumulation will depend on the precipitation_cube.
    Variable determines whether the outputted cube is a cube of snow or rain.

    The probability of rain and snow at the surfaces should only contain 1's where the
    precip type is present at the surface and 0's where the precip type is not present
    at the surface. These cubes need to be consistent with each other such that both
    rain and snow can't be present at the surface (e.g. at no grid square can both
    diagnostics have a probability of 1).

    A grid of coefficients is calculated by an arbitrary function that maps
    (0,0) -> 0.5, (1,0) -> 1, (0,1) -> 0 where the first coordinate is the probability
    for the variable that will be outputted. This grid of coefficients is then multiplied
    by the precipitation rate/accumulation to split out the contribution of the desired
    variable.

    Args:
        cubes (iris.cube.CubeList or list):
            Contains cubes of probability of rain at surface, probability of snow at
            surface and precipitation rate/accumulation.
        output_is_rain (bool):
            A boolean where True means the plugin will output rain and False means the
            output is snow.

    Returns:
        iris.cube.Cube:
            Cube of rain/snow (depending on self.output_is-rain) rate/accumulation (depending
            on precipitation cube)

    """
    from iris.cube import CubeList

    from improver.precipitation_type.snow_splitter import SnowSplitter

    return SnowSplitter(output_is_rain=output_is_rain)(CubeList(cubes))
