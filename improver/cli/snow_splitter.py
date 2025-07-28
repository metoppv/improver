# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI for SnowSplitter"""

from improver import cli


@cli.clizefy
@cli.with_output
def process(*cubes: cli.inputcube, output_is_rain: bool):
    """
    Separates the snow/rain contribution from precipitation rate/accumulation.

    Whether the output is a rate or accumulation will depend on the precipitation_cube.
    output_is_rain determines whether the outputted cube is a cube of snow or rain.

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
            Cube of rain/snow (depending on self.output_is_rain) rate/accumulation (depending
            on precipitation cube)

    """
    from improver.precipitation.snow_splitter import SnowSplitter

    return SnowSplitter(output_is_rain=output_is_rain)(*cubes)
