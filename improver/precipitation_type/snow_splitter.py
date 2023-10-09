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
"""module to separate snow and rain contributions from a precipitation diagnostic"""

from typing import Tuple

import numpy as np
from iris import Constraint
from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.cube_combiner import Combine
from improver.utilities.cube_checker import assert_spatial_coords_match


class SnowSplitter(BasePlugin):
    """A class to separate the contribution of rain or snow from the precipitation
    rate/accumulation. This is calculated using the probability of rain and snow
    at the surface to determine what fraction of the precipitation rate/accumulation
    is rain or snow.
     """

    def __init__(self, variable: str):
        """
        Sets up Class
        
        Args:
            variable:
                A string of either rain or snow depending on which diagnostic's
                rate/accumulation should be outputted
        """
        self.variable = variable

    @staticmethod
    def separate_input_cubes(cubes: CubeList) -> Tuple[Cube, Cube, Cube]:
        """Separate the input cubelist into cubes of rain and snow.

        Args:
            cubes:
                containing:
                    rain_cube:
                        Cube of the probability of rain at the surface.
                    snow_cube:
                        Cube of the probability of snow at the surface.
                    precip_cube:
                        Cube of either precipitation rate or precipitation accumulation.

        Returns:
            A tuple of rain_cube,snow_cube and precip_cube in that order.
        """

        rain_cube = cubes.extract_cube(
            Constraint(cube_func=lambda cube: "rain" in cube.name())
        )
        snow_cube = cubes.extract_cube(
            Constraint(cube_func=lambda cube: "snow" in cube.name())
        )
        precip_cube = cubes.extract_cube(
            Constraint(cube_func=lambda cube: "precipitation" in cube.name())
        )

        return (rain_cube, snow_cube, precip_cube)

    def process(self, cubes: CubeList,) -> Cube:
        """
        Splits the precipitation cube data into a snow or rain contribution.

        Whether the output is a rate or accumulation will depend on the precipitation_cube.
        self.variable will determine whether the outputted cube is a cube of snow or rain.

        The probability of rain and snow at the surfaces should only contain 1's where the
        precip type is present at the surface and0's where the precip type is not present
        at the surface. These cubes need to be consistent with eachother such that either
        rain or snow is always present at the surface (e.g. at no grid square can both
        diagnostics have a probability of 0).

        A grid of coefficients is calculated by an arbitrary function that maps
        (1,1) -> 0.5, (1,0) -> 1, (0,1) -> 0 where the first coordinate is the probability
        for the variable that will be outputted. This grid of coefficients is then multiplied
        by the precipitation rate/accumulation to split out the contribution of the desired
        variable.

        Args:
            cubes:
                containing:
                    rain_cube:
                        Cube of the probability of rain at the surface.
                    snow_cube:
                        Cube of the probability of snow at the surface.
                    precip_cube:
                        Cube of either precipitation rate or precipitation accumulation.

        Returns:
            Cube of rain/snow (depending on self.variable) rate/accumulation (depending on
            precipitation cube)
        """

        rain_cube, snow_cube, precip_cube = self.separate_input_cubes(cubes)

        assert_spatial_coords_match([rain_cube, snow_cube, precip_cube])

        if self.variable == "snow":
            required_cube = snow_cube
            other_cube = rain_cube
        elif self.variable == "rain":
            required_cube = rain_cube
            other_cube = snow_cube
        else:
            raise ValueError(
                f"""Invalid output variable provided. Provided output variable is {self.variable}.
                    Valid values for output variable are snow or rain"""
            )

        # arbitrary function that maps combinations of rain and snow probabilities
        # to an appropriate coefficient
        coefficient_cube = (required_cube - other_cube + 1) / 2
        coefficient_cube.data = coefficient_cube.data.astype(np.float32)

        new_name = precip_cube.name().replace("precipitation", self.variable)
        output_cube = Combine(operation="*", new_name=new_name)(
            [precip_cube, coefficient_cube]
        )

        return output_cube
