# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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

import iris
import numpy as np
from improver.psychrometric_calculations.psychrometric_calculations import \
    Utilities
from improver.utilities.spatial import DifferenceBetweenAdjacentGridSquares


class OrographicAlphas(object):

    """
    Class to generate alpha smoothing parameters for recursive filtering
    based on orography gradients.
    """

    def __init__(self, min_alpha=0., max_alpha=1., coefficient=1, power=1,
                 intercept=0, invert_alphas=True):
        """
        Initialise class.

        Args:
            min_alpha : float
                The minimum value of alpha that you want to go into the
                recursive filter.
            max_alpha : float
                The maximum value of alpha that you want to go into the
                recursive filter
            coefficient : float
                The coefficient for the alpha calculation
            intercept : float
                The intercept that you want for your alpha calculation
            power : float
                What power you want for your alpha equation
        """
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha
        self.coefficient = coefficient
        self.power = power
        self.intercept = intercept
        self.invert_alphas = invert_alphas

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<OrographicAlphas: min_alpha: {}; max_alpha: {};'
                  ' coefficient: {}; power: {}; intercept: {}; invert_alphas:'
                  ' {}>'.format(self.min_alpha, self.max_alpha,
                                self.coefficient, self.power, self.intercept,
                                self.invert_alphas))

        return result

    def normalise_cube(self, cubes, min_output_value=0, max_output_value=1):
        """
        This normalises a cube so that all of the numbers are between
        min and max value which can be set.

        Args:
            cubes : iris.cube.Cubelist
                A list of cubes that we need to take the cube_max and
                cube_min from.
            min_output_value : float
                The minimum value we want our alpha to be
            max_output_value : float
                The maximum value we want our alpha to be

        Returns:
            normalised_cubes : iris.cube.Cube
                A normalised cube based on the orography
        """
        cube_min = min([cube.data.min() for cube in cubes])
        cube_max = max([cube.data.max() for cube in cubes])

        normalised_cubes = iris.cube.CubeList()
        for cube in cubes:
            normalised_cube = cube.copy(data=(cube.data - cube_min) /
                                        (cube_max - cube_min))
            normalised_cube.data = (normalised_cube.data * (max_output_value
                                    - min_output_value) + min_output_value)
            normalised_cubes.append(normalised_cube)
        return normalised_cubes

    def scale_alpha_values(self, difference_cube):
        """
        This scales the alpha values depending on our equation
        for alpha.

        Args:
            difference_cube : iris.cube.Cube
                A cube of the normalised gradient

        Returns:
            difference_cube : iris.cube.Cube
                The scaled cube of normalised gradient
        """
        difference_cube.data = (
            self.coefficient * difference_cube.data**self.power +
            self.intercept)

        return difference_cube

    def process(self, cube):
        """
        This creates the alpha cubes. It returns one for the x direction and
        one for the y direction. It uses the
        DifferencBetweenAdjacentGridSquares plugin to get the height
        difference between grid spaces and then calculates a gradient,
        which is normalised between between numbers (which can be chosen).
        The gradients are then linearly regridded so that they match the
        orography dimensions and will go into the recursive filter.

        Args:
            cube: iris.cube.Cube
                A cube of the orography for the grid we want to get alphas for.

        Returns:
            alpha_x : iris.cube.Cube
               A cube of orographic dependent alphas calculated in the x
               direction.
            alpha_y : iris.cube.Cube
               A cube of orographic dependent alphas calculated in the y
               direction.
        """
        gradient_x, gradient_y = \
            DifferenceBetweenAdjacentGridSquares().process(cube, gradient=True)
        alpha_x = self.scale_alpha_values(gradient_x)
        alpha_y = self.scale_alpha_values(gradient_y)

        if self.invert_alphas is True:
            alpha_x, alpha_y = self.normalise_cube(
                [alpha_x, alpha_y], min_output_value=self.max_alpha,
                max_output_value=self.min_alpha)
        else:
            alpha_x, alpha_y = self.normalise_cube(
                [alpha_x, alpha_y], min_output_value=self.min_alpha,
                max_output_value=self.max_alpha)

        return alpha_x, alpha_y


class SaturatedVapourPressureTable(object):

    """
    Plugin to create a saturated vapour pressure lookup table.
    """

    def __init__(self, t_min=183.15, t_max=338.15, t_increment=0.1):
        """
        Initialise class.

        Args:
            t_min : float
                The minimum temperature for the range.
            t_max : float
                The maximum temperature for the range.
            t_increment : float
                The temperature increment at which to create values for the
                saturated vapour pressure between t_min and t_max.
        """
        self.t_min = t_min
        self.t_max = t_max
        self.t_increment = t_increment

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<SaturatedVapourPressureTable: t_min: {}; t_max: {}; '
                  't_increment: {}>'.format(self.t_min, self.t_max,
                                            self.t_increment))
        return result

    def process(self):
        """
        Create a saturated vapour pressure lookup table by calling the
        Utilities.saturation_vapour_pressure_goff_gratch function in
        psychrometric_calculations.Utilities.

        Returns:
            svp : iris.cube.Cube
               A cube of saturated vapour pressure values at temperature
               points defined by t_min, t_max, and t_increment (defined above).
        """
        temperatures = np.arange(self.t_min, self.t_max + 0.5*self.t_increment,
                                 self.t_increment)
        temperature = iris.cube.Cube(temperatures, 'air_temperature',
                                     units='K')

        svp = Utilities.saturation_vapour_pressure_goff_gratch(temperature)

        temperature_coord = iris.coords.DimCoord(
            temperature.data, 'air_temperature', units='K')

        svp.add_dim_coord(temperature_coord, 0)
        svp.attributes['minimum_temperature'] = self.t_min
        svp.attributes['maximum_temperature'] = self.t_max
        svp.attributes['temperature_increment'] = self.t_increment

        return svp
