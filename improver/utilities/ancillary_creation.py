# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
"""A module for creating ancillary data"""

import warnings
import numpy as np

import iris

from improver import BasePlugin
from improver.constants import TRIPLE_PT_WATER
from improver.utilities.spatial import DifferenceBetweenAdjacentGridSquares


class OrographicAlphas(BasePlugin):

    """
    Class to generate alpha smoothing parameters for recursive filtering
    based on orography gradients.
    """

    def __init__(self, min_alpha=0., max_alpha=1., coefficient=1, power=1,
                 invert_alphas=True):
        """
        Initialise class.

        Args:
            min_alpha (float):
                The minimum value of alpha that you want to go into the
                recursive filter.
            max_alpha (float):
                The maximum value of alpha that you want to go into the
                recursive filter
            coefficient (float):
                The coefficient for the alpha calculation
            power (float):
                What power you want for your alpha equation
        """
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha
        self.coefficient = coefficient
        self.power = power
        self.invert_alphas = invert_alphas

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<OrographicAlphas: min_alpha: {}; max_alpha: {};'
                  ' coefficient: {}; power: {}; invert_alphas:'
                  ' {}>'.format(self.min_alpha, self.max_alpha,
                                self.coefficient, self.power,
                                self.invert_alphas))

        return result

    @staticmethod
    def scale_alphas(cubes, min_output=0, max_output=1):
        """
        This scales a set of alphas from input cubes to range between the
        minimum and maximum alpha values.

        Args:
            cubes (iris.cube.CubeList):
                A list of alpha cubes that we need to take the cube_max and
                cube_min from.
            min_output (float):
                The minimum value we want our alpha to be
            max_output (float):
                The maximum value we want our alpha to be

        Returns:
            iris.cube.CubeList:
                A list of alpha cubes scaled to within the range specified.
        """
        cube_min = min([abs(cube.data).min() for cube in cubes])
        cube_max = max([abs(cube.data).max() for cube in cubes])

        scaled_cubes = iris.cube.CubeList()
        for cube in cubes:
            scaled_data = (abs(cube.data) - cube_min) / (cube_max - cube_min)
            scaled_data = scaled_data * (max_output - min_output) + min_output
            scaled_cube = cube.copy(data=scaled_data)
            scaled_cube.units = '1'
            scaled_cubes.append(scaled_cube)
        return scaled_cubes

    def unnormalised_alphas(self, gradient_cube):
        """
        This generates initial alpha values from gradients using a generalised
        power law, whose parameters are set at initialisation.  Current
        defaults give an output alphas_cube equal to the input gradient_cube.

        Args:
            gradient_cube (iris.cube.Cube):
                A cube of the normalised gradient

        Returns:
            iris.cube.Cube:
                The cube of initial unscaled alphas
        """
        alphas_cube = gradient_cube.copy(data=self.coefficient *
                                         gradient_cube.data**self.power)
        return alphas_cube

    @staticmethod
    def update_alphas_metadata(alphas_cube, cube_name):
        """
        Update metadata in alphas cube.  Remove any time coordinates and
        rename.

        Args:
            alphas_cube (iris.cube.Cube):
                A cube of alphas with "gradient" metadata
            cube_name (str):
                A name for the resultant cube

        Returns:
            iris.cube.Cube:
                A cube of alphas with adjusted metadata
        """
        alphas_cube.rename(cube_name)
        for coord in alphas_cube.coords(dim_coords=False):
            if 'time' in coord.name() or 'period' in coord.name():
                alphas_cube.remove_coord(coord)
        return alphas_cube

    def gradient_to_alpha(self, gradient_x, gradient_y):
        """
        Generate alpha smoothing parameters from orography gradients in the
        x- and y- directions

        Args:
            gradient_x (iris.cube.Cube):
                A cube of the normalised gradient in the x direction
            gradient_y (iris.cube.Cube):
                A cube of the normalised gradient in the y direction

        Returns:
            (tuple): tuple containing:
                **alpha_x** (iris.cube.Cube): A cube of orography-dependent
                    alphas calculated in the x direction.

                **alpha_y** (iris.cube.Cube): A cube of orography-dependent
                    alphas calculated in the y direction.
        """
        alpha_x = self.unnormalised_alphas(gradient_x)
        alpha_y = self.unnormalised_alphas(gradient_y)

        if self.invert_alphas:
            alpha_x, alpha_y = self.scale_alphas([alpha_x, alpha_y],
                                                 min_output=self.max_alpha,
                                                 max_output=self.min_alpha)
        else:
            alpha_x, alpha_y = self.scale_alphas([alpha_x, alpha_y],
                                                 min_output=self.min_alpha,
                                                 max_output=self.max_alpha)
        alpha_x = self.update_alphas_metadata(alpha_x, 'alpha_x')
        alpha_y = self.update_alphas_metadata(alpha_y, 'alpha_y')

        return alpha_x, alpha_y

    def process(self, cube):
        """
        This creates the alpha cubes. It returns one for the x direction and
        one for the y direction. It uses the
        DifferenceBetweenAdjacentGridSquares plugin to calculate an average
        gradient across each grid square.  These gradients are then used to
        calculate "alpha" smoothing arrays that are normalised between a
        user-specified max and min.

        Args:
            cube (iris.cube.Cube):
                A 2D cube of the orography for the grid we want to get alphas
                for.

        Returns:
            (iris.cube.CubeList): containing:
                **alpha_x** (iris.cube.Cube): A cube of orography-dependent
                    alphas calculated in the x direction.

                **alpha_y** (iris.cube.Cube): A cube of orography-dependent
                    alphas calculated in the y direction.
        """
        if not isinstance(cube, iris.cube.Cube):
            raise ValueError('OrographicAlphas() expects cube input, got {}'
                             .format(type(cube)))

        if len(cube.data.shape) != 2:
            raise ValueError('Expected orography on 2D grid, got {} dims'
                             .format(len(cube.data.shape)))

        gradient_x, gradient_y = \
            DifferenceBetweenAdjacentGridSquares(gradient=True).process(cube)
        alpha_x, alpha_y = self.gradient_to_alpha(gradient_x, gradient_y)

        return iris.cube.CubeList([alpha_x, alpha_y])


class SaturatedVapourPressureTable(BasePlugin):

    """
    Plugin to create a saturated vapour pressure lookup table.
    """
    MAX_VALID_TEMPERATURE = 373.
    MIN_VALID_TEMPERATURE = 173.

    def __init__(self, t_min=183.15, t_max=338.25, t_increment=0.1):
        """
        Create a table of saturated vapour pressures that can be interpolated
        through to obtain an SVP value for any temperature within the range
        t_min --> (t_max - t_increment).

        The default min/max values create a table that provides SVP values
        covering the temperature range -90C to +65.1C. Note that the last
        bin is not used, so the SVP value corresponding to +65C is the highest
        that will be used.

        Args:
            t_min (float):
                The minimum temperature for the range, in Kelvin.
            t_max (float):
                The maximum temperature for the range, in Kelvin.
            t_increment (float):
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

    def saturation_vapour_pressure_goff_gratch(self, temperature):
        """
        Saturation Vapour pressure in a water vapour system calculated using
        the Goff-Gratch Equation (WMO standard method).

        Args:
            temperature (numpy.ndarray):
                Temperature values in Kelvin. Valid from 173K to 373K

        Returns:
            numpy.ndarray:
                Corresponding values of saturation vapour pressure for a pure
                water vapour system, in hPa.

        References:
            Numerical data and functional relationships in science and
            technology. New series. Group V. Volume 4. Meteorology.
            Subvolume b. Physical and chemical properties of the air, P35.
        """
        constants = {1: 10.79574,
                     2: 5.028,
                     3: 1.50475E-4,
                     4: -8.2969,
                     5: 0.42873E-3,
                     6: 4.76955,
                     7: 0.78614,
                     8: -9.09685,
                     9: 3.56654,
                     10: 0.87682,
                     11: 0.78614}
        triple_pt = TRIPLE_PT_WATER

        # Values for which method is considered valid (see reference).
        # WetBulbTemperature.check_range(temperature.data, 173., 373.)
        if (temperature.max() > self.MAX_VALID_TEMPERATURE or
                temperature.min() < self.MIN_VALID_TEMPERATURE):
            msg = "Temperatures out of SVP table range: min {}, max {}"
            warnings.warn(msg.format(temperature.min(),
                                     temperature.max()))

        svp = temperature.copy()
        for cell in np.nditer(svp, op_flags=['readwrite']):
            if cell > triple_pt:
                n0 = constants[1] * (1. - triple_pt / cell)
                n1 = constants[2] * np.log10(cell / triple_pt)
                n2 = constants[3] * (1. - np.power(10.,
                                                   (constants[4] *
                                                    (cell / triple_pt - 1.))))
                n3 = constants[5] * (np.power(10., (constants[6] *
                                                    (1. - triple_pt / cell))) -
                                     1.)
                log_es = n0 - n1 + n2 + n3 + constants[7]
                cell[...] = (np.power(10., log_es))
            else:
                n0 = constants[8] * ((triple_pt / cell) - 1.)
                n1 = constants[9] * np.log10(triple_pt / cell)
                n2 = constants[10] * (1. - (cell / triple_pt))
                log_es = n0 - n1 + n2 + constants[11]
                cell[...] = (np.power(10., log_es))

        return svp

    def process(self):
        """
        Create a lookup table of saturation vapour pressure in a pure water
        vapour system for the range of required temperatures.

        Returns:
            iris.cube.Cube:
               A cube of saturated vapour pressure values at temperature
               points defined by t_min, t_max, and t_increment (defined above).
        """
        temperatures = np.arange(self.t_min, self.t_max + 0.5*self.t_increment,
                                 self.t_increment)
        svp_data = self.saturation_vapour_pressure_goff_gratch(temperatures)

        temperature_coord = iris.coords.DimCoord(
            temperatures, 'air_temperature', units='K')

        # Output of the Goff-Gratch is in hPa, but we want to return in Pa.
        svp = iris.cube.Cube(
            svp_data, long_name='saturated_vapour_pressure', units='hPa',
            dim_coords_and_dims=[(temperature_coord, 0)])
        svp.convert_units('Pa')
        svp.attributes['minimum_temperature'] = self.t_min
        svp.attributes['maximum_temperature'] = self.t_max
        svp.attributes['temperature_increment'] = self.t_increment

        return svp
