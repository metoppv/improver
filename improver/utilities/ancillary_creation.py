# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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

import operator
import warnings

import iris
import numpy as np

from improver import BasePlugin
from improver.constants import TRIPLE_PT_WATER
from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.metadata.utilities import create_new_diagnostic_cube
from improver.utilities.cube_manipulation import enforce_coordinate_ordering
from improver.utilities.spatial import GradientBetweenAdjacentGridSquares


class OrographicSmoothingCoefficients(BasePlugin):

    """
    Class to generate smoothing coefficients for recursive filtering based on
    orography gradients.

    A smoothing coefficient determines how much "value" of a cell
    undergoing filtering is comprised of the current value at that cell and
    how much comes from the adjacent cell preceding it in the direction in
    which filtering is being applied. A larger smoothing_coefficient results in
    a more significant proportion of a cell's new value coming from its
    neighbouring cell.

    The smoothing coefficients are calculated from the orography gradient using
    a simple equation and the user defined values for coefficient and power:

    .. math::
        \\rm{smoothing\\_coefficient} = \\rm{coefficient} \\times
        \\rm{gradient}^{\\rm{power}}

    The resulting values are scaled between min_gradient_smoothing_coefficient and
    max_gradient_smoothing_coefficient to give the desired range of
    smoothing_coefficients. These limiting values must be greater than or equal to
    zero and less than or equal to 0.5.

    Note that the smoothing coefficients are returned on a grid that is one cell
    smaller in the given dimension than the input orography, i.e. the smoothing
    coefficients in the x-direction are returned on a grid that is one cell
    smaller in x than the input. This is because the coefficients are used in
    both forward and backward passes of the recursive filter, so they need to be
    symmetric between cells in the original grid to help ensure conservation.
    """

    def __init__(
        self,
        min_gradient_smoothing_coefficient=0.5,
        max_gradient_smoothing_coefficient=0.0,
        coefficient=1,
        power=1,
        use_mask_boundary=False,
        invert_mask=False,
    ):
        """
        Initialise class.

        Args:
            min_gradient_smoothing_coefficient (float):
                The value of recursive filter smoothing_coefficient to be used
                where the orography gradient is a minimum. Generally this number
                will be larger than the max_gradient_smoothing_coefficient as
                quantities are likely to smoothed more across flat terrain.
            max_gradient_smoothing_coefficient (float):
                The value of recursive filter smoothing_coefficient to be used
                where the orography gradient is a maximum. Generally this number
                will be smaller than the min_gradient_smoothing_coefficient as
                quantities are likely to smoothed less across complex terrain.
            coefficient (float):
                The coefficient for the smoothing_coefficient equation
            power (float):
                What power you want for your smoothing_coefficient equation
            use_mask_boundary (bool):
                A mask can be provided to this plugin to define a region in which
                smoothing coefficients are set to zero, i.e. no smoothing. If this
                option is set to True then rather than the whole masked region
                being set to zero, only the cells that mark the transition from
                masked to unmasked will be set to zero. The primary purpose for
                this is to prevent smoothing across land-sea boundaries.
            invert_mask (bool):
                By default, if a mask is provided and use_mask_boundary is False,
                all the smoothing coefficients corresponding to a mask value of 1
                will be zeroed. Setting invert_mask to True reverses this behaviour
                such that mask values of 0 set the points to be zeroed in the
                smoothing coefficients. If use_mask_boundary is True this option
                has no effect.
        """
        for limit in [
            min_gradient_smoothing_coefficient,
            max_gradient_smoothing_coefficient,
        ]:
            if limit < 0 or limit > 0.5:
                msg = (
                    "min_gradient and max_gradient must be 0 <= value <=0.5, provided "
                    "values are {} and {} respectively".format(
                        min_gradient_smoothing_coefficient,
                        max_gradient_smoothing_coefficient,
                    )
                )
                raise ValueError(msg)

        self.max_gradient_smoothing_coefficient = max_gradient_smoothing_coefficient
        self.min_gradient_smoothing_coefficient = min_gradient_smoothing_coefficient
        self.coefficient = coefficient
        self.power = power
        self.use_mask_boundary = use_mask_boundary
        self.mask_comparison = operator.ge
        if invert_mask:
            self.mask_comparison = operator.le

    def scale_smoothing_coefficients(self, cubes):
        """
        This scales a set of smoothing_coefficients from input cubes to range
        between the min_gradient_smoothing_coefficient and the
        max_gradient_smoothing_coefficient.

        Args:
            cubes (iris.cube.CubeList):
                A list of smoothing_coefficient cubes that we need to take the
                minimum and maximum values from.

        Returns:
            iris.cube.CubeList:
                A list of smoothing_coefficient cubes scaled to within the
                range specified.
        """
        cube_min = min([abs(cube.data).min() for cube in cubes])
        cube_max = max([abs(cube.data).max() for cube in cubes])

        scaled_cubes = iris.cube.CubeList()
        for cube in cubes:
            scaled_data = (abs(cube.data) - cube_min) / (cube_max - cube_min)
            scaled_data = (
                scaled_data
                * (
                    self.max_gradient_smoothing_coefficient
                    - self.min_gradient_smoothing_coefficient
                )
                + self.min_gradient_smoothing_coefficient
            )
            scaled_cube = cube.copy(data=scaled_data)
            scaled_cube.units = "1"
            scaled_cubes.append(scaled_cube)
        return scaled_cubes

    def unnormalised_smoothing_coefficients(self, gradient_cube):
        """
        This generates initial smoothing_coefficient values from gradients
        using a generalised power law, whose parameters are set at
        initialisation. Current defaults give an output
        smoothing_coefficients_cube equal to the input gradient_cube.

        Args:
            gradient_cube (iris.cube.Cube):
                A cube of the normalised gradient

        Returns:
            iris.cube.Cube:
                The cube of initial unscaled smoothing_coefficients
        """
        return self.coefficient * gradient_cube.data ** self.power

    @staticmethod
    def create_coefficient_cube(data, template, cube_name, attributes):
        """
        Update metadata in smoothing_coefficients cube. Remove any time
        coordinates and rename.

        Args:
            data (numpy.array):
                The smoothin coefficient data to store in the cube.
            template (iris.cube.Cube):
                A gradient cube, the dimensions of which are used as a template
                for the coefficient cube.
            cube_name (str):
                A name for the resultant cube
            attributes (dict):
                A dictionary of attributes for the new cube.

        Returns:
            iris.cube.Cube:
                A new cube of smoothing_coefficients
        """
        for coord in template.coords(dim_coords=False):
            for coord_name in ["time", "period", "realization"]:
                if coord_name in coord.name():
                    template.remove_coord(coord)

        attributes["title"] = "Recursive filter smoothing coefficients"
        attributes.pop("history", None)

        return create_new_diagnostic_cube(
            cube_name,
            "1",
            template,
            MANDATORY_ATTRIBUTE_DEFAULTS,
            optional_attributes=attributes,
            data=data,
        )

    def zero_masked(self, smoothing_coefficient_x, smoothing_coefficient_y, mask):
        """
        Zero smoothing coefficients in regions or at boundaries defined by the
        provided mask. The changes are made in place to the input cubes. The
        behaviour is as follows:

            use_mask_boundary = True
            - The edges of the mask region are used to define where smoothing
              coefficients should be zeroed. The zeroed smoothing coefficients
              are between the masked and unmasked cells of the grid on which the
              mask is defined.
            invert_mask = False
            - All smoothing coefficients within regions for which the mask has
              value 1 are set to 0. The boundary cells between masked and
              unmasked are also set to 0. Has no effect if use_mask_boundary=True.
            invert_mask = True
            - All smoothing coefficients within regions for which the mask has
              value 0 are set to 0. The boundary cells between masked and
              unmasked are also set to 0. Has no effect if use_mask_boundary=True.

        Args:
            smoothing_coefficient_x (iris.cube.Cube):
                Smoothing coefficients calculated along the x-dimension.
            smoothing_coefficient_y (iris.cube.Cube):
                Smoothing coefficients calculated along the y-dimension.
            mask (iris.cube.Cube):
                The mask defining areas in which smoothing coefficients should
                be zeroed.
        """
        if self.use_mask_boundary:
            zero_points_x = np.diff(mask.data, axis=1) != 0
            zero_points_y = np.diff(mask.data, axis=0) != 0
        else:
            zero_points_x = self.mask_comparison(
                mask.data[:, :-1] + mask.data[:, 1:], 1
            )
            zero_points_y = self.mask_comparison(
                mask.data[:-1, :] + mask.data[1:, :], 1
            )
        smoothing_coefficient_x.data[zero_points_x] = 0.0
        smoothing_coefficient_y.data[zero_points_y] = 0.0

    def process(self, cube, mask=None):
        """
        This creates the smoothing_coefficient cubes. It returns one for the x
        direction and one for the y direction. It uses the
        DifferenceBetweenAdjacentGridSquares plugin to calculate an average
        gradient across each grid square. These gradients are then used to
        calculate "smoothing_coefficient" arrays that are normalised between a
        user-specified max and min.

        Args:
            cube (iris.cube.Cube):
                A 2D field of orography on the grid for which
                smoothing_coefficients are to be generated.
            mask (iris.cube.Cube or None):
                A mask that defines where the smoothing coefficients should
                be zeroed. How the mask is used is determined by the plugin
                configuration arguments.
        Returns:
            (iris.cube.CubeList): containing:
                **smoothing_coefficient_x** (iris.cube.Cube): A cube of
                    orography-dependent smoothing_coefficients calculated in
                    the x direction.

                **smoothing_coefficient_y** (iris.cube.Cube): A cube of
                    orography-dependent smoothing_coefficients calculated in
                    the y direction.
        """
        if not isinstance(cube, iris.cube.Cube):
            raise ValueError(
                "OrographicSmoothingCoefficients() expects cube "
                "input, got {}".format(type(cube))
            )
        if len(cube.data.shape) != 2:
            raise ValueError(
                "Expected orography on 2D grid, got {} dims".format(
                    len(cube.data.shape)
                )
            )
        if mask is not None and (
            mask.coords(dim_coords=True) != cube.coords(dim_coords=True)
        ):
            raise ValueError(
                "If a mask is provided is must have the same grid as the "
                "orography field."
            )

        # Enforce coordinate order for simpler processing.
        original_order = [crd.name() for crd in cube.coords(dim_coords=True)]
        target_order = [cube.coord(axis="y").name(), cube.coord(axis="x").name()]
        enforce_coordinate_ordering(cube, target_order)

        # Returns two cubes, ordered gradient in x and gradient in y.
        gradients = GradientBetweenAdjacentGridSquares()(cube)

        # Calculate unscaled smoothing coefficients.
        smoothing_coefficients = iris.cube.CubeList()
        iterator = zip(
            gradients, ["smoothing_coefficient_x", "smoothing_coefficient_y"]
        )
        for gradient, name in iterator:
            coefficient_data = self.unnormalised_smoothing_coefficients(gradient)
            smoothing_coefficients.append(
                self.create_coefficient_cube(
                    coefficient_data, gradient, name, cube.attributes.copy()
                )
            )

        # Scale the smoothing coefficients between provided values.
        smoothing_coefficients = self.scale_smoothing_coefficients(
            smoothing_coefficients
        )

        # If a mask has been provided, zero coefficients where required.
        if mask is not None:
            enforce_coordinate_ordering(mask, target_order)
            self.zero_masked(*smoothing_coefficients, mask)

        for smoothing_coefficient in smoothing_coefficients:
            enforce_coordinate_ordering(smoothing_coefficient, original_order)

        return smoothing_coefficients


class SaturatedVapourPressureTable(BasePlugin):

    """
    Plugin to create a saturated vapour pressure lookup table.
    """

    MAX_VALID_TEMPERATURE = 373.0
    MIN_VALID_TEMPERATURE = 173.0

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
        result = (
            "<SaturatedVapourPressureTable: t_min: {}; t_max: {}; "
            "t_increment: {}>".format(self.t_min, self.t_max, self.t_increment)
        )
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
        constants = {
            1: 10.79574,
            2: 5.028,
            3: 1.50475e-4,
            4: -8.2969,
            5: 0.42873e-3,
            6: 4.76955,
            7: 0.78614,
            8: -9.09685,
            9: 3.56654,
            10: 0.87682,
            11: 0.78614,
        }
        triple_pt = TRIPLE_PT_WATER

        # Values for which method is considered valid (see reference).
        # WetBulbTemperature.check_range(temperature.data, 173., 373.)
        if (
            temperature.max() > self.MAX_VALID_TEMPERATURE
            or temperature.min() < self.MIN_VALID_TEMPERATURE
        ):
            msg = "Temperatures out of SVP table range: min {}, max {}"
            warnings.warn(msg.format(temperature.min(), temperature.max()))

        svp = temperature.copy()
        for cell in np.nditer(svp, op_flags=["readwrite"]):
            if cell > triple_pt:
                n0 = constants[1] * (1.0 - triple_pt / cell)
                n1 = constants[2] * np.log10(cell / triple_pt)
                n2 = constants[3] * (
                    1.0 - np.power(10.0, (constants[4] * (cell / triple_pt - 1.0)))
                )
                n3 = constants[5] * (
                    np.power(10.0, (constants[6] * (1.0 - triple_pt / cell))) - 1.0
                )
                log_es = n0 - n1 + n2 + n3 + constants[7]
                cell[...] = np.power(10.0, log_es)
            else:
                n0 = constants[8] * ((triple_pt / cell) - 1.0)
                n1 = constants[9] * np.log10(triple_pt / cell)
                n2 = constants[10] * (1.0 - (cell / triple_pt))
                log_es = n0 - n1 + n2 + constants[11]
                cell[...] = np.power(10.0, log_es)

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
        temperatures = np.arange(
            self.t_min, self.t_max + 0.5 * self.t_increment, self.t_increment
        )
        svp_data = self.saturation_vapour_pressure_goff_gratch(temperatures)

        temperature_coord = iris.coords.DimCoord(
            temperatures, "air_temperature", units="K"
        )

        # Output of the Goff-Gratch is in hPa, but we want to return in Pa.
        svp = iris.cube.Cube(
            svp_data,
            long_name="saturated_vapour_pressure",
            units="hPa",
            dim_coords_and_dims=[(temperature_coord, 0)],
        )
        svp.convert_units("Pa")
        svp.attributes["minimum_temperature"] = self.t_min
        svp.attributes["maximum_temperature"] = self.t_max
        svp.attributes["temperature_increment"] = self.t_increment

        return svp
