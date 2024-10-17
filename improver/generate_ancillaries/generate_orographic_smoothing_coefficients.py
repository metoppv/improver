# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""A module for creating orographic smoothing coefficients"""

import operator
from typing import Dict, Optional

import iris
import numpy as np
from iris.cube import Cube, CubeList
from numpy import ndarray

from improver import BasePlugin
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
    a simple equation with the user defined value for the power:

    .. math::
        \\rm{smoothing\\_coefficient} = \\rm{gradient}^{\\rm{power}}

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
        min_gradient_smoothing_coefficient: float = 0.5,
        max_gradient_smoothing_coefficient: float = 0.0,
        power: float = 1,
        use_mask_boundary: bool = False,
        invert_mask: bool = False,
    ) -> None:
        """
        Initialise class.

        Args:
            min_gradient_smoothing_coefficient:
                The value of recursive filter smoothing_coefficient to be used
                where the orography gradient is a minimum. Generally this number
                will be larger than the max_gradient_smoothing_coefficient as
                quantities are likely to be smoothed more across flat terrain.
            max_gradient_smoothing_coefficient:
                The value of recursive filter smoothing_coefficient to be used
                where the orography gradient is a maximum. Generally this number
                will be smaller than the min_gradient_smoothing_coefficient as
                quantities are likely to be smoothed less across complex terrain.
            power:
                The power to be used in the smoothing_coefficient equation
            use_mask_boundary:
                A mask can be provided to this plugin to define a region in which
                smoothing coefficients are set to zero, i.e. no smoothing. If this
                option is set to True then rather than the whole masked region
                being set to zero, only the cells that mark the transition from
                masked to unmasked will be set to zero. The primary purpose for
                this is to prevent smoothing across land-sea boundaries.
            invert_mask:
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
                    "min_gradient_smoothing_coefficient and max_gradient_smoothing_coefficient "
                    "must be 0 <= value <=0.5 to help ensure better conservation across the "
                    "whole field to which the recursive filter is applied. The values provided "
                    "are {} and {} respectively".format(
                        min_gradient_smoothing_coefficient,
                        max_gradient_smoothing_coefficient,
                    )
                )
                raise ValueError(msg)

        self.max_gradient_smoothing_coefficient = max_gradient_smoothing_coefficient
        self.min_gradient_smoothing_coefficient = min_gradient_smoothing_coefficient
        self.power = power
        self.use_mask_boundary = use_mask_boundary
        self.mask_comparison = operator.ge
        if invert_mask:
            self.mask_comparison = operator.le

    def scale_smoothing_coefficients(self, cubes: CubeList) -> CubeList:
        """
        This scales a set of smoothing_coefficients from input cubes to range
        between the min_gradient_smoothing_coefficient and the
        max_gradient_smoothing_coefficient.

        Args:
            cubes:
                A list of smoothing_coefficient cubes that we need to take the
                minimum and maximum values from.

        Returns:
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

    def unnormalised_smoothing_coefficients(self, gradient_cube: Cube) -> ndarray:
        """
        This generates initial smoothing_coefficient values from gradients
        using a simple power law, for which the power is set at initialisation.
        Using a power of 1 gives an output smoothing_coefficients_cube with
        values equal to the input gradient_cube.

        Args:
            gradient_cube:
                A cube of the normalised gradient

        Returns:
            An array containing the unscaled smoothing_coefficients.
        """
        return np.abs(gradient_cube.data) ** self.power

    def create_coefficient_cube(
        self, data: ndarray, template: Cube, cube_name: str, attributes: Dict
    ) -> Cube:
        """
        Update metadata in smoothing_coefficients cube. Remove any time
        coordinates and rename.

        Args:
            data:
                The smoothing coefficient data to store in the cube.
            template:
                A gradient cube, the dimensions of which are used as a template
                for the coefficient cube.
            cube_name:
                A name for the resultant cube
            attributes:
                A dictionary of attributes for the new cube.

        Returns:
            A new cube of smoothing_coefficients
        """
        for coord in template.coords(dim_coords=False):
            for coord_name in ["time", "period", "realization"]:
                if coord_name in coord.name():
                    template.remove_coord(coord)

        attributes["title"] = "Recursive filter smoothing coefficients"
        attributes.pop("history", None)
        attributes["power"] = self.power

        return create_new_diagnostic_cube(
            cube_name,
            "1",
            template,
            MANDATORY_ATTRIBUTE_DEFAULTS.copy(),
            optional_attributes=attributes,
            data=data,
        )

    def zero_masked(
        self, smoothing_coefficient_x: Cube, smoothing_coefficient_y: Cube, mask: Cube
    ) -> None:
        """
        Zero smoothing coefficients in regions or at boundaries defined by the
        provided mask. The changes are made in place to the input cubes. The
        behaviour is as follows:

            use_mask_boundary = True:
              The edges of the mask region are used to define where smoothing
              coefficients should be zeroed. The zeroed smoothing coefficients
              are between the masked and unmasked cells of the grid on which the
              mask is defined.

            invert_mask = False:
              All smoothing coefficients within regions for which the mask has
              value 1 are set to 0. The boundary cells between masked and
              unmasked are also set to 0. Has no effect if use_mask_boundary=True.

            invert_mask = True:
              All smoothing coefficients within regions for which the mask has
              value 0 are set to 0. The boundary cells between masked and
              unmasked are also set to 0. Has no effect if use_mask_boundary=True.

        Args:
            smoothing_coefficient_x:
                Smoothing coefficients calculated along the x-dimension.
            smoothing_coefficient_y:
                Smoothing coefficients calculated along the y-dimension.
            mask:
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

    def process(self, cube: Cube, mask: Optional[Cube] = None) -> CubeList:
        """
        This creates the smoothing_coefficient cubes. It returns one for the x
        direction and one for the y direction. It uses the
        GradientBetweenAdjacentGridSquares plugin to calculate an average
        gradient across each grid square. These gradients are then used to
        calculate "smoothing_coefficient" arrays that are normalised between a
        user-specified max and min.

        Args:
            cube:
                A 2D field of orography on the grid for which
                smoothing_coefficients are to be generated.
            mask:
                A mask that defines where the smoothing coefficients should
                be zeroed. The mask must have the same spatial dimensions as
                the orography cube. How the mask is used to zero smoothing
                coefficients is determined by the plugin configuration arguments.

        Returns:
            - A cube of orography-dependent smoothing_coefficients calculated in
              the x direction.
            - A cube of orography-dependent smoothing_coefficients calculated in
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
                "If a mask is provided it must have the same grid as the "
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
