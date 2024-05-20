# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing Blending classes that blend over adjacent points, as
opposed to collapsing the whole dimension."""

from typing import Union

import iris
from cf_units import Unit
from iris.cube import Cube

from improver import PostProcessingPlugin
from improver.blending.weighted_blend import WeightedBlendAcrossWholeDimension
from improver.blending.weights import ChooseDefaultWeightsTriangular


class TriangularWeightedBlendAcrossAdjacentPoints(PostProcessingPlugin):
    """
    Applies a weighted blend to the data using a triangular weighting function
    at each point in the specified dimension. The maximum weighting is applied
    to the specified point, and weighting decreases linearly for neighbouring
    points to zero at the specified triangle width.

    Returns a cube with the same coordinates as the input cube, with each point
    in the dimension having been blended with the adjacent points according to
    a triangular weighting function of a specified width.
    """

    def __init__(
        self,
        coord: str,
        central_point: Union[int, float],
        parameter_units: str,
        width: float,
    ) -> None:
        """Set up for a Weighted Blending plugin

        Args:
            coord:
                The name of a coordinate dimension in the cube to be blended
                over.
            central_point:
                Central point at which the output from the triangular weighted
                blending will be calculated. This should be in the units of the
                units argument that is passed in. This value should be a point
                on the coordinate for blending over.
            parameter_units:
                The units of the width of the triangular weighting function
                and the units of the central_point.
                This does not need to be the same as the units of the
                coordinate being blending over, but it should be possible to
                convert between them.
            width:
                The width from the triangle’s centre point, in units of the units
                argument, which will determine the triangular weighting function
                used to blend that specified point with its adjacent points. Beyond
                this width the weighting drops to zero.
        """
        self.coord = coord
        self.central_point = central_point
        self.parameter_units = parameter_units
        self.width = width

        # Set up a plugin to calculate the triangular weights.
        self.WeightsPlugin = ChooseDefaultWeightsTriangular(
            width, units=parameter_units
        )

        # Set up the blending function, based on whether weighted blending or
        # maximum probabilities are needed.
        self.BlendingPlugin = WeightedBlendAcrossWholeDimension(
            coord, timeblending=True
        )

    def __repr__(self) -> str:
        """Represent the configured plugin instance as a string."""
        msg = (
            "<TriangularWeightedBlendAcrossAdjacentPoints:"
            " coord = {0:s}, central_point = {1:.2f}, "
            "parameter_units = {2:s}, width = {3:.2f}"
        )
        return msg.format(
            self.coord, self.central_point, self.parameter_units, self.width
        )

    def _find_central_point(self, cube: Cube) -> Cube:
        """
        Find the cube that contains the central point, otherwise, raise
        an exception.

        Args:
            cube:
                Cube containing input for blending.

        Returns:
            Cube containing central point.

        Raises:
            ValueError: Central point is not available within the input cube.
        """
        # Convert central point into the units of the cube, so that a
        # central point can be extracted.
        central_point = Unit(self.parameter_units).convert(
            self.central_point, cube.coord(self.coord).units
        )
        constr = iris.Constraint(
            coord_values={self.coord: lambda cell: cell.point == central_point}
        )
        central_point_cube = cube.extract(constr)
        if central_point_cube is None:
            msg = (
                "The central point {} in units of {} not available "
                "within input cube coordinate points: {}.".format(
                    self.central_point,
                    self.parameter_units,
                    cube.coord(self.coord).points,
                )
            )
            raise ValueError(msg)
        return central_point_cube

    def process(self, cube: Cube) -> Cube:
        """
        Apply the weighted blend for each point in the given dimension.

        Args:
            cube:
                Cube containing input for blending.

        Returns:
            A processed cube, with the same coordinates as the input
            central_cube. The points in one dimension corresponding to
            the specified coordinate will be blended with the adjacent
            points based on a triangular weighting function of the
            specified width.
        """
        # Extract the central point from the input cube.
        central_point_cube = self._find_central_point(cube)

        # Calculate weights and produce blended output.
        weights = self.WeightsPlugin(cube, self.coord, self.central_point)
        blended_cube = self.BlendingPlugin(cube, weights)

        # Copy the metadata of the central cube
        blended_cube = central_point_cube.copy(blended_cube.data)

        return blended_cube
