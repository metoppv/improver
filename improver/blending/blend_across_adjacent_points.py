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
"""Module containing Blending classes that blend over adjacent points, as
opposed to collapsing the whole dimension."""

import iris
from cf_units import Unit

from improver import PostProcessingPlugin
from improver.blending.weighted_blend import WeightedBlendAcrossWholeDimension
from improver.blending.weights import ChooseDefaultWeightsTriangular
from improver.utilities.cube_checker import check_cube_coordinates


class TriangularWeightedBlendAcrossAdjacentPoints(PostProcessingPlugin):
    """
    Apply a Weighted blend to a coordinate, using triangular weights at each
    point in the coordinate. Returns a cube with the same coordinates as the
    input cube, with each point in the coordinate of interest having been
    blended with the adjacent points according to a triangular weighting
    function of a specified width.
    """

    def __init__(self, coord, central_point, parameter_units, width):
        """Set up for a Weighted Blending plugin

        Args:
            coord (str):
                The name of a coordinate dimension in the cube that we
                will blend over.
            central_point (float or int):
                Central point at which the output from the triangular weighted
                blending will be calculated.
            parameter_units (str):
                The units of the width of the triangular weighting function
                and the units of the central_point.
                This does not need to be the same as the units of the
                coordinate we are blending over, but it should be possible to
                convert between them.
            width (float):
                The width of the triangular weighting function we will use
                to blend.
        """
        self.coord = coord
        self.central_point = central_point
        self.parameter_units = parameter_units
        self.width = width

        # Set up a plugin to calculate the triangular weights.
        self.WeightsPlugin = ChooseDefaultWeightsTriangular(
            width, units=parameter_units)

        # Set up the blending function, based on whether weighted blending or
        # maximum probabilities are needed.
        self.BlendingPlugin = (
            WeightedBlendAcrossWholeDimension(coord, timeblending=True))

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        msg = ('<TriangularWeightedBlendAcrossAdjacentPoints:'
               ' coord = {0:s}, central_point = {1:.2f}, '
               'parameter_units = {2:s}, width = {3:.2f}')
        return msg.format(
            self.coord, self.central_point, self.parameter_units, self.width)

    def _find_central_point(self, cube):
        """
        Find the cube that contains the central point, otherwise, raise
        an exception.

        Args:
            cube (iris.cube.Cube):
                Cube containing input for blending.

        Returns:
            iris.cube.Cube:
                Cube containing central point.

        Raises:
            ValueError: Central point is not available within the input cube.

        """
        # Convert central point into the units of the cube, so that a
        # central point can be extracted.
        central_point = (
            Unit(self.parameter_units).convert(
                self.central_point, cube.coord(self.coord).units))
        constr = iris.Constraint(
            coord_values={self.coord: lambda cell:
                          cell.point == central_point})
        central_point_cube = cube.extract(constr)
        if central_point_cube is None:
            msg = ("The central point {} in units of {} not available "
                   "within input cube coordinate points: {}.".format(
                       self.central_point, self.parameter_units,
                       cube.coord(self.coord).points))
            raise ValueError(msg)
        return central_point_cube

    def process(self, cube):
        """
        Apply the weighted blend for each point in the given coordinate.

        Args:
            cube (iris.cube.Cube):
                Cube containing input for blending.

        Returns:
            iris.cube.Cube:
                The processed cube, with the same coordinates as the input
                central_cube. The points in one coordinate will be blended
                with the adjacent points based on a triangular weighting
                function of the specified width.

        """
        # Extract the central point from the input cube.
        central_point_cube = self._find_central_point(cube)

        # Calculate weights and produce blended output.
        weights = self.WeightsPlugin.process(
            cube, self.coord, self.central_point)
        blended_cube = self.BlendingPlugin(cube, weights)

        # With one threshold dimension (such as for low cloud), the threshold
        # axis is demoted to a scalar co-ordinate by BlendingPlugin. This line
        # promotes threshold to match the dimensions of central_point_cube.
        blended_cube = check_cube_coordinates(central_point_cube, blended_cube)
        blended_cube = central_point_cube.copy(blended_cube.data)
        return blended_cube
