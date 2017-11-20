# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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

from improver.blending.weights import ChooseDefaultWeightsTriangular
from improver.utilities.cube_manipulation import concatenate_cubes
from improver.blending.weighted_blend import WeightedBlendAcrossWholeDimension


class TriangularWeightedBlendAcrossAdjacentPoints(object):
    """
    Apply a Weighted blend to a coordinate, using triangular weights at each
    point in the coordinate. Returns a cube with the same coordinates as the
    input cube, with each point in the coordinate of interest having been
    blended with the adjacent points according to a triangular weighting
    function of a specified width.

    There are two modes of blending:

        1. Weighted mean across the dimension of interest.
        2. Weighted maximum across the dimension of interest, where
           probabilities are multiplied by the weights and the maximum is
           taken.
    """

    def __init__(self, coord, width, parameter_units, weighting_mode):
        """Set up for a Weighted Blending plugin

        Args:
            coord (string):
                The name of a coordinate dimension in the cube that we
                will blend over.
            width (float):
                The width of the triangular weighting function we will use
                to blend.
            parameter_units (string):
                The units of the width of the triangular weighting function.
                This does not need to be the same as the units of the
                coordinate we are blending over, but it should be possible to
                convert between them.
            weighting_mode (string):
                The mode of blending, either weighted_mean or
                weighted_maximum. Weighted average finds the weighted mean
                across the dimension of interest. Maximum probability
                multiplies the values across the dimension of interest by the
                given weights and returns the maximum value.
        Raises:
            ValueError : If an invalid weighting_mode is given
        """
        self.coord = coord
        self.width = width
        self.parameter_units = parameter_units
        if weighting_mode not in ['weighted_maximum', 'weighted_mean']:
            msg = ("weighting_mode: {} is not recognised, must be either "
                   "weighted_maximum or weighted_mean").format(weighting_mode)
            raise ValueError(msg)
        self.mode = weighting_mode

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        msg = ('<TriangularWeightedBlendAcrossAdjacentPoints:'
               ' coord = {0:s}, width = {1:.2f},'
               ' parameter_units = {2:s}, mode = {3:s}>')
        return msg.format(self.coord, self.width, self.parameter_units,
                          self.mode)

    @staticmethod
    def correct_collapsed_coordinates(orig_cube, new_cube, coords_to_correct):
        """
        A helper function to replace the points and bounds in coordinates
        that have been collapsed.
        For the coordinates specified it replaces points in new_cube's
        coordinates with the points from the corresponding coordinate in
        orig_cube. The bounds are also replaced.

        Args:
            orig_cube(iris.cube.Cube):
                The cube that the original coordinates points will be taken
                from.
            new_cube(iris.cube.Cube):
                The new cube who's coordinates will be corrected. This must
                have the same number of points along the coordinates we are
                correcting as are in the orig_cube.
            coords_to_correct(list):
                A list of coordinate names to correct.
        """
        for coord in coords_to_correct:
            new_coord = new_cube.coord(coord)
            old_coord = orig_cube.coord(coord)
            new_coord.points = old_coord.points
            if old_coord.bounds is not None:
                new_coord.bounds = old_coord.bounds

    def process(self, cube):
        """
        Apply the weighted blend for each point in the given coordinate.

        Args:
            cube (iris.cube.Cube):
                Cube to blend.

        Returns:
            cube (iris.cube.Cube):
                The processed cube, with the same coordinates as the input
                cube. The points in one coordinate will be blended with the
                adjacent points based on a triangular weighting function of the
                specified width.

        """
        # We need to correct all the coordinates associated with the dimension
        # we are collapsing over, so find the relevant coordinates now.
        dimension_to_collapse = cube.coord_dims(self.coord)
        coords_to_correct = cube.coords(dimensions=dimension_to_collapse)
        coords_to_correct = [coord.name() for coord in coords_to_correct]
        # We will also need to correct the bounds on these coordinates,
        # as bounds will be added when the blending happens, so add bounds if
        # it doesn't have some already.
        for coord in coords_to_correct:
            if not cube.coord(coord).has_bounds():
                cube.coord(coord).guess_bounds()
        # Set up a plugin to calculate the triangular weights.
        WeightsPlugin = ChooseDefaultWeightsTriangular(
            self.width, units=self.parameter_units)
        # Set up the blending function, based on whether weighted blending or
        # maximum probabilities are needed.
        BlendingPlugin = WeightedBlendAcrossWholeDimension(self.coord,
                                                           self.mode)
        result = iris.cube.CubeList([])
        # Loop over each point in the coordinate we are blending over, and
        # calculate a new weighted average for it.
        for cube_slice in cube.slices_over(self.coord):
            point = cube_slice.coord(self.coord).points[0]
            weights = WeightsPlugin.process(cube, self.coord, point)
            blended_cube = BlendingPlugin.process(cube, weights)
            self.correct_collapsed_coordinates(cube_slice, blended_cube,
                                               coords_to_correct)
            result.append(blended_cube)
        result = concatenate_cubes(result)
        return result
