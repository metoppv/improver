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
"""Module to contain mathematical operations."""

import numpy as np

import iris

from improver.utilities.cube_manipulation import sort_coord_in_cube


class Integration(object):
    """Perform integration along a chosen coordinate."""

    def __init__(self, coord_name_to_integrate,
                 start_point=None, end_point=None,
                 direction_of_integration="negative"):
        """
        Initialise class.

        Args:
            coord_name_to_integrate (iris.cube.Cube):
                Name of the coordinate to be integrated.
            start_point (float or None):
                Point at which to start the interpolation.
                Default is None.
            end_point (float or None):
                Point at which to end the interpolation.
                Default is None.
            direction_of_integration (string):
                Description of the direction in which to integrate.
                Options are 'positive' or 'negative'.
        """
        self.coord_name_to_integrate = coord_name_to_integrate
        self.start_point = start_point
        self.end_point = end_point
        self.direction_of_integration = direction_of_integration
        if self.direction_of_integration not in ["positive", "negative"]:
            msg = ("The specified direction of integration should be either "
                   "'positive' or 'negative'. {} was specified.".format(
                       self.direction_of_integration))
            raise ValueError(msg)

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<Integration: coord_name_to_integrate: {}, '
                  'start_point: {}, end_point: {}, '
                  'direction_of_integration: {}>'.format(
                      self.coord_name_to_integrate, self.start_point,
                      self.end_point, self.direction_of_integration))
        return result

    def ensure_monotonic_increase_in_chosen_direction(self, cube):
        """Ensure that the chosen coordinate is monotonically increasing in
        the specified direction.

        Args:
            cube (Iris.cube.Cube):
                The cube containing the coordinate to check.

        Returns:
            cube (Iris.cube.Cube):
                The cube containing a coordinate that is monotonically
                increasing in the desired direction.

        Raises:
            ValueError: The chosen coordinate is not monotonic.

        """
        coord_name = self.coord_name_to_integrate
        direction = self.direction_of_integration
        increasing_order = np.all(np.diff(cube.coord(coord_name).points) > 0)

        if increasing_order and direction == "positive":
            pass
        elif increasing_order and direction == "negative":
            cube = sort_coord_in_cube(cube, coord_name, order="descending")
        elif not increasing_order and direction == "positive":
            cube = sort_coord_in_cube(cube, coord_name)
        elif not increasing_order and direction == "negative":
            pass
        return cube

    def process(self, cube):
        """Integrate a specified coordinate. This is calculated by defining the
        upper and lower bounds for the steps along a chosen coordinate
        within the cube.

        Integration is performed by firstly defining the stride as the
        difference between the upper and lower bound. The contribution from
        the uppermost half of the stride is calculated by multiplying the
        upper bound value by 0.5 * stride, and the contribution
        from the lowermost half of the layer is calculated by multiplying the
        lower bound value by 0.5 * stride. The contribution from the
        uppermost half of the stride and the bottom half of the stride is
        summed.

        As the coordinate is progressively integrated, the contribution of
        each stride is cumulatively summed.

        Args:
            cube (Iris.cube.Cube):
                The cube containing the coordinate to check.

        Returns:
            integrated_cube (Iris.cube.Cube):
                The cube containing the result of the integration.
                This will contain the same metadata as the input cube.

        """
        # Make coordinate monotonic in the direction desired for integration.
        cube = self.ensure_monotonic_increase_in_chosen_direction(cube)

        # Define upper and lower level cubes for the integration.
        upper_bounds = cube.coord(self.coord_name_to_integrate).points[:-1]
        lower_bounds = cube.coord(self.coord_name_to_integrate).points[1:]
        upper_bounds_cube = (
             cube.extract(
                 iris.Constraint(
                     coord_values={self.coord_name_to_integrate:
                                   upper_bounds})))
        lower_bounds_cube = (
            cube.extract(
                iris.Constraint(
                    coord_values={self.coord_name_to_integrate:
                                  lower_bounds})))

        # Determine which cube to copy for most appropriate points from the
        # coordinate that is being integrated.
        if self.direction_of_integration == "positive":
            integrated_cube = upper_bounds_cube.copy()
        elif self.direction_of_integration == "negative":
            integrated_cube = lower_bounds_cube.copy()
        integrated_cube.data = np.zeros(lower_bounds_cube.shape)

        # Create a zip for looping over.
        levels_tuple = zip(
            upper_bounds_cube.slices_over(self.coord_name_to_integrate),
            lower_bounds_cube.slices_over(self.coord_name_to_integrate),
            integrated_cube.slices_over(self.coord_name_to_integrate))

        # Perform the integration
        stride_sum = 0
        integrated_cubelist = iris.cube.CubeList([])
        for (upper_bounds_slice, lower_bounds_slice,
             integrated_slice) in levels_tuple:
            upper_bound = (
                upper_bounds_slice.coord(
                    self.coord_name_to_integrate).points.item())
            lower_bound = (
                lower_bounds_slice.coord(
                    self.coord_name_to_integrate).points.item())
            if not self.start_point and not self.end_point:
                pass
            elif self.start_point:
                if (upper_bound >= self.start_point or
                        lower_bound >= self.start_point):
                    continue
            elif self.end_point:
                if (lower_bound <= self.end_point or
                        upper_bound <= self.end_point):
                    continue
            stride = np.abs(upper_bound - lower_bound)
            upper_half_of_stride = upper_bounds_slice.data * 0.5 * stride
            lower_half_of_stride = lower_bounds_slice.data * 0.5 * stride
            stride_sum += lower_half_of_stride + upper_half_of_stride
            integrated_slice.data = stride_sum
            integrated_cubelist.append(integrated_slice.copy())

        # Merge resulting cubes back together
        integrated_cube = integrated_cubelist.merge_cube()

        # Make sure that the coordinate that has been integrated is a
        # dimension coordinate.
        for coord in integrated_cube.aux_coords[::-1]:
            if coord.name() == self.coord_name_to_integrate:
                integrated_cube = iris.util.new_axis(
                    integrated_cube, self.coord_name_to_integrate)
        integrated_cube = (
            self.ensure_monotonic_increase_in_chosen_direction(
                integrated_cube))
        return integrated_cube
