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
"""Module to apply a recursive filter to neighbourhooded data."""
import warnings

import iris
import numpy as np

from improver import PostProcessingPlugin
from improver.nbhood.square_kernel import SquareNeighbourhood
from improver.utilities.cube_checker import check_cube_coordinates
from improver.utilities.pad_spatial import (
    pad_cube_with_halo, remove_halo_from_cube)


class RecursiveFilter(PostProcessingPlugin):

    """
    Apply a recursive filter to the input cube.
    """

    def __init__(self, smoothing_coefficient_x=None,
                 smoothing_coefficient_y=None, iterations=None,
                 edge_width=1, re_mask=False):
        """
        Initialise the class.

        The smoothing_coefficient determines how much "value" of a cell
        undergoing filtering is comprised of the current value at that cell and
        how much comes from the adjacent cell preceding it in the direction in
        which filtering is being applied. A larger smoothing_coefficient
        results in a more significant proportion of a cell's new value coming
        from its neighbouring cell.

        Args:
            smoothing_coefficient_x (float or None):
                Filter parameter: A constant used to weight the
                recursive filter along the x-axis. Defined such
                that 0 < smoothing_coefficient_x < 0.5
            smoothing_coefficient_y (float or None):
                Filter parameter: A constant used to weight the
                recursive filter along the y-axis. Defined such
                that 0 < smoothing_coefficient_y < 0.5
            iterations (int or None):
                The number of iterations of the recursive filter.
            edge_width (int):
                Half the width of the padding halo applied before
                recursive filtering.
            re_mask (bool):
                If re_mask is True, the original un-recursively filtered
                mask is applied to mask out the recursively filtered cube.
                If re_mask is False, the original un-recursively filtered
                mask is not applied. Therefore, the recursive filtering
                may result in values being present in areas that were
                originally masked.
        Raises:
            ValueError: If smoothing_coefficient_x is not set such that
                        0 < smoothing_coefficient_x <= 0.5
            ValueError: If smoothing_coefficient_y is not set such that
                        0 < smoothing_coefficient_y <= 0.5
            ValueError: If number of iterations is not None and is set such
                        that iterations is less than 1.
        Warns:
            UserWarning:
                If iterations is higher than 2.
        """
        smoothing_coefficient_error = (
            "smoothing_coefficient must be less than 0.5. A large "
            "smoothing_coefficient value leads to poor conservation of "
            "probabilities: ")
        for k, smoothing_coefficient in {'x': smoothing_coefficient_x,
                                         'y': smoothing_coefficient_y}.items():
            if (smoothing_coefficient is not None and
                    not 0 < smoothing_coefficient <= 0.5):
                message = (smoothing_coefficient_error if
                           smoothing_coefficient > 0.5 else '')
                message += ("Invalid smoothing_coefficient_{}: must be > 0 "
                            "and <= 0.5: {}")
                raise ValueError(message.format(k, smoothing_coefficient))
        if iterations is not None:
            if iterations < 1:
                raise ValueError(
                    "Invalid number of iterations: must be >= 1: {}".format(
                        iterations))
            if iterations > 2:
                warnings.warn(
                    "More than two iterations degrades the conservation"
                    "of probability assumption.")
        self.smoothing_coefficient_x = smoothing_coefficient_x
        self.smoothing_coefficient_y = smoothing_coefficient_y
        self.iterations = iterations
        self.edge_width = edge_width
        self.re_mask = re_mask

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<RecursiveFilter: smoothing_coefficient_x: {}, '
                  'smoothing_coefficient_y: {}, iterations: {}, '
                  'edge_width: {}')
        return result.format(
            self.smoothing_coefficient_x, self.smoothing_coefficient_y,
            self.iterations, self.edge_width)

    @staticmethod
    def _recurse_forward(grid, smoothing_coefficients, axis):
        """
        Method to run the recursive filter in the forward direction.

        In the forward direction:
            Recursive filtering is calculated as:

        .. math::
            B_i = ((1 - \\rm{smoothing\\_coefficient}) \\times A_i) +
            (\\rm{smoothing\\_coefficient} \\times B_{i-1})

        Progressing from gridpoint i-1 to i:
            :math:`B_i` = new value at gridpoint i

            :math:`A_i` = Old value at gridpoint i

            :math:`B_{i-1}` = New value at gridpoint i - 1

        Args:
            grid (numpy.ndarray):
                2D array containing the input data to which the recursive
                filter will be applied.
            smoothing_coefficients (numpy.ndarray):
                Matching 2D array of smoothing_coefficient values that will be
                used when applying the recursive filter along the specified
                axis.
            axis (int):
                Index of the spatial axis (0 or 1) over which to recurse.

        Returns:
            numpy.ndarray:
                2D array containing the smoothed field after the recursive
                filter method has been applied to the input array in the
                forward direction along the specified axis.
        """
        lim = grid.shape[axis]
        for i in range(1, lim):
            if axis == 0:
                grid[i, :] = ((1. - smoothing_coefficients[i, :]) * grid[i, :]
                              + smoothing_coefficients[i, :] * grid[i-1, :])
            if axis == 1:
                grid[:, i] = ((1. - smoothing_coefficients[:, i]) * grid[:, i]
                              + smoothing_coefficients[:, i] * grid[:, i-1])
        return grid

    @staticmethod
    def _recurse_backward(grid, smoothing_coefficients, axis):
        """
        Method to run the recursive filter in the backwards direction.

        In the backwards direction:
            Recursive filtering is calculated as:

        .. math::
            B_i = ((1 - \\rm{smoothing\\_coefficient}) \\times A_i) +
            (\\rm{smoothing\\_coefficient} \\times B_{i+1})

        Progressing from gridpoint i+1 to i:
            :math:`B_i` = new value at gridpoint i

            :math:`A_i` = Old value at gridpoint i

            :math:`B_{i+1}` = New value at gridpoint i+1

        Args:
            grid (numpy.ndarray):
                2D array containing the input data to which the recursive
                filter will be applied.
            smoothing_coefficients (numpy.ndarray):
                Matching 2D array of smoothing_coefficient values that will be
                used when applying the recursive filter along the specified
                axis.
            axis (int):
                Index of the spatial axis (0 or 1) over which to recurse.

        Returns:
            numpy.ndarray:
                2D array containing the smoothed field after the recursive
                filter method has been applied to the input array in the
                backwards direction along the specified axis.
        """
        lim = grid.shape[axis]
        for i in range(lim-2, -1, -1):
            if axis == 0:
                grid[i, :] = ((1. - smoothing_coefficients[i, :]) * grid[i, :]
                              + smoothing_coefficients[i, :] * grid[i+1, :])
            if axis == 1:
                grid[:, i] = ((1. - smoothing_coefficients[:, i]) * grid[:, i]
                              + smoothing_coefficients[:, i] * grid[:, i+1])
        return grid

    @staticmethod
    def _run_recursion(cube, smoothing_coefficients_x,
                       smoothing_coefficients_y, iterations):
        """
        Method to run the recursive filter.

        Args:
            cube (iris.cube.Cube):
                2D cube containing the input data to which the recursive
                filter will be applied.
            smoothing_coefficients_x (iris.cube.Cube):
                2D cube containing array of smoothing_coefficient values that
                will be used when applying the recursive filter along the
                x-axis.
            smoothing_coefficients_y (iris.cube.Cube):
                2D cube containing array of smoothing_coefficient values that
                will be used when applying the recursive filter along the
                y-axis.
            iterations (int):
                The number of iterations of the recursive filter

        Returns:
            iris.cube.Cube:
                Cube containing the smoothed field after the recursive filter
                method has been applied to the input cube.
        """
        x_index, = cube.coord_dims(cube.coord(axis="x").name())
        y_index, = cube.coord_dims(cube.coord(axis="y").name())
        output = cube.data

        for _ in range(iterations):
            output = RecursiveFilter._recurse_forward(
                output, smoothing_coefficients_x.data, x_index)
            output = RecursiveFilter._recurse_backward(
                output, smoothing_coefficients_x.data, x_index)
            output = RecursiveFilter._recurse_forward(
                output, smoothing_coefficients_y.data, y_index)
            output = RecursiveFilter._recurse_backward(
                output, smoothing_coefficients_y.data, y_index)
            cube.data = output
        return cube

    def _set_smoothing_coefficients(self, cube, smoothing_coefficient,
                                    smoothing_coefficients_cube):
        """
        Set up the smoothing_coefficient parameter.

        Args:
            cube (iris.cube.Cube):
                2D cube containing the input data to which the recursive
                filter will be applied.
            smoothing_coefficient (float):
                The constant used to weight the recursive filter in that
                direction: Defined such that 0.0 < smoothing_coefficient < 1.0
            smoothing_coefficients_cube (iris.cube.Cube or None):
                Cube containing array of smoothing_coefficient values that will
                be used when applying the recursive filter in a specific
                direction.
        Raises:
            ValueError: If both smoothing_coefficients_cube and
                        smoothing_coefficient are provided.
            ValueError: If smoothing_coefficient and
                        smoothing_coefficients_cube are both set to None.
            ValueError: If the dimensions of the smoothing_coefficients array
                        do not match the dimensions of the cube data.

        Returns:
            iris.cube.Cube:
                Cube containing a padded array of smoothing_coefficient values
                for the specified direction.
        """
        if (smoothing_coefficient is not None and
                smoothing_coefficients_cube is not None):
            emsg = ("A cube of smoothing_coefficient values and a single float"
                    " value for smoothing_coefficient have both been provided."
                    " Only one of these options can be set.")
            raise ValueError(emsg)

        if smoothing_coefficients_cube is None:
            if smoothing_coefficient is None:
                emsg = ("A value for smoothing_coefficient must be set if "
                        "smoothing_coefficients_cube is set to None: "
                        "smoothing_coefficient is currently set as: {}")
                raise ValueError(emsg.format(smoothing_coefficient))
            smoothing_coefficients_cube = cube.copy(
                data=np.ones(cube.data.shape) * smoothing_coefficient)

        if smoothing_coefficients_cube is not None:
            if smoothing_coefficients_cube.data.shape != cube.data.shape:
                emsg = ("Dimensions of smoothing_coefficients array do not "
                        "match dimensions of data array: {} < {}")
                raise ValueError(
                    emsg.format(smoothing_coefficients_cube.data.shape,
                                cube.data.shape))

        smoothing_coefficients_cube = pad_cube_with_halo(
            smoothing_coefficients_cube, 2*self.edge_width, 2*self.edge_width)
        return smoothing_coefficients_cube

    def process(self, cube, smoothing_coefficients_x=None,
                smoothing_coefficients_y=None, mask_cube=None):
        """
        Set up the smoothing_coefficient parameters and run the recursive
        filter. The steps undertaken are:

        1. Split the input cube into slices determined by the co-ordinates in
           the x and y directions.
        2. Construct an array of filter parameters (smoothing_coefficients_x
           and smoothing_coefficients_y) for each cube slice that are used to
           weight the recursive filter in the x- and y-directions.
        3. Pad each cube slice with a square-neighbourhood halo and apply
           the recursive filter for the required number of iterations.
        4. Remove the halo from the cube slice and append the recursed cube
           slice to a 'recursed cube'.
        5. Merge all the cube slices in the 'recursed cube' into a 'new cube'.
        6. Modify the 'new cube' so that its scalar dimension co-ordinates are
           consistent with those in the original input cube.
        7. Return the 'new cube' which now contains the recursively filtered
           values for the original input cube.

        Args:
            cube (iris.cube.Cube):
                Cube containing the input data to which the recursive filter
                will be applied.
            smoothing_coefficients_x (iris.cube.Cube or None):
                Cube containing array of smoothing_coefficient values that will
                be used when applying the recursive filter along the x-axis.
            smoothing_coefficients_y (iris.cube.Cube or None):
                Cube containing array of smoothing_coefficient values that will
                be used when applying the recursive filter along the y-axis.
            mask_cube (iris.cube.Cube or None):
                Cube containing an external mask to apply to the cube before
                applying the recursive filter.

        Returns:
            iris.cube.Cube:
                Cube containing the smoothed field after the recursive filter
                method has been applied.

        Raises:
            ValueError: If any smoothing_coefficient cube value is over 0.5
        """
        for smoothing_coefficient in (smoothing_coefficients_x,
                                      smoothing_coefficients_y):
            if (smoothing_coefficient is not None and
                    (smoothing_coefficient.data > 0.5).any()):
                raise ValueError(
                    "All smoothing_coefficient values must be less than 0.5. "
                    "A large smoothing_coefficient value leads to poor "
                    "conservation of probabilities")

        cube_format = next(cube.slices([cube.coord(axis='y'),
                                        cube.coord(axis='x')]))
        smoothing_coefficients_x = self._set_smoothing_coefficients(
            cube_format, self.smoothing_coefficient_x,
            smoothing_coefficients_x)
        smoothing_coefficients_y = self._set_smoothing_coefficients(
            cube_format, self.smoothing_coefficient_y,
            smoothing_coefficients_y)

        recursed_cube = iris.cube.CubeList()
        for output in cube.slices([cube.coord(axis='y'),
                                   cube.coord(axis='x')]):

            # Setup cube and mask for processing.
            # This should set up a mask full of 1.0 if None is provided
            # and set the data 0.0 where mask is 0.0 or the data is NaN
            output, mask, nan_array = (
                SquareNeighbourhood().set_up_cubes_to_be_neighbourhooded(
                    output, mask_cube))
            mask = mask.data.squeeze()

            padded_cube = pad_cube_with_halo(
                output, 2*self.edge_width, 2*self.edge_width)

            new_cube = self._run_recursion(
                padded_cube, smoothing_coefficients_x,
                smoothing_coefficients_y, self.iterations)
            new_cube = remove_halo_from_cube(
                new_cube, 2*self.edge_width, 2*self.edge_width)
            if self.re_mask:
                new_cube.data[nan_array.astype(bool)] = np.nan
                new_cube.data = np.ma.masked_array(new_cube.data,
                                                   mask=np.logical_not(mask))

            recursed_cube.append(new_cube)

        new_cube = recursed_cube.merge_cube()
        new_cube = check_cube_coordinates(cube, new_cube)

        return new_cube
