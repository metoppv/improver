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
"""Module containing neighbourhood processing utilities."""

import copy
import math

import iris
from iris.exceptions import CoordinateNotFoundError
import numpy as np
import scipy.ndimage.filters

from improver.utilities.cube_checker import check_for_x_and_y_axes
from improver.utilities.cube_manipulation import concatenate_cubes
from improver.utilities.spatial import (
    convert_distance_into_number_of_grid_cells)

# Maximum radius of the neighbourhood width in grid cells.
MAX_RADIUS_IN_GRID_CELLS = 500


class Utilities(object):
    """
    Contains methods useful to multiple circular_kernel plugins.
    """

    def __init__(self):
        """
        Initialise class.
        """
        pass

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<Utilities>')
        return result

    @staticmethod
    def circular_kernel(fullranges, ranges, weighted_mode):
        """
        Method to apply a circular kernel to the data within the input cube in
        order to smooth the resulting field.

        Parameters
        ----------
        fullranges : Numpy.array
            Number of grid cells in all dimensions used to create the kernel.
            This should have the value 0 for any dimension other than x and y.
        ranges : Tuple
            Number of grid cells in the x and y direction used to create
            the kernel.
        weighted_mode : Boolean
            True when a weighted circular kernel is required.
            False will return a kernel consisting only of ones and zeroes.

        Returns
        -------
        kernel : Numpy.array
            Array containing the circular smoothing kernel.
            This will have the same number of dimensions as fullranges.

        """
        # Define the size of the kernel based on the number of grid cells
        # contained within the desired radius.
        kernel = np.ones([int(1 + x * 2) for x in fullranges])
        # Create an open multi-dimensional meshgrid.
        open_grid = np.array(np.ogrid[tuple([slice(-x, x+1) for x in ranges])])
        if weighted_mode:
            # Create a kernel, such that the central grid point has the
            # highest weighting, with the weighting decreasing with distance
            # away from the central grid point.
            open_grid_summed_squared = np.sum(open_grid**2.).astype(float)
            kernel[:] = (
                (np.prod(ranges) - open_grid_summed_squared) / np.prod(ranges))
            mask = kernel < 0.
        else:
            mask = np.reshape(
                np.sum(open_grid**2) > np.prod(ranges), np.shape(kernel))
        kernel[mask] = 0.
        return kernel


class CircularNeighbourhood(object):

    """
    Methods for use in the calculation and application of a circular
    neighbourhood.

    A maximum kernel radius of 500 grid cells is imposed in order to
    avoid computational ineffiency and possible memory errors.
    """

    def __init__(self, unweighted_mode=False):
        """
        Initialise class.

        Parameters
        ----------
        unweighted_mode : boolean
            If True, use a circle with constant weighting.
            If False, use a circle for neighbourhood kernel with
            weighting decreasing with radius.

        """
        self.unweighted_mode = unweighted_mode

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<CircularNeighbourhood: unweighted_mode: {}>')
        return result.format(self.unweighted_mode)

    def apply_circular_kernel(self, cube, ranges):
        """
        Method to apply a circular kernel to the data within the input cube in
        order to smooth the resulting field.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube containing to array to apply CircularNeighbourhood processing
            to.
        ranges : Tuple
            Number of grid cells in the x and y direction used to create
            the kernel.

        Returns
        -------
        cube : Iris.cube.Cube
            Cube containing the smoothed field after the kernel has been
            applied.

        """
        data = cube.data
        fullranges = np.zeros([np.ndim(data)])
        axes = []
        try:
            for coord_name in ['projection_x_coordinate',
                               'projection_y_coordinate']:
                axes.append(cube.coord_dims(coord_name)[0])
        except CoordinateNotFoundError:
            raise ValueError("Invalid grid: projection_x/y coords required")
        for axis_index, axis in enumerate(axes):
            fullranges[axis] = ranges[axis_index]
        kernel = Utilities.circular_kernel(fullranges, ranges,
                                           (not self.unweighted_mode))
        # Smooth the data by applying the kernel.
        cube.data = scipy.ndimage.filters.correlate(
            data, kernel, mode='nearest') / np.sum(kernel)
        return cube

    def run(self, cube, radius):
        """
        Call the methods required to calculate and apply a circular
        neighbourhood.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube containing to array to apply CircularNeighbourhood processing
            to.
        radius : Float
            Radius in metres for use in specifying the number of
            grid cells used to create a circular neighbourhood.

        Returns
        -------
        cube : Iris.cube.Cube
            Cube containing the smoothed field after the kernel has been
            applied.
        """
        ranges = convert_distance_into_number_of_grid_cells(
            cube, radius, MAX_RADIUS_IN_GRID_CELLS)
        cube = self.apply_circular_kernel(cube, ranges)
        return cube
