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
"""This module contains methods for circular neighbourhood processing."""

import numpy as np
import scipy.ndimage.filters

import iris

from improver.constants import DEFAULT_PERCENTILES
from improver.utilities.cube_checker import (
    check_cube_coordinates, find_dimension_coordinate_mismatch)
from improver.utilities.spatial import (
    check_if_grid_is_equal_area, convert_distance_into_number_of_grid_cells)


# Maximum radius of the neighbourhood width in grid cells.
MAX_RADIUS_IN_GRID_CELLS = 500


def circular_kernel(fullranges, ranges, weighted_mode):
    """

    Method to create a circular kernel.

    Parameters
    ----------
    fullranges : Numpy.array
        Number of grid cells in all dimensions used to create the kernel.
        This should have the value 0 for any dimension other than x and y.
    ranges : Tuple
        Number of grid cells in the x and y direction used to create
        the kernel.

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

    def __init__(self, weighted_mode=True):
        """
        Initialise class.

        Parameters
        ----------
        weighted_mode : boolean (optional)
            If True, use a circle for neighbourhood kernel with
            weighting decreasing with radius.
            If False, use a circle with constant weighting.
        """
        self.weighted_mode = weighted_mode

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<CircularNeighbourhood: weighted_mode: {}>')
        return result.format(self.weighted_mode)

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
        for axis in ["x", "y"]:
            coord_name = cube.coord(axis=axis).name()
            axes.append(cube.coord_dims(coord_name)[0])

        for axis_index, axis in enumerate(axes):
            fullranges[axis] = ranges[axis_index]
        kernel = circular_kernel(fullranges, ranges, self.weighted_mode)
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
        # Check that the cube has an equal area grid.
        check_if_grid_is_equal_area(cube)
        ranges = convert_distance_into_number_of_grid_cells(
            cube, radius, MAX_RADIUS_IN_GRID_CELLS)
        cube = self.apply_circular_kernel(cube, ranges)
        return cube


class GeneratePercentilesFromACircularNeighbourhood(object):
    """
    Methods for use in calculating percentiles from a 2D circular
    neighbourhood.
    A maximum kernel radius of 500 grid cells is imposed in order to
    avoid computational ineffiency and possible memory errors.
    """
    def __init__(self, percentiles=DEFAULT_PERCENTILES):
        """
        Initialise class.

        Parameters
        ----------
        percentiles : list (optional)
            Percentile values at which to calculate; if not provided uses
            DEFAULT_PERCENTILES.

        """
        self.percentiles = tuple(percentiles)

    def __repr__(self):
        """Represent the configured class instance as a string."""
        result = ('<GeneratePercentilesFromACircularNeighbourhood: '
                  'percentiles: {}>')
        return result.format(self.percentiles)

    def run(self, cube, radius):
        """
        Method to apply a circular kernel to the data within the input cube in
        order to derive percentiles over the kernel.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube containing array to apply processing to.
        radius : Float
            Radius in metres for use in specifying the number of
            grid cells used to create a circular neighbourhood.

        Returns
        -------
        result : Iris.cube.Cube
            Cube containing the percentile fields.
            Has percentile as an added dimension.

        """
        # Check that the cube has an equal area grid.
        check_if_grid_is_equal_area(cube)
        # Take data array and identify X and Y axes indices
        ranges_tuple = convert_distance_into_number_of_grid_cells(
            cube, radius, MAX_RADIUS_IN_GRID_CELLS)
        ranges_xy = np.array(ranges_tuple)
        kernel = circular_kernel(ranges_xy, ranges_tuple, weighted_mode=False)
        # Loop over each 2D slice to reduce memory demand and derive
        # percentiles on the kernel. Will return an extra dimension.
        pctcubelist = iris.cube.CubeList()
        for slice_2d in cube.slices(['projection_y_coordinate',
                                     'projection_x_coordinate']):
            # Create a 1D data array padded with repeats of the local boundary
            # mean.
            padded = np.pad(slice_2d.data, ranges_xy, mode='mean',
                            stat_length=np.max(ranges_xy))
            padshape = np.shape(padded)  # Store size to make unflatten easier
            padded = padded.flatten()
            # Add 2nd dimension with each point's neighbourhood points along it
            nbhood_slices = [
                np.roll(padded, (padshape[1]*j)+i)
                for i in range(-ranges_xy[1], ranges_xy[1]+1)
                for j in range(-ranges_xy[0], ranges_xy[0]+1)
                if kernel[..., i+ranges_xy[1], j+ranges_xy[0]] > 0.]
            # Collapse this dimension into percentiles (a new 2nd dimension)
            perc_data = np.percentile(nbhood_slices, self.percentiles, axis=0)
            # Return to 3D
            perc_data = perc_data.reshape(
                len(self.percentiles), padshape[0], padshape[1])
            # Create a cube for these data:
            pctcube = self.make_percentile_cube(slice_2d)
            # And put in data, removing the padding
            pctcube.data = perc_data[:, ranges_xy[0]:-ranges_xy[0],
                                     ranges_xy[1]:-ranges_xy[1]]
            pctcubelist.append(pctcube)
        result = pctcubelist.merge_cube()
        exception_coordinates = (
            find_dimension_coordinate_mismatch(
                cube, result, two_way_mismatch=False))
        result = (
            check_cube_coordinates(
                cube, result, exception_coordinates=exception_coordinates))

        # Arrange cube, so that the coordinate order is:
        # realization, percentile, other coordinates.
        required_order = []
        if result.coords("realization"):
            if result.coords("realization", dimensions=[]):
                result = iris.util.new_axis(result, "realization")
            required_order.append(result.coord_dims("realization")[0])
        if result.coords("percentiles_over_neighbourhood"):
            required_order.append(
                result.coord_dims("percentiles_over_neighbourhood")[0])
        other_coords = []
        for coord in result.dim_coords:
            if coord.name() not in [
                   "realization", "percentiles_over_neighbourhood"]:
                other_coords.append(result.coord_dims(coord.name())[0])
        required_order.extend(other_coords)
        result.transpose(required_order)

        return result

    def make_percentile_cube(self, cube):
        """Returns a cube with the same metadata as the sample cube
        but with an added percentile dimension.
        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube to copy meta data from.
        Returns
        -------
        cube : Iris.cube.Cube
            Cube like input but with added percentiles coordinate.
            Each slice along this coordinate is identical.
        """
        pctcubelist = iris.cube.CubeList()
        for pct in self.percentiles:
            pctcube = cube.copy()
            pctcube.add_aux_coord(iris.coords.DimCoord(
                pct, long_name="percentiles_over_neighbourhood", units='%'))
            pctcubelist.append(pctcube)
        return pctcubelist.merge_cube()
