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
from improver.percentile import PercentileConverter

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

    def __init__(self, weighted_mode=True, percentiles=None):
        """
        Initialise class.

        Parameters
        ----------
        weighted_mode : boolean
            If False, use a circle with constant weighting.
            If True, use a circle for neighbourhood kernel with
            weighting decreasing with radius.

        percentiles : list (optional)
            This is included to allow a standard interface for both the
            percentile and probability neighbourhood plugins.
        """
        self.weighted_mode = weighted_mode
        self.percentiles = percentiles

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
        try:
            for coord_name in ['projection_x_coordinate',
                               'projection_y_coordinate']:
                axes.append(cube.coord_dims(coord_name)[0])
        except CoordinateNotFoundError:
            raise ValueError("Invalid grid: projection_x/y coords required")
        for axis_index, axis in enumerate(axes):
            fullranges[axis] = ranges[axis_index]
        kernel = Utilities.circular_kernel(fullranges, ranges,
                                           self.weighted_mode)
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


class CircularKernelNumpy(object):
    """
    Methods for use in calculating percentiles from a 2D circular
    neighbourhood.

    A maximum kernel radius of 500 grid cells is imposed in order to
    avoid computational ineffiency and possible memory errors.
    """
    def __init__(self,
                 percentiles=PercentileConverter.DEFAULT_PERCENTILES):
        """

        Parameters
        ----------

        percentiles : list (optional)
            Percentile values at which to calculate; if not provided uses
            DEFAULT_PERCENTILES from percentile module.
        """
        self.percentiles = percentiles

    def __repr__(self):
        """Represent the configured class instance as a string."""
        result = ('<CircularKernelNumpy: percentiles: {}>')
        return result.format(self.percentiles)

    def run(self, cube, ranges):
        """
        Method to apply a circular kernel to the data within the input cube in
        order to derive percentiles over the kernel.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube containing array to apply processing to.
        ranges : Int
            Number of grid cells in the x and y direction used to create
            the kernel.

        Returns
        -------
        outcube : Iris.cube.Cube
            Cube containing the percentile fields.
            Has percentile as an added dimension.

        """
        # Take data array and identify X and Y axes indices
        try:
            for coord_name in ['projection_x_coordinate',
                               'projection_y_coordinate']:
                cube.coord(coord_name)
        except CoordinateNotFoundError:
            raise ValueError("Invalid grid: projection_x/y coords required")
        ranges = int(ranges)
        if ranges < 1:
            raise ValueError("Range size too small. {} < 1".format(ranges))
        ranges_xy = np.array([ranges]*2)
        ranges_tuple = tuple([ranges]*2)
        kernel = Utilities.circular_kernel(ranges_xy, ranges_tuple,
                                           weighted_mode=False)

        # Loop over each 2D slice to reduce memory demand and derive
        # percentiles on the kernel. Will return an extra dimension.
        pctcubelist = iris.cube.CubeList()
        for slice_2d in cube.slices(['projection_y_coordinate',
                                     'projection_x_coordinate']):
            # Create a 1D data array padded with repeats of the local boundary
            # mean.
            padded = np.pad(slice_2d.data, ranges, mode='mean',
                            stat_length=ranges)
            padshape = np.shape(padded)  # Store size to make unflatten easier
            padded = padded.flatten()
            # Add 2nd dimension with each point's neighbourhood points along it
            nbhood_slices = [
                np.roll(padded, padshape[1]*j+i)
                for i in range(-ranges, ranges+1)
                for j in range(-ranges, ranges+1)
                if kernel[..., i+ranges, j+ranges] > 0.]
            # Collapse this dimension into percentiles (a new 2nd dimension)
            perc_data = np.percentile(nbhood_slices, self.percentiles, axis=0)
            # Return to 3D
            perc_data = perc_data.reshape(
                len(self.percentiles), padshape[0], padshape[1])
            # Create a cube for these data:
            pctcube = self.make_percentile_cube(slice_2d)
            # And put in data, removing the padding
            pctcube.data = perc_data[:, ranges:-ranges, ranges:-ranges]
            pctcubelist.append(pctcube)
        result = pctcubelist.merge_cube()
        result = self.check_coords(result, cube)
        return result

    @staticmethod
    def check_coords(cube, cube_orig):
        """Checks the coordinates of cube match those of cube_orig
        and promotes any that are not dimensions.
        This function expects that cube will have an additional
        "percentiles" dimension.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube to ensure compliance in. May be modified if not compliant.

        cube_orig : Iris.cube.Cube
            Cube to ensure compliance against. Will NOT be modified.

        Returns
        -------
        cube : Iris.cube.Cube
            Cube after ensuring compliance.

        Exceptions
        -------
        Raises ValueError if cube cannot be made compliant.
        """

        # Promote any missing dimension coords from auxilliary coords
        for coord in cube_orig.coords():
            if len(cube_orig.coord_dims(coord)) == 0:
                continue
            try:
                cube.coord_dims(coord)[0]
            except IndexError:
                cube = iris.util.new_axis(cube, coord)
        # Now check axis order
        required_order = list(np.shape(cube.data))
        for indx, coord in enumerate(cube_orig.coords()):
            if len(cube_orig.coord_dims(coord)) == 0:
                continue
            required_order[indx+1] = cube.coord_dims(coord)[0]
        required_order[0] = cube.coord_dims("percentiles")[0]
        cube.transpose(required_order)
        return cube

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
                pct, long_name='percentiles', units='%'))
            pctcubelist.append(pctcube)
        return pctcubelist.merge_cube()
