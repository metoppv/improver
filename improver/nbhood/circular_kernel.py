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
"""This module contains methods for circular neighbourhood processing."""

import iris
import numpy as np
from scipy.ndimage.filters import correlate

from improver.constants import DEFAULT_PERCENTILES
from improver.utilities.cube_checker import (
    check_cube_coordinates, find_dimension_coordinate_mismatch)
from improver.utilities.spatial import (
    check_if_grid_is_equal_area, distance_to_number_of_grid_cells)


def check_radius_against_distance(cube, radius):
    """Check required distance isn't greater than the size of the domain.

    Args:
        cube (iris.cube.Cube):
            The cube to check.
        radius (float):
            The radius, which cannot be more than half of the
            size of the domain.

    """
    axes = []
    for axis in ['x', 'y']:
        coord = cube.coord(axis=axis).copy()
        coord.convert_units('metres')
        axes.append((max(coord.points) - min(coord.points)))

    max_allowed = np.sqrt(axes[0] ** 2 + axes[1] ** 2) * 0.5
    if radius > max_allowed:
        raise ValueError(f"Distance of {radius}m exceeds max domain "
                         f"distance of {max_allowed}m")


def circular_kernel(full_ranges, ranges, weighted_mode):
    """

    Method to create a circular kernel.

    Args:
        full_ranges (numpy.ndarray):
            Number of grid cells in all dimensions used to create the kernel.
            This should have the value 0 for any dimension other than x and y.
        ranges (int):
            Number of grid cells in the x and y direction used to create
            the kernel.
        weighted_mode (bool):
            If True, use a circle for neighbourhood kernel with
            weighting decreasing with radius.
            If False, use a circle with constant weighting.

    Returns:
        numpy.ndarray:
            Array containing the circular smoothing kernel.
            This will have the same number of dimensions as fullranges.

    """
    # The range is square

    area = ranges * ranges
    # Define the size of the kernel based on the number of grid cells
    # contained within the desired radius.
    kernel = np.ones([int(1 + x * 2) for x in full_ranges])
    # Create an open multi-dimensional meshgrid.
    open_grid = np.array(np.ogrid[[slice(-x, x+1) for x in (ranges, ranges)]])
    if weighted_mode:
        # Create a kernel, such that the central grid point has the
        # highest weighting, with the weighting decreasing with distance
        # away from the central grid point.
        open_grid_summed_squared = np.sum(open_grid**2.).astype(float)
        kernel[:] = (area - open_grid_summed_squared) / area
        mask = kernel < 0.
    else:
        mask = np.reshape(np.sum(open_grid**2) > area, np.shape(kernel))
    kernel[mask] = 0.
    return kernel


class CircularNeighbourhood:

    """
    Methods for use in the calculation and application of a circular
    neighbourhood.
    A maximum kernel radius of 500 grid cells is imposed in order to
    avoid computational ineffiency and possible memory errors.
    """

    def __init__(self, weighted_mode=True, sum_or_fraction="fraction",
                 re_mask=False):
        """
        Initialise class.

        Args:
            weighted_mode (bool):
                If True, use a circle for neighbourhood kernel with
                weighting decreasing with radius.
                If False, use a circle with constant weighting.
            sum_or_fraction (str):
                Identifier for whether sum or fraction should be returned from
                neighbourhooding. The sum represents the sum of the
                neighbourhood. The fraction represents the sum of the
                neighbourhood divided by the neighbourhood area.
                Valid options are "sum" or "fraction".
            re_mask (bool):
                If re_mask is True, the original un-neighbourhood processed
                mask is applied to mask out the neighbourhood processed cube.
                If re_mask is False, the original un-neighbourhood processed
                mask is not applied. Therefore, the neighbourhood processing
                may result in values being present in areas that were
                originally masked.
        """
        self.weighted_mode = weighted_mode
        if sum_or_fraction not in ["sum", "fraction"]:
            msg = ("The neighbourhood output can either be in the form of a "
                   "sum of all the points in the neighbourhood or a fraction "
                   "of the sum of the neighbourhood divided by the "
                   "neighbourhood area. The {} option is invalid. "
                   "Valid options are 'sum' or 'fraction'.")
            raise ValueError(msg)
        self.sum_or_fraction = sum_or_fraction
        self.re_mask = re_mask
        self.kernel = None

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<CircularNeighbourhood: weighted_mode: {}, '
                  'sum_or_fraction: {}>')
        return result.format(self.weighted_mode, self.sum_or_fraction)

    def apply_circular_kernel(self, cube, ranges):
        """
        Method to apply a circular kernel to the data within the input cube in
        order to smooth the resulting field.

        Args:
            cube (iris.cube.Cube):
                Cube containing to array to apply CircularNeighbourhood
                processing to.
            ranges (int):
                Number of grid cells in the x and y direction used to create
                the kernel.

        Returns:
            iris.cube.Cube:
                Cube containing the smoothed field after the kernel has been
                applied.

        """
        data = cube.data
        full_ranges = np.zeros([np.ndim(data)])
        axes = []
        for axis in ["x", "y"]:
            coord_name = cube.coord(axis=axis).name()
            axes.append(cube.coord_dims(coord_name)[0])

        for axis in axes:
            full_ranges[axis] = ranges
        self.kernel = circular_kernel(full_ranges, ranges, self.weighted_mode)
        # Smooth the data by applying the kernel.
        if self.sum_or_fraction == "sum":
            total_area = 1.0
        else:
            # sum_or_fraction is in fraction mode
            total_area = np.sum(self.kernel)

        cube.data = correlate(data, self.kernel, mode='nearest') / total_area
        return cube

    def run(self, cube, radius, mask_cube=None):
        """

        Call the methods required to calculate and apply a circular
        neighbourhood.

        Args:
            cube (iris.cube.Cube):
                Cube containing to array to apply CircularNeighbourhood
                processing to.
            radius (float):
                Radius in metres for use in specifying the number of
                grid cells used to create a circular neighbourhood.
            mask_cube (iris.cube.Cube or None):
                Cube containing the array to be used as a mask.

        Returns:
            iris.cube.Cube:
                Cube containing the smoothed field after the kernel has been
                applied.

        """
        if mask_cube is not None:
            msg = ("The use of a mask cube with a circular kernel is not "
                   "yet implemented.")
            raise NotImplementedError(msg)

        # Check that the cube has an equal area grid.
        check_if_grid_is_equal_area(cube)
        grid_cells = distance_to_number_of_grid_cells(cube, radius)
        cube = self.apply_circular_kernel(cube, grid_cells)
        return cube


class GeneratePercentilesFromACircularNeighbourhood:
    """
    Methods for use in calculating percentiles from a 2D circular
    neighbourhood.
    A maximum kernel radius of 500 grid cells is imposed in order to
    avoid computational ineffiency and possible memory errors.
    """
    def __init__(self, percentiles=DEFAULT_PERCENTILES):
        """
        Initialise class.

        Args:
            percentiles (list or float):
                Percentile values at which to calculate; if not provided uses
                DEFAULT_PERCENTILES.

        """
        try:
            self.percentiles = tuple(percentiles)
        except TypeError:
            self.percentiles = tuple([percentiles])

    def __repr__(self):
        """Represent the configured class instance as a string."""
        result = ('<GeneratePercentilesFromACircularNeighbourhood: '
                  'percentiles: {}>')
        return result.format(self.percentiles)

    def pad_and_unpad_cube(self, slice_2d, kernel):
        """
        Method to pad and unpad a two dimensional cube. The input array is
        padded and percentiles are calculated using a neighbourhood around
        each point. The resulting percentile data are unpadded and put into a
        cube.

        Args:
            slice_2d (iris.cube.Cube):
                2d cube to be padded with a halo.
            kernel (numpy.ndarray):
                Kernel used to specify the neighbourhood to consider when
                calculating the percentiles within a neighbourhood.

        Examples:

            1. Take the input slice_2d cube with the data, where 1 is an
               occurrence and 0 is an non-occurrence::

                    [[1., 1., 1.,],
                     [1., 0., 1.],
                     [1., 1., 1.]]

            2. Define a kernel. This kernel is effectively placed over each
               point within the input data. Note that the input data is padded
               prior to placing the kernel over each point, so that the kernel
               does not exceed the bounds of the padded data::

                    [[ 0.,  0.,  1.,  0.,  0.],
                     [ 0.,  1.,  1.,  1.,  0.],
                     [ 1.,  1.,  1.,  1.,  1.],
                     [ 0.,  1.,  1.,  1.,  0.],
                     [ 0.,  0.,  1.,  0.,  0.]]

            3. Pad the input data. The extent of the padding is given by the
               shape of the kernel. The number of values included within the
               calculation of the mean is determined by the size of the
               kernel::

                    [[ 0.75,  0.75,  1.  ,  0.5 ,  1.  ,  0.75,  0.75],
                     [ 0.75,  0.75,  1.  ,  0.5 ,  1.  ,  0.75,  0.75],
                     [ 1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ],
                     [ 0.5 ,  0.5 ,  1.  ,  0.  ,  1.  ,  0.5 ,  0.5 ],
                     [ 1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ],
                     [ 0.75,  0.75,  1.  ,  0.5 ,  1.  ,  0.75,  0.75],
                     [ 0.75,  0.75,  1.  ,  0.5 ,  1.  ,  0.75,  0.75]]

            4. Calculate the values at the percentiles: [10].
               For the point in the upper right corner within the original
               input data e.g. ::

                    [[->1.<-, 1., 1.,],
                     [  1.,   0., 1.],
                     [  1.,   1., 1.]]

               When the kernel is placed over this point within the padded
               data, then the following points are included::

                    [[   0.75,    0.75,  ->1.<-,  0.5 ,  1.  ,  0.75,  0.75],
                     [   0.75,  ->0.75,    1.  ,  0.5<-, 1.  ,  0.75,  0.75],
                     [ ->1.  ,    1.  ,    1.  ,  1.  ,  1.<-,  1.  ,  1.  ],
                     [   0.5 ,  ->0.5 ,    1.  ,  0.<-,  1.  ,  0.5 ,  0.5 ],
                     [   1.  ,    1.  ,  ->1.<-,  1.  ,  1.  ,  1.  ,  1.  ],
                     [   0.75,    0.75,    1.  ,  0.5 ,  1.  ,  0.75,  0.75],
                     [   0.75,    0.75,    1.  ,  0.5 ,  1.  ,  0.75,  0.75]]

               This gives::

                    [0, 0.5, 0.5, 0.75, 1., 1., 1., 1., 1., 1., 1., 1., 1.]

               As there are 13 points within the kernel, this gives the
               following relationship between percentiles and values.

                  ======  ==========
                  Values  Percentile
                  ======  ==========
                  0.      0
                  0.5     8.33
                  0.5     16.67
                  0.75    25.0
                  1.      33.33
                  1.      41.67
                  1.      50.0
                  1.      58.33
                  1.      66.67
                  1.      75.0
                  1.      83.33
                  1.      91.66
                  1.      100.
                  ======  ==========

               Therefore, for the 10th percentile at the value returned for
               the point in the upper right corner of the original input data
               is 0.5.
               When this process is applied to every point within the original
               input data, the result is::

                    [[[ 0.75,  0.75,  0.5 ,  0.5 ,  0.5 ,  0.75,  0.75],
                      [ 0.75,  0.55,  0.55,  0.5 ,  0.55,  0.55,  0.55],
                      [ 0.55,  0.55,  0.5 ,  0.5 ,  0.5 ,  0.5 ,  0.5 ],
                      [ 0.5 ,  0.5 ,  0.5 ,  0.5 ,  0.5 ,  0.5 ,  0.5 ],
                      [ 0.5 ,  0.5 ,  0.5 ,  0.5 ,  0.5 ,  0.55,  0.55],
                      [ 0.55,  0.55,  0.55,  0.5 ,  0.55,  0.55,  0.75],
                      [ 0.75,  0.75,  0.5 ,  0.5 ,  0.5 ,  0.75,  0.75]]],

            5. The padding is then removed to give::

                   [[[ 0.5,  0.5,  0.5],
                     [ 0.5,  0.5,  0.5],
                     [ 0.5,  0.5,  0.5]]]
        """
        ranges_xy = np.empty(2, dtype=int)
        ranges_xy[0] = int(np.floor(kernel.shape[0] / 2.0))
        ranges_xy[1] = int(np.floor(kernel.shape[1] / 2.0))
        padded = np.pad(slice_2d.data, ranges_xy, mode='mean',
                        stat_length=np.max(ranges_xy))
        padshape = np.shape(padded)  # Store size to make unflatten easier
        padded = padded.flatten()
        # Add 2nd dimension with each point's neighbourhood points along it.
        # nbhood_slices is a list of numpy arrays where each array contains the
        # total number of points within the padded array. The number of arrays
        # is equal to the number of points within the kernel.
        nbhood_slices = [
            np.roll(padded, (padshape[1]*j)+i)
            for i in range(-ranges_xy[1], ranges_xy[1]+1)
            for j in range(-ranges_xy[0], ranges_xy[0]+1)
            if kernel[..., i+ranges_xy[1], j+ranges_xy[0]] > 0.]

        # Collapse this dimension into percentiles (a new 2nd dimension)
        perc_data = np.percentile(
            nbhood_slices,
            np.array(self.percentiles, dtype=np.float32),
            axis=0
        )

        # Convert back to float32 (np.percentile always gives float64 here...)
        perc_data = perc_data.astype(np.float32)

        # Return to 3D
        perc_data = perc_data.reshape(
            len(self.percentiles), padshape[0], padshape[1])
        # Create a cube for these data:
        pctcube = self.make_percentile_cube(slice_2d)
        # And put in data, removing the padding
        pctcube.data = perc_data[:, ranges_xy[0]:-ranges_xy[0],
                                 ranges_xy[1]:-ranges_xy[1]]
        return pctcube

    def run(self, cube, radius, mask_cube=None):
        """
        Method to apply a circular kernel to the data within the input cube in
        order to derive percentiles over the kernel.

        Args:
            cube (iris.cube.Cube):
                Cube containing array to apply processing to.
            radius (float):
                Radius in metres for use in specifying the number of
                grid cells used to create a circular neighbourhood.
            mask_cube (iris.cube.Cube or None):
                Cube containing the array to be used as a mask.

        Returns:
            iris.cube.Cube:
                Cube containing the percentile fields.
                Has percentile as an added dimension.

        """
        if mask_cube is not None:
            msg = ("The use of a mask cube with a circular kernel is not "
                   "yet implemented.")
            raise NotImplementedError(msg)

        # Check that the cube has an equal area grid.
        check_if_grid_is_equal_area(cube)
        # Take data array and identify X and Y axes indices
        grid_cell = distance_to_number_of_grid_cells(cube, radius)
        check_radius_against_distance(cube, radius)
        ranges_xy = np.array((grid_cell, grid_cell))
        kernel = circular_kernel(ranges_xy, grid_cell, weighted_mode=False)
        # Loop over each 2D slice to reduce memory demand and derive
        # percentiles on the kernel. Will return an extra dimension.
        pctcubelist = iris.cube.CubeList()
        for slice_2d in cube.slices(['projection_y_coordinate',
                                     'projection_x_coordinate']):
            pctcubelist.append(self.pad_and_unpad_cube(slice_2d, kernel))

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
        if result.coords("percentile"):
            required_order.append(
                result.coord_dims("percentile")[0])
        other_coords = []
        for coord in result.dim_coords:
            if coord.name() not in ["realization",
                                    "percentile"]:
                other_coords.append(result.coord_dims(coord.name())[0])
        required_order.extend(other_coords)
        result.transpose(required_order)

        return result

    def make_percentile_cube(self, cube):
        """Returns a cube with the same metadata as the sample cube
        but with an added percentile dimension.

        Args:
            cube (iris.cube.Cube):
                Cube to copy meta data from.
        Returns:
            iris.cube.Cube:
                Cube like input but with added percentiles coordinate.
                Each slice along this coordinate is identical.
        """
        pctcubelist = iris.cube.CubeList()
        pct_coord_name = "percentile"
        for pct in self.percentiles:
            pctcube = cube.copy()
            pctcube.add_aux_coord(iris.coords.DimCoord(
                np.float32(pct), long_name=pct_coord_name, units='%'))
            pctcubelist.append(pctcube)
        result = pctcubelist.merge_cube()
        # If percentile coord is not already a dimension, promote it.
        # This is required when self.percentiles is length 1.
        if result.coord_dims(pct_coord_name) == ():
            result = iris.util.new_axis(result, scalar_coord=pct_coord_name)
        return result
