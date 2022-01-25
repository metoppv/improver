# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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

from typing import List, Optional, Union

import iris
import numpy as np
from iris.cube import Cube
from numpy import ndarray

from improver.constants import DEFAULT_PERCENTILES
from improver.nbhood.nbhood import BaseNeighbourhoodProcessing
from improver.utilities.cube_checker import (
    check_cube_coordinates,
    find_dimension_coordinate_mismatch,
)
from improver.utilities.neighbourhood_tools import pad_and_roll
from improver.utilities.spatial import (
    check_if_grid_is_equal_area,
    distance_to_number_of_grid_cells,
)


def check_radius_against_distance(cube: Cube, radius: float) -> None:
    """Check required distance isn't greater than the size of the domain.

    Args:
        cube:
            The cube to check.
        radius:
            The radius, which cannot be more than half of the
            size of the domain.
    """
    axes = []
    for axis in ["x", "y"]:
        coord = cube.coord(axis=axis).copy()
        coord.convert_units("metres")
        axes.append((max(coord.points) - min(coord.points)))

    max_allowed = np.sqrt(axes[0] ** 2 + axes[1] ** 2) * 0.5
    if radius > max_allowed:
        raise ValueError(
            f"Distance of {radius}m exceeds max domain " f"distance of {max_allowed}m"
        )


def circular_kernel(ranges: int, weighted_mode: bool) -> ndarray:
    """
    Method to create a circular kernel.

    Args:
        ranges:
            Number of grid cells in the x and y direction used to create
            the kernel.
        weighted_mode:
            If True, use a circle for neighbourhood kernel with
            weighting decreasing with radius.
            If False, use a circle with constant weighting.

    Returns:
        Array containing the circular smoothing kernel.
        This will have the same number of dimensions as fullranges.
    """
    # The range is square

    area = ranges * ranges
    # Define the size of the kernel based on the number of grid cells
    # contained within the desired radius.
    kernel = np.ones((int(1 + ranges * 2), (int(1 + ranges * 2))))
    # Create an open multi-dimensional meshgrid.
    open_grid = np.array(np.ogrid[[slice(-x, x + 1) for x in (ranges, ranges)]])
    if weighted_mode:
        # Create a kernel, such that the central grid point has the
        # highest weighting, with the weighting decreasing with distance
        # away from the central grid point.
        open_grid_summed_squared = np.sum(open_grid ** 2.0).astype(float)
        kernel[:] = (area - open_grid_summed_squared) / area
        mask = kernel < 0.0
    else:
        mask = np.reshape(np.sum(open_grid ** 2) > area, np.shape(kernel))
    kernel[mask] = 0.0
    return kernel


class GeneratePercentilesFromANeighbourhood(BaseNeighbourhoodProcessing):

    """Class for generating percentiles from a circular neighbourhood."""

    def __init__(
        self,
        radii: Union[float, List[float]],
        lead_times: Optional[List] = None,
        percentiles: List = DEFAULT_PERCENTILES,
    ) -> None:
        """
        Create a neighbourhood processing subclass that generates percentiles
        from a 2D circular neighbourhood. A maximum kernel radius of 500
        grid cells is imposed in order to avoid computational inefficiency and
        possible memory errors.

        Args:
            radii:
                The radii in metres of the neighbourhood to apply.
                Rounded up to convert into integer number of grid
                points east and north, based on the characteristic spacing
                at the zero indices of the cube projection-x and y coords.
            lead_times:
                List of lead times or forecast periods, at which the radii
                within 'radii' are defined. The lead times are expected
                in hours.
            percentiles:
                Percentile values at which to calculate; if not provided uses
                DEFAULT_PERCENTILES.
        """
        super().__init__(radii, lead_times=lead_times)
        try:
            self.percentiles = tuple(percentiles)
        except TypeError:
            self.percentiles = (percentiles,)

    def pad_and_unpad_cube(self, slice_2d: Cube, kernel: ndarray) -> Cube:
        """
        Method to pad and unpad a two dimensional cube. The input array is
        padded and percentiles are calculated using a neighbourhood around
        each point. The resulting percentile data are unpadded and put into a
        cube.

        Args:
            slice_2d:
                2d cube to be padded with a halo.
            kernel:
                Kernel used to specify the neighbourhood to consider when
                calculating the percentiles within a neighbourhood.

        Returns:
            A cube containing percentiles generated from a
            neighbourhood.

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
        kernel_mask = kernel > 0
        nb_slices = pad_and_roll(
            slice_2d.data, kernel.shape, mode="mean", stat_length=max(kernel.shape) // 2
        )
        percentiles = np.array(self.percentiles, dtype=np.float32)

        # Create cube for output percentile data.
        pctcube = self.make_percentile_cube(slice_2d)

        # Collapse neighbourhood windows into percentiles.
        # (Loop over outer dimension to reduce memory footprint.)
        for nb_chunk, perc_chunk in zip(nb_slices, pctcube.data.swapaxes(0, 1)):
            np.percentile(
                nb_chunk[..., kernel_mask],
                percentiles,
                axis=-1,
                out=perc_chunk,
                overwrite_input=True,
            )

        return pctcube

    def process(self, cube: Cube, mask_cube: Optional[Cube] = None) -> Cube:
        """
        Method to apply a circular kernel to the data within the input cube in
        order to derive percentiles over the kernel.

        Args:
            cube:
                Cube containing array to apply processing to.

        Returns:
            Cube containing the percentile fields.
            Has percentile as an added dimension.
        """
        super().process(cube)
        if np.ma.is_masked(cube.data):
            msg = (
                "The use of masked input cubes is not yet implemented in"
                " the GeneratePercentilesFromANeighbourhood plugin."
            )
            raise NotImplementedError(msg)

        # Check that the cube has an equal area grid.
        check_if_grid_is_equal_area(cube)
        # Take data array and identify X and Y axes indices
        grid_cell = distance_to_number_of_grid_cells(cube, self.radius)
        check_radius_against_distance(cube, self.radius)
        kernel = circular_kernel(grid_cell, weighted_mode=False)
        # Loop over each 2D slice to reduce memory demand and derive
        # percentiles on the kernel. Will return an extra dimension.
        pctcubelist = iris.cube.CubeList()
        for slice_2d in cube.slices(
            ["projection_y_coordinate", "projection_x_coordinate"]
        ):
            pctcubelist.append(self.pad_and_unpad_cube(slice_2d, kernel))

        result = pctcubelist.merge_cube()
        exception_coordinates = find_dimension_coordinate_mismatch(
            cube, result, two_way_mismatch=False
        )
        result = check_cube_coordinates(
            cube, result, exception_coordinates=exception_coordinates
        )

        # Arrange cube, so that the coordinate order is:
        # realization, percentile, other coordinates.
        required_order = []
        if result.coords("realization"):
            if result.coords("realization", dimensions=[]):
                result = iris.util.new_axis(result, "realization")
            required_order.append(result.coord_dims("realization")[0])
        if result.coords("percentile"):
            required_order.append(result.coord_dims("percentile")[0])
        other_coords = []
        for coord in result.dim_coords:
            if coord.name() not in ["realization", "percentile"]:
                other_coords.append(result.coord_dims(coord.name())[0])
        required_order.extend(other_coords)
        result.transpose(required_order)

        return result

    def make_percentile_cube(self, cube: Cube) -> Cube:
        """Returns a cube with the same metadata as the sample cube
        but with an added percentile dimension.

        Args:
            cube:
                Cube to copy meta data from.

        Returns:
            Cube like input but with added percentiles coordinate.
            Each slice along this coordinate is identical.
        """
        pctcubelist = iris.cube.CubeList()
        pct_coord_name = "percentile"
        for pct in self.percentiles:
            pctcube = cube.copy()
            pctcube.add_aux_coord(
                iris.coords.DimCoord(
                    np.float32(pct), long_name=pct_coord_name, units="%"
                )
            )
            pctcubelist.append(pctcube)
        result = pctcubelist.merge_cube()
        # If percentile coord is not already a dimension, promote it.
        # This is required when self.percentiles is length 1.
        if result.coord_dims(pct_coord_name) == ():
            result = iris.util.new_axis(result, scalar_coord=pct_coord_name)
        return result
