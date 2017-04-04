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


#import iris
import numpy as np
import scipy.ndimage.filters


class BasicNeighbourhoodProcessing(object):
    """
    Apply a neigbourhood processing kernel to a thresholded cube.

    When applied to a thresholded probabilistic cube, it acts like a
    low-pass filter which reduces noisiness in the probabilities.

    The kernel will presently only work with projections in which the
    x grid point spacing and y grid point spacing are constant over the
    entire domain, such as the UK national grid projection.

    A maximum kernel radius of 500 grid cells is imposed in order to
    avoid computational ineffiency and possible memory errors.

    """

    # Max extent of kernel in grid cells.
    MAX_KERNEL_CELL_RADIUS = 500

    def __init__(self, radius_in_km, unweighted_mode=False):
        """
        Create a neighbourhood processing plugin that applies a smoothing
        kernel to points in a cube.

        Parameters
        ----------

        radius_in_km : float
            The radius in kilometres of the neighbourhood kernel to
            apply. Rounded up to convert into integer number of grid
            points east and north, based on the characteristic spacing
            at the zero indices of the cube projection-x/y coords.

        unweighted_mode : boolean
            If True, use a circle with constant weighting.
            If False, use a circle for neighbourhood kernel with
            weighting decreasing with radius.

        """
        self.radius_in_km = float(radius_in_km)
        self.unweighted_mode = bool(unweighted_mode)

    def __str__(self):
        result = ('<NeighbourhoodProcessing: radius_in_km: {};' +
                  'unweighted_mode: {}>')
        return result.format(
            self.radius_in_km, self.unweighted_mode)

    def get_grid_x_y_kernel_ranges(self, cube):
        """Return grid cell numbers east and north for the kernel."""
        try:
            x_coord = cube.coord("projection_x_coordinate").copy()
            y_coord = cube.coord("projection_y_coordinate").copy()
        except iris.exceptions.CoordinateNotFoundError:
            raise ValueError("Invalid grid: projection_x/y coords required")
        x_coord.convert_units("metres")
        y_coord.convert_units("metres")
        d_north_metres = y_coord.points[1] - y_coord.points[0]
        d_east_metres = x_coord.points[1] - x_coord.points[0]
        grid_cells_y = int(self.radius_in_km * 1000 / abs(d_north_metres))
        grid_cells_x = int(self.radius_in_km * 1000 / abs(d_east_metres))
        if grid_cells_x == 0 or grid_cells_y == 0:
            raise ValueError(
                ("Neighbourhood processing radius of " +
                 "{0} km ".format(self.radius_in_km) +
                 "gives zero cell extent")
            )
        if (grid_cells_x > self.MAX_KERNEL_CELL_RADIUS or
                grid_cells_y > self.MAX_KERNEL_CELL_RADIUS):
            raise ValueError(
                ("Neighbourhood processing radius of " +
                 "{0} km ".format(self.radius_in_km) +
                 "exceeds maximum grid cell extent")
            )
        return grid_cells_x, grid_cells_y

    def process(self, cube):
        """
        Set the specified name and units metadata to the cube from the upstream
        plugin.

        Returns
        -------
        Cube
            The cube from the upstream plugin with name and units metadata
            applied.

        """
        try:
            realiz_coord = cube.coord('realization')
        except iris.exceptions.CoordinateNotFoundError:
            pass
        else:
            if len(realiz_coord.points) > 1:
                raise ValueError("Does not operate across realizations.")
        if np.isnan(cube.data).any():
            raise ValueError("Error: NaN detected in input cube data")
        data = cube.data
        ranges = self.get_grid_x_y_kernel_ranges(cube)
        fullranges = np.zeros([np.rank(data)])
        axes = []
        for coord_name in ['projection_x_coordinate',
                           'projection_y_coordinate']:
            axes.append(cube.coord_dims(coord_name)[0])
        for axis_index, axis in enumerate(axes):
            fullranges[axis] = ranges[axis_index]
        kernel = np.ones([1 + x * 2 for x in fullranges])
        n = np.ogrid[tuple([slice(-x, x+1) for x in ranges])]
        if self.unweighted_mode:
            mask = np.reshape(
                np.sum([x ** 2 for x in n]) > np.cumprod(ranges)[-1],
                np.shape(kernel)
            )
        else:
            kernel[:] = (
                (np.cumprod(ranges)[-1] - np.sum([x**2. for x in n])) /
                np.cumprod(ranges)[-1]
            )
            mask = kernel < 0.
        kernel[mask] = 0.
        cube.data = scipy.ndimage.filters.correlate(
            data, kernel, mode='nearest') / np.sum(kernel)
        return cube
