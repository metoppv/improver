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
""" Provides support utilities."""

import copy
from iris.coords import CellMethod, DimCoord
from iris.cube import Cube
import numpy as np


def convert_distance_into_number_of_grid_cells(
        cube, distance, max_distance_in_grid_cells):
    """
    Return the number of grid cells in the x and y direction based on the
    input distance in metres.

    Parameters
    ----------
    cube : Iris.cube.Cube
        Cube containing the x and y coordinates, which will be used for
        calculating the number of grid cells in the x and y direction,
        which equates to the requested distance in the x and y direction.
    distance : Float
        Distance in metres.
    max_distance_in_grid_cells : integer
        Maximum distance in grid cells.

    Returns
    -------
    grid_cells_x : Integer
        Number of grid cells in the x direction based on the requested
        distance in metres.
    grid_cells_y : Integer
        Number of grid cells in the y direction based on the requested
        distance in metres.

    """
    try:
        x_coord = cube.coord("projection_x_coordinate").copy()
        y_coord = cube.coord("projection_y_coordinate").copy()
    except CoordinateNotFoundError:
        raise ValueError("Invalid grid: projection_x/y coords required")
    x_coord.convert_units("metres")
    y_coord.convert_units("metres")
    max_distance_of_domain = np.sqrt(
        (x_coord.points.max() - x_coord.points.min())**2 +
        (y_coord.points.max() - y_coord.points.min())**2)
    if distance > max_distance_of_domain:
        raise ValueError(
            ("Distance of {0}m exceeds max domain distance of {1}m".format(
                 distance, max_distance_of_domain)))
    d_north_metres = y_coord.points[1] - y_coord.points[0]
    d_east_metres = x_coord.points[1] - x_coord.points[0]
    grid_cells_y = int(distance / abs(d_north_metres))
    grid_cells_x = int(distance / abs(d_east_metres))
    if grid_cells_x == 0 or grid_cells_y == 0:
        raise ValueError(
            "Distance of {0}m gives zero cell extent".format(distance))
    elif grid_cells_x < 0 or grid_cells_y < 0:
        raise ValueError(
            "Neighbourhood processing distance of {0}m "
            "gives a negative cell extent".format(distance))
    if (grid_cells_x > max_distance_in_grid_cells or
            grid_cells_y > max_distance_in_grid_cells):
        raise ValueError(
            "Neighbourhood processing distance of {0}m "
            "exceeds maximum grid cell extent".format(distance))
    return grid_cells_x, grid_cells_y


class DifferenceBetweenAdjacentGridSquares(object):

    """
    Calculate the difference between adjacent grid squares within
    a cube. The difference is calculated along the x and y axis
    individually.
    """

    def __init__(self):
        """
        Initialise class.
        """
        pass

    @staticmethod
    def create_difference_cube(
            cube, coord_name, diff_along_axis):
        """
        Put the difference array into a cube with the appropriate
        metadata.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube from which the differences have been calculated.
        coord_name : String
            The name of the coordinate over which the difference
            have been calculated.
        diff_along_axis : numpy array
            Array containing the differences.

        Returns
        -------
        diff_cube : Iris.cube.Cube
            Cube containing the differences calculated along the
            specified axis.
        """
        points = cube.coord(coord_name).points
        mean_points = (points[1:] + points[:-1]) / 2

        # Copy cube metadata and coordinates into a new cube.
        # Create a new coordinate for the coordinate along which the
        # difference has been calculated.
        metadata_dict = copy.deepcopy(cube.metadata._asdict())
        diff_cube = Cube(diff_along_axis, **metadata_dict)

        for coord in cube.dim_coords:
            dims = cube.coord_dims(coord)
            if coord.name() in [coord_name]:
                coord = coord.copy(points=mean_points)
            diff_cube.add_dim_coord(coord.copy(), dims)
        for coord in cube.aux_coords:
            dims = cube.coord_dims(coord)
            diff_cube.add_aux_coord(coord.copy(), dims)
        for coord in cube.derived_coords:
            dims = cube.coord_dims(coord)
            diff_cube.add_aux_coord(coord.copy(), dims)

        # Add metadata to indicate that a difference has been calculated.
        # TODO: update this metadata when proper conventions have been
        #       agreed upon.
        cell_method = CellMethod("difference", coords=[coord_name],
                                 intervals='1 grid length')
        diff_cube.add_cell_method(cell_method)
        diff_cube.attributes["form_of_difference"] = (
            "forward_difference")
        diff_cube.rename('difference_of_' + cube.name())
        return diff_cube

    def calculate_difference(self, cube, coord_axis):
        """
        Calculate the difference along the axis specified by the
        coordinate.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube from which the differences will be calculated.
        coord_axis : String
            Short-hand reference for the x or y coordinate, as allowed by
            iris.util.guess_coord_axis.

        Returns
        -------
        diff_cube : Iris.cube.Cube
            Cube after the differences have been calculated along the
            specified axis.
        """
        coord_name = cube.coord(axis=coord_axis).name()
        diff_axis = cube.coord_dims(coord_name)[0]
        diff_along_axis = np.diff(cube.data, axis=diff_axis)
        diff_cube = self.create_difference_cube(
            cube, coord_name, diff_along_axis)
        return diff_cube

    def process(self, cube):
        """
        Calculate the difference along the x and y axes and return
        the result in separate cubes. The difference along each axis is
         calculated using numpy.diff.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube from which the differences will be calculated.

        Returns
        -------
        diff_along_y_cube : Iris.cube.Cube
            Cube after the differences have been calculated along the
            y axis.
        diff_along_x_cube : Iris.cube.Cube
            Cube after the differences have been calculated along the
            x axis.

        """
        diff_along_y_cube = self.calculate_difference(cube, "y")
        diff_along_x_cube = self.calculate_difference(cube, "x")
        return diff_along_x_cube, diff_along_y_cube


class OccurrenceWithinVicinity(object):

    def __init__(self, distance):
        self.distance = distance

    def maximum_within_vicinity(self, cube):

        grid_cell_x, grid_cell_y = (
            Utilities.get_neighbourhood_width_in_grid_cells(
                cube, radius, MAX_RADIUS_IN_GRID_CELLS))

        # Option 1
        max_cube = cube.copy()
        max_cube.data = np.zeros(cube.data.shape)
        for ii in range(cube.coord(axis="y").points):
            for jj in range(cube.coord(axis="x").points):
                if ii-grid_cell_y < 0:
                    for index in range(ii-grid_cell_y, 0):
                        if index >= 0:
                            break
                    lower_y_index = index
                else:
                    lower_y_index = ii-grid_cell_y
                if jj-grid_cell_x < 0:
                    for index in range(jj-grid_cell_x, 0):
                        if index >= 0:
                            break
                    lower_x_index = index
                else:
                    lower_x_index = jj-grid_cell_x
                if ii+grid_cell_y > ylen:
                    for index in range(ii+grid_cell_y, 0):
                        if index >= 0:
                            break
                    upper_y_index = index
                else:
                    lower_y_index = ii-grid_cell_y
                if jj+grid_cell_x > xlen:
                    for index in range(jj+grid_cell_x, 0):
                        if index >= 0:
                            break
                    upper_x_index = index
                else:
                    upper_x_index = jj+grid_cell_x
                max_cube[ii, jj] = np.max(
                    cube.data[lower_y_index:upper_y_index,
                              lower_x_index:upper_x_index])

        # Option 2
        max_cube = cube.copy()
        ylen = len(max_cube.coord(axis="y").points)
        xlen = len(max_cube.coord(axis="x").points)
        nlen = xlen * ylen

        data_1d = max_cube.data.flatten()
        indices_above_zero = np.where(data_1d>0)

        for index in indices_above_zero:
            increment = ylen * self.distance
            for index_1d in range(index-increment,index+increment, ylen):
                if index_1d >= 0 and index_1d <= nlen-1:
                    data_1d[index_1d] = (
                        np.max(data_1d[index], data_1d[index_1d]))

        data = data_1d.reshape([ylen, xlen])
        max_cube.data = data
        return max_cube

    def process(self, cube):

        try:
            realization_coord = cube.coord('realization')
            slices_over_realization = cube.slices_over("realization")
        except iris.exceptions.CoordinateNotFoundError:
            slices_over_realization = iris.cube.CubeList([cube])

        max_cubes = iris.cube.CubeList([])
        for realization_slice in slices_over_realization:
            for time_slice in realization_slice.slices_over("time"):
                max_cubes.append(self.maximum_within_vicinity(time_slice))
        return max_cubes.merge_cube()
