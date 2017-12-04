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
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError
import numpy as np
import scipy.ndimage

from improver.utilities.cube_checker import check_cube_coordinates


# Maximum radius of the neighbourhood width in grid cells.
MAX_DISTANCE_IN_GRID_CELLS = 500


def check_if_grid_is_equal_area(cube):
    """Identify whether the grid is an equal area grid.
    If not, raise an error.
    Args:
        cube (Iris.cube.Cube):
            Cube with coordinates that will be cMAhecked.
    Raises:
        ValueError : Invalid grid: projection_x/y coords required
        ValueError : Intervals between points along the x and y axis vary.
                     Therefore the grid is not an equal area grid.
        ValueError : The size of the intervals along the x and y axis
                     should be equal.
    """
    try:
        for coord_name in ['projection_x_coordinate',
                           'projection_y_coordinate']:
            cube.coord(coord_name)
    except CoordinateNotFoundError:
        raise ValueError("Invalid grid: projection_x/y coords required")
    for coord_name in ['projection_x_coordinate',
                       'projection_y_coordinate']:
        if np.sum(np.diff(np.diff(cube.coord(coord_name).points))) > 0:
            msg = ("Intervals between points along the {} axis vary."
                   "Therefore the grid is not an equal area grid."
                   ).format(coord_name)
            raise ValueError(msg)
    x_diff = np.diff(cube.coord("projection_x_coordinate").points)[0]
    y_diff = np.diff(cube.coord("projection_y_coordinate").points)[0]
    if abs(x_diff) != abs(y_diff):
        msg = ("The size of the intervals along the x and y axis "
               "should be equal. x axis interval: {}, y axis interval: {}"
               ).format(x_diff, y_diff)
        raise ValueError(msg)


def convert_distance_into_number_of_grid_cells(
        cube, distance, max_distance_in_grid_cells):
    """
    Return the number of grid cells in the x and y direction based on the
    input distance in metres.

    Args:
        cube (Iris.cube.Cube):
            Cube containing the x and y coordinates, which will be used for
            calculating the number of grid cells in the x and y direction,
            which equates to the requested distance in the x and y direction.
        distance (Float):
            Distance in metres.
        max_distance_in_grid_cells (int):
            Maximum distance in grid cells.

    Returns:
        (tuple) : tuple containing:
            **grid_cells_x** (int):
                Number of grid cells in the x direction based on the requested
                distance in metres.
            **grid_cells_y** (int):
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

        Args:
            cube (Iris.cube.Cube):
                Cube from which the differences have been calculated.
            coord_name (String):
                The name of the coordinate over which the difference
                have been calculated.
            diff_along_axis (numpy array):
                Array containing the differences.

        Returns:
            diff_cube (Iris.cube.Cube):
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

        Args:
            cube (Iris.cube.Cube):
                Cube from which the differences will be calculated.
            coord_axis (String):
                Short-hand reference for the x or y coordinate, as allowed by
                iris.util.guess_coord_axis.

        Returns:
            diff_cube (Iris.cube.Cube):
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

        Args:
            cube (Iris.cube.Cube):
                Cube from which the differences will be calculated.

        Returns:
            (tuple) : tuple containing:
                **diff_along_y_cube** (Iris.cube.Cube):
                    Cube after the differences have been calculated along the
                    y axis.
                **diff_along_x_cube** (Iris.cube.Cube):
                    Cube after the differences have been calculated along the
                    x axis.

        """
        diff_along_y_cube = self.calculate_difference(cube, "y")
        diff_along_x_cube = self.calculate_difference(cube, "x")
        return diff_along_x_cube, diff_along_y_cube


class OccurrenceWithinVicinity(object):

    """Calculate whether a phenomenon occurs within the specified distance."""

    def __init__(self, distance):
        """
        Initialise the class.

        Args:
            distance (float):
                Distance in metres used to define the vicinity within which to
                search for an occurrence.

        """
        self.distance = distance

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<OccurrenceWithinVicinity: distance: {}>')
        return result.format(self.distance)

    def maximum_within_vicinity(self, cube):
        """
        Find grid points where a phenomenon occurs within a defined distance.
        The occurrences within this vicinity are maximised, such that all
        grid points within the vicinity are recorded as having an occurrence.
        For non-binary fields, if the vicinity of two occurrences overlap,
        the maximum value within the vicinity is chosen.

        Args:
            cube (Iris.cube.Cube):
                Thresholded cube.

        Returns:
            cube (Iris.cube.Cube):
                Cube where the occurrences have been spatially spread, so that
                they're equally likely to have occurred anywhere within the
                vicinity defined using the specified distance.

        """
        # The number of grid cells returned along the x and y axis will be
        # the same.
        _, grid_cell_y = (
            convert_distance_into_number_of_grid_cells(
                cube, self.distance, MAX_DISTANCE_IN_GRID_CELLS))

        # Convert the number of grid points (e.g. grid_cell_y) represented
        # by self.distance, e.g. where grid_cell_y=1 is an increment to
        # a central point, into grid_cells which is the total number of points
        # within the defined vicinity along the y axis e.g grid_cells=3.
        grid_cells = (2 * grid_cell_y) + 1

        max_cube = cube.copy()
        # The following command finds the maximum value for each grid point
        # from within a square of length "size"
        max_cube.data = (
            scipy.ndimage.filters.maximum_filter(cube.data, size=grid_cells))
        return max_cube

    def process(self, cube):
        """
        Ensure that the cube passed to the maximum_within_vicinity method is
        2d and subsequently merged back together.

        Args:
            cube (Iris.cube.Cube):
                Thresholded cube.

        Returns:
            Iris.cube.Cube
                Cube containing the occurrences within a vicinity for each
                xy 2d slice, which have been merged back together.

        """

        max_cubes = CubeList([])
        for cube_slice in cube.slices([cube.coord(axis='y'),
                                       cube.coord(axis='x')]):
            max_cubes.append(self.maximum_within_vicinity(cube_slice))
        result_cube = max_cubes.merge_cube()

        # Put dimensions back if they were there before.
        result_cube = check_cube_coordinates(cube, result_cube)

        return result_cube
