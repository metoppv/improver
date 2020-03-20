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
""" Provides support utilities."""

import copy

import cartopy.crs as ccrs
import iris
import numpy as np
from scipy.ndimage.filters import maximum_filter
from iris.coords import CellMethod
from iris.cube import Cube, CubeList

from improver import BasePlugin, PostProcessingPlugin
from improver.utilities.cube_checker import check_cube_coordinates


def check_if_grid_is_equal_area(cube, require_equal_xy_spacing=True):
    """
    Identify whether the grid is an equal area grid, by checking whether points
    are equally spaced along each of the x- and y-axes.  By default this
    function also checks whether the grid spacing is the same in both spatial
    dimensions.

    Args:
        cube (iris.cube.Cube):
            Cube with coordinates that will be checked.
        require_equal_xy_spacing (bool):
            Flag to require the grid is equally spaced in the two spatial
            dimensions (not strictly required for equal-area criterion).

    Raises:
        ValueError: If coordinate points are not equally spaced along either
            axis (from calculate_grid_spacing)
        ValueError: If point spacing is not equal for the two spatial axes
    """
    x_diff = calculate_grid_spacing(cube, 'metres', axis='x')
    y_diff = calculate_grid_spacing(cube, 'metres', axis='y')
    if require_equal_xy_spacing and not np.isclose(x_diff, y_diff):
        raise ValueError(
            "Grid does not have equal spacing in x and y dimensions")


def calculate_grid_spacing(cube, units, axis='x'):
    """
    Returns the grid spacing of a given spatial axis

    Args:
        cube (iris.cube.Cube):
            Cube of data on equal area grid
        units (str or cf_units.Unit):
            Unit in which the grid spacing is required
        axis (str):
            Axis ('x' or 'y') to use in determining grid spacing

    Returns:
        float:
            Grid spacing in required unit

    Raises:
        ValueError: If points are not equally spaced
    """
    coord = cube.coord(axis=axis).copy()
    coord.convert_units(units)
    diffs = np.unique(np.diff(coord.points))
    if len(diffs) > 1:
        raise ValueError(
            'Coordinate {} points are not equally spaced'.format(coord.name()))
    return diffs[0]


def distance_to_number_of_grid_cells(cube, distance, axis='x',
                                     return_int=True):
    """
    Return the number of grid cells in the x and y direction based on the
    input distance in metres.  Requires an equal-area grid on which the spacing
    is equal in the x- and y- directions.

    Args:
        cube (iris.cube.Cube):
            Cube containing the x and y coordinates, which will be used for
            calculating the number of grid cells in the x and y direction,
            which equates to the requested distance in the x and y direction.
        distance (float):
            Distance in metres. Must be positive.
        return_int (bool):
            If true only integer number of grid_cells are returned, rounded
            down. If false the number of grid_cells returned will be a float.
        axis (str):
            Axis ('x' or 'y') to use in determining grid spacing

    Returns:
        int or float:
            Number of grid cells along the specified (x or y) axis equal to the
            requested distance in metres.

    Raises:
        ValueError: If a non-positive distance is provided.
    """
    d_error = f"Distance of {distance}m"
    if distance <= 0:
        raise ValueError(
            f"Please specify a positive distance in metres. {d_error}")

    # calculate grid spacing along chosen axis
    grid_spacing_metres = calculate_grid_spacing(cube, 'metres', axis=axis)
    grid_cells = distance / abs(grid_spacing_metres)

    if return_int:
        grid_cells = int(grid_cells)
        if grid_cells == 0:
            zero_distance_error = f"{d_error} gives zero cell extent"
            raise ValueError(zero_distance_error)

    return grid_cells


def number_of_grid_cells_to_distance(cube, grid_points):
    """
    Calculate distance in metres equal to the given number of gridpoints
    based on the coordinates on an input cube.

    Args:
        cube (iris.cube.Cube):
            Cube for which the distance is to be calculated.
        grid_points (int):
            Number of grid points to convert.
    Returns:
        float:
            The radius in metres.
    """
    check_if_grid_is_equal_area(cube)
    spacing = calculate_grid_spacing(cube, 'metres')
    radius_in_metres = spacing * grid_points
    return radius_in_metres


class DifferenceBetweenAdjacentGridSquares(BasePlugin):

    """
    Calculate the difference between adjacent grid squares within
    a cube. The difference is calculated along the x and y axis
    individually.
    """

    def __init__(self, gradient=False):
        """
        Initialise class.
        """
        self.is_gradient = gradient

    def create_difference_cube(self, cube, coord_name, diff_along_axis):
        """
        Put the difference array into a cube with the appropriate
        metadata.

        Args:
            cube (iris.cube.Cube):
                Cube from which the differences have been calculated.
            coord_name (str):
                The name of the coordinate over which the difference
                have been calculated.
            diff_along_axis (numpy.ndarray):
                Array containing the differences.

        Returns:
            iris.cube.Cube:
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
        # TODO: update metadata for difference and add metadata for gradient
        #       when proper conventions have been agreed upon.
        if not self.is_gradient:
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
            cube (iris.cube.Cube):
                Cube from which the differences will be calculated.
            coord_axis (str):
                Short-hand reference for the x or y coordinate, as allowed by
                iris.util.guess_coord_axis.

        Returns:
            iris.cube.Cube:
                Cube after the differences have been calculated along the
                specified axis.
        """
        coord_name = cube.coord(axis=coord_axis).name()
        diff_axis = cube.coord_dims(coord_name)[0]
        diff_along_axis = np.diff(cube.data, axis=diff_axis)
        diff_cube = self.create_difference_cube(
            cube, coord_name, diff_along_axis)
        return diff_cube

    @staticmethod
    def gradient_from_diff(diff_cube, ref_cube, coord_axis):
        """
        Calculate the gradient along the x or y axis from differences between
        adjacent grid squares.

        Args:
            diff_cube (iris.cube.Cube):
                Cube containing differences along the x or y axis
            ref_cube (iris.cube.Cube):
                Cube with correct output dimensions
            coord_axis (str):
                Short-hand reference for the x or y coordinate, as allowed by
                iris.util.guess_coord_axis.


        Returns:
            iris.cube.Cube:
                A cube of the gradients in the coordinate direction specified.
        """
        grid_spacing = np.diff(diff_cube.coord(axis=coord_axis).points)[0]
        gradient = diff_cube.copy(data=diff_cube.data / grid_spacing)
        gradient = gradient.regrid(ref_cube, iris.analysis.Linear())
        gradient.rename(diff_cube.name().replace('difference_', 'gradient_'))
        return gradient

    def process(self, cube):
        """
        Calculate the difference along the x and y axes and return
        the result in separate cubes. The difference along each axis is
        calculated using numpy.diff.

        Args:
            cube (iris.cube.Cube):
                Cube from which the differences will be calculated.

        Returns:
            (tuple): tuple containing:
                **diff_along_y_cube** (iris.cube.Cube):
                    Cube after the differences have been calculated along the
                    y axis.
                **diff_along_x_cube** (iris.cube.Cube):
                    Cube after the differences have been calculated along the
                    x axis.

        """
        xy = ['x', 'y']
        diffs = [self.calculate_difference(cube, axis) for axis in xy]
        if self.is_gradient:
            diffs = [self.gradient_from_diff(diff, cube, axis)
                     for diff, axis in zip(diffs, xy)]

        return diffs[0], diffs[1]


class OccurrenceWithinVicinity(PostProcessingPlugin):

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
        result = '<OccurrenceWithinVicinity: distance: {}>'
        return result.format(self.distance)

    def maximum_within_vicinity(self, cube):
        """
        Find grid points where a phenomenon occurs within a defined distance.
        The occurrences within this vicinity are maximised, such that all
        grid points within the vicinity are recorded as having an occurrence.
        For non-binary fields, if the vicinity of two occurrences overlap,
        the maximum value within the vicinity is chosen.

        Args:
            cube (iris.cube.Cube):
                Thresholded cube.

        Returns:
            iris.cube.Cube:
                Cube where the occurrences have been spatially spread, so that
                they're equally likely to have occurred anywhere within the
                vicinity defined using the specified distance.

        """
        grid_spacing = distance_to_number_of_grid_cells(cube, self.distance)

        # Convert the number of grid points (i.e. grid_spacing) represented
        # by self.distance, e.g. where grid_spacing=1 is an increment to
        # a central point, into grid_cells which is the total number of points
        # within the defined vicinity along the y axis e.g grid_cells=3.
        grid_cells = (2 * grid_spacing) + 1

        max_cube = cube.copy()
        unmasked_cube_data = cube.data.copy()
        if np.ma.is_masked(cube.data):
            unmasked_cube_data = cube.data.data.copy()
            unmasked_cube_data[cube.data.mask] = np.nan
        # The following command finds the maximum value for each grid point
        # from within a square of length "size"
        max_data = maximum_filter(unmasked_cube_data, size=grid_cells)
        if np.ma.is_masked(cube.data):
            # Update only the unmasked values
            max_cube.data.data[~cube.data.mask] = max_data[~cube.data.mask]
        else:
            max_cube.data = max_data
        return max_cube

    def process(self, cube):
        """
        Ensure that the cube passed to the maximum_within_vicinity method is
        2d and subsequently merged back together.

        Args:
            cube (iris.cube.Cube):
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


def lat_lon_determine(cube):
    """
    Test whether a diagnostic cube is on a latitude/longitude grid or uses an
    alternative projection.

    Args:
        cube (iris.cube.Cube):
            A diagnostic cube to examine for coordinate system.

    Returns:
        cartopy.crs.CRS or None:
            Coordinate system of the diagnostic cube in a cartopy format unless
            it is already a latitude/longitude grid, in which case None is
            returned.

    """
    trg_crs = None
    if (not cube.coord(axis='x').name() == 'longitude' or
            not cube.coord(axis='y').name() == 'latitude'):
        trg_crs = cube.coord_system().as_cartopy_crs()
    return trg_crs


def transform_grid_to_lat_lon(cube):
    """
    Calculate the latitudes and longitudes of each points in the cube.

    Args:
        cube (iris.cube.Cube):
            Cube with points to transform

    Returns
        (tuple): tuple containing:
            **lats** (numpy.ndarray):
                Array of cube.data.shape of Latitude values
            **lons** (numpy.ndarray):
                Array of cube.data.shape of Longitude values

    """
    trg_latlon = ccrs.PlateCarree()
    trg_crs = cube.coord_system().as_cartopy_crs()
    x_points = cube.coord(axis='x').points
    y_points = cube.coord(axis='y').points
    x_zeros = np.zeros_like(x_points)
    y_zeros = np.zeros_like(y_points)

    # Broadcast x points and y points onto grid
    all_x_points = y_zeros.reshape(len(y_zeros), 1) + x_points
    all_y_points = y_points.reshape(len(y_points), 1) + x_zeros

    # Transform points
    points = trg_latlon.transform_points(trg_crs,
                                         all_x_points,
                                         all_y_points)
    lons = points[..., 0]
    lats = points[..., 1]

    return lats, lons
