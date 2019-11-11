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
import scipy.ndimage
from iris.coords import CellMethod
from iris.cube import Cube, CubeList
from iris.exceptions import CoordinateNotFoundError
from scipy.interpolate import griddata

from improver.threshold import BasicThreshold
from improver.utilities.cube_checker import (
    check_cube_coordinates, spatial_coords_match)

# Maximum radius of the neighbourhood width in grid cells.
MAX_DISTANCE_IN_GRID_CELLS = 500


def check_if_grid_is_equal_area(cube, require_equal_xy_spacing=True):
    """
    Identify whether the grid is an equal area grid, by checking whether points
    are equally spaced along each of the x- and y-axes.  By default this
    function also checks whether the grid spacing is the same in both spatial
    dimensions.

    Args:
        cube (iris.cube.Cube):
            Cube with coordinates that will be checked.
        require_equal_spacing (bool):
            Flag to require the grid is equally spaced in the two spatial
            dimensions (not strictly required for equal-area criterion).

    Raises:
        ValueError: If coordinate points are not equally spaced along either
            axis (from calculate_grid_spacing)
        ValueError: If point spacing is not equal for the two spatial axes
    """
    xdiff = calculate_grid_spacing(cube, 'metres', axis='x')
    ydiff = calculate_grid_spacing(cube, 'metres', axis='y')
    if require_equal_xy_spacing and not np.isclose(xdiff, ydiff):
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


def convert_distance_into_number_of_grid_cells(
        cube, distance, axis='x', max_distance_in_grid_cells=None,
        int_grid_cells=True):
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
            Distance in metres.  Must be positive.
        axis (str):
            Axis ('x' or 'y') to use in determining grid spacing
        max_distance_in_grid_cells (int or None):
            Maximum distance in grid cells.  Defaults to None, which bypasses
            the check.
        int_grid_cells (bool):
            If true only integer number of grid_cells are returned, rounded
            down. If false the number of grid_cells returned will be a float.

    Returns:
        int or float:
            Number of grid cells along the specified (x or y) axis equal to the
            requested distance in metres.

    Raises:
        ValueError: If a negative distance is provided
        ValueError: If the projection is not equal-area
        ValueError:
            If the distance in grid cells is larger than the maximum dimension
            of the rectangular domain (measured across the diagonal).  Needed
            for neighbourhood processing.
        ValueError: If the distance in grid cells is zero.
        Value Error:
            If max_distance_in_grid_cells is set and the distance in grid cells
            exceeds this value.  Needed for neighbourhood processing.
    """
    d_error = "Distance of {}m".format(distance)
    zero_distance_error = ("{} gives zero cell extent".format(d_error))
    if distance == 0:
        raise ValueError(zero_distance_error)
    if distance < 0:
        raise ValueError("Please specify a positive distance in metres")

    # calculate grid spacing along chosen axis
    grid_spacing_metres = calculate_grid_spacing(cube, 'metres', axis=axis)

    # check required distance isn't greater than the size of the domain
    # (note: this implicitly assumes equal x- and y-spacing)
    def calculate_domain_extent(coord):
        """Calculates the coordinate extent in metres"""
        new_coord = coord.copy()
        new_coord.convert_units('metres')
        return max(new_coord.points) - min(new_coord.points)

    x_extent_metres = calculate_domain_extent(cube.coord(axis='x'))
    y_extent_metres = calculate_domain_extent(cube.coord(axis='y'))
    max_distance_of_domain = np.sqrt(x_extent_metres**2 + y_extent_metres**2)
    if distance > max_distance_of_domain:
        raise ValueError(
            "{} exceeds max domain distance of {}m".format(
                d_error, max_distance_of_domain))

    # calculate distance in grid squares
    grid_cells = distance / abs(grid_spacing_metres)

    if int_grid_cells:
        grid_cells = int(grid_cells)
        if grid_cells == 0:
            raise ValueError(zero_distance_error)

    if max_distance_in_grid_cells is not None:
        if grid_cells > max_distance_in_grid_cells:
            raise ValueError(
                "{} exceeds maximum permitted grid cell extent".format(
                    d_error))

    return grid_cells


def convert_number_of_grid_cells_into_distance(cube, grid_points):
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
    check_if_grid_is_equal_area(cube, require_equal_xy_spacing=True)
    spacing = calculate_grid_spacing(cube, 'metres')
    radius_in_metres = spacing * grid_points
    return radius_in_metres


class DifferenceBetweenAdjacentGridSquares(object):

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
        gradient = diff_cube.copy(data=(diff_cube.data) / grid_spacing)
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
        diff_along_y_cube = self.calculate_difference(cube, "y")
        diff_along_x_cube = self.calculate_difference(cube, "x")

        if self.is_gradient:
            diff_along_y_cube = self.gradient_from_diff(diff_along_y_cube,
                                                        cube, "y")
            diff_along_x_cube = self.gradient_from_diff(diff_along_x_cube,
                                                        cube, "x")
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
            cube (iris.cube.Cube):
                Thresholded cube.

        Returns:
            iris.cube.Cube:
                Cube where the occurrences have been spatially spread, so that
                they're equally likely to have occurred anywhere within the
                vicinity defined using the specified distance.

        """
        grid_spacing = (
            convert_distance_into_number_of_grid_cells(
                cube, self.distance,
                max_distance_in_grid_cells=MAX_DISTANCE_IN_GRID_CELLS))

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
        max_data = (
            scipy.ndimage.filters.maximum_filter(unmasked_cube_data,
                                                 size=grid_cells))
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


class RegridLandSea():
    """
    Replace data values at points where the nearest-regridding technique
    selects a source grid-point with an opposite land-sea-mask value to the
    target grid-point.
    The replacement data values are selected from a vicinity of points on the
    source-grid and the closest point of the correct mask is used.
    Where no match is found within the vicinity, the data value is not changed.
    """

    def __init__(self, extrapolation_mode="nanmask", vicinity_radius=25000.):
        """
        Initialise class

        Args:
            extrapolation_mode (str):
                Mode to use for extrapolating data into regions
                beyond the limits of the source_data domain.
                Available modes are documented in
                `iris.analysis <https://scitools.org.uk/iris/docs/latest/iris/
                iris/analysis.html#iris.analysis.Nearest>`_

                Defaults to "nanmask".
            vicinity_radius (float):
                Distance in metres to search for a sea or land point.
        """
        self.input_land = None
        self.nearest_cube = None
        self.output_land = None
        self.output_cube = None
        self.regridder = iris.analysis.Nearest(
            extrapolation_mode=extrapolation_mode)
        self.vicinity = OccurrenceWithinVicinity(vicinity_radius)

    def __repr__(self):
        """
        Print a human-readable representation of the instantiated object.
        """
        return "<RegridLandSea: regridder: {}; vicinity: {}>".format(
            self.regridder, self.vicinity)

    def correct_where_input_true(self, selector_val):
        """
        Replace points in the output_cube where output_land matches the
        selector_val and the input_land does not match, but has matching
        points in the vicinity, with the nearest matching point in the
        vicinity in the original nearest_cube.

        Updates self.output_cube.data

        Args:
            selector_val (int):
                Value of mask to replace if needed.
                Intended to be 1 for filling land points near the coast
                and 0 for filling sea points near the coast.
        """
        # Find all points on output grid matching selector_val
        use_points = np.where(self.input_land.data == selector_val)

        # If there are no matching points on the input grid, no alteration can
        # be made. This tests the size of the y-coordinate of use_points.
        if use_points[0].size is 0:
            return

        # Get shape of output grid
        ynum, xnum = self.output_land.shape

        # Using only these points, extrapolate to fill domain using nearest
        # neighbour. This will generate a grid where the non-selector_val
        # points are filled with the nearest value in the same mask
        # classification.
        (y_points, x_points) = np.mgrid[0:ynum, 0:xnum]
        selector_data = griddata(use_points,
                                 self.nearest_cube.data[use_points],
                                 (y_points, x_points), method="nearest")

        # Identify nearby points on regridded input_land that match the
        # selector_value
        if selector_val > 0.5:
            thresholder = BasicThreshold(0.5)
        else:
            thresholder = BasicThreshold(0.5, comparison_operator='<=')
        in_vicinity = self.vicinity.process(
            thresholder.process(self.input_land))

        # Identify those points sourced from the opposite mask that are
        # close to a source point of the correct mask
        mismatch_points, = np.logical_and(
            np.logical_and(self.output_land.data == selector_val,
                           self.input_land.data != selector_val),
            in_vicinity.data > 0.5)

        # Replace these points with the filled-domain data
        self.output_cube.data[mismatch_points] = (
            selector_data[mismatch_points])

    def process(self, cube, input_land, output_land):
        """
        Update cube.data so that output_land and sea points match an input_land
        or sea point respectively so long as one is present within the
        specified vicinity radius.

        Args:
            cube (iris.cube.Cube):
                Cube of data to be updated (on same grid as output_land).
            input_land (iris.cube.Cube):
                Cube of land_binary_mask data on the grid from which "cube" has
                been reprojected (it is expected that the iris.analysis.Nearest
                method would have been used).
                This is used to determine where the input model data is
                representing land and sea points.
            output_land (iris.cube.Cube):
                Cube of land_binary_mask data on target grid.
        """
        # Check cube and output_land are on the same grid:
        if not spatial_coords_match(cube, output_land):
            raise ValueError('X and Y coordinates do not match for cubes {}'
                             'and {}'.format(repr(cube), repr(output_land)))
        self.output_land = output_land

        # Regrid input_land to output_land grid.
        self.input_land = input_land.regrid(self.output_land, self.regridder)

        # Slice over x-y grids for multi-realization data.
        result = iris.cube.CubeList()
        for xyslice in cube.slices(
                [cube.coord(axis='y'), cube.coord(axis='x')]):

            # Store and copy cube ready for the output data
            self.nearest_cube = xyslice
            self.output_cube = self.nearest_cube.copy()

            # Update sea points that were incorrectly sourced from land points
            self.correct_where_input_true(0)

            # Update land points that were incorrectly sourced from sea points
            self.correct_where_input_true(1)

            result.append(self.output_cube)

        result = result.merge_cube()
        return result
