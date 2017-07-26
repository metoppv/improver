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

from improver.utilities.cube_manipulation import concatenate_cubes
from improver.utilities.spatial import (
    convert_distance_into_number_of_grid_cells)

# Maximum radius of the neighbourhood width in grid cells.
MAX_RADIUS_IN_GRID_CELLS = 500


class Utilities(object):

    """
    Utilities for neighbourhood processing.
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
    def find_required_lead_times(cube):
        """
        Determine the lead times within a cube, either by reading the
        forecast_period coordinate, or by calculating the difference between
        the time and the forecast_reference_time. If the forecast_period
        coordinate is present, the points are assumed to represent the
        desired lead times with the bounds not being considered. The units of
        the forecast_period, time and forecast_reference_time coordinates are
        converted, if required.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube from which the lead times will be determined.

        Returns
        -------
        required_lead_times : Numpy array
            Array containing the lead times, at which the radii need to be
            calculated.

        """
        if cube.coords("forecast_period"):
            try:
                cube.coord("forecast_period").convert_units("hours")
            except ValueError as err:
                msg = "For forecast_period: {}".format(err)
                raise ValueError(msg)
            required_lead_times = cube.coord("forecast_period").points
        else:
            if cube.coords("time") and cube.coords("forecast_reference_time"):
                try:
                    cube.coord("time").convert_units(
                        "hours since 1970-01-01 00:00:00")
                    cube.coord("forecast_reference_time").convert_units(
                        "hours since 1970-01-01 00:00:00")
                except ValueError as err:
                    msg = "For time/forecast_reference_time: {}".format(err)
                    raise ValueError(msg)
                required_lead_times = (
                    cube.coord("time").points -
                    cube.coord("forecast_reference_time").points)
            else:
                msg = ("The forecast period coordinate is not available "
                       "within {}."
                       "The time coordinate and forecast_reference_time "
                       "coordinate were also not available for calculating "
                       "the forecast_period.".format(cube))
                raise CoordinateNotFoundError(msg)
        return required_lead_times

    @staticmethod
    def adjust_nsize_for_ens(ens_factor, num_ens, width):
        """
        Adjust neighbourhood size according to ensemble size.

        Parameters
        ----------
        ens_factor : float
            The factor with which to adjust the neighbourhood size
            for more than one ensemble member.
            If ens_factor = 1.0 this essentially conserves ensemble
            members if every grid square is considered to be the
            equivalent of an ensemble member.
        num_ens : float
            Number of realizations or ensemble members.
        width : float
            radius or width appropriate for a single forecast in m.

        Returns
        -------
        new_width : float
            new neighbourhood radius (m).

        """
        if num_ens <= 1.0:
            new_width = width
        else:
            new_width = (ens_factor *
                         math.sqrt((width**2.0)/num_ens))
        return new_width


class SquareNeighbourhood(object):

    """
    Methods for use in application of a square neighbourhood.
    """

    def __init__(self, unweighted_mode=False):
        """
        Initialise class.

        Parameters
        ----------
        unweighted_mode : boolean
            This is included to allow a standard interface for both the
            square and circular neighbourhood plugins.
        """
        self.unweighted_mode = unweighted_mode

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<SquareNeighbourhood: unweighted_mode: {}>')
        return result.format(self.unweighted_mode)

    @staticmethod
    def cumulate_array(cube):
        """
        Method to calculate the cumulative sum of an m x n array, by first
        cumulating along the y direction so that the largest values
        are in the nth row, and then cumulating along the x direction,
        so that the largest values are in the mth column. Each grid point
        will contain the cumulative sum from the origin to that grid point.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube to which the cumulative summing along the y and x direction
            will be applied.

        Returns
        -------
        cube : Iris.cube.Cube
            Cube to which the cumulative summing along the y and x direction
            has been applied.
        nan_masks : list
            List of numpy arrays to be used to set the values within the data
            of the output cube to be NaN.
        """
        yname = cube.coord(axis="y").name()
        xname = cube.coord(axis="x").name()
        cubelist = iris.cube.CubeList([])
        nan_masks = []
        for slice_2d in cube.slices([yname, xname]):
            data = slice_2d.data
            nan_mask = np.isnan(data)
            data[nan_mask] = 0
            data_summed_along_y = np.cumsum(data, axis=0)
            data_summed_along_x = (
                np.cumsum(data_summed_along_y, axis=1))
            slice_2d.data = data_summed_along_x
            cubelist.append(slice_2d)
            nan_masks.append(nan_mask)
        return cubelist.merge_cube(), nan_masks

    @staticmethod
    def _check_for_x_and_y_axes(cube):
        """
        Check whether the cube has an x and y axis, otherwise raise an error.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube to be checked for x and y axes.

        Raises
        ------
        ValueError : Raise an error if non-uniform increments exist between
                     grid points.
        """
        for axis in ["x", "y"]:
            if cube.coords(axis=axis):
                pass
            else:
                msg = ("The cube does not contain the expected {}"
                       "coordinates.".format(axis))
                raise ValueError(msg)

    @staticmethod
    def pad_coord(coord, width, method):
        """
        Construct a new coordinate by extending the current coordinate by the
        padding width.

        Parameters
        ----------
        coord : iris.coord
            Original coordinate which will be used as the basis of the
            new extended coordinate.
        width : integer
            The width of padding in grid cells (the extent of the
            neighbourhood radius in grid cells in a given direction).
        method : string
            A string determining whether the coordinate is being expanded
            or contracted. Options: 'remove' to remove points from coord;
            'add' to add points to coord.

        Returns
        -------
        iris.coord
            Coordinate with expanded or contracted length, to be added to the
            padded or unpadded iris cube.

        Raises
        ------
        ValueError : Raise an error if non-uniform increments exist between
                     grid points.
        """
        orig_points = coord.points
        if method == 'add':
            increment = orig_points[1:] - orig_points[:-1]
            if np.isclose(np.sum(np.diff(increment)), 0):
                increment = increment[0]
            else:
                msg = ("Non-uniform increments between grid points: "
                       "{}.".format(increment))
                raise ValueError(msg)
            num_of_new_points = len(orig_points) + 2*2*width
            new_points = (
                np.linspace(
                    orig_points.min() - (2*width+1)*increment,
                    orig_points.max() + (2*width+1)*increment,
                    num_of_new_points))
        elif method == 'remove':
            end_width = -2*width if width != 0 else None
            new_points = np.float32(orig_points[2*width:end_width])
        return coord.copy(points=new_points)

    def _create_cube_with_new_data(self, cube, data, coord_x, coord_y):
        """
        Create a cube with newly created data where the metadata is copied from
        the input cube and the supplied x and y coordinates are added to the
        cube.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Template cube used for copying metadata and non x and y axes
            coordinates.
        data : Numpy array
            Data to be put into the new cube.
        coord_x : Iris.coords.DimCoord
            Coordinate to be added to the new cube to represent the x axis.
        coord_y : Iris.coords.DimCoord
            Coordinate to be added to the new cube to represent the y axis.

        Returns
        -------
        new_cube : Iris.cube.Cube
            Cube built from the template cube using the requested data and the
            supplied x and y axis coordinates.
        """
        self._check_for_x_and_y_axes(cube)

        yname = cube.coord(axis='y').name()
        xname = cube.coord(axis='x').name()
        ycoord_dim = cube.coord_dims(yname)
        xcoord_dim = cube.coord_dims(xname)
        metadata_dict = copy.deepcopy(cube.metadata._asdict())
        new_cube = iris.cube.Cube(data, **metadata_dict)
        for coord in cube.coords():
            if coord.name() not in [yname, xname]:
                if cube.coords(coord, dim_coords=True):
                    coord_dim = cube.coord_dims(coord)
                    new_cube.add_dim_coord(coord, coord_dim)
                else:
                    new_cube.add_aux_coord(coord)
        if len(xcoord_dim) > 0:
            new_cube.add_dim_coord(coord_x, xcoord_dim)
        else:
            new_cube.add_aux_coord(coord_x)
        if len(ycoord_dim) > 0:
            new_cube.add_dim_coord(coord_y, ycoord_dim)
        else:
            new_cube.add_aux_coord(coord_y)
        return new_cube

    def pad_cube_with_halo(self, cube, width_x, width_y):
        """
        Method to pad a halo around the data in an iris cube with 0s.

        Parameters
        ----------
        cube : iris.cube.Cube
            The original cube to be padded out with zeros.
        width_x, width_y : integer
            The width in x and y directions of the neighbourhood radius in
            grid cells. This will be the width of zeros added to the numpy
            array.

        Returns
        -------
        iris.cube.Cube
            Cube containing the new zero padded cube, with appropriate
            changes to the cube's dimension coordinates.
        """
        self._check_for_x_and_y_axes(cube)

        yname = cube.coord(axis='y').name()
        xname = cube.coord(axis='x').name()
        cubelist = iris.cube.CubeList([])
        for slice_2d in cube.slices([yname, xname]):
            # Pad a halo around the original data with the extent of the halo
            # given by width_y and width_x.
            padded_data = np.pad(
                slice_2d.data,
                ((2*width_y, 2*width_y), (2*width_x, 2*width_x)),
                "mean", stat_length=width_y)
            coord_x = cube.coord(axis='x')
            padded_x_coord = (
                SquareNeighbourhood.pad_coord(coord_x, width_x, 'add'))
            coord_y = cube.coord(axis='y')
            padded_y_coord = (
                SquareNeighbourhood.pad_coord(coord_y, width_y, 'add'))
            cubelist.append(
                self._create_cube_with_new_data(
                    slice_2d, padded_data, padded_x_coord, padded_y_coord))
        return cubelist.merge_cube()

    def remove_halo_from_cube(self, cube, width_x, width_y):
        """
        Method to remove rows/columnds from the edge of an iris cube.
        Used to 'unpad' cubes which have been previously padded by
        pad_cube_with_halo.

        Parameters
        ----------
        cube : iris.cube.Cube
            The original cube to be trimmed of edge data.
        width_x, width_y : integer
            The width in x and y directions of the neighbourhood radius in
            grid cells. This will be the width removed from the numpy
            array.

        Returns
        -------
        iris.cube.Cube
            Cube containing the new trimmed cube, with appropriate
            changes to the cube's dimension coordinates.
        """
        self._check_for_x_and_y_axes(cube)

        yname = cube.coord(axis='y')
        xname = cube.coord(axis='x')
        cubelist = iris.cube.CubeList([])
        for slice_2d in cube.slices([yname, xname]):
            end_y = -2*width_y if width_y != 0 else None
            end_x = -2*width_x if width_x != 0 else None
            trimmed_data = slice_2d.data[2*width_y:end_y,
                                         2*width_x:end_x]
            coord_x = slice_2d.coord(axis='x')
            trimmed_x_coord = (
                SquareNeighbourhood.pad_coord(coord_x, width_x, 'remove'))
            coord_y = slice_2d.coord(axis='y')
            trimmed_y_coord = (
                SquareNeighbourhood.pad_coord(coord_y, width_y, 'remove'))
            cubelist.append(
                self._create_cube_with_new_data(
                    slice_2d, trimmed_data, trimmed_x_coord, trimmed_y_coord))
        return cubelist.merge_cube()

    def mean_over_neighbourhood(self, cube, cells_x, cells_y, nan_masks):
        """
        Method to calculate the average value in a square neighbourhood using
        the 4-point algorithm to find the total sum over the neighbourhood.

        The output from the cumulate_array method can be used to
        calculate the sum over a neighbourhood of size
        (2*cells_x+1)*(2*cells_y+1). This sum is then divided by the area of
        the neighbourhood to calculate the mean value in the neighbourhood.

        For non-edge points, a faster, vectorised
        approach is taken:
        1. The displacements between the four points used to calculate the
           neighbourhood total sum and the central grid point are calculated.
        2. Four copies of the cumulate array output are flattened and rolled
           by these displacements to align the four terms used in the
           neighbourhood total sum calculation.
        3. The neighbourhood total at all non-edge points can then be
           calculated simultaneously in a single vector sum.

        Displacements are calculated as follows for the following input array,
        where the accumulation has occurred from left to right and top to
        bottom.

        | 2 | 4 | 6 | 7 |
        | 2 | 4 | 5 | 6 |
        | 1 | 3 | 4 | 4 |
        | 1 | 2 | 2 | 2 |

        For a 3x3 neighbourhood centred around the point with a value of 5:

        | 2 (A) | 4 | 6                 | 7 (B) |
        | 2     | 4 | 5 (Central point) | 6     |
        | 1     | 3 | 4                 | 4     |
        | 1 (C) | 2 | 2                 | 2 (D) |

        To calculate the value for the neighbourhood sum at the "Central point"
        with a value of 5, calculate:
        Neighbourhood sum = B - A - D + C
        At the central point, this will yield:
        Neighbourhood sum = 7 - 2 - 2 +1 => 4
        Neighbourhood mean = Neighbourhood sum
                             -----------------
                             (2 * nb_width +1)
        where nb_width is the neighbourhood width, which is equal to 1 for a
        3x3 neighbourhood.

        Neighbourhood mean = 4. / 9.

        Parameters
        ----------
        cube : iris.cube.Cube
            Cube to which neighbourhood processing is being applied. Must
            be passed through cumulate_array method first.
        cells_x, cells_y : integer
            The radius of the neighbourhood in grid points, in the x and y
            directions (excluding the central grid point).
        nan_masks : list
            List of numpy arrays to be used to set the values within the data
            of the output cube to be NaN.

        Returns
        -------
        cube : iris.cube.Cube
            Cube to which square neighbourhood has been applied.
        """
        self._check_for_x_and_y_axes(cube)

        yname = cube.coord(axis="y").name()
        xname = cube.coord(axis="x").name()

        # Calculate displacement factors to find 4-points after flattening the
        # array.
        n_rows = len(cube.coord(axis="y").points)
        n_columns = len(cube.coord(axis="x").points)

        # Displacements from the point at the centre of the neighbourhood.
        # Equivalent to point B in the docstring example.
        ymax_xmax_disp = (cells_y*n_columns) + cells_x
        # Equivalent to point A in the docstring example.
        ymax_xmin_disp = (cells_y*n_columns) - cells_x - 1

        # Equivalent to point D in the docstring example.
        ymin_xmax_disp = (-1*(cells_y+1)*n_columns) + cells_x
        # Equivalent to point C in the docstring example.
        ymin_xmin_disp = (-1*(cells_y+1)*n_columns) - cells_x - 1

        cubelist = iris.cube.CubeList([])
        for slice_2d, nan_mask in zip(cube.slices([yname, xname]), nan_masks):
            # Flatten the 2d slice and calculate the sum over the array for
            # non-edge cases. This is done by creating 4 copies of the
            # flattened array which are rolled to allign the 4-points which
            # are needed for the calculation.
            flattened = slice_2d.data.flatten()
            ymax_xmax_array = np.roll(flattened, -ymax_xmax_disp)
            ymin_xmax_array = np.roll(flattened, -ymin_xmax_disp)
            ymin_xmin_array = np.roll(flattened, -ymin_xmin_disp)
            ymax_xmin_array = np.roll(flattened, -ymax_xmin_disp)
            neighbourhood_total = (ymax_xmax_array - ymin_xmax_array +
                                   ymin_xmin_array - ymax_xmin_array)
            neighbourhood_total.resize(n_rows, n_columns)
            # Initialise the neighbourhood size array and calculate
            # neighbourhood size for non edge cases.
            neighbourhood_area = np.zeros(neighbourhood_total.shape)
            neighbourhood_area.fill((2*cells_x+1) * (2*cells_y+1))
            with np.errstate(invalid='ignore', divide='ignore'):
                slice_2d.data = (neighbourhood_total.astype(float) /
                                 neighbourhood_area.astype(float))
            slice_2d.data[nan_mask.astype(bool)] = np.NaN
            cubelist.append(slice_2d)
        return cubelist.merge_cube()

    @staticmethod
    def _set_up_cubes_to_be_neighbourhooded(cube):
        """
        Set up a cubelist containing either the input cube, or the input cube
        and a mask cube.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube that will be checked for whether the data is masked.

        Returns
        -------
        cubes_to_sum : Iris.cube.CubeList
            CubeList containing either the input cube, or the input cube and
            a mask cube.
        """
        # If there is no mask, make a mask of ones. This will speed up the
        # calculation.
        if isinstance(cube.data, np.ma.MaskedArray):
            mask_cube = cube.copy()
            mask_cube.rename('mask_data')
            mask_cube.data = np.logical_not(cube.data.mask.astype(int))
            cube.data = cube.data.data * mask_cube.data
            cubes_to_sum = iris.cube.CubeList([cube, mask_cube])
        else:
            cubes_to_sum = iris.cube.CubeList([cube])
        return cubes_to_sum

    def _pad_and_calculate_neighbourhood(
            self, cubes_to_sum, grid_cells_x, grid_cells_y):
        """
        Apply neighbourhood processing consisting of the following steps:
        1. Pad a halo around the input cube to allow vectorised
           neighbourhooding at edgepoints.
        2. Cumulate the array along the x and y axes.
        3. Apply neighbourhood processing to the cumulated array.

        Parameters
        ----------
        cubes_to_sum : Iris.cube.CubeList
            CubeList containing either the input cube, or the input cube and
            a mask cube.
        grid_cells_x : Float
            The number of grid cells along the x axis used to create a square
            neighbourhood.
        grid_cells_y : Float
            The number of grid cells along the y axis used to create a square
            neighbourhood.

        Returns
        -------
        neighhourhood_averaged_cubes : Iris.cube.CubeList
            CubeList containing the smoothed field after the square
            neighbourhood method has been applied to either the input cube, or
            both the input cube and a mask cube.
        """
        neighbourhood_averaged_cubes = iris.cube.CubeList([])
        for cube_to_process in cubes_to_sum:
            # Pad the iris cube with zeros. This way, the edge effects produced
            # by the vectorisation of the 4-point method will appear outside
            # our domain of interest. These unwanted points can be trimmed off
            # later.
            cube_to_process = self.pad_cube_with_halo(
                cube_to_process, grid_cells_x, grid_cells_y)
            summed_up_cube, nan_masks = self.cumulate_array(
                cube_to_process)
            neighbourhood_averaged_cubes.append(
                self.mean_over_neighbourhood(
                    summed_up_cube, grid_cells_x, grid_cells_y, nan_masks))
        return neighbourhood_averaged_cubes

    def _remove_padding_and_mask(
            self, neighbourhood_averaged_cubes, cube_name,
            grid_cells_x, grid_cells_y):
        """
        Remove the halo from the padded array and apply the mask, if required.

        Parameters
        ----------
        neighhourhood_averaged_cubes : Iris.cube.CubeList
            CubeList containing the smoothed field after the square
            neighbourhood method has been applied to either the input cube, or
            both the input cube and a mask cube.
        cube_name : String
            Name of the variable that has been neighbourhooded.
        grid_cells_x : Float
            The number of grid cells along the x axis used to create a square
            neighbourhood.
        grid_cells_y : Float
            The number of grid cells along the y axis used to create a square
            neighbourhood.

        Returns
        -------
        neighhourhood_averaged_cube : Iris.cube.Cube
            Cube containing the smoothed field after the square
            neighbourhood method has been applied.
        """
        # Correct neighbourhood averages for masked data, which may have been
        # calculated using larger neighbourhood areas than are present in
        # reality.
        neighbourhood_averaged_cube, = neighbourhood_averaged_cubes.extract(
            cube_name)
        neighbourhood_averaged_cube = self.remove_halo_from_cube(
            neighbourhood_averaged_cube, grid_cells_x, grid_cells_y)
        if len(neighbourhood_averaged_cubes) > 1:
            mask_cube, = neighbourhood_averaged_cubes.extract('mask_data')
            mask_cube = self.remove_halo_from_cube(
                mask_cube, grid_cells_x, grid_cells_y)
            neighbourhood_averaged_cube.data = (
                neighbourhood_averaged_cube.data * mask_cube.data)
            neighbourhood_averaged_cube.data = (
                np.ma.masked_where(np.logical_not(mask_cube.data.squeeze()),
                                   neighbourhood_averaged_cube.data))
        return neighbourhood_averaged_cube

    def run(self, cube, radius):
        """
        Call the methods required to apply a square neighbourhood
        method to a cube.

        The steps for undertaken are:
        1. Set to cubes by determining, if the arrays are masked.
        2. Pad the input array with a halo and then calculate the neighbourhood
           of the haloed array.
        3. Remove the halo from the neighbourhooded array and deal with a mask,
           if required.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube containing the array to which the square neighbourhood
            will be applied.
        radius : Float
            Radius in metres for use in specifying the number of
            grid cells used to create a square neighbourhood.

        Returns
        -------
        neighhourhood_averaged_cube : Iris.cube.Cube
            Cube containing the smoothed field after the square neighbourhood
            method has been applied.
        """
        # If the data is masked, the mask will be processed as well as the
        # original_data * mask array.
        original_attributes = cube.attributes
        original_methods = cube.cell_methods

        grid_cells_x, grid_cells_y = (
            convert_distance_into_number_of_grid_cells(
                cube_to_process, radius, MAX_RADIUS_IN_GRID_CELLS))
        cubes_to_sum = (
            self._set_up_cubes_to_be_neighbourhooded(cube))
        neighbourhood_averaged_cubes = (
            self._pad_and_calculate_neighbourhood(
                cubes_to_sum, grid_cells_x, grid_cells_y))
        neighbourhood_averaged_cube = (
            self._remove_padding_and_mask(
                neighbourhood_averaged_cubes, cube.name(),
                grid_cells_x, grid_cells_y))

        neighbourhood_averaged_cube.cell_methods = original_methods
        neighbourhood_averaged_cube.attributes = original_attributes
        return neighbourhood_averaged_cube


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
        # Define the size of the kernel based on the number of grid cells
        # contained within the desired radius.
        kernel = np.ones([int(1 + x * 2) for x in fullranges])
        # Create an open multi-dimensional meshgrid.
        open_grid = np.array(np.ogrid[tuple([slice(-x, x+1) for x in ranges])])
        if self.unweighted_mode:
            mask = np.reshape(
                np.sum(open_grid**2) > np.prod(ranges), np.shape(kernel))
        else:
            # Create a kernel, such that the central grid point has the
            # highest weighting, with the weighting decreasing with distance
            # away from the central grid point.
            open_grid_summed_squared = np.sum(open_grid**2.).astype(float)
            kernel[:] = (
                (np.prod(ranges) - open_grid_summed_squared) / np.prod(ranges))
            mask = kernel < 0.
        kernel[mask] = 0.
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


class NeighbourhoodProcessing(object):
    """
    Apply a neigbourhood processing method to a thresholded cube.

    When applied to a thresholded probabilistic cube, it acts like a
    low-pass filter which reduces noisiness in the probabilities.

    The neighbourhood methods will presently only work with projections in
    which the x grid point spacing and y grid point spacing are constant
    over the entire domain, such as the UK national grid projection

    """

    def __init__(self, neighbourhood_method, radii, lead_times=None,
                 unweighted_mode=False, ens_factor=1.0):
        """
        Create a neighbourhood processing plugin that applies a smoothing
        to points in a cube.

        Parameters
        ----------

        neighbourhood_method : str
            Name of the neighbourhood method to use. Options: 'circular',
            'square'.
        radii : float or List (if defining lead times)
            The radii in metres of the neighbourhood to apply.
            Rounded up to convert into integer number of grid
            points east and north, based on the characteristic spacing
            at the zero indices of the cube projection-x and y coords.
        lead_times : None or List
            List of lead times or forecast periods, at which the radii
            within 'radii' are defined. The lead times are expected
            in hours.
        unweighted_mode : boolean
            If True, use a circle with constant weighting.
            If False, use a circle for neighbourhood kernel with
            weighting decreasing with radius.
        ens_factor : float
            The factor with which to adjust the neighbourhood size
            for more than one ensemble member.
            If ens_factor = 1.0 this essentially conserves ensemble
            members if every grid square is considered to be the
            equivalent of an ensemble member.
            Optional, defaults to 1.0
        """
        self.neighbourhood_method_key = neighbourhood_method
        methods = {
            "circular": CircularNeighbourhood,
            "square": SquareNeighbourhood}
        try:
            method = methods[neighbourhood_method]
            self.neighbourhood_method = method(unweighted_mode)
        except KeyError:
            msg = ("The neighbourhood_method requested: {} is not a "
                   "supported method. Please choose from: {}".format(
                       neighbourhood_method, methods.keys()))
            raise KeyError(msg)

        if isinstance(radii, list):
            self.radii = [float(x) for x in radii]
        else:
            self.radii = float(radii)
        self.lead_times = lead_times
        if self.lead_times is not None:
            if len(radii) != len(lead_times):
                msg = ("There is a mismatch in the number of radii "
                       "and the number of lead times. "
                       "Unable to continue due to mismatch.")
                raise ValueError(msg)
        self.unweighted_mode = bool(unweighted_mode)
        self.ens_factor = float(ens_factor)

    def _find_radii(self, num_ens, cube_lead_times=None):
        """Revise radius or radii for found lead times and ensemble members

        If cube_lead_times is None just adjust for ensemble
        members if necessary.
        Otherwise interpolate to find radius at each cube
        lead time and adjust for ensemble members if necessary.

        Parameters
        ----------
        num_ens : float
            Number of ensemble members or realizations.
        cube_lead_times : np.array
            Array of forecast times found in cube.

        Returns
        -------
        radii : float or np.array of float
            Required neighbourhood sizes.
        """
        if cube_lead_times is None:
            radii = Utilities.adjust_nsize_for_ens(self.ens_factor,
                                                   num_ens, self.radii)
        else:
            # Interpolate to find the radius at each required lead time.
            radii = (
                np.interp(
                    cube_lead_times, self.lead_times, self.radii))
            for i, val in enumerate(radii):
                radii[i] = Utilities.adjust_nsize_for_ens(self.ens_factor,
                                                          num_ens, val)
        return radii

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<NeighbourhoodProcessing: neighbourhood_method: {}; '
                  'radii: {}; lead_times: {}; '
                  'unweighted_mode: {}; ens_factor: {}>')
        return result.format(
            self.neighbourhood_method_key, self.radii, self.lead_times,
            self.unweighted_mode, self.ens_factor)

    def process(self, cube):
        """
        Supply neighbourhood processing method, in order to smooth the
        input cube.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube to apply a neighbourhood processing method to, in order to
            generate a smoother field.

        Returns
        -------
        cube : Iris.cube.Cube
            Cube after applying a neighbourhood processing method, so that the
            resulting field is smoothed.

        """
        # Check if the realization coordinate exists. If there are multiple
        # values for the realization, then an exception is raised. Otherwise,
        # the cube is sliced, so that the realization becomes a scalar
        # coordinate.
        try:
            realiz_coord = cube.coord('realization')
        except iris.exceptions.CoordinateNotFoundError:
            if 'source_realizations' in cube.attributes:
                num_ens = len(cube.attributes['source_realizations'])
            else:
                num_ens = 1.0
            slices_over_realization = [cube]
        else:
            num_ens = len(realiz_coord.points)
            slices_over_realization = cube.slices_over("realization")
            if 'source_realizations' in cube.attributes:
                msg = ("Realizations and attribute source_realizations "
                       "should not both be set in input cube")
                raise ValueError(msg)

        if np.isnan(cube.data).any():
            raise ValueError("Error: NaN detected in input cube data")

        cubelist = iris.cube.CubeList([])
        for cube_realization in slices_over_realization:
            if self.lead_times is None:
                radius = self._find_radii(num_ens)
                cube_new = self.neighbourhood_method.run(cube_realization,
                                                         radius)
            else:
                cube_lead_times = (
                    Utilities.find_required_lead_times(cube_realization))
                # Interpolate to find the radius at each required lead time.
                required_radii = (
                    self._find_radii(num_ens,
                                     cube_lead_times=cube_lead_times))

                cubes = iris.cube.CubeList([])
                # Find the number of grid cells required for creating the
                # neighbourhood, and then apply the neighbourhood
                # processing method to smooth the field.
                for cube_slice, radius in (
                        zip(cube_realization.slices_over("time"),
                            required_radii)):
                    cube_slice = self.neighbourhood_method.run(
                        cube_slice, radius)
                    cube_slice = iris.util.new_axis(cube_slice, "time")
                    cubes.append(cube_slice)
                cube_new = concatenate_cubes(cubes,
                                             coords_to_slice_over=["time"])

            cubelist.append(cube_new)
        cube = cubelist.merge_cube()

        return cube
