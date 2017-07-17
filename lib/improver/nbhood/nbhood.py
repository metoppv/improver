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

import math

import iris
from iris.exceptions import CoordinateNotFoundError
import numpy as np
import scipy.ndimage.filters
import time

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
    def mean_over_neighbourhood(cube, cells_x, cells_y, nan_mask):
        """
        Method to calculate the average value in a square neighbourhood using
        the 4-point algorithm to find the total sum over the neighbourhood.

        The output from the cumulate_array method can be used to
        calculate the sum over a neighbourhood of size
        (2*cells_x+1)*(2*cells_y+1). This sum is then divided by the area of
        the neighbourhood to calculate the mean value in the neighbourhood.

        At edge points, the sum and area of the neighbourhood are calculated
        for each point individually. For non-edge points, a faster, vectorised
        approach is taken:
        1. The displacements between the four points used to calculate the
           neighbourhood total sum and the central grid point are calculated.
        2. Four copies of the cumulate array output are flattened and rolled
           by these displacements to align the four terms used in the
           neighbourhood total sum calculation.
        3. The neighbourhood total at all non-edge points can then be
           calculated simultaneously in a single vector sum.

        Parameters
        ----------
        cube : iris.cube.Cube
            Cube to which neighbourhood processing is being applied. Must
            be passed through cumulate_array method first.
        cells_x, cells_y : integer
            The radius of the neighbourhood in grid points, in the x and y
            directions (excluding the central grid point).
        nan_mask : numpy.array
            Mask of where the original input data array had nans. Nans will
            be reapplied at these points after neighbourhood processing has
             been carried out.

        Returns
        -------
        cube : iris.cube.Cube
            Cube to which square neighbourhood has been applied.
        """
        def _sum_and_area_for_edge_cases(cube, cells_x, cells_y, i, j):
            """
            Function to calculate the total sum and area of neighbourhoods
            surrounding edge grid point (i,j) which can't use the flatten and
            roll method. These values are used to calculate the mean value of
            the neighbourhood.

            Parameters
            ----------
            cube : iris.cube.Cube
                Cube to which neighbourhood processing is being applied. Must
                be passed through cumulate_array method first.
            cells_x, cells_y : integer
                The maximum radius of the neighbourhood in grid points, in the
                x and y directions (excluding the central grid point). For
                edge cases, the radius may be less than this, if the
                neighbourhood falls off the domain edge.
            i, j : integer
                x and y indices of the grid point for which the total sum and
                neighbourhood area is sought.

            Returns
            -------
            total : float
                The sum of all values in the neighbourhood surrounding grid
                point (i,j).
            area : integer
                The area of the neighbourhood surrounding grid point (i,j).
                This accounts for cases where the neighbourhood extends beyond
                domain bounds.
            """
            x_min = i-cells_x-1
            x_max = min(len(cube.coord(axis="x").points)-1, i+cells_x)
            y_min = j-cells_y-1
            y_max = min(len(cube.coord(axis="y").points)-1, j+cells_y)
            summed_array = cube.data
            # The neighbourhood of some edge-points will fall off the edge of
            # the domain which will necessitate modifying formulae to calculate
            # the sum over the  at these points. The equation below simplifies
            # the formulae needed for edge points by using masks to remove
            # terms when a particular domain edge is exceeded.
            total = (summed_array[y_max, x_max] -
                     summed_array[y_min, x_max]*(y_min >= 0) -
                     summed_array[y_max, x_min]*(x_min >= 0) +
                     summed_array[y_min, x_min]*(y_min >= 0 and x_min >= 0))
            x_min = max(-1, x_min)
            y_min = max(-1, y_min)
            area = (y_max - y_min) * (x_max - x_min)
            return total, area

        yname = cube.coord(axis="y").name()
        xname = cube.coord(axis="x").name()
        #print cube.data
        # Calculate displacement factors to find 4-points after flattening the
        # array.
        n_rows = len(cube.coord(axis="y").points)
        n_columns = len(cube.coord(axis="x").points)
        ymax_xmax_disp = (cells_y*n_columns) + cells_x
        ymin_xmax_disp = (-1*(cells_y+1)*n_columns) + cells_x
        ymin_xmin_disp = (-1*(cells_y+1)*n_columns) - cells_x - 1
        ymax_xmin_disp = (cells_y*n_columns) - cells_x - 1

        cubelist = iris.cube.CubeList([])
        for slice_2d in cube.slices([yname, xname]):
            # Flatten the 2d slice and calculate the sum over the array for
            # non-edge cases. This is done by creating 4 copies of the
            # flattened array which are rolled to allign the 4-points which
            # are needed for the calculation.
            flattened = slice_2d.data.flatten()
            ymax_xmax_array = np.roll(flattened, -ymax_xmax_disp)
            ymin_xmax_array = np.roll(flattened, -ymin_xmax_disp)
            ymin_xmin_array = np.roll(flattened, -ymin_xmin_disp)
            ymax_ymin_array = np.roll(flattened, -ymax_xmin_disp)
            neighbourhood_total = (ymax_xmax_array - ymin_xmax_array +
                                   ymin_xmin_array - ymax_ymin_array)
            neighbourhood_total.resize(n_rows, n_columns)
            #print neighbourhood_total
            # Initialise the neighbourhood size array and calculate
            # neighbourhood size for non edge cases.
            neighbourhood_area = np.zeros(neighbourhood_total.shape)
            neighbourhood_area.fill((2*cells_x+1) * (2*cells_y+1))
            # Calculate total sum and area of neighbourhood for edge cases.
            # NOTE: edge cases could be dealt with more efficiently.
            # If neighbourhoods get large, this method will need revision.
            #if not mask_flag:
                #edge_rows = (range(min(cells_x*2, n_rows)) +
                              #range(max(n_rows-2*cells_x, 0), n_rows))
                #edge_columns = (range(min(cells_x*2, n_columns)) +
                                #range(max(n_columns-2*cells_x, 0), n_columns))
                #for j in range(n_rows):
                    #for i in (edge_columns):
                        #neighbourhood_total[j, i], neighbourhood_area[j, i] = (
                            #_sum_and_area_for_edge_cases(
                                #slice_2d, cells_x, cells_y, i, j))
                #for i in range(n_columns):
                    #for j in (edge_rows):
                        #neighbourhood_total[j, i], neighbourhood_area[j, i] = (
                            #_sum_and_area_for_edge_cases(
                                #slice_2d, cells_x, cells_y, i, j))
            with np.errstate(invalid='ignore', divide='ignore'):
                slice_2d.data = (neighbourhood_total.astype(float) /
                                  neighbourhood_area.astype(float))
            slice_2d.data[nan_mask] = np.NaN
            cubelist.append(slice_2d)
            #print slice_2d.data
        return cubelist.merge_cube()

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
        nan_mask : numpy array
            Mask of where the original input data array had nans. Nans will
            be reapplied at these points after neighbourhood processing has
             been carried out.
        """
        yname = cube.coord(axis="y").name()
        xname = cube.coord(axis="x").name()
        cubelist = iris.cube.CubeList([])
        for slice_2d in cube.slices([yname, xname]):
            data = slice_2d.data
            nan_mask = np.isnan(data)
            data[nan_mask] = 0
            data_summed_along_y = np.cumsum(data, axis=0)
            data_summed_along_x = (
                np.cumsum(data_summed_along_y, axis=1))
            slice_2d.data = data_summed_along_x
            cubelist.append(slice_2d)
        return cubelist.merge_cube(), nan_mask

    @staticmethod
    def pad_coord(coord, width, method):
        """
        Construct a new coordinate by extending the current coordinate by the
        padding width.

        Args:
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

        Returns:
          new_coord : iris.coord
              Coordinate with expanded or contracted length, to be added to the
              padded or unpadded iris cube.
        """
        orig_points = coord.points
        if method == 'add':
            orig_del = orig_points[1] - orig_points[0]
            pre_points = np.linspace(orig_points.min() - (2*width + 1)*orig_del,
                                    orig_points.min() - orig_del, 2*width)
            post_points = np.linspace(orig_points.max() + orig_del,
                                      orig_points.max() + (2*width + 1)*orig_del, 2*width)
            new_points = np.float32(np.append(
                np.append(pre_points, orig_points), post_points))
        elif method == 'remove':
            new_points = np.float32(orig_points[2*width:-2*width])
        new_coord = iris.coords.DimCoord(
            new_points, coord.name(), coord_system=coord.coord_system, units=coord.units)
        return new_coord

    @staticmethod
    def pad_cube(cube, width_x, width_y):
        """
        Method to pad an iris cube with 0s.

        Args:
            cube : iris.cube.Cube
                The original cube to be padded out with zeros.
            width_x, width_y : integer
                The width in x and y directions of the neighbourhood radius in
                grid cells. This will be the width of zeros added to the numpy
                array.

        Returns:
            iris.cube.Cube
                Cube containing the new zero padded cube, with appropriate
                changes to the cube's dimension coordinates.
        """
        yname = cube.coord(axis='y')
        xname = cube.coord(axis='x')
        cubelist = iris.cube.CubeList([])
        for slice_2d in cube.slices([yname, xname]):
            padded_data = np.pad(slice_2d.data.squeeze(), (2*width_y, 2*width_x), 'constant')
            orig_x = cube.coord(axis='x')
            new_x = SquareNeighbourhood.pad_coord(orig_x, width_x, 'add')
            orig_y = cube.coord(axis='y')
            new_y = SquareNeighbourhood.pad_coord(orig_y, width_y, 'add')
            new_cube = iris.cube.Cube(padded_data, long_name=cube.name())
            for coord in slice_2d.coords():
                if coord.name() not in [yname.name(), xname.name()]:
                    dim = slice_2d.coord_dims(coord)
                    if dim:
                        new_cube.add_dim_coord(coord, dim)
                    else:
                        new_cube.add_aux_coord(coord)
            new_cube.add_dim_coord(new_x, 1)    
            new_cube.add_dim_coord(new_y, 0)
            cubelist.append(new_cube)
        return cubelist.merge_cube()

    @staticmethod
    def unpad_cube(cube, width_x, width_y):
        """
        Method to remove rows/columnds from the edge of an iris cube.
        Used to 'unpad' cubes which have been previously padded by pad_cube.

        Args:
            cube : iris.cube.Cube
                The original cube to be trimmed of edge data.
            width_x, width_y : integer
                The width in x and y directions of the neighbourhood radius in
                grid cells. This will be the width removed from the numpy
                array.

        Returns:
            iris.cube.Cube
                Cube containing the new trimmed cube, with appropriate
                changes to the cube's dimension coordinates.
        """
        yname = cube.coord(axis='y')
        xname = cube.coord(axis='x')
        cubelist = iris.cube.CubeList([])
        for slice_2d in cube.slices([yname, xname]):
            unpadded_data = slice_2d.data[2*width_y:-2*width_y,
                                          2*width_x:-2*width_x]
            orig_x = slice_2d.coord(axis='x')
            new_x = SquareNeighbourhood.pad_coord(orig_x, width_x, 'remove')
            orig_y = slice_2d.coord(axis='y')
            new_y = SquareNeighbourhood.pad_coord(orig_y, width_y, 'remove')
            new_cube = iris.cube.Cube(unpadded_data, long_name=cube.name())
            for coord in slice_2d.coords():
                if coord.name() not in [yname.name(), xname.name()]:
                    dim = slice_2d.coord_dims(coord)
                    if dim:
                        new_cube.add_dim_coord(coord, dim)
                    else:
                        new_cube.add_aux_coord(coord)
            new_cube.add_dim_coord(new_x, 1)
            new_cube.add_dim_coord(new_y, 0)
            cubelist.append(new_cube)
        return cubelist.merge_cube()

    @staticmethod
    def run(cube, radius):
        """
        Call the methods required to apply a square neighbourhood
        method to a cube.

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
        cube : Iris.cube.Cube
            Cube containing the smoothed field after the square neighbourhood
            method has been applied.
        """
        masked_flag = False
        if np.any(np.isnan(cube.data)):
            msg = ('Data array contains NaNs which are not currently ',
                   'supported in SquareNeighbourhood.')
            raise ValueError(msg)
        # If the data is masked, the mask will be processed as well as the
        # original_data * mask array.
        original_attributes = cube.attributes
        original_methods = cube.cell_methods
        #print cube.data
        # If there is no mask, make a mask of ones. This will speed up the
        # calculation.
        mask_cube = cube.copy()
        mask_cube.rename('mask_data')
        cubes_to_sum = iris.cube.CubeList([cube, mask_cube])
        if isinstance(cube.data, np.ma.MaskedArray):
            masked_flag = True
            mask_cube.data = np.logical_not(cube.data.mask.astype(int))
            cube.data = cube.data.data * mask_cube.data
        #print mask_cube.data
        #print cube.data
        cubes_to_sum = iris.cube.CubeList([cube, mask_cube])
        neighbourhood_averaged_cubes = iris.cube.CubeList([])
        for cube_to_process in cubes_to_sum:
            grid_cells_x, grid_cells_y = (
                convert_distance_into_number_of_grid_cells(
                    cube_to_process, radius, MAX_RADIUS_IN_GRID_CELLS))
            # Pad the iris cube with zeros. This way, the edge effects produced
            # by the vectorisation of the 4-point method will appear outside
            # our domain of interest. These unwanted points can be trimmed off
            # later.
            cube_to_process = SquareNeighbourhood.pad_cube(
                cube_to_process, grid_cells_x, grid_cells_y)
            #print cube.data
            summed_up_cube, nan_mask = SquareNeighbourhood.cumulate_array(
                cube_to_process)
            #print summed_up_cube.data
            neighbourhood_averaged_cubes.append(
                SquareNeighbourhood.mean_over_neighbourhood(
                    summed_up_cube, grid_cells_x, grid_cells_y, nan_mask))

        # Correct neighbourhood averages for masked data, which may have been
        # calculated using larger neighbourhood areas than are present in
        # reality.
        neighbourhood_averaged_cube, = neighbourhood_averaged_cubes.extract(
            cube.name())
        neighbourhood_averaged_cube = SquareNeighbourhood.unpad_cube(
            neighbourhood_averaged_cube, grid_cells_x, grid_cells_y)
        mask_probs, = neighbourhood_averaged_cubes.extract('mask_data')
        mask_probs = SquareNeighbourhood.unpad_cube(
            mask_probs, grid_cells_x, grid_cells_y)
        with np.errstate(invalid='ignore', divide='ignore'):
            neighbourhood_averaged_cube.data = (
                neighbourhood_averaged_cube.data/mask_probs.data)
        neighbourhood_averaged_cube.data = (
            np.ma.masked_where(np.logical_not(mask_cube.data.squeeze()),
                                neighbourhood_averaged_cube.data))
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
                print 'start'
                t0=time.time()
                cube_new = self.neighbourhood_method.run(cube_realization,
                                                         radius)
                t1=time.time()
                print 'end:',t1-t0
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
                    print 'start'
                    t0 = time.time()
                    cube_slice = self.neighbourhood_method.run(
                        cube_slice, radius)
                    t1=time.time()
                    print 'end:', t1-t0
                    cube_slice = iris.util.new_axis(cube_slice, "time")
                    cubes.append(cube_slice)
                cube_new = concatenate_cubes(cubes,
                                             coords_to_slice_over=["time"])

            cubelist.append(cube_new)
        cube = cubelist.merge_cube()

        return cube
