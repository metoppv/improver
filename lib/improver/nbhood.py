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

import iris
from iris.exceptions import CoordinateNotFoundError
import numpy as np
import scipy.ndimage.filters

from improver.ensemble_calibration.ensemble_calibration_utilities import (
    concatenate_cubes)

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
    def get_neighbourhood_width_in_grid_cells(
            cube, radius_in_km, max_radius_in_grid_cells):
        """
        Return the number of grid cells in the x and y direction
        used to define the neighbourhood width in the x and y direction
        based on the input radius in km.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube containing the x and y coordinates, which will be used for
            calculating the number of grid cells in the x and y direction,
            which equates to the size of the desired radius.
        radius_in_km : Float
            Radius in kilometres for use in specifying the number of
            grid cells used to create a circular neighbourhood.
        max_radius_in_grid_cells : integer
            Maximum radius of the neighbourhood width in grid cells.

        Returns
        -------
        grid_cells_x : Integer
            Number of grid cells in the x direction based on the requested
            radius in km.
        grid_cells_y : Integer
            Number of grid cells in the y direction based on the requested
            radius in km.

        """
        try:
            x_coord = cube.coord("projection_x_coordinate").copy()
            y_coord = cube.coord("projection_y_coordinate").copy()
        except CoordinateNotFoundError:
            raise ValueError("Invalid grid: projection_x/y coords required")
        x_coord.convert_units("metres")
        y_coord.convert_units("metres")
        d_north_metres = y_coord.points[1] - y_coord.points[0]
        d_east_metres = x_coord.points[1] - x_coord.points[0]
        grid_cells_y = int(radius_in_km * 1000 / abs(d_north_metres))
        grid_cells_x = int(radius_in_km * 1000 / abs(d_east_metres))
        if grid_cells_x == 0 or grid_cells_y == 0:
            raise ValueError(
                ("Neighbourhood processing radius of " +
                 "{0} km ".format(radius_in_km) +
                 "gives zero cell extent")
            )
        elif grid_cells_x < 0 or grid_cells_y < 0:
            raise ValueError(
                ("Neighbourhood processing radius of " +
                 "{0} km ".format(radius_in_km) +
                 "gives a negative cell extent")
            )
        if (grid_cells_x > max_radius_in_grid_cells or
                grid_cells_y > max_radius_in_grid_cells):
            raise ValueError(
                ("Neighbourhood processing radius of " +
                 "{0} km ".format(radius_in_km) +
                 "exceeds maximum grid cell extent")
            )
        return grid_cells_x, grid_cells_y


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
    def mean_over_neighbourhood(cube, cells_x, cells_y):
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

        Returns
        -------
        cube : iris.cube.Cube
            Cube to which square neighbourhood has been applied.
        """
        def _sum_and_area_for_edge_cases(cube, cells_x, cells_y, i, j):
            """
            Function to calculate the total sum and area of neighbourhoods
            surrounding edge cases which can't use the flatten and roll method.

            Calculates the total sum and area of the neighbourhood surrounding
            grid point (i,j) using the 4-point method.
            """
            x_min = i-cells_x-1
            x_max = min(cube.shape[1]-1, i+cells_x)
            y_min = j-cells_y-1
            y_max = min(cube.shape[0]-1, j+cells_y)
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

            # Initialise the neighbourhood size array and calculate
            # neighbourhood size for non edge cases.
            neighbourhood_area = np.zeros(neighbourhood_total.shape)
            neighbourhood_area.fill((2*cells_x+1) * (2*cells_y+1))
            # Calculate total sum and area of neighbourhood for edge cases.
            edge_rows = range(cells_x*2) + range(n_rows-2*cells_x, n_rows)
            edge_columns = range(cells_x*2) + range(n_columns-2*cells_x,
                                                    n_columns)
            for j in range(n_rows):
                for i in (edge_columns):
                    neighbourhood_total[j, i], neighbourhood_area[j, i] = (
                        _sum_and_area_for_edge_cases(
                            slice_2d, cells_x, cells_y, i, j))
            for i in range(n_columns):
                for j in (edge_rows):
                    neighbourhood_total[j, i], neighbourhood_area[j, i] = (
                        _sum_and_area_for_edge_cases(
                            slice_2d, cells_x, cells_y, i, j))
            slice_2d.data = neighbourhood_total/neighbourhood_area
            cubelist.append(slice_2d)
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

        """
        yname = cube.coord(axis="y").name()
        xname = cube.coord(axis="x").name()
        cubelist = iris.cube.CubeList([])
        for slice_2d in cube.slices([yname, xname]):
            data = slice_2d.data
            data_summed_along_y = np.cumsum(data, axis=0)
            data_summed_along_x = (
                np.cumsum(data_summed_along_y, axis=1))
            slice_2d.data = data_summed_along_x
            cubelist.append(slice_2d)
        return cubelist.merge_cube()

    @staticmethod
    def run(cube, radius_in_km):
        """
        Call the methods required to apply a square neighbourhood
        method to a cube.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube containing the array to which the square neighbourhood
            will be applied.
        radius_in_km : Float
            Radius in kilometres for use in specifying the number of
            grid cells used to create a square neighbourhood.

        Returns
        -------
        cube : Iris.cube.Cube
            Cube containing the smoothed field after the square neighbourhood
            method has been applied.
        """
        if isinstance(cube.data, np.ma.MaskedArray):
            msg = ('Masked data is not currently supported in ',
                   'SquareNeighbourhood.')
            raise ValueError(msg)
        if np.any(np.isnan(cube.data)):
            msg = ('Data array contains NaNs which are not currently ',
                   'supported in SquareNeighbourhood.')
            raise ValueError(msg)
        summed_up_cube = SquareNeighbourhood.cumulate_array(cube)
        grid_cells_x, grid_cells_y = (
            Utilities.get_neighbourhood_width_in_grid_cells(
                summed_up_cube, radius_in_km, MAX_RADIUS_IN_GRID_CELLS))
        neighbourhood_averaged_cube = (
            SquareNeighbourhood.mean_over_neighbourhood(
                summed_up_cube, grid_cells_x, grid_cells_y))
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

    def run(self, cube, radius_in_km):
        """
        Call the methods required to calculate and apply a circular
        neighbourhood.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube containing to array to apply CircularNeighbourhood processing
            to.
        radius_in_km : Float
            Radius in kilometres for use in specifying the number of
            grid cells used to create a circular neighbourhood.

        Returns
        -------
        cube : Iris.cube.Cube
            Cube containing the smoothed field after the kernel has been
            applied.
        """
        ranges = Utilities.get_neighbourhood_width_in_grid_cells(
            cube, radius_in_km, MAX_RADIUS_IN_GRID_CELLS)
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

    def __init__(self, neighbourhood_method, radii_in_km, lead_times=None,
                 unweighted_mode=False):
        """
        Create a neighbourhood processing plugin that applies a smoothing
        to points in a cube.

        Parameters
        ----------

        neighbourhood_method : str
            Name of the neighbourhood method to use. Options: 'circular'.
        radii_in_km : float or List (if defining lead times)
            The radii in kilometres of the neighbourhood to apply.
            Rounded up to convert into integer number of grid
            points east and north, based on the characteristic spacing
            at the zero indices of the cube projection-x/y coords.
        lead_times : None or List
            List of lead times or forecast periods, at which thel radii
            within radii_in_km are defined. The lead times are expected
            in hours.
        unweighted_mode : boolean
            If True, use a circle with constant weighting.
            If False, use a circle for neighbourhood kernel with
            weighting decreasing with radius.

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

        if isinstance(radii_in_km, list):
            self.radii_in_km = [float(x) for x in radii_in_km]
        else:
            self.radii_in_km = float(radii_in_km)
        self.lead_times = lead_times
        if self.lead_times is not None:
            if len(radii_in_km) != len(lead_times):
                msg = ("There is a mismatch in the number of radii "
                       "and the number of lead times. "
                       "Unable to continue due to mismatch.")
                raise ValueError(msg)
        self.unweighted_mode = bool(unweighted_mode)

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<NeighbourhoodProcessing: neighbourhood_method: {}; '
                  'radii_in_km: {}; lead_times: {}; '
                  'unweighted_mode: {}>')
        return result.format(
            self.neighbourhood_method_key, self.radii_in_km, self.lead_times,
            self.unweighted_mode)

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
            pass
        else:
            if len(realiz_coord.points) > 1:
                raise ValueError("Does not operate across realizations.")
            else:
                for cube_slice in cube.slices_over("realization"):
                    cube = cube_slice
        if np.isnan(cube.data).any():
            raise ValueError("Error: NaN detected in input cube data")

        if self.lead_times is None:
            radius_in_km = self.radii_in_km
            cube = self.neighbourhood_method.run(cube, radius_in_km)
        else:
            required_lead_times = Utilities.find_required_lead_times(cube)
            # Interpolate to find the radius at each required lead time.
            required_radii_in_km = (
                np.interp(
                    required_lead_times, self.lead_times, self.radii_in_km))
            cubes = iris.cube.CubeList([])
            # Find the number of grid cells required for creating the
            # neighbourhood, and then apply the neighbourhood processing method
            # to smooth the field.
            for cube_slice, radius_in_km in (
                    zip(cube.slices_over("time"), required_radii_in_km)):
                cube_slice = self.neighbourhood_method.run(
                    cube_slice, radius_in_km)
                cube_slice = iris.util.new_axis(cube_slice, "time")
                cubes.append(cube_slice)
            cube = concatenate_cubes(cubes, coords_to_slice_over=["time"])
        return cube
