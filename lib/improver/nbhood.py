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


class Utilities(object):

    """
    Utilities for neighbourhood processing.

    The methods available in this class are:
    * find_required_lead_times
    """

    def __init__(self):
        pass

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


class SquareNeighbourhood(object):

    def __init__(self, cube):
        """
        Initialise class.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube containing the array to which the square neighbourhood
            will be applied

        """
        self.cube = cube

    def cumulate_array(self):
        """
        Method to calculate the cumulative sum of an array, by first
        cumulating along the y direction so that the largest values
        are in the upper row, and then cumulating along the x direction,
        so that the largest values are in the right hand column.

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
        yname = self.cube.coord(axis="y").name()
        xname = self.cube.coord(axis="x").name()
        cubelist = iris.cube.CubeList([])
        for slice_2d in self.cube.slices([yname, xname]):
            data = slice_2d.data
            data_summed_along_y = (
                np.flipud(np.cumsum(np.flipud(data), axis=0)))
            data_summed_along_x = (
                np.cumsum(data_summed_along_y, axis=1))
            slice_2d.data = data_summed_along_x
            cubelist.append(slice_2d)
        return cubelist.merge_cube()

    def run(self):
        """
        Call the methods required to apply a square neighbourhood
        method to a cube.

        Returns
        -------
        cube : Iris.cube.Cube
            Cube containing the smoothed field after the kernel has been
            applied.
        """
        summed_up_cube = self.cumulate_array()
        return summed_up_cube


class CircularNeighbourhood(object):

    """
    Methods for use in the calculation and application of a circular
    neighbourhood.
    """

    # Max extent of kernel in grid cells.
    MAX_KERNEL_CELL_RADIUS = 500

    def __init__(self, cube, radius_in_km, unweighted_mode=False):
        """
        Initialise class.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube containing the x and y coordinates, which will be used for
            calculating the number of grid cells in the x and y direction,
            which equates to the size of the desired radii.
        radius_in_km : Float
            Radius in kilometres for use in specifying the number of
            grid cells used to create a kernel.
        unweighted_mode : boolean
            If True, use a circle with constant weighting.
            If False, use a circle for neighbourhood kernel with
            weighting decreasing with radius.

        """
        self.cube = cube
        self.radius_in_km = radius_in_km
        self.unweighted_mode = unweighted_mode

    def apply_circular_kernel(self, ranges):
        """
        Method to apply a circular kernel to the data within the input cube in
        order to smooth the resulting field.

        Parameters
        ----------
        ranges : Tuple
            Number of grid cells in the x and y direction used to create
            the kernel.

        Returns
        -------
        cube : Iris.cube.Cube
            Cube containing the smoothed field after the kernel has been
            applied.

        """
        cube = self.cube
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

    def get_grid_x_y_kernel_ranges(self):
        """
        Return the number of grid cells in the x and y direction
        to be used to create the kernel.

        Returns
        -------
        grid_cells_x : Integer
            Number of grid cells in the x direction to be used to create the
            kernel.
        grid_cells_y : Integer
            Number of grid cells in the y direction to be used to create the
            kernel.

        """
        try:
            x_coord = self.cube.coord("projection_x_coordinate").copy()
            y_coord = self.cube.coord("projection_y_coordinate").copy()
        except CoordinateNotFoundError:
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
        elif grid_cells_x < 0 or grid_cells_y < 0:
            raise ValueError(
                ("Neighbourhood processing radius of " +
                 "{0} km ".format(self.radius_in_km) +
                 "gives a negative cell extent")
            )
        if (grid_cells_x > self.MAX_KERNEL_CELL_RADIUS or
                grid_cells_y > self.MAX_KERNEL_CELL_RADIUS):
            raise ValueError(
                ("Neighbourhood processing radius of " +
                 "{0} km ".format(self.radius_in_km) +
                 "exceeds maximum grid cell extent")
            )
        return grid_cells_x, grid_cells_y

    def run(self):
        """
        Call the methods required to calculate and apply a circular
        neighbourhood.

        Returns
        -------
        cube : Iris.cube.Cube
            Cube containing the smoothed field after the kernel has been
            applied.
        """
        ranges = self.get_grid_x_y_kernel_ranges()
        cube = self.apply_circular_kernel(ranges)
        return cube


class NeighbourhoodProcessing(object):
    """
    Apply a neigbourhood processing kernel to a thresholded cube.

    When applied to a thresholded probabilistic cube, it acts like a
    low-pass filter which reduces noisiness in the probabilities.

    The kernel will presently only work with projections in which the
    x grid point spacing and y grid point spacing are constant over the
      entire domain, such as the UK national grid projection

    A maximum kernel radius of 500 grid cells is imposed in order to
    avoid computational ineffiency and possible memory errors.

    """

    def __init__(self, neighbourhood_method, radii_in_km, lead_times=None,
                 unweighted_mode=False):
        """
        Create a neighbourhood processing plugin that applies a smoothing
        kernel to points in a cube.

        Parameters
        ----------

        neighbourhood_method : str
            Name of the neighbourhood method to use. Options: 'circular'.
        radii_in_km : float or List (if defining lead times)
            The radii in kilometres of the neighbourhood kernel to
            apply. Rounded up to convert into integer number of grid
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
        methods = {
            "circular": CircularNeighbourhood, "square": SquareNeighbourhood}
        try:
            self.neighbourhood_method = methods[neighbourhood_method]
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

    def __str__(self):
        result = ('<NeighbourhoodProcessing: neighbourhood_method: {}; ' +
                  'radii_in_km: {}; lead_times: {};' +
                  'unweighted_mode: {}>')
        return result.format(
            self.neighbourhood_method, self.radii_in_km, self.lead_times,
            self.unweighted_mode)

    def process(self, cube):
        """
        Calculate a kernel to apply, in order to smooth the input cube.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube to apply a kernel to, in order to generate a smoother field.

        Returns
        -------
        cube : Iris.cube.Cube
            Cube after applying a kernel, so that the resulting field is
            smoothed.

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
            cube = self.neighbourhood_method(
                cube, radius_in_km, self.unweighted_mode).run()
        else:
            required_lead_times = Utilities.find_required_lead_times(cube)
            # Interpolate to find the radius at each required lead time.
            required_radii_in_km = (
                np.interp(
                    required_lead_times, self.lead_times, self.radii_in_km))
            cubes = iris.cube.CubeList([])
            # Find the number of grid cells required for creating the kernel,
            # and then apply the kernel to smooth the field.
            for cube_slice, radius_in_km in (
                    zip(cube.slices_over("time"), required_radii_in_km)):
                cube_slice = self.neighbourhood_method(
                    cube_slice, radius_in_km, self.unweighted_mode).run()
                cube_slice = iris.util.new_axis(cube_slice, "time")
                cubes.append(cube_slice)
            cube = concatenate_cubes(cubes, coords_to_slice_over=["time"])
        return cube
