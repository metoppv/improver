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

from improver.ensemble_calibration.ensemble_calibration_utilities import (
    concatenate_cubes)


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

    @staticmethod
    def adjust_nsize_for_ens(ens_factor, num_ens, width_in_km):
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
        width_in_km : float
            radius or width appropriate for a single forecast in km.

        Returns
        -------
        new_width : float
            new neighbourhood radius (km).

        """
        if num_ens <= 1.0:
            new_width = width_in_km
        else:
            new_width = (ens_factor *
                         math.sqrt((width_in_km**2.0)/num_ens))
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
        summed_up_cube = SquareNeighbourhood.cumulate_array(cube)
        return summed_up_cube


class CircularNeighbourhood(object):

    """
    Methods for use in the calculation and application of a circular
    neighbourhood.

    A maximum kernel radius of 500 grid cells is imposed in order to
    avoid computational ineffiency and possible memory errors.
    """

    # Maximum radius of the neighbourhood width in grid cells.
    MAX_RADIUS_IN_GRID_CELLS = 500

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
            cube, radius_in_km, self.MAX_RADIUS_IN_GRID_CELLS)
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
                 unweighted_mode=False, ens_factor=1.0):
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
            at the zero indices of the cube projection-x and y coords.
        lead_times : None or List
            List of lead times or forecast periods, at which thel radii
            within radii_in_km are defined. The lead times are expected
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
            "circular": CircularNeighbourhood}
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
                                                   num_ens,
                                                   self.radii_in_km)
        else:
            # Interpolate to find the radius at each required lead time.
            radii = (
                np.interp(
                    cube_lead_times, self.lead_times, self.radii_in_km))
            for i, val in enumerate(radii):
                radii[i] = Utilities.adjust_nsize_for_ens(self.ens_factor,
                                                          num_ens, val)
        return radii

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<NeighbourhoodProcessing: neighbourhood_method: {}; '
                  'radii_in_km: {}; lead_times: {}; '
                  'unweighted_mode: {}; ens_factor: {}>')
        return result.format(
            self.neighbourhood_method_key, self.radii_in_km, self.lead_times,
            self.unweighted_mode, self.ens_factor)

    def process(self, cube):
        """
        Spply neighbourhood processing method, in order to smooth the
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
                radius_in_km = self._find_radii(num_ens)
                cube_new = self.neighbourhood_method.run(cube_realization,
                                                         radius_in_km)
            else:
                cube_lead_times = (
                    Utilities.find_required_lead_times(cube_realization))
                # Interpolate to find the radius at each required lead time.
                required_radii_in_km = (
                    self._find_radii(num_ens,
                                     cube_lead_times=cube_lead_times))

                cubes = iris.cube.CubeList([])
                # Find the number of grid cells required for creating the
                # neighbourhood, and then apply the neighbourhood
                # processing method to smooth the field.
                for cube_slice, radius_in_km in (
                        zip(cube_realization.slices_over("time"),
                            required_radii_in_km)):
                    cube_slice = self.neighbourhood_method.run(
                        cube_slice, radius_in_km)
                    cube_slice = iris.util.new_axis(cube_slice, "time")
                    cubes.append(cube_slice)
                cube_new = concatenate_cubes(cubes,
                                             coords_to_slice_over=["time"])

            cubelist.append(cube_new)
        cube = cubelist.merge_cube()

        return cube
