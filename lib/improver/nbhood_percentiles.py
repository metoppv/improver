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

from improver.utilities.cube_manipulation import concatenate_cubes
from improver.nbhood import Utilities
from percentile import PercentileConverter

# Maximum radius of the neighbourhood width in grid cells.
MAX_RADIUS_IN_GRID_CELLS = 500


class NeighbourhoodPercentiles(object):
    """
    Apply a neigbourhood processing method to each 2D slice in a cube.

    When applied to a cube, it samples all points in a neighbourhood as
    equal realizations and derives the percentile distribution.

    The neighbourhood methods will presently only work with projections in
    which the x grid point spacing and y grid point spacing are constant
    over the entire domain, such as the UK national grid projection

    """

    def __init__(self, method, radii, lead_times=None,
                 unweighted_mode=False, ens_factor=1.0,
                 percentiles=PercentileConverter.DEFAULT_PERCENTILES):
        """
        Create a neighbourhood processing plugin that applies a smoothing
        to points in a cube.

        Parameters
        ----------

        neighbourhood_method : str
            Name of the neighbourhood method to use. Options: TBC.
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
        percentiles : float or List
        """
        self.percentiles = percentiles
        self.method_key = method
        methods = {
            "circular_scipy": CircularKernelScipy,
            "circular_numpy": CircularKernelNumpy}
        try:
            usemethod = methods[self.method_key]
            self.method = usemethod(unweighted_mode, percentiles=self.percentiles)
        except KeyError:
            msg = ("The method requested: {} is not a "
                   "supported method. Please choose from: {}".format(
                       self.method_key, methods.keys()))
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
        result = ('<NeighbourhoodPercentiles: '
                  'radii: {}; lead_times: {}; '
                  'unweighted_mode: {}; ens_factor: {}; percentile-count: {}>')
        return result.format(
            self.radii, self.lead_times,
            self.unweighted_mode, self.ens_factor, len(self.percentiles))

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

        ranges = Utilities.get_neighbourhood_width_in_grid_cells(
            cube, self.radii, MAX_RADIUS_IN_GRID_CELLS)
        if self.lead_times is None:
            radius = self._find_radii(num_ens)
            cube_new = self.method.run(cube, ranges)
        else:
            cube_lead_times = (
                Utilities.find_required_lead_times(cube))
            # Interpolate to find the radius at each required lead time.
            required_radii = (
                self._find_radii(num_ens,
                                  cube_lead_times=cube_lead_times))

            cubes = iris.cube.CubeList([])
            # Find the number of grid cells required for creating the
            # neighbourhood
            for cube_slice, radius in (
                    zip(cube.slices_over("time"),
                        required_radii)):
                cube_perc = self.method.run(cube_slice, ranges)
                cube_perc = iris.util.new_axis(cube_perc, "time")
                cubes.append(cube_perc)
                cube_new = concatenate_cubes(cubes,
                                             coords_to_slice_over=["time"])


        return cube_new

class CircularKernelScipy(object):
    def __init__(self, unweighted_mode=False,
                 percentiles=PercentileConverter.DEFAULT_PERCENTILES):
        """

        Parameters
        ----------

        unweighted_mode : boolean
            If True, use a circle with constant weighting.
            If False, use a circle for neighbourhood kernel with
            weighting decreasing with radius.
        percentiles : float or List
        """
        self.percentiles = percentiles
        self.unweighted_mode = bool(unweighted_mode)

    def run(self, cube, ranges):
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
        outcube : Iris.cube.Cube
            Cube containing the percentile fields.

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
        # Derive percentiles on the kernel. Will return an extra dimension.
        pctcubelist = iris.cube.CubeList()
        for pct in self.percentiles:
            pctcube = cube.copy()
            pctcube.data = np.array(scipy.ndimage.filters.percentile_filter(
                data, pct, footprint=kernel, mode='nearest'))
            pctcube.add_aux_coord(iris.coords.DimCoord(pct, long_name='percentiles', units='%'))
            pctcubelist.append(pctcube)
        result = pctcubelist.merge_cube()
        return result


class CircularKernelNumpy(object):
    def __init__(self, unweighted_mode=False,
                 percentiles=PercentileConverter.DEFAULT_PERCENTILES):
        """

        Parameters
        ----------

        unweighted_mode : boolean
            If True, use a circle with constant weighting.
            If False, use a circle for neighbourhood kernel with
            weighting decreasing with radius.
        percentiles : float or List
        """
        self.percentiles = percentiles
        self.unweighted_mode = bool(unweighted_mode)

    def run(self, cube, ranges):
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
        outcube : Iris.cube.Cube
            Cube containing the percentile fields.

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
            fullranges[axis] = int(max(np.array(ranges)))
        fullranges = fullranges.astype(int)
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
        pctcubelist = iris.cube.CubeList()
        for slice_2d in cube.slices(['projection_x_coordinate',
                                     'projection_y_coordinate']):
            # Derive percentiles on the kernel. Will return an extra dimension.
            # Create a 1D data array padded with repeats of the local boundary mean.
            datashape = np.shape(cube.data)
            padded = np.pad(cube.data, max(fullranges), mode='mean', stat_length=fullranges[0])
            padshape = np.shape(padded) # Store size to make unflatten easier
            padded = padded.flatten()
            # Add 2nd dimension with each point's neighbourhood points along it
            nbhood_slices = [np.roll(padded, padshape[1]*j+i) for i in range(-fullranges[0], fullranges[0]+1) for j in range(-fullranges[1], fullranges[1]+1) if kernel[i+fullranges[0], j+fullranges[1]]>0.]
            # Collapse this dimension into percentiles (a new 2nd dimension)
            perc_data = np.percentile(nbhood_slices, self.percentiles, axis=0)
            # Return to 3D
            perc_data = perc_data.reshape(len(self.percentiles), padshape[0], padshape[1])
            # Create a cube for these data:
            pctcube = self.make_percentile_cube(cube)
            # And put in data, removing the padding
            pctcube.data = perc_data[:,fullranges[0]:-fullranges[0], fullranges[1]:-fullranges[1]]
            pctcubelist.append(pctcube)
        result = pctcubelist.merge_cube()
        return result

    def make_percentile_cube(self, cube):
        """Returns a cube with the same metadata as the sample cube, but with an added percentile dimension"""
        pctcubelist = iris.cube.CubeList()
        for pct in self.percentiles:
            pctcube = cube.copy()
            pctcube.add_aux_coord(iris.coords.DimCoord(pct, long_name='percentiles', units='%'))
            pctcubelist.append(pctcube)
        return pctcubelist.merge_cube()
