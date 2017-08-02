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
"""Module containing neighbourhood processing percentiles classes."""

import iris
from iris.exceptions import CoordinateNotFoundError
import numpy as np

from improver.utilities.cube_manipulation import concatenate_cubes
from improver.nbhood.nbhood import Utilities as NBUtilities
from improver.nbhood.circular_kernel import (Utilities, CircularKernelNumpy)
from improver.utilities.spatial import (
    convert_distance_into_number_of_grid_cells)
from improver.percentile import PercentileConverter

# Maximum radius of the neighbourhood width in grid cells.
MAX_RADIUS_IN_GRID_CELLS = 500


class NeighbourhoodPercentiles(object):
    """
    Apply a neigbourhood processing method to each 2D slice in a cube.

    When applied to a cube, it samples all points in a neighbourhood as
    equal realizations and derives the requested percentile distribution.

    The neighbourhood methods will presently only work with projections in
    which the x grid point spacing and y grid point spacing are constant
    over the entire domain, such as the UK Standard Grid projection

    """

    def __init__(self, method, radii, lead_times=None,
                 ens_factor=1.0,
                 percentiles=PercentileConverter.DEFAULT_PERCENTILES):
        """
        Create a neighbourhood processing plugin that applies a smoothing
        to points in a cube.

        Parameters
        ----------

        method : str
            Name of the method to use. Options: circular.
        radii : float or List (if defining lead times)
            The radii in metres of the neighbourhood to apply.
            Rounded up to convert into integer number of grid
            points east and north, based on the characteristic spacing
            at the zero indices of the cube projection-x and y coords.
        lead_times : None or List (optional)
            List of lead times or forecast periods, at which the radii
            within 'radii' are defined. The lead times are expected
            in hours.
        ens_factor : float (optional)
            The factor with which to adjust the neighbourhood size
            for more than one ensemble member.
            If ens_factor = 1.0 this essentially conserves ensemble
            members if every grid square is considered to be the
            equivalent of an ensemble member.
            Optional, defaults to 1.0
        percentiles : list (optional)
            Percentile values at which to calculate; if not provided uses
            DEFAULT_PERCENTILES from percentile module.
        """
        self.percentiles = percentiles
        self.method_key = method
        methods = {
            "circular": CircularKernelNumpy}
        try:
            usemethod = methods[self.method_key]
            self.method = usemethod(percentiles=self.percentiles)
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
            radii = NBUtilities.adjust_nsize_for_ens(self.ens_factor,
                                                     num_ens, self.radii)
        else:
            # Interpolate to find the radius at each required lead time.
            radii = (
                np.interp(
                    cube_lead_times, self.lead_times, self.radii))
            for i, val in enumerate(radii):
                radii[i] = NBUtilities.adjust_nsize_for_ens(self.ens_factor,
                                                            num_ens, val)
        return radii

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<NeighbourhoodPercentiles: method: {}; '
                  'radii: {}; lead_times: {}; '
                  'ens_factor: {}; percentile-count: {}>')
        return result.format(
            self.method_key, self.radii, self.lead_times,
            self.ens_factor, len(self.percentiles))

    def process(self, cube):
        """
        Apply neighbourhood processing method to return percentiles over area.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube to apply a neighbourhood processing method to, in order to
            generate percentiles.

        Returns
        -------
        cube : Iris.cube.Cube
            Cube after applying a neighbourhood processing method with
            additional percentile coordinate.

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
        else:
            num_ens = len(realiz_coord.points)
            if 'source_realizations' in cube.attributes:
                msg = ("Realizations and attribute source_realizations "
                       "should not both be set in input cube")
                raise ValueError(msg)

        if np.isnan(cube.data).any():
            raise ValueError("Error: NaN detected in input cube data")

        # Find the number of grid cells required for creating the
        # neighbourhood (take largest value from returned tuple)
        if self.lead_times is None:
            radius = self._find_radii(num_ens)
            ranges = max(convert_distance_into_number_of_grid_cells(
                cube, radius, MAX_RADIUS_IN_GRID_CELLS))
            cube_new = self.method.run(cube, ranges)
        else:
            cube_lead_times = (
                NBUtilities.find_required_lead_times(cube))
            # Interpolate to find the radius at each required lead time.
            required_radii = (
                self._find_radii(num_ens, cube_lead_times=cube_lead_times))

            cubes = iris.cube.CubeList([])
            for cube_slice, radius in (
                    zip(cube.slices_over("time"),
                        required_radii)):
                ranges = max(convert_distance_into_number_of_grid_cells(
                    cube, radius, MAX_RADIUS_IN_GRID_CELLS))
                cube_perc = self.method.run(cube_slice, ranges)
                cube_perc = iris.util.new_axis(cube_perc, "time")
                cubes.append(cube_perc)
                cube_new = concatenate_cubes(cubes,
                                             coords_to_slice_over=["time"])

        return cube_new
