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

from improver.utilities.cube_checker import (
    check_cube_coordinates, find_dimension_coordinate_mismatch)
from improver.utilities.cube_manipulation import concatenate_cubes

from improver.nbhood.circular_kernel import (
    CircularProbabilities, CircularPercentiles)
from improver.nbhood.square_kernel import SquareProbabilities
from improver.nbhood.utilities import Utilities
from improver.percentile import PercentileConverter


class NeighbourhoodProcessing(object):
    """
    Apply a neigbourhood processing method to a thresholded cube.

    When applied to a thresholded probabilistic cube, it acts like a
    low-pass filter which reduces noisiness in the probabilities.

    The neighbourhood methods will presently only work with projections in
    which the x grid point spacing and y grid point spacing are constant
    over the entire domain, such as the UK national grid projection

    """

    def __init__(self, neighbourhood_method, neighbourhood_shape,
                 radii, lead_times=None, weighted_mode=True, ens_factor=1.0,
                 percentiles=PercentileConverter.DEFAULT_PERCENTILES):
        """
        Create a neighbourhood processing plugin that applies a smoothing
        to points in a cube.

        Parameters
        ----------

        neighbourhood_method : str
            Name of the neighbourhood method to use.
            Options: 'probabilities', 'percentiles'.
        neighbourhood_shape : str
            Name of the neighbourhood shape to use.
            Options: 'circular', 'square'.
        radii : float or List (if defining lead times)
            The radii in metres of the neighbourhood to apply.
            Rounded up to convert into integer number of grid
            points east and north, based on the characteristic spacing
            at the zero indices of the cube projection-x and y coords.
        lead_times : None or List (optional)
            List of lead times or forecast periods, at which the radii
            within 'radii' are defined. The lead times are expected
            in hours.
        weighted_mode : boolean (optional)
            If False, use a circle with constant weighting.
            If True, use a circle for neighbourhood kernel with
            weighting decreasing with radius.
            This value only has an effect with the circular_probabilities
            method.
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
            This value is ignored for probability methods.
        """
        self.percentiles = tuple(percentiles)
        self.weighted_mode = bool(weighted_mode)
        self.neighbourhood_method_key = neighbourhood_method
        self.neighbourhood_shape = neighbourhood_shape

        methods = {
            "circular_probabilities": CircularProbabilities,
            "circular_percentiles": CircularPercentiles,
            "square_probabilities": SquareProbabilities}

        method_shape = "_".join([neighbourhood_shape, neighbourhood_method])

        try:
            method = methods[method_shape]
            self.neighbourhood_method = method(
                weighted_mode=self.weighted_mode,
                percentiles=self.percentiles)
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
        self.ens_factor = float(ens_factor)

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<NeighbourhoodProcessing: neighbourhood_method: {}; '
                  'neighbourhood_shape: {}; radii: {}; lead_times: {}; '
                  'weighted_mode: {}; ens_factor: {}; '
                  'percentiles: {}>')
        return result.format(
            self.neighbourhood_method_key, self.neighbourhood_shape,
            self.radii, self.lead_times,
            self.weighted_mode, self.ens_factor,
            PercentileConverter.DEFAULT_PERCENTILES)

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

    def process(self, cube):
        """
        Supply neighbourhood processing method, in order to smooth the
        input cube.

        Parameters
        ----------
        cube : Iris.cube.Cube
            Cube to apply a neighbourhood processing method to, in order to
            generate a smoother field.
            For probability methods, the cube should have already been
            thresholded.

        Returns
        -------
        cube : Iris.cube.Cube
            Cube after applying a neighbourhood processing method, so that the
            resulting field is either:
            Smoothed probability threshold data.
            Smoothed percentile data (with percentiles as an added dimension)

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
        merged_cube = cubelist.merge_cube()
        # Promote dimensional coordinates that have been demoted to scalars.
        exception_coordinates = (
            find_dimension_coordinate_mismatch(
                cube, merged_cube, two_way_mismatch=False))
        merged_cube = check_cube_coordinates(
            cube, merged_cube, exception_coordinates=exception_coordinates)
        return merged_cube
