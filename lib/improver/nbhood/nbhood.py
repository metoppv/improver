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
import numpy as np

from improver.nbhood.circular_kernel import (
    CircularNeighbourhood, GeneratePercentilesFromACircularNeighbourhood)
from improver.nbhood.square_kernel import SquareNeighbourhood

from improver.constants import DEFAULT_PERCENTILES
from improver.utilities.cube_checker import (
    check_cube_coordinates, find_dimension_coordinate_mismatch)
from improver.utilities.cube_manipulation import concatenate_cubes
from improver.utilities.temporal import forecast_period_coord


class BaseNeighbourhoodProcessing(object):
    """
    Apply a neighbourhood processing method to a thresholded cube. This is a
    base class for usage with a subclass that will inherit the functionality
    within this base class.

    When applied to a thresholded probabilistic cube, it acts like a
    low-pass filter which reduces noisiness in the probabilities.

    The neighbourhood methods will presently only work with projections in
    which the x grid point spacing and y grid point spacing are constant
    over the entire domain, such as the UK national grid projection

    """

    def __init__(self, neighbourhood_method, radii, lead_times=None,
                 ens_factor=1.0):
        """
        Create a neighbourhood processing plugin that applies a smoothing
        to points in a cube.

        Args:
            neighbourhood_method (Class object):
                Instance of the class containing the method that will be used
                for the neighbourhood processing.
            radii (float or List if defining lead times):
                The radii in metres of the neighbourhood to apply.
                Rounded up to convert into integer number of grid
                points east and north, based on the characteristic spacing
                at the zero indices of the cube projection-x and y coords.

        Keyword Args:
            lead_times (None or List):
                List of lead times or forecast periods, at which the radii
                within 'radii' are defined. The lead times are expected
                in hours.
            ens_factor (float):
                The factor with which to adjust the neighbourhood size
                for more than one ensemble member.
                If ens_factor = 1.0 this essentially conserves ensemble
                members if every grid square is considered to be the
                equivalent of an ensemble member.
                Optional, defaults to 1.0
        """
        self.neighbourhood_method = neighbourhood_method

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

    def adjust_nsize_for_ens(self, num_ens, width):
        """
        Adjust neighbourhood size according to ensemble size.

        Args:
            num_ens (float):
                Number of realizations or ensemble members.
            width (float):
                radius or width appropriate for a single forecast in m.

        Returns:
            new_width (float):
                new neighbourhood radius (m).

        """
        if num_ens <= 1.0:
            new_width = width
        else:
            new_width = (self.ens_factor *
                         math.sqrt((width**2.0)/num_ens))
        return new_width

    def _find_radii(self, num_ens, cube_lead_times=None):
        """Revise radius or radii for found lead times and ensemble members

        If cube_lead_times is None just adjust for ensemble
        members if necessary.
        Otherwise interpolate to find radius at each cube
        lead time and adjust for ensemble members if necessary.

        Args:
            num_ens (float):
                Number of ensemble members or realizations.

        Keyword Args:
            cube_lead_times (np.array):
                Array of forecast times found in cube.

        Returns:
            radii (float or np.array of float):
                Required neighbourhood sizes.
        """
        if cube_lead_times is None:
            radii = self.adjust_nsize_for_ens(num_ens, self.radii)
        else:
            # Interpolate to find the radius at each required lead time.
            radii = (
                np.interp(
                    cube_lead_times, self.lead_times, self.radii))
            for i, val in enumerate(radii):
                radii[i] = self.adjust_nsize_for_ens(num_ens, val)
        return radii

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        if callable(self.neighbourhood_method):
            neighbourhood_method = self.neighbourhood_method()
        else:
            neighbourhood_method = self.neighbourhood_method
        result = ('<BaseNeighbourhoodProcessing: neighbourhood_method: {}; '
                  'radii: {}; lead_times: {}; ens_factor: {}>')
        return result.format(
            neighbourhood_method, self.radii, self.lead_times,
            self.ens_factor)

    def process(self, cube, mask_cube=None):
        """
        Supply neighbourhood processing method, in order to smooth the
        input cube.

        Args:
            cube (Iris.cube.Cube):
                Cube to apply a neighbourhood processing method to, in order to
                generate a smoother field.

        Keyword Args:
            mask_cube (Iris.cube.Cube):
                Cube containing the array to be used as a mask.

        Returns:
            cube (Iris.cube.Cube):
                Cube after applying a neighbourhood processing method, so that
                the resulting field is smoothed.

        """
        if (not getattr(self.neighbourhood_method, "run", None) or
                not callable(self.neighbourhood_method.run)):
            msg = ("{} is not valid as a neighbourhood_method. "
                   "Please choose a valid neighbourhood_method with a "
                   "run method.".format(
                       self.neighbourhood_method))
            raise ValueError(msg)

        # Check if a dimensional realization coordinate exists. If so, the
        # cube is sliced, so that it becomes a scalar coordinate.
        try:
            realiz_coord = cube.coord('realization', dim_coords=True)
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

        cubes_real = []
        for cube_realization in slices_over_realization:
            if self.lead_times is None:
                radius = self._find_radii(num_ens)
                cube_new = self.neighbourhood_method.run(
                    cube_realization, radius, mask_cube=mask_cube)
            else:
                # Interpolate to find the radius at each required lead time.
                fp_coord = forecast_period_coord(cube_realization)
                fp_coord.convert_units("hours")
                required_radii = self._find_radii(
                    num_ens,
                    cube_lead_times=fp_coord.points
                )

                cubes_time = iris.cube.CubeList([])
                # Find the number of grid cells required for creating the
                # neighbourhood, and then apply the neighbourhood
                # processing method to smooth the field.
                for cube_slice, radius in (
                        zip(cube_realization.slices_over("time"),
                            required_radii)):
                    cube_slice = self.neighbourhood_method.run(
                        cube_slice, radius, mask_cube=mask_cube)
                    cubes_time.append(cube_slice)
                if len(cubes_time) > 1:
                    cube_new = concatenate_cubes(
                        cubes_time, coords_to_slice_over=["time"])
                else:
                    cube_new = cubes_time[0]
            cubes_real.append(cube_new)
        if len(cubes_real) > 1:
            combined_cube = concatenate_cubes(
                cubes_real, coords_to_slice_over=["realization"])
        else:
            combined_cube = cubes_real[0]

        # Promote dimensional coordinates that used to be present.
        exception_coordinates = (
            find_dimension_coordinate_mismatch(
                cube, combined_cube, two_way_mismatch=False))
        combined_cube = check_cube_coordinates(
            cube, combined_cube, exception_coordinates=exception_coordinates)
        return combined_cube


class GeneratePercentilesFromANeighbourhood(BaseNeighbourhoodProcessing):

    """Class for generating percentiles from a neighbourhood."""

    def __init__(
            self, neighbourhood_method, radii, lead_times=None,
            ens_factor=1.0, percentiles=DEFAULT_PERCENTILES):
        """
        Create a neighbourhood processing subclass that generates percentiles
        from a neighbourhood of points.

        Args:
            neighbourhood_method (str):
                Name of the neighbourhood method to use. Options: 'circular'.
            radii (float or List if defining lead times):
                The radii in metres of the neighbourhood to apply.
                Rounded up to convert into integer number of grid
                points east and north, based on the characteristic spacing
                at the zero indices of the cube projection-x and y coords.

        Keyword Args:
            lead_times (None or List):
                List of lead times or forecast periods, at which the radii
                within 'radii' are defined. The lead times are expected
                in hours.
            ens_factor (float):
                The factor with which to adjust the neighbourhood size
                for more than one ensemble member.
                If ens_factor = 1.0 this essentially conserves ensemble
                members if every grid square is considered to be the
                equivalent of an ensemble member.
                Optional, defaults to 1.0
            percentiles (list):
                Percentile values at which to calculate; if not provided uses
                DEFAULT_PERCENTILES.
        """
        super(GeneratePercentilesFromANeighbourhood, self).__init__(
            neighbourhood_method, radii, lead_times=lead_times,
            ens_factor=ens_factor)

        methods = {
            "circular": GeneratePercentilesFromACircularNeighbourhood}
        try:
            method = methods[neighbourhood_method]
            self.neighbourhood_method = method(percentiles=percentiles)
        except KeyError:
            msg = ("The neighbourhood_method requested: {} is not a "
                   "supported method. Please choose from: {}".format(
                       neighbourhood_method, methods.keys()))
            raise KeyError(msg)


class NeighbourhoodProcessing(BaseNeighbourhoodProcessing):

    """Class for applying neighbourhood processing to produce a smoothed field
    within the chosen neighbourhood."""

    def __init__(
            self, neighbourhood_method, radii, lead_times=None,
            ens_factor=1.0, weighted_mode=True, sum_or_fraction="fraction",
            re_mask=False):
        """
        Create a neighbourhood processing subclass that applies a smoothing
        to points in a cube.

        Args:
            neighbourhood_method (str):
                Name of the neighbourhood method to use. Options: 'circular',
                'square'.
            radii (float or List if defining lead times):
                The radii in metres of the neighbourhood to apply.
                Rounded up to convert into integer number of grid
                points east and north, based on the characteristic spacing
                at the zero indices of the cube projection-x and y coords.

        Keyword Args:
            lead_times (None or List):
                List of lead times or forecast periods, at which the radii
                within 'radii' are defined. The lead times are expected
                in hours.
            ens_factor (float):
                The factor with which to adjust the neighbourhood size
                for more than one ensemble member.
                If ens_factor = 1.0 this essentially conserves ensemble
                members if every grid square is considered to be the
                equivalent of an ensemble member.
                Optional, defaults to 1.0
            weighted_mode (boolean):
                If True, use a circle for neighbourhood kernel with
                weighting decreasing with radius.
                If False, use a circle with constant weighting.
            sum_or_fraction (string):
                Identifier for whether sum or fraction should be returned from
                neighbourhooding. The sum represents the sum of the
                neighbourhood. The fraction represents the sum of the
                neighbourhood divided by the neighbourhood area.
                "fraction" is the default.
                Valid options are "sum" or "fraction".
            re_mask (boolean):
                If re_mask is True, the original un-neighbourhood processed
                mask is applied to mask out the neighbourhood processed cube.
                If re_mask is False, the original un-neighbourhood processed
                mask is not applied. Therefore, the neighbourhood processing
                may result in values being present in areas that were
                originally masked.
        """
        super(NeighbourhoodProcessing, self).__init__(
            neighbourhood_method, radii, lead_times=lead_times,
            ens_factor=ens_factor)

        methods = {
            "circular": CircularNeighbourhood,
            "square": SquareNeighbourhood}
        try:
            method = methods[neighbourhood_method]
            self.neighbourhood_method = method(
                weighted_mode, sum_or_fraction, re_mask)
        except KeyError:
            msg = ("The neighbourhood_method requested: {} is not a "
                   "supported method. Please choose from: {}".format(
                       neighbourhood_method, methods.keys()))
            raise KeyError(msg)
