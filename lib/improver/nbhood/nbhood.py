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

from improver.nbhood.circular_kernel import (
    CircularNeighbourhood, GeneratePercentilesFromACircularNeighbourhood)
from improver.nbhood.square_kernel import SquareNeighbourhood

from improver.constants import DEFAULT_PERCENTILES
from improver.utilities.cube_checker import (
    check_cube_coordinates, find_dimension_coordinate_mismatch)
from improver.utilities.cube_manipulation import concatenate_cubes
from improver.utilities.spatial import (
    convert_distance_into_number_of_grid_cells)


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


class BaseNeighbourhoodProcessing(object):
    """
    Apply a neighbourhood processing method to a thresholded cube.

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

        Parameters
        ----------
        neighbourhood_method : Class object
            Instance of the class containing the method that will be used for
            the neighbourhood processing.
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
        if callable(self.neighbourhood_method):
            neighbourhood_method = self.neighbourhood_method()
        else:
            neighbourhood_method = self.neighbourhood_method

        result = ('<NeighbourhoodProcessing: neighbourhood_method: {}; '
                  'radii: {}; lead_times: {}; ens_factor: {}>')
        return result.format(
            neighbourhood_method, self.radii, self.lead_times,
            self.ens_factor)

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
        merged_cube = cubelist.merge_cube()
        # Promote dimensional coordinates that have been demoted to scalars.
        exception_coordinates = (
            find_dimension_coordinate_mismatch(
                cube, merged_cube, two_way_mismatch=False))
        merged_cube = check_cube_coordinates(
            cube, merged_cube, exception_coordinates=exception_coordinates)
        return merged_cube


class GeneratePercentilesFromANeighbourhood(BaseNeighbourhoodProcessing):

    """Class for generating percentiles from a neighbourhood."""

    def __init__(
            self, neighbourhood_method, radii, lead_times=None,
            ens_factor=1.0, percentiles=DEFAULT_PERCENTILES):
        """
        Create a neighbourhood processing subclass that generates percentiles
        from a neighbourhood of points.

        Parameters
        ----------
        neighbourhood_method : str
            Name of the neighbourhood method to use. Options: 'circular'.
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
            ens_factor=1.0, weighted_mode=True):
        """
        Create a neighbourhood processing subclass that applies a smoothing
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
        weighted_mode : boolean (optional)
            If True, use a circle for neighbourhood kernel with
            weighting decreasing with radius.
            If False, use a circle with constant weighting.

        """
        super(NeighbourhoodProcessing, self).__init__(
            neighbourhood_method, radii, lead_times=lead_times,
            ens_factor=ens_factor)

        methods = {
            "circular": CircularNeighbourhood,
            "square": SquareNeighbourhood}
        try:
            method = methods[neighbourhood_method]
            self.neighbourhood_method = method(weighted_mode)
        except KeyError:
            msg = ("The neighbourhood_method requested: {} is not a "
                   "supported method. Please choose from: {}".format(
                       neighbourhood_method, methods.keys()))
            raise KeyError(msg)
