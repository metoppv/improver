#!/usr/bin/env python2.7
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
"""Module for NowcastLightning class and associated functions."""
import numpy as np
import iris
from improver.nbhood.nbhood import NeighbourhoodProcessing
from improver.utilities.cube_checker import check_cube_coordinates
from improver.utilities.rescale import rescale


class NowcastLightning(object):
    """Produce Nowcast of lightning probability.

    This Plugin selects a first-guess lightning probability field from
    MOGREPS-UK data matching the nowcast validity-time and modifies this
    based on information from the nowcast on:
     * prob(precipitation): no rain ==> no lightning
     * lightning rate from ATDNet: recent activity ==> increased prob(lightning)

    Keyword Args:
        radius (float):
            This value controls the halo radius (metres)
            The value supplied applies at T+0
            and increases to 2*radius at T+6 hours
            The radius is applied using the circular neighbourhood plugin.

        lightning_thresholds (tuple):
            Lightning rate thresholds for adjusting the first-guess
            lightning probability.
            First element must be a function that takes one argument and
            returns a float of the lightning rate threshold for increasing
            first-guess lightning probability to risk 1 when given an int/float
            forecast-lead-time in minutes.
            Second element must be a float for the lightning rate threshold
            for increasing first-guess lightning probability to risk 2.
            Default value is (lambda mins: 0.5 + mins * 2. / 360., 0.)

        problightning_values (dict):
            Lightning probability values to increase first-guess to if
            the lightning_thresholds are exceeded in the nowcast data.
            Dict must have keys 1 and 2 and contain float values.

        probprecip_thresholds (tuple):
            Values for limiting prob(lightning) with prob(precip)
            These are the three prob(precip) thresholds

        problightning_scaling (tuple):
            Values for limiting prob(lightning) with prob(precip)
            These are the three prob(lightning) values to scale to.

        debug (boolean):
            True results in verbose output for debugging purposes.
    """
    def __init__(self, radius=10000.,
                 lightning_thresholds=(
                     lambda mins: 0.5 + mins * 2. / 360., 0.),
                 problightning_values={1: 1., 2: 0.25},
                 probprecip_thresholds=(0.0, 0.05, 0.1),
                 problightning_scaling=(0.0067, 0.2, 1.),
                 debug=False):
        """
        Set up class for Nowcast of lightning probability.
        """
        self.debug = debug
        self.radius = radius
        lead_times = [0., 6.]
        radii = [self.radius, 2*self.radius]
        self.neighbourhood = NeighbourhoodProcessing(
            'circular', radii, lead_times=lead_times)

        # Lightning-rate threshold for Lightning Risk 2 level
        # Lightning-rate threshold for Lightning Risk 1 level
        # (dependent on forecast-length)
        self.lrt_lev1, self.lrt_lev2 = lightning_thresholds
        # Prob(lightning) value for Lightning Risk 1 & 2 levels
        self.pl_dict = problightning_values

        self.precipthr = probprecip_thresholds
        self.ltngthr = problightning_scaling

    def __repr__(self):
        """
        Docstring to describe the repr, which should return a
        printable representation of the object.
        """
        return "<NowcastLightning: radius={radius}, debug={debug}>".format(
            radius=self.radius, debug=self.debug)

    def _process_haloes(self, cube):
        """
        Adjust data so that lightning probability does not decrease too rapidly
        with distance.

        Args:
            cube (iris.cube.Cube):
                Radius will be applied equally on x and y dimensions.

        Returns:
            new_cube (iris.cube.Cube):
                Output cube of same shape as cube with haloes applied.
        """
        new_cube = self.neighbourhood.process(cube.copy())
        return new_cube

    def _update_meta(self, cube):
        """
        Modify the meta data of input cube to resemble a Nowcast of lightning
        probability

        Args:
            cube (iris.cube.Cube):
                An input cube

        Returns:
            new_cube (iris.cube.Cube):
                Output cube - a copy of input cube with meta-data relating to
                a Nowcast of lightning probability.
                The data array will be a copy of the input cube.data
        """
        new_cube = cube.copy()
        new_cube.rename("lightning_probability")
        new_cube.attributes = {}
        if self.debug:
            print('In {}, new_cube is {}'.format(self, new_cube))
        return new_cube

    def _modify_first_guess(self, cube, fg_cube, ltng_cube, precip_cube):
        """
        Modify first-guess lightning probability with nowcast data

        Args:
            cube (iris.cube.Cube):
                Provides the meta-data for the Nowcast lightning probability
                output cube

            fg_cube (iris.cube.Cube):
                First-guess lightning probability
                Must have same x & y dimensions as cube
                Time dimension should overlap that of cube

            ltng_cube (iris.cube.Cube):
                Nowcast lightning rate
                Must have same dimensions as cube

            precip_cube (iris.cube.Cube):
                Nowcast precipitation probability (threshold > 0)
                Must have same dimensions as cube

        Returns:
            new_cube (iris.cube.Cube):
                Output cube containing Nowcast lightning probability.
        """
        new_cube_list = iris.cube.CubeList([])
        for cube_slice in cube.slices_over('time'):
            thistime = cube_slice.coord('time').points
            this_ltng = ltng_cube.extract(iris.Constraint(time=thistime))
            this_precip = precip_cube.extract(iris.Constraint(time=thistime))
            fg_time = fg_cube.coord('time').points[
                fg_cube.coord('time').nearest_neighbour_index(thistime)]
            this_fg = fg_cube.extract(iris.Constraint(time=fg_time))
            err_string = "No matching {} cube for {}"
            assert isinstance(this_ltng,
                              iris.cube.Cube), err_string.format("lightning",
                                                                 thistime)
            assert isinstance(this_precip,
                              iris.cube.Cube), err_string.format("precip",
                                                                 thistime)
            assert isinstance(cube_slice,
                              iris.cube.Cube), err_string.format("output",
                                                                 thistime)
            assert isinstance(this_fg,
                              iris.cube.Cube), err_string.format("first-guess",
                                                                 thistime)
            cube_slice.data = this_fg.data
            cube_slice.coord('forecast_period').convert_units('minutes')
            fcmins = cube_slice.coord('forecast_period').points[0]

            # Increase prob(lightning) to Risk 2 (pl_dict[2]) when within
            #   lightning halo (lrt_lev2; 50km of an observed ATDNet strike):
            cube_slice.data = np.where(
                np.logical_and(this_ltng.data >= self.lrt_lev2,
                               cube_slice.data < self.pl_dict[2]),
                self.pl_dict[2], cube_slice.data)
            lratethresh = self.lrt_lev1(fcmins)
            if self.debug:
                print 'LRate threshold is {} strikes per minute'.format(
                    lratethresh)

            # Increase prob(lightning) to Risk 1 (pl_dict[1]) when within
            #   lightning storm (lrt_lev1; ~5km of an observed ATDNet strike):
            cube_slice.data = np.where(this_ltng.data >= lratethresh,
                                       self.pl_dict[1],
                                       cube_slice.data)

            # Set up an array of prob(lightning) upper-limits based on
            # prob(precip) by rescaling the prob(precip) array based on an
            # upper, mid and lower threshold.
            # precipthr supplies the prob(precip) points
            # ltngthr supplies the equivalent prob(lightning) points.
            preciplimit = np.where(
                this_precip.data < self.precipthr[1],
                rescale(this_precip.data,
                        data_range=(self.precipthr[0], self.precipthr[1]),
                        scale_range=(self.ltngthr[0], self.ltngthr[1]),
                        clip=True),
                rescale(this_precip.data,
                        data_range=(self.precipthr[1], self.precipthr[2]),
                        scale_range=(self.ltngthr[1], self.ltngthr[2]),
                        clip=True))
            # Ensure prob(lightning) is no larger than the local upper-limit:
            cube_slice.data = np.minimum(cube_slice.data, preciplimit)

            new_cube_list.append(cube_slice)

        merged_cube = new_cube_list.merge_cube()
        merged_cube = check_cube_coordinates(
            cube, merged_cube)
        return merged_cube

    def process(self, cubelist):
        """
        Produce Nowcast of lightning probability

        Args:
            cubelist (iris.cube.CubeList):
                Contains cubes of
                    * First-guess lightning probability
                    * Nowcast precipitation probability (threshold > 0)
                    * Nowcast lightning rate

        Returns:
            new_cube (iris.cube.Cube):
                Output cube containing Nowcast lightning probability.
                This cube will have the same dimensions as the input
                Nowcast precipitation probability.
        """
        fg_cube, = cubelist.extract("probability_of_lightning")
        ltng_cube, = cubelist.extract("rate_of_lightning")
        precip_cube, = cubelist.extract("probability_of_precipitation")
        precip_cube = precip_cube.extract(iris.Constraint(threshold=0.5))
        new_cube = self._update_meta(precip_cube)
        new_cube = self._modify_first_guess(
            new_cube, fg_cube, ltng_cube, precip_cube)
        new_cube = self._process_haloes(new_cube)
        return new_cube
