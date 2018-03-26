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
from improver.nowcast.convection.handle_vii import ApplyIce
from improver.nowcast.convection.handle_precip import ApplyPrecip


class NowcastLightning(object):
    """Produce Nowcast of lightning probability.

    This Plugin selects a first-guess lightning probability field from
    MOGREPS-UK data matching the nowcast validity-time and modifies this
    based on information from the nowcast. The default behaviour makes
    these adjustments:
    lightning mapping (lightning rate in "min^-1"):
        upper: lightning rate <function> => min lightning prob 1.0
            The <function> returns a linear value from 0.5 to 2.5
            over the 6-hour forecast_period.
        lower: lightning rate 0.0 => min lightning prob 0.25
            Zero is a special value indicating that lightning is
            present within 50km.
    precipitation mapping:
        upper:  precip probability 0.1 => max lightning prob 1.0
        middle: precip probability 0.05 => max lightning prob 0.2
        lower:  precip probability 0.0 => max lightning prob 0.0067

        heavy:  prob(precip>7mm/hr)  0.4 => min lightning prob 0.25
                equiv radar refl 37dBZ
        intense:prob(precip>35mm/hr) 0.2 => min lightning prob 1.0
                equiv radar refl 48dBZ

    Keyword Args:
        radius (float):
            This value controls the halo radius (metres)
            The value supplied applies at T+0
            and increases to 2*radius at T+6 hours
            The radius is applied using the circular neighbourhood plugin.

        lightning_thresholds (tuple):
            Lightning rate thresholds for adjusting the first-guess
            lightning probability (strikes per minute == "min^-1").
            First element must be a function that takes one argument and
            returns a float of the lightning rate threshold for increasing
            first-guess lightning probability to risk 1 when given an int/float
            forecast-lead-time in minutes.
            Second element must be a float for the lightning rate threshold
            for increasing first-guess lightning probability to risk 2.
            There are two special values in the lightning rate field:
                0: No lightning at point, but lightning present within 50km.
                -1: No lightning at point or within 50km halo.
            Default value is (lambda mins: 0.5 + mins * 2. / 360., 0.)
            This gives a decreasing influence on the extrapolated lightning
            nowcast over forecast_period while retaining an influence from the
            50km halo..

        problightning_values (dict):
            Lightning probability values to increase first-guess to if
            the lightning_thresholds are exceeded in the nowcast data.
            Dict must have keys 1 and 2 and contain float values.
            The default values are selected to represent lightning risk
            index values of 1 and 2 relating to the key.

        precip_method = (improver.nowcast.lightning.
                         handle_precip.ApplyPrecip()):
            Initiated plugin with a process method that takes two
            iris.cube.Cube arguments of lightning probability and precipitation
            probability and returns an updated lightning probability cube.
            Default value of None causes a plugin to be initiated using the
            problightning_values available in this plugin.

        ice_method = (improver.nowcast.lightning.handle_vii.ApplyIce()):
            Initiated plugin with a process method that takes two
            iris.cube.Cube arguments of lightning probability and vertically
            integrated ice and returns an updated lightning probability cube.

        debug (boolean):
            True results in verbose output for debugging purposes.
    """
    def __init__(self, radius=10000.,
                 lightning_thresholds=(
                     lambda mins: 0.5 + mins * 2. / 360., 0.),
                 problightning_values={1: 1., 2: 0.25},
                 precip_method=None,
                 ice_method=ApplyIce(),
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

        if precip_method is None:
            self.precip_plugin = ApplyPrecip(problightning_values=self.pl_dict)
        else:
            self.precip_plugin = precip_method
        self.ice_plugin = ice_method

    def __repr__(self):
        """
        Docstring to describe the repr, which should return a
        printable representation of the object.
        """
        return """<NowcastLightning: radius={radius}, debug={debug},
 lightning mapping (lightning rate in "min^-1"):
   upper: lightning rate {lthru} => min lightning prob {lprobu}
   lower: lightning rate {lthrl} => min lightning prob {lprobl}
With:
{precip}
{ice}
>""".format(radius=self.radius, debug=self.debug,
            lthru=self.lrt_lev1, lthrl=self.lrt_lev2,
            lprobu=self.pl_dict[1], lprobl=self.pl_dict[2],
            precip=self.precip_plugin,
            ice=self.ice_plugin)

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
        new_cube.remove_coord('threshold')
        new_cube.attributes = {}
        if self.debug:
            print('In {}, new_cube is {}'.format(self, new_cube))
        return new_cube

    def _modify_first_guess(self, cube, fg_cube, ltng_cube, precip_cube,
                            vii_cube):
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
                Nowcast precipitation probability (threshold > 0.5, 7, 35)
                Must have same other dimensions as cube

            vii_cube (iris.cube.Cube):
                Radar-derived vertically integrated ice content (VII)
                Must have same x and y dimensions as cube
                Time should be a scalar coordinate
                Must have a threshold coordinate with points matching
                self.vii_thresholds
                Can be <No cube> or None or anything that evaluates to False

        Returns:
            new_cube (iris.cube.Cube):
                Output cube containing Nowcast lightning probability.
        """
        new_cube_list = iris.cube.CubeList([])
        for cube_slice in cube.slices_over('time'):
            thistime = cube_slice.coord('time').points
            this_ltng = ltng_cube.extract(iris.Constraint(time=thistime))
            fg_time = fg_cube.coord('time').points[
                fg_cube.coord('time').nearest_neighbour_index(thistime)]
            this_fg = fg_cube.extract(iris.Constraint(time=fg_time))
            err_string = "No matching {} cube for {}"
            assert isinstance(this_ltng,
                              iris.cube.Cube), err_string.format("lightning",
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
            cube_slice.data = np.where(
                np.logical_and(this_ltng.data >= lratethresh,
                               cube_slice.data < self.pl_dict[1]),
                self.pl_dict[1], cube_slice.data)

            new_cube_list.append(cube_slice)

        merged_cube = new_cube_list.merge_cube()
        merged_cube = check_cube_coordinates(
            cube, merged_cube)

        # Apply precipitation adjustments.
        merged_cube = self.precip_plugin.process(merged_cube, precip_cube)

        # If we have VII data, increase prob(lightning) accordingly.
        if vii_cube:
            merged_cube = self.ice_plugin.process(merged_cube, vii_cube)
        return merged_cube

    def process(self, cubelist):
        """
        Produce Nowcast of lightning probability

        Args:
            cubelist (iris.cube.CubeList):
                Contains cubes of
                    * First-guess lightning probability
                    * Nowcast precipitation probability
                        (threshold > 0.5, 7., 35.)
                    * Nowcast lightning rate
                    * (optional) Analysis of vertically integrated ice (VII)
                      from radar

        Returns:
            new_cube (iris.cube.Cube):
                Output cube containing Nowcast lightning probability.
                This cube will have the same dimensions as the input
                Nowcast precipitation probability.
        """
        fg_cube = cubelist.extract("probability_of_lightning").merge_cube()
        ltng_cube = cubelist.extract("rate_of_lightning").merge_cube()
        ltng_cube.convert_units("min^-1")  # Ensure units are correct.
        precip_cube = cubelist.extract("probability_of_precipitation")
        precip_cube = precip_cube.merge_cube()
        vii_cube = cubelist.extract("probability_of_vertical_integral_of_ice")
        if vii_cube:
            vii_cube = vii_cube.merge_cube()
        precip_slice = precip_cube.extract(iris.Constraint(threshold=0.5))
        new_cube = self._update_meta(precip_slice)
        new_cube = self._modify_first_guess(
            new_cube, fg_cube, ltng_cube, precip_cube, vii_cube)
        new_cube = self._process_haloes(new_cube)
        return new_cube
