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

        probprecip_thresholds (tuple):
            Values for limiting prob(lightning) with prob(precip)
            These are the three prob(precip) thresholds and are designed
            to prevent a large probability of lightning being output if
            the probability of precipitation is very low.

        problightning_scaling (tuple):
            Values for limiting prob(lightning) with prob(precip)
            These are the three prob(lightning) values to scale to.

        hvyprecip_threshs (tuple):
            probability thresholds for increasing the prob(lightning)
            First value for heavy precip (>7mm/hr)
                relates to problightning_values[2]
            Second value for intense precip (>35mm/hr)
                relates to problightning_values[1]
        vii_thresholds (tuple):
            Values for increasing prob(lightning) with column-ice data.
            These are the three vertically-integrated ice thresholds in kg/m2.

        vii_scaling (tuple):
            Values for increasing prob(lightning) with column-ice data.
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
                 hvyprecip_threshs=(0.4, 0.2),
                 vii_thresholds=(0.5, 1.0, 2.0),
                 vii_scaling=(0.1, 0.5, 0.9),
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
        self.phighthresh = hvyprecip_threshs[0]
        self.ptorrthresh = hvyprecip_threshs[1]
        self.vii_thresholds = vii_thresholds
        self.vii_scaling = vii_scaling

    def __repr__(self):
        """
        Docstring to describe the repr, which should return a
        printable representation of the object.
        """
        return """
<NowcastLightning: radius={radius}, debug={debug},
 lightning mapping (lightning rate in "min^-1"):
   upper: lightning rate {lthru} => min lightning prob {lprobu}
   lower: lightning rate {lthrl} => min lightning prob {lprobl}
 precipitation mapping:
   upper:  precip probability {precu} => max lightning prob {lprecu}
   middle: precip probability {precm} => max lightning prob {lprecm}
   lower:  precip probability {precl} => max lightning prob {lprecl}

   heavy:  prob(precip>7mm/hr)  {pphvy} => min lightning prob {lprobl}
   intense:prob(precip>35mm/hr) {ppint} => min lightning prob {lprobu}
 VII (ice) mapping:
   upper:  VII {viiu} => max lightning prob {lviiu}
   middle: VII {viim} => max lightning prob {lviim}
   lower:  VII {viil} => max lightning prob {lviil}
>""".format(
            radius=self.radius, debug=self.debug,
            lthru=self.lrt_lev1, lthrl=self.lrt_lev2,
            lprobu=self.pl_dict[1], lprobl=self.pl_dict[2],
            precu=self.precipthr[2], precm=self.precipthr[1],
            precl=self.precipthr[0],
            lprecu=self.ltngthr[2], lprecm=self.ltngthr[1],
            lprecl=self.ltngthr[0],
            pphvy=self.phighthresh, ppint=self.ptorrthresh,
            viiu=self.vii_thresholds[2], viim=self.vii_thresholds[1],
            viil=self.vii_thresholds[0],
            lviiu=self.vii_scaling[2], lviim=self.vii_scaling[1],
            lviil=self.vii_scaling[0])

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
            this_precip = precip_cube.extract(iris.Constraint(time=thistime) &
                                              iris.Constraint(threshold=0.5))
            high_precip = precip_cube.extract(iris.Constraint(time=thistime) &
                                              iris.Constraint(threshold=7.))
            torr_precip = precip_cube.extract(iris.Constraint(time=thistime) &
                                              iris.Constraint(threshold=35.))
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
            cube_slice.data = np.where(
                np.logical_and(this_ltng.data >= lratethresh,
                               cube_slice.data < self.pl_dict[1]),
                self.pl_dict[1], cube_slice.data)

            # Increase prob(lightning) to Risk 2 (pl_dict[2]) when
            #   prob(precip > 7mm/hr) > phighthresh
            cube_slice.data = np.where(
                np.logical_and(high_precip.data >= self.phighthresh,
                               cube_slice.data < self.pl_dict[2]),
                self.pl_dict[2], cube_slice.data)
            # Increase prob(lightning) to Risk 1 (pl_dict[1]) when
            #   prob(precip > 35mm/hr) > ptorrthresh
            cube_slice.data = np.where(
                np.logical_and(torr_precip.data >= self.ptorrthresh,
                               cube_slice.data < self.pl_dict[1]),
                self.pl_dict[1], cube_slice.data)

            # Decrease prob(lightning) where prob(precip > 0) is low.
            cube_slice.data = self._apply_double_scaling(
                this_precip, cube_slice, self.precipthr, self.ltngthr)

            # If we have VII data, increase prob(lightning) accordingly.
            if vii_cube:
                for threshold, prob_max in zip(self.vii_thresholds,
                                               self.vii_scaling):
                    vii_slice = vii_cube.extract(
                        iris.Constraint(threshold=threshold))
                    vii_scaling = [0., (prob_max * (1. - (fcmins / 150.)))]
                    cube_slice.data = np.maximum(
                        rescale(vii_slice.data,
                                data_range=(0., 1.),
                                scale_range=vii_scaling,
                                clip=True),
                        cube_slice.data)

            new_cube_list.append(cube_slice)

        merged_cube = new_cube_list.merge_cube()
        merged_cube = check_cube_coordinates(
            cube, merged_cube)
        return merged_cube

    def _apply_double_scaling(self, data_cube, scaled_cube,
                              data_vals, scaling_vals,
                              combine_function=np.minimum):
        """
        Update scaled_cube based on the contents of data_cube so that
        scaled_cube is at least the value of the data_cube after rescaling
        based on an upper, mid and lower threshold.

        Args:
            data_cube (iris.cube.Cube):
                Data with which to modify scaled_cube.data
            scaled_cube (iris.cube.Cube):
                Input cube to modify
            data_vals (tuple of three values):
                Lower, mid and upper points to rescale data_cube from
            scaling_vals (tuple of three values):
                Lower, mid and upper points to rescale data_cube to

        Returns:
            data (numpy.array):
                Output data from scaled_cube after modification.
                This array will have the same dimensions as scaled_cube.
        """
        local_limit = np.where(
            data_cube.data < data_vals[1],
            rescale(data_cube.data,
                    data_range=(data_vals[0], data_vals[1]),
                    scale_range=(scaling_vals[0], scaling_vals[1]),
                    clip=True),
            rescale(data_cube.data,
                    data_range=(data_vals[1], data_vals[2]),
                    scale_range=(scaling_vals[1], scaling_vals[2]),
                    clip=True))
        # Ensure prob(lightning) is no larger than the local upper-limit:
        return combine_function(scaled_cube.data, local_limit)

    def process(self, cubelist):
        """
        Produce Nowcast of lightning probability

        Args:
            cubelist (iris.cube.CubeList):
                Contains cubes of
                    * First-guess lightning probability
                    * Nowcast precipitation probability (threshold > 0.5, 7., 35.)
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
        precip_cube = precip_cube.extract(iris.Constraint(threshold=0.5))
        new_cube = self._update_meta(precip_cube)
        new_cube = self._modify_first_guess(
            new_cube, fg_cube, ltng_cube, precip_cube, vii_cube)
        new_cube = self._process_haloes(new_cube)
        return new_cube
