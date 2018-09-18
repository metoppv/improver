# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
from improver.utilities.temporal import iris_time_to_datetime
from improver.utilities.rescale import apply_double_scaling, rescale


class NowcastLightning(object):
    """Produce Nowcast of lightning probability.

    This Plugin selects a first-guess lightning probability field from
    MOGREPS-UK data matching the nowcast validity-time, and modifies this
    based on information from the nowcast. The default behaviour makes
    these adjustments to the upper and lower limits of lightning probability:

    lightning mapping (lightning rate in "min^-1"):
        upper: lightning rate >= <function> => lightning prob = 1.0 (LR1)
            The <function> returns a linear value from 0.5 to 2.5 over a
            6-hour forecast_period.
        lower: lightning rate == 0.0 => min lightning prob 0.25

        There are two special values in the lightning rate field:
            0: No lightning at point, but lightning present within 50 km
            -1: No lightning at point or within 50 km

    precipitation mapping (for prob(precip > 0.5 mm/hr)):
        upper:  precip probability >= 0.1 => max lightning prob 1.0 (LR1)
        middle: precip probability >= 0.05 => max lightning prob 0.2
        lower:  precip probability >= 0.0 => max lightning prob 0.0067

        heavy:  prob(precip > 7mm/hr) >= 0.4 => min lightning prob 0.25
                equiv radar refl 37dBZ
        intense:prob(precip > 35mm/hr) >= 0.2 => min lightning prob 1.0
                equiv radar refl 48dBZ

    """
    def __init__(self, radius=10000.,
                 lightning_thresholds=(
                     lambda mins: 0.5 + mins * 2. / 360., 0.),
                 problightning_values={1: 1., 2: 0.25},
                 debug=False):
        """
        Initialise class for Nowcast of lightning probability.

        Keyword Args:
            radius (float):
                Radius (metres) over which to neighbourhood process the output
                lightning probability.  The value supplied applies at T+0
                and increases to 2*radius at T+6 hours.  The radius is applied
                in "process" using the circular neighbourhood plugin.

            lightning_thresholds (tuple):
                Lightning rate thresholds for adjusting the first-guess
                lightning probability (strikes per minute == "min^-1").
                First element must be a function that takes "forecast_period"
                in minutes and returns the lightning rate threshold for
                increasing first-guess lightning probability to risk 1 (LR1).
                Default value is (lambda mins: 0.5 + mins * 2. / 360., 0.).
                This gives a decreasing influence on the extrapolated lightning
                nowcast over forecast_period while retaining an influence from
                the 50 km halo.
                Second element is the lightning rate threshold (as float) for
                increasing first-guess lightning probability to risk 2 (LR2).

            problightning_values (dict):
                Lightning probability values to increase first-guess to if
                the lightning_thresholds are exceeded in the nowcast data.
                Dict must have keys 1 and 2 and contain float values.
                The default values are selected to represent lightning risk
                index values of 1 and 2 relating to the key.

            debug (boolean):
                True results in verbose output for debugging purposes.

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

        # Set values for handling precipitation rate data
        #    precipthr (tuple):
        #        Values for limiting prob(lightning) with prob(precip).
        #        These are the three prob(precip) thresholds and are designed
        #        to prevent a large probability of lightning being output if
        #        the probability of precipitation is very low.
        self.precipthr = (0.0, 0.05, 0.1)
        #    ltngthr (tuple):
        #        Values for limiting prob(lightning) with prob(precip)
        #        These are the three prob(lightning) values to scale to.
        self.ltngthr = (0.0067, 0.2, 1.)
        #    probability thresholds for increasing the prob(lightning)
        #        phighthresh for heavy precip (>7mm/hr)
        #            relates to problightning_values[2]
        #        ptorrthresh for intense precip (>35mm/hr)
        #            relates to problightning_values[1]
        self.phighthresh = 0.4
        self.ptorrthresh = 0.2
        #    pl_dict (dict):
        #        Lightning probability values to increase first-guess to if
        #        the lightning_thresholds are exceeded in the nowcast data.
        #        Dict must have keys 1 and 2 and contain float values.
        #        The default values are selected to represent lightning risk
        #        index values of 1 and 2 relating to the key.
        self.pl_dict = {1: 1., 2: 0.25}

        # Set values for handling vertically-integrated-ice (VII) data
        #    ice_thresholds (tuple):
        #        Values for increasing prob(lightning) with column-ice data.
        #        These are the three VII thresholds in kg/m2.
        self.ice_thresholds = (0.5, 1.0, 2.0)
        #    ice_scaling (tuple):
        #        Values for increasing prob(lightning) with VII data.
        #        These are the three prob(lightning) values to scale to.
        self.ice_scaling = (0.1, 0.5, 0.9)

    def __repr__(self):
        """
        Docstring to describe the repr, which should return a
        printable representation of the object.
        """
        return """<NowcastLightning: radius={radius}, debug={debug},
 lightning mapping (lightning rate in "min^-1"):
   upper: lightning rate {lthru} => min lightning prob {lprobu}
   lower: lightning rate {lthrl} => min lightning prob {lprobl}
>""".format(radius=self.radius, debug=self.debug,
            lthru=self.lrt_lev1, lthrl=self.lrt_lev2,
            lprobu=self.pl_dict[1], lprobl=self.pl_dict[2])

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
            print(('In {}, new_cube is {}'.format(self, new_cube)))
        return new_cube

    def _modify_first_guess(self, cube, fg_cube, ltng_cube, prob_precip_cube,
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

            prob_precip_cube (iris.cube.Cube):
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
            thispoint = cube_slice.coord('time').points[0]
            thistime = iris_time_to_datetime(
                cube_slice.coord('time').copy())[0]
            this_ltng = ltng_cube.extract(iris.Constraint(time=thistime))
            fg_time = iris_time_to_datetime(
                fg_cube.coord('time').copy())[
                fg_cube.coord('time').nearest_neighbour_index(thispoint)]
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
                print('LRate threshold is {} strikes per minute'.format(
                    lratethresh))

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
        merged_cube = self.apply_precip(merged_cube, prob_precip_cube)

        # If we have VII data, increase prob(lightning) accordingly.
        if vii_cube:
            merged_cube = self.apply_ice(merged_cube, vii_cube)
        return merged_cube

    def apply_precip(self, prob_lightning_cube, prob_precip_cube):
        """
        Modify Nowcast of lightning probability with precipitation rate
        probabilities at thresholds of 0.5, 7 and 35 mm/h.

        Args:
            prob_lightning_cube (iris.cube.Cube):
                First-guess lightning probability.
                This is modified in-place.

            prob_precip_cube (iris.cube.Cube):
                Nowcast precipitation probability
                    (threshold > 0.5, 7., 35.)

        Returns:
            new_cube (iris.cube.Cube):
                Output cube containing updated nowcast lightning probability.
                This cube will have the same dimensions and meta-data as
                prob_lightning_cube.
        """
        prob_lightning_cube.coord('forecast_period').convert_units('minutes')
        new_cube_list = iris.cube.CubeList([])
        # check prob-precip threshold units are as expected
        prob_precip_cube.coord('threshold').convert_units('mm hr-1')
        # extract precipitation probabilities at required thresholds
        for cube_slice in prob_lightning_cube.slices_over('time'):
            thistime = iris_time_to_datetime(
                cube_slice.coord('time').copy())[0]
            this_precip = prob_precip_cube.extract(
                iris.Constraint(time=thistime) &
                iris.Constraint(threshold=0.5))
            high_precip = prob_precip_cube.extract(
                iris.Constraint(time=thistime) &
                iris.Constraint(threshold=7.))
            torr_precip = prob_precip_cube.extract(
                iris.Constraint(time=thistime) &
                iris.Constraint(threshold=35.))
            err_string = "No matching {} cube for {}"
            assert isinstance(this_precip,
                              iris.cube.Cube), err_string.format("any precip",
                                                                 thistime)
            assert isinstance(high_precip,
                              iris.cube.Cube), err_string.format("high precip",
                                                                 thistime)
            assert isinstance(torr_precip,
                              iris.cube.Cube), err_string.format(
                                  "intense precip",
                                  thistime)
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
            cube_slice.data = apply_double_scaling(
                this_precip, cube_slice, self.precipthr, self.ltngthr)

            new_cube_list.append(cube_slice)

        new_cube = new_cube_list.merge_cube()
        new_cube = check_cube_coordinates(
            prob_lightning_cube, new_cube)
        return new_cube

    def apply_ice(self, prob_lightning_cube, ice_cube):
        """
        Modify Nowcast of lightning probability with ice data from radarnet
        composite (VII; Vertically Integrated Ice)

        Args:
            prob_lightning_cube (iris.cube.Cube):
                First-guess lightning probability.
                This is modified in-place.
            ice_cube (iris.cube.Cube):
                Analysis of vertically integrated ice (VII) from radar

        Returns:
            new_cube (iris.cube.Cube):
                Output cube containing updated nowcast lightning probability.
                This cube will have the same dimensions and meta-data as
                prob_lightning_cube.
        """
        prob_lightning_cube.coord('forecast_period').convert_units('minutes')
        # check prob-ice threshold units are as expected
        ice_cube.coord('threshold').convert_units('kg m^-2')
        new_cube_list = iris.cube.CubeList([])
        for cube_slice in prob_lightning_cube.slices_over('time'):
            fcmins = cube_slice.coord('forecast_period').points[0]
            for threshold, prob_max in zip(self.ice_thresholds,
                                           self.ice_scaling):
                ice_slice = ice_cube.extract(
                    iris.Constraint(threshold=threshold))
                ice_scaling = [0., (prob_max * (1. - (fcmins / 150.)))]
                cube_slice.data = np.maximum(
                    rescale(ice_slice.data,
                            data_range=(0., 1.),
                            scale_range=ice_scaling,
                            clip=True),
                    cube_slice.data)
            new_cube_list.append(cube_slice)

        new_cube = new_cube_list.merge_cube()
        new_cube = check_cube_coordinates(
            prob_lightning_cube, new_cube)
        return new_cube

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
        prob_precip_cube = cubelist.extract("probability_of_precipitation")
        prob_precip_cube = prob_precip_cube.merge_cube()
        vii_cube = cubelist.extract("probability_of_vertical_integral_of_ice")
        if vii_cube:
            vii_cube = vii_cube.merge_cube()
        precip_slice = prob_precip_cube.extract(iris.Constraint(threshold=0.5))
        new_cube = self._update_meta(precip_slice)
        new_cube = self._modify_first_guess(
            new_cube, fg_cube, ltng_cube, prob_precip_cube, vii_cube)
        new_cube = self._process_haloes(new_cube)
        return new_cube
