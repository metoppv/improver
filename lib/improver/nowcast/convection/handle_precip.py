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
"""Module for adding impact of precipitation rate into the nowcast lightning
probability."""

import iris
import numpy as np
from improver.utilities.rescale import apply_double_scaling
from improver.utilities.cube_checker import check_cube_coordinates


class ApplyPrecip(object):
    """

    Keyword Args:
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

        problightning_values (dict):
            Lightning probability values to increase first-guess to if
            the lightning_thresholds are exceeded in the nowcast data.
            Dict must have keys 1 and 2 and contain float values.
            The default values are selected to represent lightning risk
            index values of 1 and 2 relating to the key.

    """
    def __init__(self,
                 problightning_values={1: 1., 2: 0.25},
                 probprecip_thresholds=(0.0, 0.05, 0.1),
                 problightning_scaling=(0.0067, 0.2, 1.),
                 hvyprecip_threshs=(0.4, 0.2)):
        """
        Set up class for modifying lightning probability with ice data.
        """
        self.precipthr = probprecip_thresholds
        self.ltngthr = problightning_scaling
        self.phighthresh = hvyprecip_threshs[0]
        self.ptorrthresh = hvyprecip_threshs[1]
        # Prob(lightning) value for Lightning Risk 1 & 2 levels
        self.pl_dict = problightning_values

    def __repr__(self):
        """
        Docstring to describe the repr, which should return a
        printable representation of the object.
        """
        return """<ApplyPrecip:
 precipitation mapping:
   upper:  precip probability {precu} => max lightning prob {lprecu}
   middle: precip probability {precm} => max lightning prob {lprecm}
   lower:  precip probability {precl} => max lightning prob {lprecl}

   heavy:  prob(precip>7mm/hr)  {pphvy} => min lightning prob {lprobl}
   intense:prob(precip>35mm/hr) {ppint} => min lightning prob {lprobu}
>""".format(lprobu=self.pl_dict[1], lprobl=self.pl_dict[2],
            precu=self.precipthr[2], precm=self.precipthr[1],
            precl=self.precipthr[0],
            lprecu=self.ltngthr[2], lprecm=self.ltngthr[1],
            lprecl=self.ltngthr[0],
            pphvy=self.phighthresh, ppint=self.ptorrthresh)

    def process(self, first_guess_cube, precip_cube):
        """
        Modify Nowcast of lightning probability with ice data from radarnet
        composite (VII; Vertically Integrated Ice)

        Args:
            first_guess_cube (iris.cube.Cube):
                First-guess lightning probability.
                This is modified in-place.

            precip_cube (iris.cube.Cube):
                Nowcast precipitation probability
                    (threshold > 0.5, 7., 35.)

        Returns:
            new_cube (iris.cube.Cube):
                Output cube containing Nowcast lightning probability.
                This cube will have the same dimensions and meta-data as
                first_guess_cube.
        """
        first_guess_cube.coord('forecast_period').convert_units('minutes')
        new_cube_list = iris.cube.CubeList([])
        for cube_slice in first_guess_cube.slices_over('time'):
            thistime = cube_slice.coord('time').points
            fcmins = cube_slice.coord('forecast_period').points[0]
            this_precip = precip_cube.extract(iris.Constraint(time=thistime) &
                                              iris.Constraint(threshold=0.5))
            high_precip = precip_cube.extract(iris.Constraint(time=thistime) &
                                              iris.Constraint(threshold=7.))
            torr_precip = precip_cube.extract(iris.Constraint(time=thistime) &
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
            first_guess_cube, new_cube)
        return new_cube
