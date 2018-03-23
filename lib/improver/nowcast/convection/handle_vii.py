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
"""Module for adding impact of Vertically Integrated Ice from the radar
composite into the nowcast lightning probability."""

import iris
import numpy as np
from improver.utilities.rescale import rescale
from improver.utilities.cube_checker import check_cube_coordinates

class ApplyIce(object):
    """

    Keyword Args:
        ice_thresholds (tuple):
            Values for increasing prob(lightning) with column-ice data.
            These are the three vertically-integrated ice thresholds in kg/m2.

        ice_scaling (tuple):
            Values for increasing prob(lightning) with column-ice data.
            These are the three prob(lightning) values to scale to.
    """
    def __init__(self,
                 ice_thresholds=(0.5, 1.0, 2.0),
                 ice_scaling=(0.1, 0.5, 0.9)):
        """
        Set up class for modifying lightning probability with ice data.
        """
        self.ice_thresholds = ice_thresholds
        self.ice_scaling = ice_scaling

    def __repr__(self):
        """
        Docstring to describe the repr, which should return a
        printable representation of the object.
        """
        return """<ApplyIce:
 VII (ice) mapping (kg/m2):
   upper:  VII {viiu} => max lightning prob {lviiu}
   middle: VII {viim} => max lightning prob {lviim}
   lower:  VII {viil} => max lightning prob {lviil}
>""".format(viiu=self.ice_thresholds[2], viim=self.ice_thresholds[1],
            viil=self.ice_thresholds[0],
            lviiu=self.ice_scaling[2], lviim=self.ice_scaling[1],
            lviil=self.ice_scaling[0])

    def process(self, first_guess_cube, ice_cube):
        """
        Modify Nowcast of lightning probability with ice data from radarnet
        composite (VII; Vertically Integrated Ice)

        Args:
            first_guess_cube (iris.cube.Cube):
                First-guess lightning probability.
                This is modified in-place.
            ice_cube (iris.cube.Cube):
                Analysis of vertically integrated ice (VII) from radar

        Returns:
            new_cube (iris.cube.Cube):
                Output cube containing Nowcast lightning probability.
                This cube will have the same dimensions and meta-data as
                first_guess_cube.
        """
        first_guess_cube.coord('forecast_period').convert_units('minutes')
        new_cube_list = iris.cube.CubeList([])
        for cube_slice in first_guess_cube.slices_over('time'):
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
            first_guess_cube, new_cube)
        return new_cube
