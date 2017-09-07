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
import numpy as np
from scipy.ndimage import correlate
from math import ceil
from improver.nbhood.circular_kernel import CircularNeighbourhood
import iris

class NowcastLightning(object):
    """Produce Nowcast of lightning probability.

    Need something meaningful here.
    """
    def __init__(self, radius=10000., debug=False):
        """Set up class for Nowcast of lightning probability.

        Args:
            radius : float
                This value controls the halo radius (metres)

            debug : boolean
                True results in verbose output for debugging purposes.
        """
        self.debug = debug
        self.radius = radius
        self.Neighbourhood = CircularNeighbourhood()

    def __repr__(self):
        """
        Docstring to describe the repr, which should return a
        printable representation of the object.
        """
        return "<NowcastLightning: radius={radius}, debug={debug}>".format(radius=self.radius,
                                                                           debug=self.debug)

    def _process_haloes(self, cube):
        """
        Adjust data so that lightning probability does not decrease too rapidly with distance

        Args:
            cube : iris.cube.Cube
                Radius will be applied equally on all dimensions.

        Returns:
            new_data : Numpy array of same shape as data
                Output data with haloes applied
        """
        new_data = self.Neighbourhood.run(cube.copy(), self.radius)
        return new_data

    def _update_meta(self, cube):
        """
        Modify the meta data of input cube to resemble a Nowcast of lightning probability

        Args:
            cube : iris.cube.Cube
                An input cube

        Returns:
            new_cube : iris.cube.Cube
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
            cube : iris.cube.Cube
                Provides the meta-data for the Nowcast lightning probability output cube

            ltng_cube : iris.cube.Cube
                Nowcast lightning rate
                Must have same dimensions as cube

            precip_cube : iris.cube.Cube
                Nowcast precipitation probability (threshold > 0)
                Must have same dimensions as cube

            fg_cube : iris.cube.Cube
                First-guess lightning probability
                Must have same x & y dimensions as cube
                Time dimension should overlap that of cube

        Returns:
            new_cube : iris.cube.Cube
                Output cube containing Nowcast lightning probability.
        """
        new_cube = cube.copy()
        timeunit = cube.coord('time').units
        for thistime in cube.coord('time').points:
            this_ltng = ltng_cube.extract(iris.Constraint(time=thistime))
            this_precip = precip_cube.extract(iris.Constraint(time=thistime))
            this_out = new_cube.extract(iris.Constraint(time=thistime))
            fg_time = fg_cube.coord('time').points[fg_cube.coord('time').nearest_neighbour_index(thistime)]
            this_fg = fg_cube.extract(iris.Constraint(time=fg_time))
            print fg_cube.coord('time').nearest_neighbour_index(thistime)
            assert isinstance(this_ltng, iris.cube.Cube), "No matching lightning cube for {}".format(thistime)
            assert isinstance(this_precip, iris.cube.Cube), "No matching precip cube for {}".format(thistime)
            assert isinstance(this_out, iris.cube.Cube), "No matching output cube for {}".format(thistime)
            assert isinstance(this_fg, iris.cube.Cube), "No matching first-guess cube for {}".format(thistime)
            this_out.data = this_fg.data
            this_out.coord('forecast_period').convert_units('minutes')
            fcmins = this_out.coord('forecast_period').points[0]

            lratethresh = 0.
            # Increase to LR2 prob when within lightning halo:
            this_out.data = np.where(np.logical_and(this_ltng.data >= lratethresh,
                                                    this_out.data < 0.25), 0.25, this_out.data)
            lratethresh = 0.5 + fcmins * 2. / 360.
            if self.debug:
                print 'LRate threshold is {} strikes per minute'.format(lratethresh)
            # Increase to LR1 when within thunderstorm:
            this_out.data = np.where(this_ltng.data >= lratethresh, 1., this_out.data)
            preciplimit = np.where(this_precip.data < 0.05,
                                   rescale(this_precip.data,
                                           datamin=0.00, datamax=0.05,
                                           scalemax=0.2, scalemin=0.0067,
                                           clip=True, debug=self.debug),
                                   rescale(this_precip.data,
                                           datamin=0.05, datamax=0.10,
                                           scalemax=1.0, scalemin=0.2000,
                                           clip=True, debug=self.debug))
            # Reduce to LR2 prob when Prob(rain > 0) is low and LR3 when very low:
            this_out.data = np.minimum(this_out.data, preciplimit)
        return new_cube

    def process(self, cubelist):
        """
        Produce Nowcast of lightning probability

        Args:
            cubelist : iris.cube.CubeList
                Contains cubes of
                    First-guess lightning probability
                    Nowcast precipitation probability (threshold > 0)
                    Nowcast lightning rate

        Returns:
            new_cube : iris.cube.Cube
                Output cube containing Nowcast lightning probability.
                This cube will have the same dimensions as the input Nowcast precipitation probability.
        """
        precip_cube = cubelist.extract(long_name="precipitation_rate_probability", threshold="0.")[0]
        fg_cube = cubelist.extract(long_name="lightning_probability")[0]
        ltng_cube = cubelist.extract(long_name="lightning_rate")[0]
        new_cube = _update_meta(precip_cube)
        new_cube = _modify_first_guess(new_cube, fg_cube, ltng_cube, precip_cube)
        new_cube = _process_haloes(new_cube)
        return new_cube


def rescale(data, datamin=None, datamax=None, scalemin=0., scalemax=1., clip=False, debug=False):
    """Rescale data array so that datamin => scalemin and datamax => scale max.
       All adjustments are linear"""
    datamin = np.min(data) if datamin is None else datamin
    datamax = np.max(data) if datamax is None else datamax
    if debug:
        print "Rescaling data so that {} -> {} and {} -> {}".format(datamin,
                                                                    scalemin,
                                                                    datamax,
                                                                    scalemax)
    result = ((data - datamin) * (scalemax - scalemin) / (datamax - datamin)) + scalemin
    if clip:
        result = np.clip(result, scalemin, scalemax)
    return result
