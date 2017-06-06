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
"""Module to create the weights used to Blend data."""


import numpy as np
import iris


def normalise_weights(weights):
    """Ensures all weights add up to one.

        Args:
            weights : array of weights.

        Returns:
            normalised_weights : array of weights
                                 where sum = 1.0
    """
    if weights.min() < 0.0:
        msg = 'Weights must be positive, at least one value < 0.0'
        raise ValueError(msg)

    sumval = weights.sum()
    if sumval == 0:
        msg = 'Sum of weights must be > 0.0'
        raise ValueError(msg)

    normalised_weights = weights / sumval
    return normalised_weights


def nonlinear_weights(num_of_weights, cval):
    """Create nonlinear weights.

        Args:
            num_of_weights : Positive Integer
                             Number of weights to create.

            cval : Float
                   greater than 0.0 but less than or equal to 1,0,
                   to be used for the nonlinear weights function.
                   1.0 = equal weights for all.

                   Weights will be calculated as
                     cval**(tval-1)/Sum(of all weights)
                   tval is the value of 1 to num_of_weights,

        Returns:
            weights : array of weights, sum of all weights = 1.0

    """
    if not isinstance(num_of_weights, int) or num_of_weights <= 0:
        msg = ('Number of weights must be integer > 0, '
               'num = {0:s}'.format(str(num_of_weights)))
        raise ValueError(msg)
    if cval <= 0.0 or cval > 1.0:
        msg = ('cval must be greater than 0.0 and less '
               'than or equal to 1.0 '
               'cval = {0:s}'.format(str(cval)))
        raise ValueError(msg)

    weights_list = []
    for tval_minus1 in range(0, num_of_weights):
        weights_list.append(cval**(tval_minus1))

    weights = normalise_weights(np.array(weights_list))

    return weights


def linear_weights(num_of_weights, y0val=1.0, slope=0.0,
                   ynval=None):
    """Create linear weights

        Args:
            num_of_weights : Positive Integer
                             Number of weights to create.
            y0val : Positive float
                    relative value of starting point. Default = 1.0

            AND EITHER:
            slope : float
                    slope of the line. Default = 0.0 (equal weights)
            OR
            ynval : Positive float or None
                    Relative weights of last point.
                    Default value is None

        Returns:
            weights : array of weights
                      sum of all weights = 1.0

    """
    if not isinstance(num_of_weights, int) or num_of_weights <= 0:
        msg = ('Number of weights must be integer > 0 '
               'num = {0:s}'.format(str(num_of_weights)))
        raise ValueError(msg)
    # Special case num_of_weighs == 1 i.e. Scalar coordinate.
    if num_of_weights == 1:
        weights = np.array([1.0])
        return weights
    if not isinstance(y0val, float) or y0val <= 0.0:
        msg = ('y0val must be a float > 0.0, '
               'y0val = {0:s}'.format(str(y0val)))
        raise ValueError(msg)
    if ynval is not None:
        if slope != 0.0:
            msg = ('Relative end point weight or slope must be set'
                   ' but not both.')
            raise ValueError(msg)
        else:
            slope = (ynval - y0val)/(num_of_weights - 1.0)

    weights_list = []
    for tval in range(0, num_of_weights):
        weights_list.append(slope*tval + y0val)

    weights = normalise_weights(np.array(weights_list))

    return weights


class ChooseDefaultWeightsLinear(object):
    """ Calculate Default Weights using Linear Function. """

    def __init__(self, y0val=None, slope=0.0, ynval=None):
        """Set up for calculating default weights using linear function

            Args:
                y0val : None or positive float
                        Relative value of starting point.
                slope : float
                        Slope of the line. Default = 0.0 (equal weights).
                ynval : float or None
                        Relative weights of last point.
                        Default value is None

            slope OR ynval should be set but NOT BOTH.

            If y0val value is not set or set to None then the code
            assumes that the ultimate default values of
            y0val = 20.0 and ynval = 2.0 are required.

            equal weights when slope = 0.0 or y0val = ynval
        """
        self.slope = slope
        self.ynval = ynval
        if y0val is None:
            self.y0val = 20.0
            self.ynval = 2.0
        else:
            self.y0val = y0val

    def process(self, cube, coord):
        """Calculated weights for a given cube and coord.

            Args:
                cube : iris.cube.Cube
                       Cube to blend across the coord.
                coord : string
                        The name of a coordinate dimension in the cube.
            Returns:
                weights : array of weights, sum of all weights = 1.0
        """
        if not isinstance(cube, iris.cube.Cube):
            msg = ('The first argument must be an instance of '
                   'iris.cube.Cube but is'
                   ' {0:s}'.format(type(cube)))
            raise ValueError(msg)

        if not cube.coords(coord):
            msg = ('The coord for this plugin must be '
                   'an existing coordinate in the input cube.')
            raise ValueError(msg)

        num_of_weights = len(cube.coord(coord).points)

        weights = linear_weights(num_of_weights, y0val=self.y0val,
                                 slope=self.slope,
                                 ynval=self.ynval)

        return weights

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        desc = '<ChooseDefaultWeightsLinear y0val={0:4.1f}'.format(self.y0val)
        if self.ynval is None:
            desc += ', slope={0:6.2f}>'.format(self.slope)
        else:
            desc += ', ynval={0:4.1f}>'.format(self.slope)
        return desc


class ChooseDefaultWeightsNonLinear(object):
    """ Calculate Default Weights using NonLinear Function. """
    def __init__(self, cval=0.85):
        """Set up for calculating default weights using non-linear function.

            Args:
                cval = float
                       Value greater than 0, less than equal 1.0
                       default = 0.85
                       equal weights when cval = 1.0
        """
        self.cval = cval

    def process(self, cube, coord):
        """Calculated weights for a given cube and coord.

            Args:
                cube : iris.cube.Cube
                       Cube to blend across the coord.
                coord : string
                        The name of a coordinate dimension in the cube.
            Returns:
                weights : array of weights, sum of all weights = 1.0
        """
        if not isinstance(cube, iris.cube.Cube):
            msg = ('The first argument must be an instance of '
                   'iris.cube.Cube but is'
                   ' {0:s}'.format(type(cube)))
            raise ValueError(msg)
        if not cube.coords(coord):
            msg = ('The coord for this plugin must be '
                   'an existing coordinate in the input cube')
            raise ValueError(msg)
        num_of_weights = len(cube.coord(coord).points)

        weights = nonlinear_weights(num_of_weights, cval=self.cval)

        return weights

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        desc = ('<ChooseDefaultWeightsNonLinear '
                'cval={0:4.1f}>'.format(self.cval))
        return desc
