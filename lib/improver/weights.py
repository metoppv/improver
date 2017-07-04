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
import cf_units


class WeightsUtilities(object):
    """ Utilities for Weight processing. """

    def __init__(self):
        """Initialise class."""
        pass

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<WeightsUtilities>')
        return result

    @staticmethod
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

    @staticmethod
    def redistribute_weights(weights, missing_weights, method='evenly'):
        """Redistribute weights if any of the forecasts are missing.

            Args:
                weights : (numpy.ndarray) of weights.
                missing_weights : (numpy.ndarray) size of weights
                                  with values set as
                                  1.0 for present
                                  0.0 for missing.
                method : (string) method to redistribute weights
                         Options are
                         evenly - adding the weights from the
                                   missing forecasts evenly across
                                   the remaining forecasts.
                         proportional - re-weight according to the
                                   proportion of the previous weights.
                         Default is evenly
            Returns:
                redistributed_weights : (numpy.ndarray) of weights
                                       where sum = 1.0
                                       and missing weights are set to -1.0
        """
        sumval = weights.sum()

        if abs(sumval - 1.0) > 0.0001:
            msg = 'Sum of weights must be 1.0'
            raise ValueError(msg)

        if weights.min() < 0.0:
            msg = 'Weights should be positive or at least one > 0.0'
            raise ValueError(msg)

        if len(weights) != len(missing_weights):
            msg = ('Arrays weights and missing_weights not the same size'
                   ' weights is len {0:}'.format(len(weights)) +
                   ' missing_weights is len {0:}'.format(len(missing_weights)))
            raise ValueError(msg)

        missing_sum = missing_weights.sum()
        if missing_sum > 0.0:
            combined_weights = weights*missing_weights
            if method == 'evenly':
                missing_avg_weight = (1.0 - combined_weights.sum())/missing_sum
                redistributed_weights = combined_weights + missing_avg_weight
            elif method == 'proportional':
                redistributed_weights = (
                    WeightsUtilities.normalise_weights(combined_weights))
            else:
                msg = ('Unknown weights redistribution method'
                       ': {}'.format(method))
                raise ValueError(msg)
            # Set missing values to -1.0
            redistributed_weights[np.where(missing_weights == 0.0)] = -1.0
        else:
            redistributed_weights = weights
        return redistributed_weights

    @staticmethod
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

        weights = WeightsUtilities.normalise_weights(np.array(weights_list))

        return weights

    @staticmethod
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

        weights = WeightsUtilities.normalise_weights(np.array(weights_list))

        return weights

    @staticmethod
    def process_coord(cube, coord):
        """Calculated weights for a given cube and coord.

            Args:
                cube : iris.cube.Cube
                       Cube to blend across the coord.
                coord : iris.corrd.Coord
                        Coordinate in the cube to be blended.
            Returns:
                weights :  (numpy.ndarray) of weights
                           where sum = 1.0
                           and missing weights are set to -1.0
                           size is the same as points in coord if
                           points set in coord otherwise defaults to
                           point in cube.coord
        """
        if not isinstance(coord, iris.coords.Coord):
            msg = ('The second argument must be an instance of '
                   'iris.coords.Coord but is'
                   ' {0:s}'.format(type(coord)))
            raise ValueError(msg)

        req_coord_name = str(coord.name())
        if not cube.coords(req_coord_name):
            msg = ('The coord for this plugin must be '
                   'an existing coordinate in the input cube.')
            raise ValueError(msg)

        cube_coord = cube.coord(req_coord_name)

        if len(coord.points) == 0:
            # coord points unset will default to all points in cube
            num_of_weights = len(cube_coord.points)
        elif len(coord.points) < len(cube_coord.points):
            msg = ('The cube coordinate has more points '
                   'than requested coord, '
                   'len coord points = {0:d} '.format(len(coord.points)) +
                   'len cube points = {0:d}'.format(len(cube_coord.points)))
            raise ValueError(msg)
        else:
            num_of_weights = len(coord.points)

        # Find missing points
        missing_weights = np.ones(num_of_weights)
        if len(cube_coord.points < num_of_weights):
            # Firstly check that coord is in the right units
            # Do not try if coord.units not set
            if (coord.units != cf_units.Unit('1') and
                    str(coord.units) != 'no_unit'):
                if coord.units != cube_coord.units:
                    try:
                        coord.convert_units(cube_coord.units)
                    except:
                        msg = ('Failed to convert coord units '
                               'requested coord units '
                               '= {0:s} '.format(str(coord.units)) +
                               'cube units '
                               '= {0:s}'.format(str(cube_coord.units)))
                        raise ValueError(msg)
            for i, val in enumerate(coord.points):
                if val not in cube_coord.points:
                    missing_weights[i] = 0.0
        return (num_of_weights, missing_weights)


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
                cube : (iris.cube.Cube)
                       Cube to blend across the coord.
                coord : (iris.coords.Coord)
                        coordinate in the cube.
            Returns:
                weights : array of weights, sum of all weights = 1.0
        """
        if not isinstance(cube, iris.cube.Cube):
            msg = ('The first argument must be an instance of '
                   'iris.cube.Cube but is'
                   ' {0:s}'.format(type(cube)))
            raise ValueError(msg)

        (num_of_weights,
         missing_weights) = WeightsUtilities.process_coord(cube, coord)

        weights_in = WeightsUtilities.linear_weights(num_of_weights,
                                                     y0val=self.y0val,
                                                     slope=self.slope,
                                                     ynval=self.ynval)

        redist_weights = WeightsUtilities.redistribute_weights(weights_in,
                                                               missing_weights)
        weights = redist_weights[np.where(redist_weights != -1.0)]

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
                coord : (iris.coords.Coord)
                        coordinate in the cube.
            Returns:
                weights : array of weights, sum of all weights = 1.0
        """
        if not isinstance(cube, iris.cube.Cube):
            msg = ('The first argument must be an instance of '
                   'iris.cube.Cube but is'
                   ' {0:s}'.format(type(cube)))
            raise ValueError(msg)

        (num_of_weights,
         missing_weights) = WeightsUtilities.process_coord(cube, coord)

        weights_in = WeightsUtilities.nonlinear_weights(num_of_weights,
                                                        cval=self.cval)

        redist_weights = WeightsUtilities.redistribute_weights(weights_in,
                                                               missing_weights)
        weights = redist_weights[np.where(redist_weights != -1.0)]

        return weights

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        desc = ('<ChooseDefaultWeightsNonLinear '
                'cval={0:4.1f}>'.format(self.cval))
        return desc
