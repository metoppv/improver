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
                weights (numpy.array):
                    array of weights

            Returns:
                normalised_weights (numpy.array):
                    array of weights where sum = 1.0

            Raises:
                ValueError: any negative weights are found in input.
                ValueError: sum of weights in the input is 0.
        """
        if weights.min() < 0.0:
            msg = ('Weights must be positive. The weights have at least one '
                   'value < 0.0: {}'.format(weights))
            raise ValueError(msg)

        sumval = weights.sum()
        if sumval == 0:
            msg = 'Sum of weights must be > 0.0'
            raise ValueError(msg)

        normalised_weights = weights / sumval
        return normalised_weights

    @staticmethod
    def redistribute_weights(weights, forecast_present, method='evenly'):
        """Redistribute weights if any of the forecasts are missing.

            Args:
                weights (numpy.ndarray):
                    Array of weights.
                forecast_present (numpy.ndarray):
                    Size of weights with values set as
                                   1.0 for present
                                   0.0 for missing.
                method (string):
                    Method to redistribute weights, default evenly.

                    Options are:
                        evenly - adding the weights from the
                                 missing forecasts evenly across
                                 the remaining forecasts.
                        proportional - re-weight according to the
                                       proportion of the previous
                                       weights.

            Returns:
                redistributed_weights (numpy.ndarray):
                    Array of weights where sum = 1.0 and missing weights are
                    set to -1.0

            Raises:
                ValueError: the weights input do not add up to 1.
                ValueError: any of the input weights are negative.
                ValueError: an unexpected number of weights are input.
                ValueError: none of the forecasts expected (according to
                            the user input coord_exp_vals) were found on the
                            cube being blended.
                ValueError: an unknown weights redistribution method is
                            entered (only recognised methods are 'evenly'
                            and 'proportional').
        """
        sumval = weights.sum()

        if abs(sumval - 1.0) > 0.0001:
            msg = 'Sum of weights must be 1.0'
            raise ValueError(msg)

        if weights.min() < 0.0:
            msg = 'Weights should be positive or at least one > 0.0'
            raise ValueError(msg)

        if len(weights) != len(forecast_present):
            msg = ('Arrays weights and forecast_present not the same size'
                   ' weights is len {0:}'.format(len(weights)) +
                   ' forecast_present is len {0:}'.format(
                      len(forecast_present)))
            raise ValueError(msg)

        num_forecasts_present = forecast_present.sum()
        if num_forecasts_present == 0:
            msg = 'None of the expected forecasts were found.'
            raise ValueError(msg)
        elif num_forecasts_present < len(forecast_present):
            combined_weights = weights*forecast_present
            if method == 'evenly':
                missing_avg_weight = (
                    1.0 - combined_weights.sum())/num_forecasts_present
                redistributed_weights = combined_weights + missing_avg_weight
            elif method == 'proportional':
                redistributed_weights = (
                    WeightsUtilities.normalise_weights(combined_weights))
            else:
                msg = ('Unknown weights redistribution method'
                       ': {}'.format(method))
                raise ValueError(msg)
            # Set missing values to -1.0
            redistributed_weights = redistributed_weights[np.where(
                forecast_present == 1)]
        elif num_forecasts_present == len(forecast_present):
            redistributed_weights = weights
        return redistributed_weights

    @staticmethod
    def process_coord(cube, coordinate, coord_exp_vals=None,
                      coord_unit='no_unit'):
        """Calculated weights for a given cube and coord.

            Args:
                cube (iris.cube.Cube):
                       Cube to blend across the coord.
                coordinate (string):
                       Name of coordinate in the cube to be blended.
                coord_exp_vals (string):
                       String list of values which are expected on the
                       coordinate to be blended over.
                coord_unit (cf_units.Unit):
                       The unit in which the coord_exp_vals have been passed
                       in.

            Returns:
                (tuple) : tuple containing:
                    **exp_coord_len** (int):
                           The number of forecasts we expect to blend, based on
                           the length of the coordinate we are going to blend
                           over.
                    **exp_forecast_found** (binary mask):
                           Array showing where the input cube coordinate values
                           agree with the input expected coordinate values.

            Raises:
                ValueError: the coordinate to blend over does not exist on
                            the cube being blended.
                ValueError: the length of the expected coordinate input is
                            less than the length of the corresponding cube
                            coordinate.
                ValueError: the input coordinate units cannot be converted
                            to the units of the corresponding cube
                            coordinate.
        """
        if not cube.coords(coordinate):
            msg = ('The coord for this plugin must be '
                   'an existing coordinate in the input cube.')
            raise ValueError(msg)
        cube_coord = cube.coord(coordinate)
        if coord_exp_vals is not None:
            coord_values = [float(x) for x in coord_exp_vals.split(',')]
            if len(coord_values) < len(cube_coord.points):
                msg = ('The cube coordinate has more points '
                       'than requested coord, '
                       'len coord points = {0:d} '.format(len(coord_values)) +
                       'len cube points = {0:d}'.format(
                          len(cube_coord.points)))
                raise ValueError(msg)
            else:
                exp_coord = iris.coords.AuxCoord(coord_values,
                                                 long_name=coordinate,
                                                 units=coord_unit)
                exp_coord_len = len(exp_coord.points)
        else:
            exp_coord_len = len(cube_coord.points)
        # Find which coordinates are present in exp_coord but not in cube_coord
        # ie: find missing forecasts.
        if len(cube_coord.points) < exp_coord_len:
            # Firstly check that coord is in the right units
            # Do not try if coord.units not set
            if (exp_coord.units != cf_units.Unit('1') and
                    str(exp_coord.units) != 'no_unit'):
                if exp_coord.units != cube_coord.units:
                    try:
                        exp_coord.convert_units(cube_coord.units)
                    except ValueError:
                        msg = ('Failed to convert coord units '
                               'requested coord units '
                               '= {0:s} '.format(str(exp_coord.units)) +
                               'cube units '
                               '= {0:s}'.format(str(cube_coord.units)))
                        raise ValueError(msg)
            exp_forecast_found = np.array([int(x in cube_coord.points)
                                          for x in exp_coord.points])
        else:
            exp_forecast_found = np.ones(exp_coord_len)
        return (exp_coord_len, exp_forecast_found)


class ChooseDefaultWeightsLinear(object):
    """ Calculate Default Weights using Linear Function. """

    def __init__(self, y0val=None, slope=0.0, ynval=None):
        """Set up for calculating default weights using linear function

            Keyword Args:
                y0val (None or positive float):
                    Relative value of starting point.
                slope (float):
                    Slope of the line. Default = 0.0 (equal weights).
                ynval (float or None):
                    Relative weights of last point.
                    Default value is None

            slope OR ynval should be set but NOT BOTH.

            If y0val value is not set or set to None then the code
            uses default values of y0val = 20.0 and ynval = 2.0.

            equal weights when slope = 0.0 or y0val = ynval
        """
        self.slope = slope
        self.ynval = ynval
        if y0val is None:
            self.y0val = 20.0
            self.ynval = 2.0
        else:
            self.y0val = y0val

    def linear_weights(self, num_of_weights):
        """Create linear weights

            Args:
                num_of_weights (Positive Integer):
                                 Number of weights to create.
                y0val (Positive float):
                        relative value of starting point. Default = 1.0
                slope (float):
                        slope of the line. Default = 0.0 (equal weights)
                ynval (Positive float or None):
                        Relative weights of last point.
                        Default value is None

            Returns:
                weights (numpy.array):
                    array of weights, sum of all weights = 1.0

            Raises:
                ValueError: an inappropriate value of y0val is input.
                ValueError: both slope and ynval are set at input.

        """
        # Special case num_of_weighs == 1 i.e. Scalar coordinate.
        if num_of_weights == 1:
            weights = np.array([1.0])
            return weights
        if not isinstance(self.y0val, float) or self.y0val < 0.0:
            msg = ('y0val must be a float >= 0.0, '
                   'y0val = {0:s}'.format(str(self.y0val)))
            raise ValueError(msg)
        if self.ynval is not None:
            if self.slope == 0.0:
                self.slope = (self.ynval - self.y0val)/(num_of_weights - 1.0)
            else:
                msg = ('Relative end point weight or slope must be set'
                       ' but not both.')
                raise ValueError(msg)

        weights_list = []
        for tval in range(0, num_of_weights):
            weights_list.append(self.slope*tval + self.y0val)

        weights = WeightsUtilities.normalise_weights(np.array(weights_list))

        return weights

    def process(self, cube, coord_name, coord_vals=None, coord_unit='no_unit',
                weights_distrib_method='evenly'):
        """Calculated weights for a given cube and coord.

            Args:
                cube (iris.cube.Cube):
                       Cube to blend across the coord.
                coord_name (string):
                       Name of coordinate in the cube to be blended.
                coord_vals (string):
                       String list of values which are expected on the
                       coordinate to be blended over.
                coord_unit (cf_units.Unit):
                       The unit in which the coord_exp_vals have been passed
                       in.
                weights_distrib_method (string):
                       The method to use when redistributing weights in cases
                       where there are some forecasts missing. Options:
                       "evenly", "proportional".
            Returns:
                weights (numpy.array):
                    array of weights, sum of all weights = 1.0

            Raises:
                TypeError : input is not a cube
        """
        if not isinstance(cube, iris.cube.Cube):
            msg = ('The first argument must be an instance of '
                   'iris.cube.Cube but is'
                   ' {0:s}'.format(type(cube)))
            raise TypeError(msg)

        (num_of_weights,
         exp_coord_found) = WeightsUtilities.process_coord(
            cube, coord_name, coord_vals, coord_unit)

        weights_in = self.linear_weights(num_of_weights)

        weights = WeightsUtilities.redistribute_weights(
            weights_in, exp_coord_found, weights_distrib_method)

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
                cval (float):
                       Value greater than 0, less than equal 1.0
                       default = 0.85
                       equal weights when cval = 1.0
        """
        self.cval = cval

    def nonlinear_weights(self, num_of_weights):
        """Create nonlinear weights.

            Args:
                num_of_weights (Positive Integer):
                                 Number of weights to create.

            Returns:
                weights (numpy.array):
                    array of weights, sum of all weights = 1.0

            Raises:
                ValueError: an inappropriate value of cval is input.
        """
        if self.cval <= 0.0 or self.cval > 1.0:
            msg = ('cval must be greater than 0.0 and less '
                   'than or equal to 1.0 '
                   'cval = {0:s}'.format(str(self.cval)))
            raise ValueError(msg)

        weights_list = []
        for tval_minus1 in range(0, num_of_weights):
            weights_list.append(self.cval**(tval_minus1))

        weights = WeightsUtilities.normalise_weights(np.array(weights_list))

        return weights

    def process(self, cube, coord_name, coord_vals=None, coord_unit='no_unit',
                weights_distrib_method='evenly'):
        """Calculated weights for a given cube and coord.

            Args:
                cube (iris.cube.Cube):
                       Cube to blend across the coord.
                coord_name (string):
                       Name of coordinate in the cube to be blended.
                coord_vals (string):
                       String list of values which are expected on the
                       coordinate to be blended over.
                coord_unit (cf_units.Unit):
                       The unit in which the coord_exp_vals have been passed
                       in.
                weights_distrib_method (string):
                        The method to use when redistributing weights in cases
                        where there are some forecasts missing. Options:
                        "evenly", "proportional".
            Returns:
                weights (numpy.array):
                    array of weights, sum of all weights = 1.0

            Raises:
                TypeError : input is not a cube
        """
        if not isinstance(cube, iris.cube.Cube):
            msg = ('The first argument must be an instance of '
                   'iris.cube.Cube but is'
                   ' {0:s}'.format(type(cube)))
            raise TypeError(msg)

        (num_of_weights,
         exp_coord_found) = WeightsUtilities.process_coord(
            cube, coord_name, coord_vals, coord_unit)

        weights_in = self.nonlinear_weights(num_of_weights)

        weights = WeightsUtilities.redistribute_weights(
            weights_in, exp_coord_found, weights_distrib_method)

        return weights

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        desc = ('<ChooseDefaultWeightsNonLinear '
                'cval={0:4.1f}>'.format(self.cval))
        return desc


class ChooseDefaultWeightsTriangular(object):
    """ Calculate Default Weights using a Triangular Function. """
    def __init__(self, width, units="no_unit"):
        """Set up for calculating default weights using triangular function.

            Args:
                width (float):
                    The width of the triangular function from the centre point.
                units (cf_units.Unit):
                    The cf units of the width and midpoint.
        """
        self.width = width
        if not isinstance(units, cf_units.Unit):
            units = cf_units.Unit(units)
        self.parameters_units = units

    def triangular_weights(self, coord_vals, midpoint):
        """Create triangular weights.

            Args:
                coord_vals (numpy array):
                    An array of coordinate values that we want to calculate
                    weights for.

            Returns:
                weights (numpy.array):
                    array of weights, sum of all weights should equal 1.0.
        """

        def calculate_weight(point, slope):
            """
            A helper function to calculate the weights for each point using a
            piecewise function to build up the triangular function.
            Args:
                point (float):
                    The point in the coordinate from the cube for
                    which we want to calculate a weight for.
                slope (float):
                    The gradient of the triangle, calculated from
                    1/(width of triangle).

            Returns:
                weight (float):
                    The individual weight calculated by the function.
            """
            if point == midpoint:
                weight = 1
            else:
                weight = 1-abs(point-midpoint)*slope
            return weight

        slope = 1.0/self.width
        weights = np.zeros(coord_vals.shape)
        # Find the indices of the points where there will be non-zero weights.
        condition = ((coord_vals >= (midpoint-self.width)) &
                     (coord_vals <= (midpoint+self.width)))
        points_with_weights = np.where(condition)[0]
        # Calculate for weights for points where we want a non-zero weight.
        for index in points_with_weights:
            weights[index] = calculate_weight(coord_vals[index], slope)
        # Normalise the weights.
        weights = WeightsUtilities.normalise_weights(weights)

        return weights

    def process(self, cube, coord_name, midpoint):
        """Calculate triangular weights for a given cube and coord.

            Args:
                cube (iris.cube.Cube):
                    Cube to blend across the coord.
                coord_name (string):
                    Name of coordinate in the cube to be blended.
                midpoint (float):
                    The centre point of the triangular function.

            Returns:
                weights (numpy.array):
                    array of weights, sum of all weights = 1.0

            Raises:
                TypeError : input is not a cube
        """
        if not isinstance(cube, iris.cube.Cube):
            msg = ('The first argument must be an instance of '
                   'iris.cube.Cube but is'
                   ' {0:s}'.format(type(cube)))
            raise TypeError(msg)

        cube_coord = cube.coord(coord_name)
        coord_vals = cube_coord.points
        coord_units = cube_coord.units

        # Rescale width and midpoint if in different units to the coordinate
        if coord_units != self.parameters_units:
            self.width = self.parameters_units.convert(self.width, coord_units)

        weights = self.triangular_weights(coord_vals, midpoint)

        return weights

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        msg = ("<ChooseDefaultTriangularWeights width={:4.1f},"
               " parameters_units={:s}>")
        desc = msg.format(self.width, self.parameters_units)
        return desc
