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
"""Unit tests for the weights.ChooseDefaultWeightsLinear plugin."""


import unittest

from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests import IrisTest
import iris
import numpy as np

from improver.blending.weights import WeightsUtilities


def set_up_cube():
    """Create a cube for testing
        Returns:
            cube : iris.cube.Cube
                   dummy cube for testing
    """
    data = np.zeros((2, 2, 2))
    data[0][:][:] = 0.0
    data[1][:][:] = 1.0
    cube = Cube(data, standard_name="precipitation_amount",
                units="kg m^-2 s^-1")
    cube.add_dim_coord(DimCoord(np.linspace(-45.0, 45.0, 2), 'latitude',
                                units='degrees'), 1)
    cube.add_dim_coord(DimCoord(np.linspace(120, 180, 2), 'longitude',
                                units='degrees'), 2)
    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"
    tunit = Unit(time_origin, calendar)
    time_coord = DimCoord([402192.5, 402193.5],
                          "time", units=tunit)
    cube.add_dim_coord(time_coord, 0)
    dummy_scalar_coord = iris.coords.AuxCoord(1,
                                              long_name='scalar_coord',
                                              units='no_unit')
    cube.add_aux_coord(dummy_scalar_coord)
    return cube


def add_realizations(cube, num):
    """Create num realizations of input cube.
        Args:
            cube : iris.cube.Cube
                   input cube.
            num : integer
                   Number of realizations.
        Returns:
            cubeout : iris.cube.Cube
                      copy of cube with num realizations added.
    """
    cubelist = iris.cube.CubeList()
    for i in range(0, num):
        newcube = cube.copy()
        new_ensemble_coord = iris.coords.AuxCoord(i,
                                                  standard_name='realization')
        newcube.add_aux_coord(new_ensemble_coord)
        cubelist.append(newcube)
    cubeout = cubelist.merge_cube()
    return cubeout


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(WeightsUtilities())
        msg = '<WeightsUtilities>'
        self.assertEqual(result, msg)


class Test_normalise_weights(IrisTest):
    """Test the normalise_weights function. """

    def test_basic(self):
        """Test that the function returns an array of weights. """
        weights_in = np.array([1.0, 2.0, 3.0])
        result = WeightsUtilities.normalise_weights(weights_in)
        self.assertIsInstance(result, np.ndarray)

    def test_array_sum_equals_one(self):
        """Test that the resulting weights add up to one. """
        weights_in = np.array([1.0, 2.0, 3.0])
        result = WeightsUtilities.normalise_weights(weights_in)
        self.assertAlmostEqual(result.sum(), 1.0)

    def test_fails_weight_less_than_zero(self):
        """Test it fails if weight less than zero. """
        weights_in = np.array([-1.0, 0.1])
        msg = ('Weights must be positive')
        with self.assertRaisesRegex(ValueError, msg):
            WeightsUtilities.normalise_weights(weights_in)

    def test_fails_sum_equals_zero(self):
        """Test it fails if sum of input weights is zero. """
        weights_in = np.array([0.0, 0.0, 0.0])
        msg = ('Sum of weights must be > 0.0')
        with self.assertRaisesRegex(ValueError, msg):
            WeightsUtilities.normalise_weights(weights_in)

    def test_returns_correct_values(self):
        """Test it returns the correct values. """
        weights_in = np.array([6.0, 3.0, 1.0])
        result = WeightsUtilities.normalise_weights(weights_in)
        expected_result = np.array([0.6, 0.3, 0.1])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_returns_correct_values_2darray_axis0(self):
        """Test normalizing along the columns of the array."""
        weights_in = np.array([[6.0, 3.0, 1.0],
                               [4.0, 1.0, 3.0]])
        result = WeightsUtilities.normalise_weights(weights_in, axis=0)
        expected_result = np.array([[0.6, 0.75, 0.25],
                                    [0.4, 0.25, 0.75]])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_returns_correct_values_2darray_axis1(self):
        """Test normalizing along the rows of the array."""
        weights_in = np.array([[6.0, 3.0, 1.0],
                               [4.0, 1.0, 3.0]])
        result = WeightsUtilities.normalise_weights(weights_in, axis=1)
        expected_result = np.array([[0.6, 0.3, 0.1],
                                    [0.5, 0.125, 0.375]])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_returns_correct_values_2darray_zero_weights(self):
        """Test normalizing along the columns of the array when there are
           zeros in the input array."""
        weights_in = np.array([[6.0, 3.0, 0.0],
                               [0.0, 1.0, 3.0]])
        result = WeightsUtilities.normalise_weights(weights_in, axis=0)
        expected_result = np.array([[1.0, 0.75, 0.0],
                                    [0.0, 0.25, 1.0]])
        self.assertArrayAlmostEqual(result, expected_result)


class Test_redistribute_weights(IrisTest):
    """Test the redistribute weights function. """

    def test_basic(self):
        """Test that the function returns an array of weights. """
        weights_in = np.array([0.6, 0.3, 0.1])
        missing_weights = np.ones(3)
        result = WeightsUtilities.redistribute_weights(weights_in,
                                                       missing_weights)
        self.assertIsInstance(result, np.ndarray)

    def test_fails_sum__not_equal_to_one(self):
        """Test it fails if sum of input weights not equal to one. """
        weights_in = np.array([3.0, 2.0, 1.0])
        missing_weights = np.ones(3)
        msg = ('Sum of weights must be 1.0')
        with self.assertRaisesRegex(ValueError, msg):
            WeightsUtilities.redistribute_weights(weights_in,
                                                  missing_weights)

    def test_fails_weight_less_than_zero(self):
        """Test it fails if weight less than zero. """
        weights_in = np.array([-0.1, 1.1])
        missing_weights = np.ones(2)
        msg = ('Weights should be positive or at least one > 0.0')
        with self.assertRaisesRegex(ValueError, msg):
            WeightsUtilities.redistribute_weights(weights_in,
                                                  missing_weights)

    def test_fails_mismatch_array_sizes(self):
        """Test it fails if weights and missing_weights not the same size."""
        weights_in = np.array([0.7, 0.2, 0.1])
        missing_weights = np.ones(2)
        msg = ('Arrays weights and forecast_present not the same size')
        with self.assertRaisesRegex(ValueError, msg):
            WeightsUtilities.redistribute_weights(weights_in,
                                                  missing_weights)

    def test_returns_correct_values_evenly(self):
        """Test it returns the correct values, method is evenly."""
        weights_in = np.array([0.41957573, 0.25174544,
                               0.15104726, 0.09062836,
                               0.05437701, 0.03262621])
        missing_weights = np.ones(6)
        missing_weights[2] = 0.0
        result = WeightsUtilities.redistribute_weights(weights_in,
                                                       missing_weights)
        expected_result = np.array([0.44978518, 0.28195489,
                                    0.12083781,
                                    0.08458647, 0.06283566])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_returns_correct_values_proportional(self):
        """Test it returns the correct values, method is proportional."""
        weights_in = np.array([0.41957573, 0.25174544,
                               0.15104726, 0.09062836,
                               0.05437701, 0.03262621])
        missing_weights = np.ones(6)
        missing_weights[2] = 0.0
        result = WeightsUtilities.redistribute_weights(weights_in,
                                                       missing_weights,
                                                       method='proportional')
        expected_result = np.array([0.49422742, 0.29653645,
                                    0.10675312,
                                    0.06405187, 0.03843112])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_all_missing(self):
        """Test it raises the correct error when none of the expected
           coordinate values are present on the cube."""
        weights_in = np.array([0.6, 0.3, 0.1])
        missing_weights = np.zeros(3)
        msg = 'None of the expected forecasts were found.'
        with self.assertRaisesRegex(ValueError, msg):
            WeightsUtilities.redistribute_weights(weights_in, missing_weights)


class Test_process_coord(IrisTest):
    """Test the linear weights function. """

    def setUp(self):
        """Setup for testing process coord"""
        self.cube = set_up_cube()
        self.cube_coord = self.cube.coord("time")
        self.coordinate = self.cube_coord.name()
        self.exp_coord_vals = ','.join(
            [str(x) for x in self.cube_coord.points])

    def test_basic(self):
        """Test process_cord returns num and array of missing_weights. """
        (result_num_of_weights,
         result_missing) = WeightsUtilities.process_coord(self.cube,
                                                          self.coordinate,
                                                          self.exp_coord_vals)
        self.assertIsInstance(result_num_of_weights, int)
        self.assertIsInstance(result_missing, np.ndarray)

    def test_fails_coord_not_in_cube(self):
        """ Test process_cord fails if coord not in cube """
        msg = ('The coord for this plugin must be '
               'an existing coordinate in the input cube.')
        with self.assertRaisesRegex(ValueError, msg):
            WeightsUtilities.process_coord(self.cube, 'not_in_cube', '0')

    def test_no_points_set_in_coord(self):
        """Test returns num in coordinate if no points set in coord. """
        expected_num = len(self.cube_coord.points)
        expected_array = np.ones(expected_num)
        (result_num_of_weights,
         result_missing) = WeightsUtilities.process_coord(self.cube,
                                                          self.coordinate)
        self.assertAlmostEqual(result_num_of_weights, expected_num)
        self.assertArrayAlmostEqual(result_missing, expected_array)

    def test_fails_less_points_in_coord(self):
        """Test fails if less points in coord than in cube. """
        exp_coord_vals = self.exp_coord_vals[0]
        msg = ('The cube coordinate has more points '
               'than requested coord, ')
        with self.assertRaisesRegex(ValueError, msg):
            WeightsUtilities.process_coord(self.cube, self.coordinate,
                                           exp_coord_vals)

    def test_fails_if_can_not_convert_units(self):
        """Test fails if it can not convert units """
        units = 'mm'
        exp_coord_vals = '402191.5, 402192.5, 402193.5'
        msg = ('Failed to convert coord units ')
        with self.assertRaisesRegex(ValueError, msg):
            WeightsUtilities.process_coord(
                self.cube, self.coordinate, exp_coord_vals, units)

    def test_finds_missing_points(self):
        """Test correct values are returned for case where not all expected
           coordinate values are present in cube."""
        expected_num = 3
        expected_array = np.ones(expected_num)
        expected_array[0] = 0.0
        exp_coord_vals = '402191.5, 402192.5, 402193.5'
        (result_num_of_weights,
         result_missing) = WeightsUtilities.process_coord(
             self.cube, self.coordinate, exp_coord_vals)
        self.assertAlmostEqual(result_num_of_weights, expected_num)
        self.assertArrayAlmostEqual(result_missing, expected_array)


if __name__ == '__main__':
    unittest.main()
