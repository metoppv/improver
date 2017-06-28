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
"""Unit tests for the weights.ChooseDefaultWeightsLinear plugin."""


import unittest

from cf_units import Unit
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.tests import IrisTest
import iris
import numpy as np

from improver.weights import WeightsUtilities


def set_up_cube():
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
        time_coord = AuxCoord([402192.5, 402193.5],
                              "time", units=tunit)
        cube.add_aux_coord(time_coord, 0)
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
        self.assertAlmostEquals(result.sum(), 1.0)

    def test_fails_weight_less_than_zero(self):
        """Test it fails if weight less than zero. """
        weights_in = np.array([-1.0, 0.1])
        msg = ('Weights must be positive, at least one value < 0.0')
        with self.assertRaisesRegexp(ValueError, msg):
            WeightsUtilities.normalise_weights(weights_in)

    def test_fails_sum_equals_zero(self):
        """Test it fails if sum of input weights is zero. """
        weights_in = np.array([0.0, 0.0, 0.0])
        msg = ('Sum of weights must be > 0.0')
        with self.assertRaisesRegexp(ValueError, msg):
            WeightsUtilities.normalise_weights(weights_in)

    def test_returns_correct_values(self):
        """Test it returns the correct values. """
        weights_in = np.array([6.0, 3.0, 1.0])
        result = WeightsUtilities.normalise_weights(weights_in)
        expected_result = np.array([0.6, 0.3, 0.1])
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
        with self.assertRaisesRegexp(ValueError, msg):
            WeightsUtilities.redistribute_weights(weights_in,
                                                  missing_weights)

    def test_fails_weight_less_than_zero(self):
        """Test it fails if weight less than zero. """
        weights_in = np.array([-0.1, 1.1])
        missing_weights = np.ones(2)
        msg = ('Weights should be positive or at least one > 0.0')
        with self.assertRaisesRegexp(ValueError, msg):
            WeightsUtilities.redistribute_weights(weights_in,
                                                  missing_weights)

    def test_fails_mismatch_array_sizes(self):
        """Test it fails if weights and missing_weights not the same size."""
        weights_in = np.array([0.7, 0.2, 0.1])
        missing_weights = np.ones(2)
        msg = ('Arrays weights and missing_weights not the same size')
        with self.assertRaisesRegexp(ValueError, msg):
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
                                    -1.0, 0.12083781,
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
                                    -1.0, 0.10675312,
                                    0.06405187, 0.03843112])
        self.assertArrayAlmostEqual(result, expected_result)


class Test_nonlinear_weights(IrisTest):
    """Test the nonlinear weights function. """

    def test_basic(self):
        """Test that the function returns an array of weights. """
        result = WeightsUtilities.nonlinear_weights(3, 0.85)
        self.assertIsInstance(result, np.ndarray)

    def test_fails_num_of_weights_set_wrong(self):
        """Test it fails if num_of_weights not an integer or > 0. """
        msg = ('Number of weights must be integer > 0')
        with self.assertRaisesRegexp(ValueError, msg):
            WeightsUtilities.nonlinear_weights(3.0, 0.85)
        with self.assertRaisesRegexp(ValueError, msg):
            WeightsUtilities.nonlinear_weights(-1, 0.85)

    def test_fails_cval_set_wrong(self):
        """Test it fails if < """
        msg = ('cval must be greater than 0.0')
        with self.assertRaisesRegexp(ValueError, msg):
            WeightsUtilities.nonlinear_weights(3, -0.1)
        with self.assertRaisesRegexp(ValueError, msg):
            WeightsUtilities.nonlinear_weights(3, 1.85)

    def test_returns_correct_values(self):
        """Test it returns the correct values for num_of_weights 6, cval 0.6"""
        result = WeightsUtilities.nonlinear_weights(6, 0.6)
        expected_result = np.array([0.41957573, 0.25174544,
                                    0.15104726, 0.09062836,
                                    0.05437701, 0.03262621])
        self.assertArrayAlmostEqual(result, expected_result)


class Test_linear_weights(IrisTest):
    """Test the linear weights function. """

    def test_basic(self):
        """Test that the function returns an array of weights. """
        result = WeightsUtilities.linear_weights(3)
        self.assertIsInstance(result, np.ndarray)

    def test_fails_num_of_weights_set_wrong(self):
        """Test it fails if num_of_weights not an integer or > 0. """
        msg = ('Number of weights must be integer > 0')
        with self.assertRaisesRegexp(ValueError, msg):
            WeightsUtilities.linear_weights(3.0)
        with self.assertRaisesRegexp(ValueError, msg):
            WeightsUtilities.linear_weights(-1)

    def test_fails_y0val_set_wrong(self):
        """Test it fails if y0val not set properly """
        msg = ('y0val must be a float > 0.0')
        with self.assertRaisesRegexp(ValueError, msg):
            WeightsUtilities.linear_weights(3, y0val=-0.1)
        with self.assertRaisesRegexp(ValueError, msg):
            WeightsUtilities.linear_weights(3, y0val=2)

    def test_fails_ynval_and_slope_set(self):
        """Test it fails if y0val not set properly """
        msg = ('Relative end point weight or slope must be set'
               ' but not both.')
        with self.assertRaisesRegexp(ValueError, msg):
            WeightsUtilities.linear_weights(3, ynval=3.0, slope=-1.0)

    def test_returns_correct_values_num_of_weights_one(self):
        """Test it returns the correct values, method is proportional."""
        result = WeightsUtilities.linear_weights(1)
        expected_result = np.array([1.0])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_returns_correct_values_y0val_ynval_set(self):
        """Test it returns the correct values when y0val and ynval set"""
        result = WeightsUtilities.linear_weights(6,
                                                 y0val=100.0,
                                                 ynval=10.0)
        expected_result = np.array([0.3030303, 0.24848485,
                                    0.19393939, 0.13939394,
                                    0.08484848, 0.0303030])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_returns_correct_values_y0val_slope_set(self):
        """Test it returns the correct values when y0val and slope set"""
        result = WeightsUtilities.linear_weights(6,
                                                 y0val=10.0,
                                                 slope=-1.0)
        expected_result = np.array([0.22222222, 0.2,
                                    0.17777778, 0.15555556,
                                    0.13333333, 0.11111111])
        self.assertArrayAlmostEqual(result, expected_result)


class Test_process_coord(IrisTest):
    """Test the linear weights function. """

    def setUp(self):
        self.cube = set_up_cube()
        self.coord = self.cube.coord("time")

    def test_basic(self):
        """Test it returns num and array of missing_weights. """
        (result_num_of_weights,
         result_missing) = WeightsUtilities.process_coord(self.cube,
                                                          self.coord)
        self.assertIsInstance(result_num_of_weights, int)
        self.assertIsInstance(result_missing, np.ndarray)

if __name__ == '__main__':
    unittest.main()
