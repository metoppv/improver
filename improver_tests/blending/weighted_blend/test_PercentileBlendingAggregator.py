# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
"""Unit tests for the weighted_blend.PercentileBlendingAggregator class."""


import unittest

import numpy as np
from iris.tests import IrisTest

from improver.blending.weighted_blend import PercentileBlendingAggregator

# The PERCENTILE_DATA below were generated using a call to np.random.rand
# The numbers were then scaled between 12 and 18, envisaged as Spring or
# Autumn temperatures in Celsius.

PERCENTILE_DATA = np.array([
    17.458706, 13.732982, 15.138694, 13.966815, 16.187801,
    15.125104, 12.560181, 14.662473, 13.505879, 14.229357,
    16.645939, 16.348572, 17.298779, 17.408989, 14.526242,
    17.002329, 17.33035, 16.923946, 16.454231, 16.48794,
    15.292369, 14.879623, 16.818222, 16.288244, 14.501231,
    15.792644, 14.74469, 13.747394, 16.2813, 15.025502,
    16.620153, 15.497392, 14.028551, 16.490143, 12.824328,
    16.97861, 17.247797, 15.923066, 16.534174, 14.043188,
    15.108195, 15.579895, 16.051695, 16.475237, 13.344669,
    15.433237, 13.313879, 15.678431, 17.403114, 13.770423,
    17.443968, 17.0385, 15.021733, 16.863739, 15.647017,
    16.435345, 12.968588, 13.497512, 14.2414055, 14.173083,
    14.522574, 14.454596, 13.354028, 13.807901, 13.009074,
    12.984587, 15.867088, 12.503394, 14.164387, 16.018044,
    17.481287, 12.66411], dtype=np.float32)

WEIGHTS = np.array([
    [[0.8, 0.8],
     [0.8, 0.8]],
    [[0.5, 0.5],
     [0.5, 0.5]],
    [[0.2, 0.2],
     [0.2, 0.2]]], dtype=np.float32)

BLENDED_PERCENTILE_DATA = np.array([
    [[12.968588, 12.984587],
     [12.560181, 12.503394]],
    [[12.990671, 12.984587],
     [14.356173, 12.503394]],
    [[14.164387, 13.835985],
     [14.607758, 12.66411]],
    [[14.855347, 14.404217],
     [14.736798, 13.913844]],
    [[16.250134, 15.728171],
     [16.480879, 15.219085]],
    [[17.458706, 17.408989],
     [17.481287, 17.0385]]], dtype=np.float32)

BLENDED_PERCENTILE_DATA_EQUAL_WEIGHTS = (
    np.array([[[12.968588, 12.984587],
               [12.560181, 12.503394]],
              [[12.968588, 12.984587],
               [14.439088, 12.503394]],
              [[13.425274, 13.764813],
               [15.138694, 12.535538]],
              [[14.096469, 14.454596],
               [16.454231, 12.631967]],
              [[16.187801, 16.018042],
               [17.027607, 15.497392]],
              [[17.458706, 17.408989],
               [17.481287, 17.0385]]], dtype=np.float32))

BLENDED_PERCENTILE_DATA_SPATIAL_WEIGHTS = (
    np.array([[[12.968588, 12.984587],
               [12.560181, 12.503394]],
              [[13.138149, 12.984587],
               [14.172956, 12.503394]],
              [[13.452143, 13.801561],
               [16.620153, 12.503394]],
              [[14.07383, 14.909795],
               [16.723688, 13.807901]],
              [[14.3716755, 15.994956],
               [17.06419, 15.497392]],
              [[17.458706, 17.408989],
               [17.481287, 17.0385]]], dtype=np.float32))

PERCENTILE_VALUES = np.array(
    [[12.70237152, 14.83664335, 16.23242317, 17.42014139, 18.42036664,
      19.10276753, 19.61048008, 20.27459352, 20.886425, 21.41928051,
      22.60297787],
     [17.4934137, 20.56739689, 20.96798405, 21.4865958, 21.53586395,
      21.55643557, 22.31650746, 23.26993755, 23.62817599, 23.6783294,
      24.64542338],
     [16.24727652, 17.57784376, 17.9637658, 18.52589225, 18.99357526,
      20.50915582, 21.82791334, 21.90645982, 21.95860878, 23.52203933,
      23.71409191]])


def generate_matching_weights_array(weights, shape):
    """Create an array of weights that matches the shape of the cube.

    Args:
        weights (numpy.ndarray):
            An array of weights that needs to be broadcast to match the
            specified shape.
        shape (tuple):
            A tuple that specifies the shape to which weights should be
            broadcast. If broadcasting to this shape is not possible numpy will
            raise a broadcast error.
    """
    weights_array = np.broadcast_to(weights, shape)
    return weights_array.astype(np.float32)


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(PercentileBlendingAggregator())
        msg = '<PercentileBlendingAggregator>'
        self.assertEqual(result, msg)


class Test_aggregate(IrisTest):
    """Test the aggregate method"""
    def test_blend_percentile_aggregate(self):
        """Test blend_percentile_aggregate function works"""
        weights = np.array([0.6, 0.3, 0.1])
        weights = generate_matching_weights_array(weights, (4, 6, 3))
        weights = np.moveaxis(weights, (0, 1, 2), (2, 1, 0))

        percentiles = np.array([0, 20, 40, 60, 80, 100]).astype(np.float32)
        result = PercentileBlendingAggregator.aggregate(
            np.reshape(PERCENTILE_DATA, (6, 3, 2, 2)), 1,
            percentiles,
            weights, 0)
        self.assertArrayAlmostEqual(result, BLENDED_PERCENTILE_DATA,)

    def test_blend_percentile_aggregate_reorder1(self):
        """Test blend_percentile_aggregate works with out of order dims 1"""
        weights = np.array([0.6, 0.3, 0.1])
        weights = generate_matching_weights_array(weights, (4, 6, 3))
        weights = np.moveaxis(weights, (0, 1, 2), (2, 1, 0))

        percentiles = np.array([0, 20, 40, 60, 80, 100])
        perc_data = np.reshape(PERCENTILE_DATA, (6, 3, 2, 2))
        perc_data = np.moveaxis(perc_data, [0, 1], [3, 1])
        result = PercentileBlendingAggregator.aggregate(
            perc_data, 1,
            percentiles,
            weights, 3)
        expected_result_array = BLENDED_PERCENTILE_DATA
        expected_result_array = np.moveaxis(expected_result_array, 0, 2)
        self.assertArrayAlmostEqual(result, expected_result_array)

    def test_blend_percentile_aggregate_reorder2(self):
        """Test blend_percentile_aggregate works with out of order dims 2"""
        weights = np.array([0.6, 0.3, 0.1])
        weights = generate_matching_weights_array(weights, (4, 6, 3))
        weights = np.moveaxis(weights, (0, 1, 2), (2, 1, 0))

        percentiles = np.array([0, 20, 40, 60, 80, 100])
        perc_data = np.reshape(PERCENTILE_DATA, (6, 3, 2, 2))
        perc_data = np.moveaxis(perc_data, [0, 1], [1, 2])
        result = PercentileBlendingAggregator.aggregate(
            perc_data, 2,
            percentiles,
            weights, 1)
        expected_result_array = BLENDED_PERCENTILE_DATA
        expected_result_array = np.moveaxis(expected_result_array, 0, 1)
        self.assertArrayAlmostEqual(result, expected_result_array)

    def test_2D_simple_case(self):
        """ Test that for a simple case with only one point in the resulting
            array the function behaves as expected"""
        weights = np.array([0.8, 0.2])
        weights = generate_matching_weights_array(weights, (1, 3, 2))

        percentiles = np.array([0, 50, 100])
        perc_data = np.array([[1.0, 2.0], [5.0, 5.0], [10.0, 9.0]])
        result = PercentileBlendingAggregator.aggregate(
            perc_data, 1,
            percentiles,
            weights, 0)
        expected_result = np.array([1.0, 5.0, 10.0])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_3D_simple_case(self):
        """ Test that for a simple case with only one point and an extra
            internal dimension behaves as expected"""
        weights = np.array([0.5, 0.5])
        weights = generate_matching_weights_array(weights, (1, 3, 2))

        percentiles = np.array([0, 50, 100])
        perc_data = np.array([[[1.0], [2.0]],
                              [[5.0], [6.0]],
                              [[10.0], [9.0]]])
        result = PercentileBlendingAggregator.aggregate(
            perc_data, 1,
            percentiles,
            weights, 0)
        expected_result = np.array([[1.0], [5.555555], [10.0]])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_4D_simple_case(self):
        """ Test that for a simple case with only one point and 4D input data
            it behaves as expected"""
        weights = np.array([0.5, 0.5])
        weights = generate_matching_weights_array(weights, (1, 3, 2))

        percentiles = np.array([0, 50, 100])
        perc_data = np.array([1.0, 3.0, 2.0,
                              4.0, 5.0, 6.0])
        input_shape = (3, 2, 1, 1)
        perc_data = perc_data.reshape(input_shape)
        result = PercentileBlendingAggregator.aggregate(
            perc_data, 1,
            percentiles,
            weights, 0)
        expected_result = np.array([[[1.0]], [[3.5]], [[6.0]]])
        expected_result_shape = (3, 1, 1)
        self.assertArrayAlmostEqual(result, expected_result)
        self.assertEqual(result.shape, expected_result_shape)


class Test_blend_percentiles(IrisTest):
    """Test the blend_percentiles method"""
    def test_blend_percentiles(self):
        """Test blend_percentile function works"""
        weights = np.array([0.38872692, 0.33041788, 0.2808552])
        percentiles = np.array([0., 10., 20., 30., 40., 50.,
                                60., 70., 80., 90., 100.])
        result = PercentileBlendingAggregator.blend_percentiles(
            PERCENTILE_VALUES, percentiles, weights)
        expected_result_array = np.array([12.70237152, 16.65161847,
                                          17.97408712, 18.86356829,
                                          19.84089805, 20.77406153,
                                          21.39078426, 21.73778353,
                                          22.22440125, 23.53863876,
                                          24.64542338])
        self.assertArrayAlmostEqual(result, expected_result_array)

    def test_two_percentiles(self):
        """Test that when two percentiles are provided, the extreme values in
           the set of thresholds we are blending are returned"""
        weights = np.array([0.5, 0.5])
        percentiles = np.array([30., 60.])
        percentile_values = np.array([[5.0, 8.0], [6.0, 7.0]])
        result = PercentileBlendingAggregator.blend_percentiles(
            percentile_values, percentiles, weights)
        expected_result = np.array([5.0, 8.0])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_three_percentiles_symmetric_case(self):
        """Test that when three percentiles are provided the correct values
           are returned, not a simple average"""
        weights = np.array([0.5, 0.5])
        percentiles = np.array([20.0, 50.0, 80.0])
        percentile_values = np.array([[5.0, 6.0, 7.0], [5.0, 6.5, 7.0]])
        result = PercentileBlendingAggregator.blend_percentiles(
            percentile_values, percentiles, weights)
        expected_result = np.array([5.0, 6.2, 7.0])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_only_one_point_to_blend(self):
        """Test case where there is only one point in the coordinate we are
           blending over."""
        weights = np.array([1.0])
        percentiles = np.array([20.0, 50.0, 80.0])
        percentile_values = np.array([[5.0, 6.0, 7.0]])
        result = PercentileBlendingAggregator.blend_percentiles(
            percentile_values, percentiles, weights)
        expected_result = np.array([5.0, 6.0, 7.0])
        self.assertArrayAlmostEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
