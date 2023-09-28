# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
# Autumn temperatures in Celsius. These data have been reshaped and sorted so that
# the data are in ascending order along the first axis, suitable for a cube
# with a leading "percentile" dimension.
PERCENTILE_DATA = np.array(
    [
        [
            [[14.501231, 13.732982], [13.354028, 13.747394]],
            [[13.009074, 12.984587], [12.560181, 12.503394]],
            [[12.968588, 13.497512], [12.824328, 12.66411]],
        ],
        [
            [[14.522574, 13.770423], [14.526242, 13.807901]],
            [[15.021733, 15.025502], [15.647017, 14.662473]],
            [[13.344669, 14.229357], [13.313879, 14.173083]],
        ],
        [
            [[17.247797, 14.454596], [14.744690, 13.966815]],
            [[15.108195, 15.125104], [15.867088, 15.497392]],
            [[13.505879, 14.879623], [14.2414055, 15.678431]],
        ],
        [
            [[17.298779, 15.792644], [15.138694, 14.043188]],
            [[16.187801, 15.579895], [16.051695, 16.435345]],
            [[14.028551, 15.433237], [16.645939, 16.288244]],
        ],
        [
            [[17.403114, 15.923066], [16.534174, 17.002329]],
            [[16.281300, 16.863739], [16.454231, 16.475237]],
            [[14.164387, 16.018044], [16.818222, 16.348572]],
        ],
        [
            [[17.458706, 17.408989], [17.443968, 17.03850]],
            [[17.330350, 16.923946], [16.620153, 16.48794]],
            [[15.292369, 16.490143], [17.481287, 16.97861]],
        ],
    ],
    dtype=np.float32,
)

WEIGHTS = np.array(
    [[[0.8, 0.8], [0.8, 0.8]], [[0.5, 0.5], [0.5, 0.5]], [[0.2, 0.2], [0.2, 0.2]]],
    dtype=np.float32,
)

BLENDED_PERCENTILE_DATA = np.array(
    [
        [[12.968588, 12.984587], [12.560181, 12.503394]],
        [[14.513496, 13.983413], [14.521752, 13.843809]],
        [[15.451672, 15.030349], [14.965419, 14.030121]],
        [[16.899292, 15.662327], [15.876251, 15.6937685]],
        [[17.333557, 16.205572], [16.516674, 16.478855]],
        [[17.458706, 17.408989], [17.481285, 17.0385]],
    ],
    dtype=np.float32,
)

BLENDED_PERCENTILE_DATA_EQUAL_WEIGHTS = np.array(
    [
        [[12.968588, 12.984587], [12.560181, 12.503394]],
        [[13.818380, 14.144759], [14.107645, 13.893876]],
        [[14.521132, 15.050104], [15.0750885, 14.874123]],
        [[15.715300, 15.560184], [15.989533, 16.09052]],
        [[17.251623, 16.184650], [16.609045, 16.474388]],
        [[17.458706, 17.408987], [17.481287, 17.0385]],
    ],
    dtype=np.float32,
)

BLENDED_PERCENTILE_DATA_SPATIAL_WEIGHTS = np.array(
    [
        [[12.968588, 12.984587], [12.560181, 12.503394]],
        [[13.381246, 14.15497], [13.719661, 13.880822]],
        [[13.743379, 15.042325], [14.906357, 14.731093]],
        [[14.121047, 15.546338], [16.071285, 16.017628]],
        [[14.9885845, 16.15714], [16.74734, 16.47621]],
        [[17.458706, 17.408987], [17.481287, 17.0385]],
    ],
    dtype=np.float32,
)

PERCENTILE_VALUES = np.array(
    [
        [
            12.70237152,
            14.83664335,
            16.23242317,
            17.42014139,
            18.42036664,
            19.10276753,
            19.61048008,
            20.27459352,
            20.886425,
            21.41928051,
            22.60297787,
        ],
        [
            17.4934137,
            20.56739689,
            20.96798405,
            21.4865958,
            21.53586395,
            21.55643557,
            22.31650746,
            23.26993755,
            23.62817599,
            23.6783294,
            24.64542338,
        ],
        [
            16.24727652,
            17.57784376,
            17.9637658,
            18.52589225,
            18.99357526,
            20.50915582,
            21.82791334,
            21.90645982,
            21.95860878,
            23.52203933,
            23.71409191,
        ],
    ]
)


def generate_matching_weights_array(weights, other_dim_length):
    """Broadcast an array of 1D weights (varying along the blend dimension) to
    the shape required to match the percentile cube

    Args:
        weights (numpy.ndarray):
            A 1D array of weights varying along the blend dimension
        other_dim_length (int):
            Length of second dimension required to match percentile cube

    Returns:
        numpy.ndarray:
            Weights that vary along the first (blend) dimension, with second
            dimension of required length
    """
    shape_t = (other_dim_length, len(weights))
    weights_array = np.broadcast_to(weights, shape_t)
    return weights_array.astype(np.float32).T


class Test_aggregate(IrisTest):
    """Test the aggregate method"""

    def test_blend_percentile_aggregate(self):
        """Test blend_percentile_aggregate function works"""
        weights = generate_matching_weights_array([0.6, 0.3, 0.1], 4)
        percentiles = np.array([0, 20, 40, 60, 80, 100]).astype(np.float32)
        result = PercentileBlendingAggregator.aggregate(
            PERCENTILE_DATA, 1, percentiles, weights
        )
        self.assertArrayAlmostEqual(
            result,
            BLENDED_PERCENTILE_DATA,
        )

    def test_2D_simple_case(self):
        """Test that for a simple case with only one point in the resulting
        array the function behaves as expected"""
        weights = generate_matching_weights_array([0.8, 0.2], 1)
        percentiles = np.array([0, 50, 100])
        perc_data = np.array([[1.0, 2.0], [5.0, 5.0], [10.0, 9.0]])
        result = PercentileBlendingAggregator.aggregate(
            perc_data, 1, percentiles, weights
        )
        expected_result = np.array([1.0, 5.0, 10.0])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_3D_simple_case(self):
        """Test that for a simple case with only one point and an extra
        internal dimension behaves as expected"""
        weights = generate_matching_weights_array([0.5, 0.5], 1)
        percentiles = np.array([0, 50, 100])
        perc_data = np.array([[[1.0], [2.0]], [[5.0], [6.0]], [[10.0], [9.0]]])
        result = PercentileBlendingAggregator.aggregate(
            perc_data, 1, percentiles, weights
        )
        expected_result = np.array([[1.0], [5.555555], [10.0]])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_4D_simple_case(self):
        """Test that for a simple case with only one point and 4D input data
        it behaves as expected"""
        weights = generate_matching_weights_array([0.5, 0.5], 1)
        percentiles = np.array([0, 50, 100])
        perc_data = np.array([1.0, 3.0, 2.0, 4.0, 5.0, 6.0])
        input_shape = (3, 2, 1, 1)
        perc_data = perc_data.reshape(input_shape)
        result = PercentileBlendingAggregator.aggregate(
            perc_data, 1, percentiles, weights
        )
        expected_result = np.array([[[1.0]], [[3.5]], [[6.0]]])
        expected_result_shape = (3, 1, 1)
        self.assertArrayAlmostEqual(result, expected_result)
        self.assertEqual(result.shape, expected_result_shape)

    def test_error_unmatched_weights(self):
        """Test error when weights shape doesn't match length of blend dimension
        (in this case 3 weights for 2 blend slices)"""
        weights = generate_matching_weights_array([0.7, 0.1, 0.2], 1)
        percentiles = np.array([0, 50, 100])
        perc_data = np.array([[1.0, 2.0], [5.0, 5.0], [10.0, 9.0]])
        with self.assertRaisesRegex(ValueError, "Weights shape does not match data"):
            PercentileBlendingAggregator.aggregate(perc_data, 1, percentiles, weights)


class Test_blend_percentiles(IrisTest):
    """Test the blend_percentiles method"""

    def test_blend_percentiles(self):
        """Test blend_percentile function works"""
        weights = np.array([0.38872692, 0.33041788, 0.2808552])
        percentiles = np.array(
            [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        )
        result = PercentileBlendingAggregator.blend_percentiles(
            PERCENTILE_VALUES, percentiles, weights
        )
        expected_result_array = np.array(
            [
                12.70237152,
                16.65161847,
                17.97408712,
                18.86356829,
                19.84089805,
                20.77406153,
                21.39078426,
                21.73778353,
                22.22440125,
                23.53863876,
                24.64542338,
            ]
        )
        self.assertArrayAlmostEqual(result, expected_result_array)

    def test_two_percentiles(self):
        """Test that when two percentiles are provided, the extreme values in
        the set of thresholds we are blending are returned"""
        weights = np.array([0.5, 0.5])
        percentiles = np.array([30.0, 60.0])
        percentile_values = np.array([[5.0, 8.0], [6.0, 7.0]])
        result = PercentileBlendingAggregator.blend_percentiles(
            percentile_values, percentiles, weights
        )
        expected_result = np.array([5.0, 8.0])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_three_percentiles_symmetric_case(self):
        """Test that when three percentiles are provided the correct values
        are returned, not a simple average"""
        weights = np.array([0.5, 0.5])
        percentiles = np.array([20.0, 50.0, 80.0])
        percentile_values = np.array([[5.0, 6.0, 7.0], [5.0, 6.5, 7.0]])
        result = PercentileBlendingAggregator.blend_percentiles(
            percentile_values, percentiles, weights
        )
        expected_result = np.array([5.0, 6.2, 7.0])
        self.assertArrayAlmostEqual(result, expected_result)

    def test_only_one_point_to_blend(self):
        """Test case where there is only one point in the coordinate we are
        blending over."""
        weights = np.array([1.0])
        percentiles = np.array([20.0, 50.0, 80.0])
        percentile_values = np.array([[5.0, 6.0, 7.0]])
        result = PercentileBlendingAggregator.blend_percentiles(
            percentile_values, percentiles, weights
        )
        expected_result = np.array([5.0, 6.0, 7.0])
        self.assertArrayAlmostEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
