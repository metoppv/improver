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
"""Unit tests for the weighted_blend.PercentileBlendingAggregator class."""


import unittest

from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests import IrisTest
import numpy as np

from improver.weighted_blend import PercentileBlendingAggregator

PERCENTILE_DATA = np.array([
    15.3077946, 14.65380361, 15.91478244, 15.10887522,
    12.30817311, 14.61048935, 17.80547685, 15.16351283,
    17.65229058, 18.16963972, 19.96189077, 19.87125985,
    18.14138889, 16.66496889, 19.12200888, 18.97425367,
    18.88194635, 18.63268387, 21.23771814, 21.22492499,
    18.64451978, 18.08331902, 20.0182035, 19.95292475,
    21.4430201, 19.89146966, 22.11073569, 22.84817844,
    19.50214911, 19.63833851, 21.1260302, 21.19184589,
    22.64466784, 21.18604047, 22.55922723, 24.00842263,
    21.11200487, 20.99792316, 22.07955526, 21.71937432,
    23.12910858, 22.94415916, 25.11682828, 25.76481447,
    23.82016292, 22.11742526, 26.30864878, 23.81492893])

BLENDED_PERCENTILE_DATA1 = np.array([
    12.308173106352402, 14.610489348891615,
    15.914782442039872, 15.108875220912573,
    17.737443276652936, 17.097492097934627,
    19.274275011968893, 19.127165271387128,
    18.742074248084133, 18.50663225732617,
    20.584984938812934, 20.560776453138736,
    20.25126805925584, 19.769776454949138,
    21.64008410795689, 21.598099315407907,
    22.17361605670607, 21.085160674433627,
    22.51323580222509, 23.192691941719808,
    23.82016292415821, 22.944159163666185,
    26.30864878110861, 25.76481446912205])

BLENDED_PERCENTILE_DATA2 = np.array([
    12.30817311, 14.61048935,
    15.91478244, 15.10887522,
    17.676775957972925, 17.593947064474417,
    19.51651034470578, 19.378940557706372,
    18.819297828987274, 18.594620794715546,
    20.965096170815123, 20.95201851550872,
    20.890749057567294, 19.843891344879836,
    21.927325497880343, 22.32506953618696,
    22.491628272758373, 21.143852884603188,
    22.546838489123576, 23.730102965015696,
    23.82016292, 22.94415916,
    26.30864878, 25.76481447])

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


def percentile_cube():
    """Create a percentile cube for testing."""
    data = np.reshape(PERCENTILE_DATA, (6, 2, 2, 2))
    cube = Cube(data, standard_name="air_temperature",
                units="C")
    cube.add_dim_coord(DimCoord(np.linspace(-45.0, 45.0, 2), 'latitude',
                                units='degrees'), 2)
    cube.add_dim_coord(DimCoord(np.linspace(120, 180, 2), 'longitude',
                                units='degrees'), 3)
    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"
    tunit = Unit(time_origin, calendar)
    cube.add_dim_coord(DimCoord([402192.5, 402193.5],
                                "time", units=tunit), 1)
    cube.add_dim_coord(DimCoord([0, 20, 40, 60, 80, 100],
                                long_name="percentile_over_realization"), 0)
    return cube


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
        weights = np.array([0.8, 0.2])
        percentiles = np.array([0, 20, 40, 60, 80, 100])
        result = PercentileBlendingAggregator.aggregate(
            np.reshape(PERCENTILE_DATA, (6, 2, 2, 2)), 1,
            percentiles,
            weights, 0)
        expected_result_array = np.reshape(BLENDED_PERCENTILE_DATA2,
                                           (6, 2, 2))
        self.assertArrayAlmostEqual(result, expected_result_array)

    def test_blend_percentile_aggregate_reorder1(self):
        """Test blend_percentile_aggregate works with out of order dims 1"""
        weights = np.array([0.8, 0.2])
        percentiles = np.array([0, 20, 40, 60, 80, 100])
        perc_data = np.reshape(PERCENTILE_DATA, (6, 2, 2, 2))
        perc_data = np.moveaxis(perc_data, [0, 1], [3, 1])
        result = PercentileBlendingAggregator.aggregate(
            perc_data, 1,
            percentiles,
            weights, 3)
        expected_result_array = np.reshape(BLENDED_PERCENTILE_DATA2,
                                           (6, 2, 2))
        expected_result_array = np.moveaxis(expected_result_array, 0, 2)
        self.assertArrayAlmostEqual(result, expected_result_array)

    def test_blend_percentile_aggregate_reorder2(self):
        """Test blend_percentile_aggregate works with out of order dims 2"""
        weights = np.array([0.8, 0.2])
        percentiles = np.array([0, 20, 40, 60, 80, 100])
        perc_data = np.reshape(PERCENTILE_DATA, (6, 2, 2, 2))
        perc_data = np.moveaxis(perc_data, [0, 1], [1, 2])
        result = PercentileBlendingAggregator.aggregate(
            perc_data, 2,
            percentiles,
            weights, 1)
        expected_result_array = np.reshape(BLENDED_PERCENTILE_DATA2,
                                           (6, 2, 2))
        expected_result_array = np.moveaxis(expected_result_array, 0, 1)
        self.assertArrayAlmostEqual(result, expected_result_array)

    def test_2D_simple_case(self):
        """ Test that for a simple case with only one point in the resulting
            array the function behaves as expected"""
        weights = np.array([0.8, 0.2])
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
