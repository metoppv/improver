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
"""Unit tests for the ConstructReliabilityCalibrationTables plugin."""

import unittest

from datetime import datetime
import iris
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from improver.utilities.cube_manipulation import merge_cubes
from improver.calibration.reliability_calibration import (
    ConstructReliabilityCalibrationTables as Plugin)
from improver_tests.set_up_test_cubes import set_up_probability_cube


class Test_Setup(unittest.TestCase):

    """Test class for the Test_ConstructReliabilityCalibrationTables tests,
    setting up cubes to use as inputs."""

    def setUp(self):
        """Create forecast and truth cubes for use in testing the reliability
        calibration plugin. Two forecast and two truth cubes are created, each
        pair containing the same data but given different forecast reference
        times and validity times. These times maintain the same forecast period
        for each forecast cube.

        The truth data for reliability calibration is thresholded data, giving
        fields of zeroes and ones.

        Each forecast cube in conjunction with the contemporaneous truth cube
        will be used to produce a reliability calibration table. When testing
        the process method here we expect the final reliability calibration
        table for a given threshold (we are only using 283K in the value
        comparisons) to be the sum of two of these identical tables."""

        thresholds = [283, 288]
        forecast_data = np.arange(9, dtype=np.float32).reshape(3, 3) / 8.
        forecast_data = np.stack([forecast_data, forecast_data])
        truth_data = np.linspace(281, 285, 9, dtype=np.float32).reshape(3, 3)
        # Threshold the truths, giving fields of zeroes and ones.
        truth_data_a = (truth_data > thresholds[0]).astype(int)
        truth_data_b = (truth_data > thresholds[1]).astype(int)
        truth_data = np.stack([truth_data_a, truth_data_b])

        self.forecast_1 = set_up_probability_cube(forecast_data, thresholds)
        self.forecast_2 = set_up_probability_cube(
            forecast_data, thresholds, time=datetime(2017, 11, 11, 4, 0),
            frt=datetime(2017, 11, 11, 0, 0))
        self.forecasts = merge_cubes([self.forecast_1, self.forecast_2])
        self.truth_1 = set_up_probability_cube(
            truth_data, thresholds, frt=datetime(2017, 11, 10, 4, 0))
        self.truth_2 = set_up_probability_cube(
            truth_data, thresholds, time=datetime(2017, 11, 11, 4, 0),
            frt=datetime(2017, 11, 11, 4, 0))
        self.truths = merge_cubes([self.truth_1, self.truth_2])
        self.expected_threshold_coord = self.forecasts.coord(
            var_name='threshold')
        self.expected_table_shape = (3, 5, 3, 3)
        self.expected_attributes = (
            {'title': 'Reliability calibration data table',
             'source': 'IMPROVER',
             'institution': 'unknown'})

        # Note the structure of the expected_table is non-trivial to interpret
        # due to the dimension ordering.
        self.expected_table = np.array([[[[0., 0., 0.],
                                          [0., 0., 0.],
                                          [0., 0., 0.]],
                                         [[0., 0., 0.],
                                          [0., 0., 0.],
                                          [0., 0., 0.]],
                                         [[0., 0., 0.],
                                          [0., 0., 1.],
                                          [0., 0., 0.]],
                                         [[0., 0., 0.],
                                          [0., 0., 0.],
                                          [1., 1., 0.]],
                                         [[0., 0., 0.],
                                          [0., 0., 0.],
                                          [0., 0., 1.]]],
                                        [[[0., 0., 0.],
                                          [0., 0., 0.],
                                          [0., 0., 0.]],
                                         [[0., 0.125, 0.25],
                                          [0., 0., 0.],
                                          [0., 0., 0.]],
                                         [[0., 0., 0.],
                                          [0.375, 0.5, 0.625],
                                          [0., 0., 0.]],
                                         [[0., 0., 0.],
                                          [0., 0., 0.],
                                          [0.75, 0.875, 0.]],
                                         [[0., 0., 0.],
                                          [0., 0., 0.],
                                          [0., 0., 1.]]],
                                        [[[1., 0., 0.],
                                          [0., 0., 0.],
                                          [0., 0., 0.]],
                                         [[0., 1., 1.],
                                          [0., 0., 0.],
                                          [0., 0., 0.]],
                                         [[0., 0., 0.],
                                          [1., 1., 1.],
                                          [0., 0., 0.]],
                                         [[0., 0., 0.],
                                          [0., 0., 0.],
                                          [1., 1., 0.]],
                                         [[0., 0., 0.],
                                          [0., 0., 0.],
                                          [0., 0., 1.]]]], dtype=np.float32)


class Test__init__(unittest.TestCase):

    """Test the __init__ method."""

    def test_using_defaults(self):
        """Test without providing any arguments."""
        plugin = Plugin()
        self.assertEqual(len(plugin.probability_bins), 5)
        self.assertEqual(plugin.expected_table_shape, (3, 5))

    def test_with_arguments(self):
        """Test with specified arguments."""
        plugin = Plugin(n_probability_bins=4, single_value_limits=False)
        self.assertEqual(len(plugin.probability_bins), 4)
        self.assertEqual(plugin.expected_table_shape, (3, 4))


class Test__repr__(unittest.TestCase):

    """Test the __repr__ method."""

    def test_basic(self):
        """Test repr is as expected."""
        plugin = Plugin(n_probability_bins=2, single_value_limits=False)
        self.assertEqual(
            str(plugin),
            '<ConstructReliabilityCalibrationTables: probability_bins: '
            '[0.00 --> 0.50], [0.50 --> 1.00]>')


class Test__define_probability_bins(unittest.TestCase):

    """Test the _define_probability_bins method."""

    @staticmethod
    def test_without_single_value_limits():
        """Test the generation of probability bins without single value end
        bins. The range 0 to 1 will be divided into 4 equally sized bins."""
        expected = np.array(
            [[0., 0.24999999],
             [0.25, 0.49999997],
             [0.5, 0.74999994],
             [0.75, 1.]])
        result = Plugin()._define_probability_bins(n_probability_bins=4,
                                                   single_value_limits=False)
        assert_allclose(result, expected)

    @staticmethod
    def test_with_single_value_limits():
        """Test the generation of probability bins with single value end
        bins. The range 0 to 1 will be divided into 2 equally sized bins,
        with 2 end bins holding values approximately equal to 0 and 1."""
        expected = np.array(
            [[0.0000000e+00, 1.0000000e-06],
             [1.0000001e-06, 4.9999997e-01],
             [5.0000000e-01, 9.9999893e-01],
             [9.9999899e-01, 1.0000000e+00]])
        result = Plugin()._define_probability_bins(n_probability_bins=4,
                                                   single_value_limits=True)
        assert_allclose(result, expected)

    def test_with_single_value_limits_too_few_bins(self):
        """In this test the single_value_limits are requested whilst also
        trying to use 2 bins. This would leave no bins to cover the range 0 to
        1, so an error is raised."""

        msg = 'Cannot use single_value_limits with 2 or fewer probability bins'
        with self.assertRaisesRegex(ValueError, msg):
            Plugin()._define_probability_bins(n_probability_bins=2,
                                              single_value_limits=True)


class Test__create_probability_bins_coord(unittest.TestCase):

    """Test the _create_probability_bins_coord method."""

    def test_coordinate(self):
        """Test the probability_bins coordinate has the expected values and
        type."""
        expected_bounds = np.array([[0, 0.5], [0.5, 1]])
        expected_points = np.mean(expected_bounds, axis=1)
        plugin = Plugin(n_probability_bins=2, single_value_limits=False)
        result = plugin._create_probability_bins_coord()

        self.assertIsInstance(result, iris.coords.DimCoord)
        assert_allclose(result.points, expected_points)
        assert_allclose(result.bounds, expected_bounds)


class Test__create_reliability_table_coords(unittest.TestCase):

    """Test the _create_reliability_table_coords method."""

    def test_coordinates(self):
        """Test the reliability table coordinates have the expected values and
        type."""
        expected_indices = np.array([0, 1, 2], dtype=np.int32)
        expected_names = np.array(
            ['observation_count', 'sum_of_forecast_probabilities',
             'forecast_count'])
        index_coord, name_coord = Plugin()._create_reliability_table_coords()

        self.assertIsInstance(index_coord, iris.coords.DimCoord)
        self.assertIsInstance(name_coord, iris.coords.AuxCoord)
        assert_array_equal(index_coord.points, expected_indices)
        assert_array_equal(name_coord.points, expected_names)


class Test__get_cycle_hours(Test_Setup):

    """Test the _get_cycle_hours method."""

    def test_single_value(self):
        """Test that the expected cycle hour value is returned in a set."""

        frt = self.forecast_1.coord('forecast_reference_time')
        result = Plugin()._get_cycle_hours(frt)
        self.assertEqual(result, set([0]))

    def test_multiple_values(self):
        """Test that the expected cycle hour values are returned in a set."""

        frt = self.forecast_1.coord('forecast_reference_time')
        expected = np.array([0, 1, 4])
        frt = frt.copy(points=3600*expected + frt.points[0])
        result = Plugin()._get_cycle_hours(frt)
        self.assertEqual(result, set(expected))


class Test__check_forecast_consistency(Test_Setup):

    """Test the _check_forecast_consistency method."""

    def test_matching_forecasts(self):
        """Test case in which forecasts share cycle hour and forecast period
        values. No result is expected in this case, hence there is no value
        comparison; the test is the absence of an exception."""

        Plugin()._check_forecast_consistency(self.forecasts)

    def test_unmatching_cycle_hours(self):
        """Test case in which forecasts do not share consistent cycle hours."""

        self.forecasts.coord('forecast_reference_time').points = [3600, 7200]

        msg = ('Forecasts have been provided from differing cycle hours '
               'or forecast periods, or without these coordinates. These '
               'coordinates should be present and consistent between '
               'forecasts. Number of cycle hours found: 2, number of '
               'forecast periods found: 1.')

        with self.assertRaisesRegex(ValueError, msg):
            Plugin()._check_forecast_consistency(self.forecasts)

    def test_unmatching_forecast_periods(self):
        """Test case in which forecasts do not share consistent forecast
        periods."""

        forecasts = iris.cube.CubeList()
        forecast = self.forecasts[0].copy()
        forecast.coord('forecast_period').points = [3600]
        forecasts.append(forecast)
        forecasts.append(self.forecasts[1])
        forecasts = merge_cubes(forecasts)

        msg = ('Forecasts have been provided from differing cycle hours '
               'or forecast periods, or without these coordinates. These '
               'coordinates should be present and consistent between '
               'forecasts. Number of cycle hours found: 1, number of '
               'forecast periods found: 2.')

        with self.assertRaisesRegex(ValueError, msg):
            Plugin()._check_forecast_consistency(forecasts)

    def test_missing_forecast_period(self):
        """Test case in which forecasts do not have a forecast period
        coordinate."""

        self.forecasts.remove_coord('forecast_period')

        msg = ('Forecasts have been provided from differing cycle hours '
               'or forecast periods, or without these coordinates. These '
               'coordinates should be present and consistent between '
               'forecasts. Number of cycle hours found: 1, number of '
               'forecast periods found: 0.')

        with self.assertRaisesRegex(ValueError, msg):
            Plugin()._check_forecast_consistency(self.forecasts)


class Test__define_metadata(Test_Setup):

    """Test the _define_metadata method."""

    def test_metadata_with_complete_inputs(self):
        """Test the metadata returned is complete and as expected when the
        forecast cube contains the required metadata to copy."""

        self.forecast_1.attributes['institution'] = 'Kitten Inc'
        self.expected_attributes['institution'] = 'Kitten Inc'

        result = Plugin._define_metadata(self.forecast_1)

        self.assertIsInstance(result, dict)
        self.assertEqual(result, self.expected_attributes)

    def test_metadata_with_incomplete_inputs(self):
        """Test the metadata returned is complete and as expected when the
        forecast cube does not contain all the required metadata to copy."""

        result = Plugin._define_metadata(self.forecast_1)

        self.assertIsInstance(result, dict)
        self.assertEqual(result, self.expected_attributes)


class Test__create_reliability_table_cube(Test_Setup):

    """Test the _create_reliability_table_cube method."""

    def test_valid_inputs(self):
        """Test the cube returned has the structure expected."""

        forecast_slice = next(self.forecast_1.slices_over('air_temperature'))
        result = Plugin()._create_reliability_table_cube(
            forecast_slice, forecast_slice.coord(var_name='threshold'))
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertSequenceEqual(result.shape, self.expected_table_shape)
        self.assertEqual(result.name(), "reliability_calibration_table")
        self.assertEqual(result.attributes, self.expected_attributes)


class Test__populate_reliability_bins(Test_Setup):

    """Test the _populate_reliability_bins method."""

    def test_table_values(self):
        """Test the reliability table returned has the expected values for the
        given inputs."""

        forecast_slice = next(self.forecast_1.slices_over('air_temperature'))
        truth_slice = next(self.truth_1.slices_over('air_temperature'))
        result = Plugin()._populate_reliability_bins(
            forecast_slice.data, truth_slice.data)

        self.assertSequenceEqual(result.shape, self.expected_table_shape)
        assert_array_equal(result, self.expected_table)


class Test_process(Test_Setup):

    """Test the process method."""

    def test_return_type(self):
        """Test the process method returns a reliability table cube."""

        result = Plugin().process(self.forecasts, self.truths)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), "reliability_calibration_table")
        self.assertEqual(
            result.coord('air_temperature'), self.expected_threshold_coord)
        self.assertEqual(result.coord_dims('air_temperature')[0], 0)

    def test_table_values(self):
        """Test that cube values are as expected when process has sliced the
        inputs up for processing and then summed the contributions from the
        two dates. Note that the values tested here are for only one of the
        two processed thresholds (283K). The results contain contributions
        from two forecast/truth pairs."""

        expected = np.sum([self.expected_table, self.expected_table], axis=0)
        result = Plugin().process(self.forecasts, self.truths)

        assert_array_equal(result[0].data, expected)

    def test_mismatching_threshold_coordinates(self):
        """Test that an exception is raised if the forecast and truth cubes
        have differing threshold coordinates."""

        self.truths = self.truths[:, 0, ...]
        msg = "Threshold coordinates differ between forecasts and truths."
        with self.assertRaisesRegex(ValueError, msg):
            Plugin().process(self.forecasts, self.truths)


if __name__ == '__main__':
    unittest.main()
