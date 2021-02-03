# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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

from improver.calibration.reliability_calibration import (
    ConstructReliabilityCalibrationTables as Plugin,
)
from improver.synthetic_data.set_up_test_cubes import set_up_probability_cube
from improver.utilities.cube_manipulation import MergeCubes


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
        forecast_data = np.arange(9, dtype=np.float32).reshape(3, 3) / 8.0
        forecast_data = np.stack([forecast_data, forecast_data])
        truth_data = np.linspace(281, 285, 9, dtype=np.float32).reshape(3, 3)
        # Threshold the truths, giving fields of zeroes and ones.
        truth_data_a = (truth_data > thresholds[0]).astype(int)
        truth_data_b = (truth_data > thresholds[1]).astype(int)
        truth_data = np.stack([truth_data_a, truth_data_b])

        self.forecast_1 = set_up_probability_cube(forecast_data, thresholds)
        self.forecast_2 = set_up_probability_cube(
            forecast_data,
            thresholds,
            time=datetime(2017, 11, 11, 4, 0),
            frt=datetime(2017, 11, 11, 0, 0),
        )
        self.forecasts = MergeCubes()([self.forecast_1, self.forecast_2])
        self.truth_1 = set_up_probability_cube(
            truth_data, thresholds, frt=datetime(2017, 11, 10, 4, 0)
        )
        self.truth_2 = set_up_probability_cube(
            truth_data,
            thresholds,
            time=datetime(2017, 11, 11, 4, 0),
            frt=datetime(2017, 11, 11, 4, 0),
        )
        self.truths = MergeCubes()([self.truth_1, self.truth_2])

        masked_array = np.zeros(truth_data.shape, dtype=bool)
        masked_array[:, 0, :2] = True
        masked_truth_data_1 = np.ma.array(truth_data, mask=masked_array)
        masked_array = np.zeros(truth_data.shape, dtype=bool)
        masked_array[:, :2, 0] = True
        masked_truth_data_2 = np.ma.array(truth_data, mask=masked_array)

        self.masked_truth_1 = set_up_probability_cube(
            masked_truth_data_1, thresholds, frt=datetime(2017, 11, 10, 4, 0)
        )
        self.masked_truth_2 = set_up_probability_cube(
            masked_truth_data_2,
            thresholds,
            time=datetime(2017, 11, 11, 4, 0),
            frt=datetime(2017, 11, 11, 4, 0),
        )
        self.masked_truths = MergeCubes()([self.masked_truth_1, self.masked_truth_2])
        self.expected_threshold_coord = self.forecasts.coord(var_name="threshold")
        self.expected_table_shape = (3, 5, 3, 3)
        self.expected_attributes = {
            "title": "Reliability calibration data table",
            "source": "IMPROVER",
            "institution": "unknown",
        }

        # Note the structure of the expected_table is non-trivial to interpret
        # due to the dimension ordering.
        self.expected_table = np.array(
            [
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                ],
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.125, 0.25], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.375, 0.5, 0.625], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.75, 0.875, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                ],
                [
                    [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                ],
            ],
            dtype=np.float32,
        )
        self.expected_table_for_mask = np.array(
            [
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                ],
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.25], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.375, 0.5, 0.625], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.75, 0.875, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                ],
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                ],
            ],
            dtype=np.float32,
        )


class Test__init__(unittest.TestCase):

    """Test the __init__ method."""

    def test_using_defaults(self):
        """Test without providing any arguments."""
        plugin = Plugin()
        self.assertEqual(len(plugin.probability_bins), 5)
        self.assertEqual(plugin.expected_table_shape, (3, 5))

    def test_with_arguments(self):
        """Test with specified arguments."""
        plugin = Plugin(
            n_probability_bins=4,
            single_value_lower_limit=False,
            single_value_upper_limit=False,
        )
        self.assertEqual(len(plugin.probability_bins), 4)
        self.assertEqual(plugin.expected_table_shape, (3, 4))


class Test__repr__(unittest.TestCase):

    """Test the __repr__ method."""

    def test_basic(self):
        """Test repr is as expected."""
        plugin = Plugin(
            n_probability_bins=2,
            single_value_lower_limit=False,
            single_value_upper_limit=False,
        )
        self.assertEqual(
            str(plugin),
            "<ConstructReliabilityCalibrationTables: probability_bins: "
            "[0.00 --> 0.50], [0.50 --> 1.00]>",
        )


class Test__define_probability_bins(unittest.TestCase):

    """Test the _define_probability_bins method."""

    @staticmethod
    def test_without_single_value_limits():
        """Test the generation of probability bins without single value end
        bins. The range 0 to 1 will be divided into 4 equally sized bins."""
        expected = np.array(
            [[0.0, 0.24999999], [0.25, 0.49999997], [0.5, 0.74999994], [0.75, 1.0]]
        )
        result = Plugin()._define_probability_bins(
            n_probability_bins=4,
            single_value_lower_limit=False,
            single_value_upper_limit=False,
        )
        assert_allclose(result, expected)

    @staticmethod
    def test_with_both_single_value_limits():
        """Test the generation of probability bins with both upper and lower
        single value end bins. The range 0 to 1 will be divided into 2 equally
        sized bins, with 2 end bins holding values approximately equal to 0 and 1."""
        expected = np.array(
            [
                [0.0000000e00, 1.0000000e-06],
                [1.0000001e-06, 4.9999997e-01],
                [5.0000000e-01, 9.9999893e-01],
                [9.9999899e-01, 1.0000000e00],
            ]
        )
        result = Plugin()._define_probability_bins(
            n_probability_bins=4,
            single_value_lower_limit=True,
            single_value_upper_limit=True,
        )
        assert_allclose(result, expected)

    @staticmethod
    def test_with_lower_single_value_limit():
        """Test the generation of probability bins with only the lower single value
        limit bin. The range 0 to 1 will be divided into 4 equally sized bins,
        with 1 lower bin holding values approximately equal to 0."""
        expected = np.array(
            [
                [0.0000000e00, 1.0000000e-06],
                [1.0000001e-06, 3.3333331e-01],
                [3.3333334e-01, 6.6666663e-01],
                [6.6666669e-01, 1.0000000e00],
            ],
            dtype=np.float32,
        )

        result = Plugin()._define_probability_bins(
            n_probability_bins=4,
            single_value_lower_limit=True,
            single_value_upper_limit=False,
        )
        assert_allclose(result, expected)

    @staticmethod
    def test_with_upper_single_value_limit():
        """Test the generation of probability bins with only the upper single value
        limit bin. The range 0 to 1 will be divided into 4 equally sized bins,
        with 1 upper bin holding values approximately equal to 1."""
        expected = np.array(
            [
                [0.0, 0.3333333],
                [0.33333334, 0.6666666],
                [0.6666667, 0.9999989],
                [0.999999, 1.0],
            ],
            dtype=np.float32,
        )

        result = Plugin()._define_probability_bins(
            n_probability_bins=4,
            single_value_lower_limit=False,
            single_value_upper_limit=True,
        )
        assert_allclose(result, expected)

    def test_with_both_single_value_limits_too_few_bins(self):
        """In this test both lower and uppper single_value_limits are requested
        whilst also trying to use 2 bins. This would leave no bins to cover the
        range 0 to 1, so an error is raised."""

        msg = (
            "Cannot use both single_value_lower_limit and "
            "single_value_upper_limit with 2 or fewer probability bins."
        )
        with self.assertRaisesRegex(ValueError, msg):
            Plugin()._define_probability_bins(
                n_probability_bins=2,
                single_value_lower_limit=True,
                single_value_upper_limit=True,
            )


class Test__create_probability_bins_coord(unittest.TestCase):

    """Test the _create_probability_bins_coord method."""

    def test_coordinate_no_single_value_bins(self):
        """Test the probability_bins coordinate has the expected values and
        type with no single value lower and upper bins."""
        expected_bounds = np.array([[0, 0.5], [0.5, 1]])
        expected_points = np.mean(expected_bounds, axis=1)
        plugin = Plugin(n_probability_bins=2,)
        result = plugin._create_probability_bins_coord()

        self.assertIsInstance(result, iris.coords.DimCoord)
        assert_allclose(result.points, expected_points)
        assert_allclose(result.bounds, expected_bounds)

    def test_coordinate_single_value_bins(self):
        """Test the probability_bins coordinate has the expected values and
        type when using the single value lower and upper bins."""
        expected_bounds = np.array(
            [
                [0.0000000e00, 1.0000000e-06],
                [1.0000001e-06, 4.9999997e-01],
                [5.0000000e-01, 9.9999893e-01],
                [9.9999899e-01, 1.0000000e00],
            ]
        )
        expected_points = np.mean(expected_bounds, axis=1)
        plugin = Plugin(
            n_probability_bins=4,
            single_value_lower_limit=True,
            single_value_upper_limit=True,
        )
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
            ["observation_count", "sum_of_forecast_probabilities", "forecast_count"]
        )
        index_coord, name_coord = Plugin()._create_reliability_table_coords()

        self.assertIsInstance(index_coord, iris.coords.DimCoord)
        self.assertIsInstance(name_coord, iris.coords.AuxCoord)
        assert_array_equal(index_coord.points, expected_indices)
        assert_array_equal(name_coord.points, expected_names)


class Test__define_metadata(Test_Setup):

    """Test the _define_metadata method."""

    def test_metadata_with_complete_inputs(self):
        """Test the metadata returned is complete and as expected when the
        forecast cube contains the required metadata to copy."""

        self.forecast_1.attributes["institution"] = "Kitten Inc"
        self.expected_attributes["institution"] = "Kitten Inc"

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

        forecast_slice = next(self.forecast_1.slices_over("air_temperature"))
        result = Plugin()._create_reliability_table_cube(
            forecast_slice, forecast_slice.coord(var_name="threshold")
        )
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertSequenceEqual(result.shape, self.expected_table_shape)
        self.assertEqual(result.name(), "reliability_calibration_table")
        self.assertEqual(result.attributes, self.expected_attributes)


class Test__populate_reliability_bins(Test_Setup):

    """Test the _populate_reliability_bins method."""

    def test_table_values(self):
        """Test the reliability table returned has the expected values for the
        given inputs."""

        forecast_slice = next(self.forecast_1.slices_over("air_temperature"))
        truth_slice = next(self.truth_1.slices_over("air_temperature"))
        result = Plugin(
            single_value_lower_limit=True, single_value_upper_limit=True
        )._populate_reliability_bins(forecast_slice.data, truth_slice.data)

        self.assertSequenceEqual(result.shape, self.expected_table_shape)
        assert_array_equal(result, self.expected_table)


class Test__populate_masked_reliability_bins(Test_Setup):

    """Test the _populate_masked_reliability_bins method."""

    def test_table_values_masked_truth(self):
        """Test the reliability table returned has the expected values when a
        masked truth is input."""

        forecast_slice = next(self.forecast_1.slices_over("air_temperature"))
        truth_slice = next(self.masked_truth_1.slices_over("air_temperature"))
        result = Plugin(
            single_value_lower_limit=True, single_value_upper_limit=True
        )._populate_masked_reliability_bins(forecast_slice.data, truth_slice.data)

        self.assertSequenceEqual(result.shape, self.expected_table_shape)
        self.assertTrue(np.ma.is_masked(result))
        assert_array_equal(result.data, self.expected_table_for_mask)
        expected_mask = np.zeros(self.expected_table_for_mask.shape, dtype=bool)
        expected_mask[:, :, 0, :2] = True
        assert_array_equal(result.mask, expected_mask)


class Test_process(Test_Setup):

    """Test the process method."""

    def test_return_type(self):
        """Test the process method returns a reliability table cube."""

        result = Plugin().process(self.forecasts, self.truths)

        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), "reliability_calibration_table")
        self.assertEqual(result.coord("air_temperature"), self.expected_threshold_coord)
        self.assertEqual(result.coord_dims("air_temperature")[0], 0)

    def test_table_values(self):
        """Test that cube values are as expected when process has sliced the
        inputs up for processing and then summed the contributions from the
        two dates. Note that the values tested here are for only one of the
        two processed thresholds (283K). The results contain contributions
        from two forecast/truth pairs."""

        expected = np.sum([self.expected_table, self.expected_table], axis=0)
        result = Plugin(
            single_value_lower_limit=True, single_value_upper_limit=True
        ).process(self.forecasts, self.truths)

        assert_array_equal(result[0].data, expected)

    def test_table_values_masked_truth(self):
        """Test, similar to test_table_values, using masked arrays. The
        mask is different for different timesteps, reflecting the potential
        for masked areas in e.g. a radar truth to differ between timesteps.
        At timestep 1, two grid points are masked. At timestep 2, two
        grid points are also masked with one masked grid point in common
        between timesteps. As a result, only one grid point is masked (
        within the upper left corner) within the resulting reliability table."""

        expected_table_for_second_mask = np.array(
            [
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                ],
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.125, 0.25], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.5, 0.625], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.75, 0.875, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                ],
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                ],
            ],
            dtype=np.float32,
        )
        expected = np.sum(
            [self.expected_table_for_mask, expected_table_for_second_mask], axis=0
        )
        expected_mask = np.zeros(expected.shape, dtype=bool)
        expected_mask[:, :, 0, 0] = True
        result = Plugin(
            single_value_lower_limit=True, single_value_upper_limit=True
        ).process(self.forecasts, self.masked_truths)
        self.assertIsInstance(result.data, np.ma.MaskedArray)
        assert_array_equal(result[0].data.data, expected)
        assert_array_equal(result[0].data.mask, expected_mask)
        # Different thresholds must have the same mask.
        assert_array_equal(result[0].data.mask, result[1].data.mask)

    def test_mismatching_threshold_coordinates(self):
        """Test that an exception is raised if the forecast and truth cubes
        have differing threshold coordinates."""

        self.truths = self.truths[:, 0, ...]
        msg = "Threshold coordinates differ between forecasts and truths."
        with self.assertRaisesRegex(ValueError, msg):
            Plugin().process(self.forecasts, self.truths)


if __name__ == "__main__":
    unittest.main()
