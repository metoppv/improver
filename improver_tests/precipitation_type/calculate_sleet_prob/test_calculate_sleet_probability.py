# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the calculate_sleet_probability plugin."""

import unittest

import numpy as np
from iris.tests import IrisTest

from improver.precipitation_type.calculate_sleet_prob import calculate_sleet_probability
from improver.synthetic_data.set_up_test_cubes import set_up_probability_cube


class Test_calculate_sleet_probability(IrisTest):
    """Tests the calculate sleet probability function."""

    def setUp(self):
        """Create cubes to input into the function."""

        self.thresholds = np.array([276, 277], dtype=np.float32)
        self.rain_name = "probability_of_falling_rain_level_above_surface"
        self.snow_name = "probability_of_falling_snow_level_below_surface"

        rain_prob = np.array(
            [
                [[0.5, 0.1, 1.0], [0.0, 0.2, 0.5], [0.1, 0.1, 0.3]],
                [[0.5, 0.1, 1.0], [0.0, 0.2, 0.5], [0.1, 0.1, 0.3]],
            ],
            dtype=np.float32,
        )
        self.rain_prob_cube = set_up_probability_cube(
            rain_prob, self.thresholds, variable_name=self.rain_name
        )

        snow_prob = np.array(
            [
                [[0.0, 0.4, 0.0], [0.5, 0.3, 0.1], [0.0, 0.4, 0.3]],
                [[0.0, 0.4, 0.0], [0.5, 0.3, 0.1], [0.0, 0.4, 0.3]],
            ],
            dtype=np.float32,
        )
        self.snow_prob_cube = set_up_probability_cube(
            snow_prob, self.thresholds, variable_name=self.snow_name
        )

        high_prob = np.array(
            [
                [[1.0, 0.7, 0.2], [0.8, 0.8, 0.7], [0.9, 0.9, 0.7]],
                [[1.0, 0.7, 0.2], [0.8, 0.8, 0.7], [0.9, 0.9, 0.7]],
            ],
            dtype=np.float32,
        )
        self.high_prob_cube = set_up_probability_cube(
            high_prob, self.thresholds, variable_name=self.snow_name
        )

    def test_basic_calculation(self):
        """Test the basic sleet calculation works."""
        expected_result = np.array(
            [
                [[0.5, 0.5, 0.0], [0.5, 0.5, 0.4], [0.9, 0.5, 0.4]],
                [[0.5, 0.5, 0.0], [0.5, 0.5, 0.4], [0.9, 0.5, 0.4]],
            ],
            dtype=np.float32,
        )
        result = calculate_sleet_probability(self.rain_prob_cube, self.snow_prob_cube)
        self.assertArrayAlmostEqual(result.data, expected_result)
        self.assertTrue(result.dtype == np.float32)

    def test_with_ints(self):
        """Test the basic sleet calculation works with int8 data."""
        rain_prob_cube = self.rain_prob_cube.copy(
            np.array(
                [[[1, 0, 0], [0, 1, 1], [0, 0, 1]], [[1, 0, 0], [0, 1, 1], [0, 0, 1]]],
                dtype=np.int8,
            )
        )
        snow_prob_cube = self.snow_prob_cube.copy(
            np.array(
                [[[0, 1, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 0], [1, 0, 0], [0, 1, 0]]],
                dtype=np.int8,
            )
        )
        expected_result = np.array(
            [[[0, 0, 1], [0, 0, 0], [1, 0, 0]], [[0, 0, 1], [0, 0, 0], [1, 0, 0]]],
            dtype=np.int8,
        )
        result = calculate_sleet_probability(rain_prob_cube, snow_prob_cube)
        self.assertArrayAlmostEqual(result.data, expected_result)
        self.assertTrue(result.dtype == np.int8)

    def test_negative_values(self):
        """Test that an exception is raised for negative values of
        probability_of_sleet in the cube."""
        rain = self.rain_prob_cube
        high_prob = self.high_prob_cube
        msg = "Negative values of sleet probability have been calculated."
        with self.assertRaisesRegex(ValueError, msg):
            calculate_sleet_probability(rain, high_prob)

    def test_name_of_cube(self):
        """Test that the name has been changed to sleet_probability"""
        result = calculate_sleet_probability(self.snow_prob_cube, self.rain_prob_cube)
        name = "probability_of_sleet"
        self.assertEqual(result.long_name, name)


if __name__ == "__main__":
    unittest.main()
