# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for the OccurrenceBetweenThresholds plugin"""

import unittest

import iris
import numpy as np
from iris.tests import IrisTest

from improver.between_thresholds import OccurrenceBetweenThresholds
from improver.synthetic_data.set_up_test_cubes import (
    set_up_percentile_cube,
    set_up_probability_cube,
)


class Test_process(IrisTest):
    """Test the process method"""

    def setUp(self):
        """Set up a test cube with probability data"""
        data = np.array(
            [
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[0.9, 0.9, 0.9], [0.8, 0.8, 0.8], [0.7, 0.7, 0.7]],
                [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
                [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.1, 0.2, 0.2]],
            ],
            dtype=np.float32,
        )
        temp_thresholds = np.array([279, 280, 281, 282], dtype=np.float32)
        vis_thresholds = np.array([100, 1000, 5000, 10000], dtype=np.float32)

        self.temp_cube = set_up_probability_cube(data, temp_thresholds)
        self.vis_cube = set_up_probability_cube(
            np.flip(data, axis=0),
            vis_thresholds,
            variable_name="visibility",
            threshold_units="m",
            spp__relative_to_threshold="below",
        )

        # set up a cube of rainfall rates in m s-1 (~1e-8 values)
        self.precip_cube = self.temp_cube.copy()
        self.precip_cube.coord("air_temperature").rename("rainfall_rate")
        self.precip_cube.coord("rainfall_rate").var_name = "threshold"
        self.precip_cube.coord("rainfall_rate").points = np.array(
            [0, 0.25, 0.5, 1], dtype=np.float32
        )
        self.precip_cube.coord("rainfall_rate").units = "mm h-1"
        self.precip_cube.coord("rainfall_rate").convert_units("m s-1")

    def test_above_threshold(self):
        """Test values from an "above threshold" cube"""
        threshold_ranges = [[280, 281], [281, 282]]
        expected_data = np.array(
            [
                [[0.8, 0.7, 0.6], [0.7, 0.6, 0.5], [0.6, 0.5, 0.4]],
                [[0.1, 0.2, 0.3], [0.0, 0.1, 0.2], [0.0, 0.0, 0.1]],
            ],
            dtype=np.float32,
        )
        plugin = OccurrenceBetweenThresholds(threshold_ranges.copy(), "K")
        result = plugin(self.temp_cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(
            result.name(), "probability_of_air_temperature_between_thresholds"
        )
        self.assertArrayAlmostEqual(result.data, expected_data)
        thresh_coord = result.coord("air_temperature")
        self.assertArrayAlmostEqual(thresh_coord.points, [281.0, 282.0])
        self.assertArrayAlmostEqual(thresh_coord.bounds, threshold_ranges)
        self.assertEqual(
            thresh_coord.attributes["spp__relative_to_threshold"], "between_thresholds"
        )

    def test_below_threshold(self):
        """Test values from a "below threshold" cube"""
        threshold_ranges = [[1000, 5000]]
        expected_data = np.array(
            [[0.8, 0.7, 0.6], [0.7, 0.6, 0.5], [0.6, 0.5, 0.4]], dtype=np.float32
        )
        plugin = OccurrenceBetweenThresholds(threshold_ranges.copy(), "m")
        result = plugin(self.vis_cube)
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertArrayAlmostEqual(result.coord("visibility").points, [5000.0])
        self.assertArrayAlmostEqual(result.coord("visibility").bounds, threshold_ranges)

    def test_skip_threshold(self):
        """Test calculation works for non-adjacent thresholds"""
        threshold_ranges = [[100, 1000], [1000, 10000]]
        expected_data = np.array(
            [
                [[0.1, 0.2, 0.3], [0.0, 0.1, 0.2], [0.0, 0.0, 0.1]],
                [[0.9, 0.8, 0.7], [0.9, 0.8, 0.7], [0.9, 0.8, 0.7]],
            ],
            dtype=np.float32,
        )
        plugin = OccurrenceBetweenThresholds(threshold_ranges, "m")
        result = plugin(self.vis_cube)
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_threshold_units(self):
        """Test calculation works for thresholds specified in different units
        from the cube data"""
        threshold_ranges = [[0.1, 1], [1, 10]]
        expected_data = np.array(
            [
                [[0.1, 0.2, 0.3], [0.0, 0.1, 0.2], [0.0, 0.0, 0.1]],
                [[0.9, 0.8, 0.7], [0.9, 0.8, 0.7], [0.9, 0.8, 0.7]],
            ],
            dtype=np.float32,
        )
        plugin = OccurrenceBetweenThresholds(threshold_ranges, "km")
        result = plugin(self.vis_cube)
        self.assertArrayAlmostEqual(result.data, expected_data)
        # check original cube units are not modified
        self.assertEqual(self.vis_cube.coord("visibility").units, "m")
        # check output cube units match original cube
        self.assertEqual(result.coord("visibility").units, "m")
        self.assertArrayAlmostEqual(result.coord("visibility").points, [1000, 10000])

    def test_error_non_probability_cube(self):
        """Test failure if cube doesn't contain probabilities"""
        perc_cube = set_up_percentile_cube(
            np.ones((3, 3, 3), dtype=np.float32),
            np.array((25, 50, 75), dtype=np.float32),
        )
        plugin = OccurrenceBetweenThresholds([[25, 50]], "K")
        msg = "Input is not a probability cube"
        with self.assertRaisesRegex(ValueError, msg):
            plugin(perc_cube)

    def test_error_between_thresholds_cube(self):
        """Test failure if cube isn't above or below threshold"""
        # use plugin to generate a "between_thresholds" cube...
        between_thresholds_cube = OccurrenceBetweenThresholds(
            [[280, 281], [281, 282]], "K"
        )(self.temp_cube)
        plugin = OccurrenceBetweenThresholds([[281, 282]], "K")
        msg = "Input cube must contain"
        with self.assertRaisesRegex(ValueError, msg):
            plugin(between_thresholds_cube)

    def test_error_thresholds_unavailable(self):
        """Test error if cube doesn't contain the required thresholds"""
        threshold_ranges = [[10, 100], [1000, 30000]]
        plugin = OccurrenceBetweenThresholds(threshold_ranges, "m")
        msg = (
            "visibility threshold 10 m is not available\n"
            "visibility threshold 30000 m is not available"
        )
        with self.assertRaisesRegex(ValueError, msg):
            plugin(self.vis_cube)

    def test_threshold_matching_tolerance(self):
        """Test threshold matching succeeds for absolute values close to
        zero"""
        new_thresholds = np.array([272.15, 273.15, 274.15, 275.15], dtype=np.float32)
        self.temp_cube.coord("air_temperature").points = new_thresholds
        threshold_ranges = [[-1, 0], [0, 2]]
        expected_data = np.array(
            [
                [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]],
                [[0.9, 0.9, 0.9], [0.7, 0.7, 0.7], [0.6, 0.5, 0.5]],
            ],
            dtype=np.float32,
        )
        plugin = OccurrenceBetweenThresholds(threshold_ranges, "degC")
        result = plugin(self.temp_cube)
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_thresholds_indistinguishable(self):
        """Test behaviour in a case where cube extraction cannot work within a
        tolerance of 1e-5"""
        # set threshold ranges in m s-1
        points = self.precip_cube.coord("rainfall_rate").points.copy()
        threshold_ranges = [[points[1], points[2]]]
        msg = "Plugin cannot distinguish between thresholds at"
        with self.assertRaisesRegex(ValueError, msg):
            OccurrenceBetweenThresholds(threshold_ranges, "m s-1")

    def test_original_units_indistinguishable(self):
        """Test cubes where thresholds are indistinguisable in SI units can be
        correctly processed using threshold ranges specified in a unit with
        more than 1e-5 discrimination"""
        expected_data = np.array(
            [[0.8, 0.7, 0.6], [0.7, 0.6, 0.5], [0.6, 0.5, 0.4]], dtype=np.float32
        )
        threshold_ranges = [[0.25, 0.5]]
        plugin = OccurrenceBetweenThresholds(threshold_ranges, "mm h-1")
        result = plugin(self.precip_cube)
        self.assertArrayAlmostEqual(result.data, expected_data)


if __name__ == "__main__":
    unittest.main()
