# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2020 Met Office.
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
"""Unit tests for the ApplyReliabilityCalibration plugin."""

import unittest

import iris
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from improver.calibration.reliability_calibration import (
    ConstructReliabilityCalibrationTables as CalPlugin,
)
from improver.calibration.reliability_calibration import (
    ManipulateReliabilityTable as Plugin,
)
from improver.synthetic_data.set_up_test_cubes import set_up_probability_cube


class Test_setup(unittest.TestCase):

    """Test class for the Test_ManipulateReliabilityTable tests,
    setting up cubes to use as inputs."""

    def setUp(self):
        """Set up forecast count"""
        # Set up a reliabilty table with a single threshold
        thresholds = [275.0]
        self.forecast = set_up_probability_cube(
            np.ones((1, 3, 3), dtype=np.float32), thresholds
        )
        reliability_cube_format = CalPlugin()._create_reliability_table_cube(
            self.forecast[0], self.forecast.coord(var_name="threshold")
        )
        reliability_cube_format = reliability_cube_format.collapsed(
            [
                reliability_cube_format.coord(axis="x"),
                reliability_cube_format.coord(axis="y"),
            ],
            iris.analysis.SUM,
        )
        self.obs_count = np.array([0, 0, 250, 500, 750], dtype=np.float32)
        self.forecast_probability_sum = np.array(
            [0, 250, 500, 750, 1000], dtype=np.float32
        )

        self.forecast_count = np.array([1000, 1000, 1000, 1000, 1000], dtype=np.float32)
        reliability_data_0 = np.stack(
            [self.obs_count, self.forecast_probability_sum, self.forecast_count]
        )
        self.reliability_table = reliability_cube_format.copy(data=reliability_data_0)
        self.probability_bin_coord = self.reliability_table.coord("probability_bin")

        # Set up a reliablity table cube with two thresholds
        reliability_data_1 = np.array(
            [
                [250, 500, 750, 1000, 1000],  # Observation count
                [0, 250, 500, 750, 1000],  # Sum of forecast probability
                [1000, 1000, 1000, 1000, 1000],  # Forecast count
            ],
            dtype=np.float32,
        )
        reliability_table_1 = self.reliability_table.copy(data=reliability_data_1)
        reliability_table_1.coord("air_temperature").points = np.array(
            278.0, dtype=np.float32
        )
        self.multi_threshold_rt = iris.cube.CubeList(
            [self.reliability_table, reliability_table_1]
        ).merge_cube()
        # Set up expected resulting reliablity table data for Test__combine_bin_pair
        self.expected_enforced_monotonic = np.array(
            [
                [0, 250, 500, 1750],  # Observation count
                [0, 250, 500, 1750],  # Sum of forecast probability
                [1000, 1000, 1000, 2000],  # Forecast count
            ]
        )


class Test__combine_bin_pair(Test_setup):

    """Test the _combine_bin_pair."""

    def test_monotonic(self):
        """Test no bin pairs are combined, if all bin pairs are monotonic."""
        obs_count = np.array([0, 250, 500, 750, 1000], dtype=np.float32)
        result = Plugin()._combine_bin_pair(
            obs_count,
            self.forecast_probability_sum,
            self.forecast_count,
            self.probability_bin_coord,
        )
        assert_array_equal(
            result[:3], [obs_count, self.forecast_probability_sum, self.forecast_count],
        )
        self.assertEqual(result[3], self.probability_bin_coord)

    def test_one_non_monotonic_bin_pair(self):
        """Test one bin pair is combined, if one bin pair is non-monotonic."""
        obs_count = np.array([0, 250, 500, 1000, 750], dtype=np.float32)
        result = Plugin()._combine_bin_pair(
            obs_count,
            self.forecast_probability_sum,
            self.forecast_count,
            self.probability_bin_coord,
        )
        assert_array_equal(result[:3], self.expected_enforced_monotonic)
        expected_bin_coord_points = np.array(
            [0.09999999, 0.29999998, 0.5, 0.8], dtype=np.float32
        )
        expected_bin_coord_bounds = np.array(
            [[0.0, 0.19999999], [0.2, 0.39999998], [0.4, 0.59999996], [0.6, 1.0]],
            dtype=np.float32,
        )
        assert_allclose(expected_bin_coord_bounds, result[3].bounds)
        assert_allclose(expected_bin_coord_points, result[3].points)

    def test_two_non_monotonic_bin_pairs(self):
        """Test one bin pair is combined, if two bin pairs are non-monotonic.
        As only a single bin pair is combined, the resulting observation
        count will still yield a non-monotonic observation frequency."""
        obs_count = np.array([0, 750, 500, 1000, 750], dtype=np.float32)
        self.expected_enforced_monotonic[0][1] = 750  # Amend observation count
        result = Plugin()._combine_bin_pair(
            obs_count,
            self.forecast_probability_sum,
            self.forecast_count,
            self.probability_bin_coord,
        )
        assert_array_equal(result[:3], self.expected_enforced_monotonic)
        expected_bin_coord_points = np.array(
            [0.09999999, 0.29999998, 0.5, 0.8], dtype=np.float32
        )
        expected_bin_coord_bounds = np.array(
            [[0.0, 0.19999999], [0.2, 0.39999998], [0.4, 0.59999996], [0.6, 1.0]],
            dtype=np.float32,
        )
        assert_allclose(expected_bin_coord_bounds, result[3].bounds)
        assert_allclose(expected_bin_coord_points, result[3].points)


class Test__assume_constant_observation_frequency(Test_setup):
    """Test the _assume_constant_observation_frequency method"""

    def test_monotonic(self):
        """Test no change to observation frequency, if already monotonic."""
        result = Plugin()._assume_constant_observation_frequency(
            self.obs_count, self.forecast_count
        )
        assert_array_equal(result.data, self.obs_count)

    def test_non_monotonic(self):
        """Test enforcement of monotonicity for observation frequency."""
        obs_count = np.array([0, 750, 500, 1000, 750], dtype=np.float32)
        expected_result = np.array([0, 750, 750, 1000, 1000], dtype=np.float32)
        result = Plugin()._assume_constant_observation_frequency(
            obs_count, self.forecast_count
        )
        assert_array_equal(result.data, expected_result)


class Test_process(Test_setup):

    """Test the process method."""

    def test_no_change(self):
        """Test with on changes required to preserve monotonicity"""
        result = Plugin().process(self.multi_threshold_rt.copy())
        assert_array_equal(result[0].data, self.multi_threshold_rt[0].data)
        self.assertEqual(result[0].coords(), self.multi_threshold_rt[0].coords())
        assert_array_equal(result[1].data, self.multi_threshold_rt[1].data)
        self.assertEqual(result[1].coords(), self.multi_threshold_rt[1].coords())

    def test_highest_bin_non_monotonic(self):
        """Test expected values are returned where the highest observation
        count bin is non-monotonic."""

        expected_data = np.array(
            [[0, 250, 500, 1750], [0, 250, 500, 1750], [1000, 1000, 1000, 2000]]
        )
        expected_bin_coord_points = np.array([0.1, 0.3, 0.5, 0.8], dtype=np.float32)
        expected_bin_coord_bounds = np.array(
            [[0.0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 1.0]], dtype=np.float32,
        )
        self.multi_threshold_rt.data[1] = np.array(
            [
                [0, 250, 500, 1000, 750],  # Observation count
                [0, 250, 500, 750, 1000],  # Sum of forecast probability
                [1000, 1000, 1000, 1000, 1000],  # Forecast count
            ]
        )

        result = Plugin().process(self.multi_threshold_rt.copy())
        assert_array_equal(result[0].data, self.multi_threshold_rt[0].data)
        self.assertEqual(result[0].coords(), self.multi_threshold_rt[0].coords())
        assert_array_equal(result[1].data, expected_data)
        assert_allclose(
            result[1].coord("probability_bin").points, expected_bin_coord_points
        )
        assert_allclose(
            result[1].coord("probability_bin").bounds, expected_bin_coord_bounds
        )

    def test_central_bin_non_monotonic(self):
        """Test expected values are returned where a central observation
        count bin is non-monotonic."""

        expected_data = np.array(
            [[0, 750, 750, 1000], [0, 750, 750, 1000], [1000, 2000, 1000, 1000]]
        )

        expected_bin_coord_points = np.array([0.1, 0.4, 0.7, 0.9], dtype=np.float32)

        expected_bin_coord_bounds = np.array(
            [[0.0, 0.2], [0.2, 0.6], [0.6, 0.8], [0.8, 1.0]], dtype=np.float32,
        )
        self.multi_threshold_rt.data[1] = np.array(
            [
                [0, 500, 250, 750, 1000],  # Observation count
                [0, 250, 500, 750, 1000],  # Sum of forecast probability
                [1000, 1000, 1000, 1000, 1000],  # Forecast count
            ]
        )

        result = Plugin().process(self.multi_threshold_rt.copy())
        assert_array_equal(result[0].data, self.multi_threshold_rt[0].data)
        self.assertEqual(result[0].coords(), self.multi_threshold_rt[0].coords())
        assert_array_equal(result[1].data, expected_data)
        assert_allclose(
            result[1].coord("probability_bin").points, expected_bin_coord_points
        )
        assert_allclose(
            result[1].coord("probability_bin").bounds, expected_bin_coord_bounds
        )

    def test_upper_bins_non_monotonic(self):
        """Test expected values are returned where the upper observation
        count bins are non-monotonic."""
        expected_data = np.array(
            [[0, 1000, 1000, 2000], [0, 250, 500, 1750], [1000, 1000, 1000, 2000]]
        )
        expected_bin_coord_points = np.array([0.1, 0.3, 0.5, 0.8], dtype=np.float32)

        expected_bin_coord_bounds = np.array(
            [[0.0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 1.0]], dtype=np.float32,
        )

        self.multi_threshold_rt.data[1] = np.array(
            [
                [0, 1000, 750, 500, 250],  # Observation count
                [0, 250, 500, 750, 1000],  # Sum of forecast probability
                [1000, 1000, 1000, 1000, 1000],  # Forecast count
            ]
        )

        result = Plugin().process(self.multi_threshold_rt.copy())
        assert_array_equal(result[0].data, self.multi_threshold_rt[0].data)
        self.assertEqual(result[0].coords(), self.multi_threshold_rt[0].coords())
        assert_array_equal(result[1].data, expected_data)
        assert_allclose(
            result[1].coord("probability_bin").points, expected_bin_coord_points
        )
        assert_allclose(
            result[1].coord("probability_bin").bounds, expected_bin_coord_bounds
        )

    def test_lowest_bin_non_monotonic(self):
        """Test expected values are returned where the lowest observation
        count bin is non-monotonic."""
        expected_data = np.array(
            [[1000, 500, 500, 750], [250, 500, 750, 1000], [2000, 1000, 1000, 1000]]
        )

        expected_bin_coord_points = np.array([0.2, 0.5, 0.7, 0.9], dtype=np.float32)

        expected_bin_coord_bounds = np.array(
            [[0.0, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0]], dtype=np.float32,
        )

        self.multi_threshold_rt.data[1] = np.array(
            [
                [1000, 0, 250, 500, 750],  # Observation count
                [0, 250, 500, 750, 1000],  # Sum of forecast probability
                [1000, 1000, 1000, 1000, 1000],  # Forecast count
            ]
        )
        result = Plugin().process(self.multi_threshold_rt.copy())
        assert_array_equal(result[0].data, self.multi_threshold_rt[0].data)
        self.assertEqual(result[0].coords(), self.multi_threshold_rt[0].coords())
        assert_array_equal(result[1].data, expected_data)
        assert_allclose(
            result[1].coord("probability_bin").points, expected_bin_coord_points
        )
        assert_allclose(
            result[1].coord("probability_bin").bounds, expected_bin_coord_bounds
        )


if __name__ == "__main__":
    unittest.main()
