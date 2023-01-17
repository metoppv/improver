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
"""Unit tests for the ManipulateReliabilityTable plugin."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from improver.calibration.reliability_calibration import (
    ManipulateReliabilityTable as Plugin,
)


def test_init_using_defaults():
    """Test init without providing any arguments."""
    plugin = Plugin()
    assert plugin.minimum_forecast_count == 200


def test_init_with_arguments():
    """Test init with specified arguments."""
    plugin = Plugin(minimum_forecast_count=100)
    assert plugin.minimum_forecast_count == 100


def test_init_with_invalid_minimum_forecast_count():
    """Test an exception is raised if the minimum_forecast_count value is
    less than 1."""
    msg = "The minimum_forecast_count must be at least 1"
    with pytest.raises(ValueError, match=msg):
        Plugin(minimum_forecast_count=0)


def test_cub_monotonic_no_undersampled_bins(
    default_obs_counts, default_fcst_counts, probability_bin_coord
):
    """Test no bins are combined when no bins are under-sampled and all bin
    pairs are monotonic."""
    obs_count = forecast_probability_sum = default_obs_counts

    result = Plugin()._combine_undersampled_bins(
        obs_count, forecast_probability_sum, default_fcst_counts, probability_bin_coord,
    )

    assert_array_equal(
        result[:3], [obs_count, forecast_probability_sum, default_fcst_counts]
    )
    assert result[3] == probability_bin_coord


def test_cub_poorly_sampled_bins(probability_bin_coord):
    """Test when all bins are poorly sampled and the minimum forecast count
    cannot be reached."""
    obs_count = forecast_probability_sum = np.array([0, 2, 5, 8, 10], dtype=np.float32)
    forecast_count = np.array([10, 10, 10, 10, 10], dtype=np.float32)
    expected = np.array(
        [
            [25],  # Observation count
            [25],  # Sum of forecast probability
            [50],  # Forecast count
        ]
    )

    result = Plugin()._combine_undersampled_bins(
        obs_count, forecast_probability_sum, forecast_count, probability_bin_coord,
    )

    assert_array_equal(result[:3], expected)
    expected_bin_coord_points = np.array([0.5], dtype=np.float32)
    expected_bin_coord_bounds = np.array([[0.0, 1.0]], dtype=np.float32,)
    assert_allclose(expected_bin_coord_points, result[3].points)
    assert_allclose(expected_bin_coord_bounds, result[3].bounds)


def test_cub_one_undersampled_bin_at_top(probability_bin_coord):
    """Test when the highest probability bin is under-sampled."""
    obs_count = forecast_probability_sum = np.array(
        [0, 250, 500, 750, 100], dtype=np.float32
    )
    forecast_count = np.array([1000, 1000, 1000, 1000, 100], dtype=np.float32)
    expected = np.array(
        [
            [0, 250, 500, 850],  # Observation count
            [0, 250, 500, 850],  # Sum of forecast probability
            [1000, 1000, 1000, 1100],  # Forecast count
        ]
    )

    result = Plugin()._combine_undersampled_bins(
        obs_count, forecast_probability_sum, forecast_count, probability_bin_coord,
    )

    assert_array_equal(result[:3], expected)
    expected_bin_coord_points = np.array([0.1, 0.3, 0.5, 0.8], dtype=np.float32)
    expected_bin_coord_bounds = np.array(
        [[0.0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 1.0]], dtype=np.float32,
    )
    assert_allclose(expected_bin_coord_points, result[3].points)
    assert_allclose(expected_bin_coord_bounds, result[3].bounds)


def test_cub_one_undersampled_bin_at_bottom(default_obs_counts, probability_bin_coord):
    """Test when the lowest probability bin is under-sampled."""
    obs_count = forecast_probability_sum = default_obs_counts
    forecast_count = np.array([100, 1000, 1000, 1000, 1000], dtype=np.float32)

    expected = np.array(
        [
            [250, 500, 750, 1000],  # Observation count
            [250, 500, 750, 1000],  # Sum of forecast probability
            [1100, 1000, 1000, 1000],  # Forecast count
        ]
    )
    result = Plugin()._combine_undersampled_bins(
        obs_count, forecast_probability_sum, forecast_count, probability_bin_coord,
    )

    assert_array_equal(result[:3], expected)
    expected_bin_coord_points = np.array([0.2, 0.5, 0.7, 0.9], dtype=np.float32)
    expected_bin_coord_bounds = np.array(
        [[0.0, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0]], dtype=np.float32,
    )
    assert_allclose(expected_bin_coord_points, result[3].points)
    assert_allclose(expected_bin_coord_bounds, result[3].bounds)


def test_cub_one_undersampled_bin_lower_neighbour(probability_bin_coord):
    """Test for one under-sampled bin that is combined with its lower
    neighbour."""
    obs_count = np.array([0, 250, 50, 1500, 1000], dtype=np.float32)
    forecast_probability_sum = np.array([0, 250, 50, 1500, 1000], dtype=np.float32)
    forecast_count = np.array([1000, 1000, 100, 2000, 1000], dtype=np.float32)

    expected = np.array(
        [
            [0, 300, 1500, 1000],  # Observation count
            [0, 300, 1500, 1000],  # Sum of forecast probability
            [1000, 1100, 2000, 1000],  # Forecast count
        ]
    )
    result = Plugin()._combine_undersampled_bins(
        obs_count, forecast_probability_sum, forecast_count, probability_bin_coord,
    )

    assert_array_equal(result[:3], expected)
    expected_bin_coord_points = np.array([0.1, 0.4, 0.7, 0.9], dtype=np.float32)
    expected_bin_coord_bounds = np.array(
        [[0.0, 0.2], [0.2, 0.6], [0.6, 0.8], [0.8, 1.0]], dtype=np.float32,
    )
    assert_allclose(expected_bin_coord_points, result[3].points)
    assert_allclose(expected_bin_coord_bounds, result[3].bounds)


def test_cub_one_undersampled_bin_upper_neighbour(probability_bin_coord):
    """Test for one under-sampled bin that is combined with its upper
    neighbour."""
    obs_count = np.array([0, 500, 50, 750, 1000], dtype=np.float32)
    forecast_probability_sum = np.array([0, 500, 50, 750, 1000], dtype=np.float32)
    forecast_count = np.array([1000, 2000, 100, 1000, 1000], dtype=np.float32)

    expected = np.array(
        [
            [0, 500, 800, 1000],  # Observation count
            [0, 500, 800, 1000],  # Sum of forecast probability
            [1000, 2000, 1100, 1000],  # Forecast count
        ]
    )
    result = Plugin()._combine_undersampled_bins(
        obs_count, forecast_probability_sum, forecast_count, probability_bin_coord,
    )

    assert_array_equal(result[:3], expected)
    expected_bin_coord_points = np.array([0.1, 0.3, 0.6, 0.9], dtype=np.float32)
    expected_bin_coord_bounds = np.array(
        [[0.0, 0.2], [0.2, 0.4], [0.4, 0.8], [0.8, 1.0]], dtype=np.float32,
    )
    assert_allclose(expected_bin_coord_points, result[3].points)
    assert_allclose(expected_bin_coord_bounds, result[3].bounds)


def test_cub_two_undersampled_bins(probability_bin_coord):
    """Test when two bins are under-sampled."""
    obs_count = np.array([0, 12, 250, 75, 250], dtype=np.float32)
    forecast_probability_sum = np.array([0, 12, 250, 75, 250], dtype=np.float32)
    forecast_count = np.array([1000, 50, 500, 100, 250], dtype=np.float32)

    expected = np.array(
        [
            [0, 262, 325],  # Observation count
            [0, 262, 325],  # Sum of forecast probability
            [1000, 550, 350],  # Forecast count
        ]
    )
    result = Plugin()._combine_undersampled_bins(
        obs_count, forecast_probability_sum, forecast_count, probability_bin_coord,
    )

    assert_array_equal(result[:3], expected)
    expected_bin_coord_points = np.array([0.1, 0.4, 0.8], dtype=np.float32)
    expected_bin_coord_bounds = np.array(
        [[0.0, 0.2], [0.2, 0.6], [0.6, 1.0]], dtype=np.float32,
    )
    assert_allclose(expected_bin_coord_points, result[3].points)
    assert_allclose(expected_bin_coord_bounds, result[3].bounds)


def test_cub_two_equal_undersampled_bins(probability_bin_coord):
    """Test when two bins are under-sampled and the under-sampled bins have
    an equal forecast count."""
    obs_count = np.array([0, 25, 250, 75, 250], dtype=np.float32)
    forecast_probability_sum = np.array([0, 25, 250, 75, 250], dtype=np.float32)
    forecast_count = np.array([1000, 100, 500, 100, 250], dtype=np.float32)

    expected = np.array(
        [
            [0, 275, 325],  # Observation count
            [0, 275, 325],  # Sum of forecast probability
            [1000, 600, 350],  # Forecast count
        ]
    )

    result = Plugin()._combine_undersampled_bins(
        obs_count, forecast_probability_sum, forecast_count, probability_bin_coord,
    )

    assert_array_equal(result[:3], expected)
    expected_bin_coord_points = np.array([0.1, 0.4, 0.8], dtype=np.float32)
    expected_bin_coord_bounds = np.array(
        [[0.0, 0.2], [0.2, 0.6], [0.6, 1.0]], dtype=np.float32,
    )
    assert_allclose(expected_bin_coord_points, result[3].points)
    assert_allclose(expected_bin_coord_bounds, result[3].bounds)


def test_cub_three_equal_undersampled_bin_neighbours(probability_bin_coord):
    """Test when three neighbouring bins are under-sampled."""
    obs_count = np.array([0, 25, 50, 75, 250], dtype=np.float32)
    forecast_probability_sum = np.array([0, 25, 50, 75, 250], dtype=np.float32)
    forecast_count = np.array([1000, 100, 100, 100, 250], dtype=np.float32)

    expected = np.array(
        [
            [0, 150, 250],  # Observation count
            [0, 150, 250],  # Sum of forecast probability
            [1000, 300, 250],  # Forecast count
        ]
    )

    result = Plugin()._combine_undersampled_bins(
        obs_count, forecast_probability_sum, forecast_count, probability_bin_coord,
    )

    assert_array_equal(result[:3], expected)
    expected_bin_coord_points = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    expected_bin_coord_bounds = np.array(
        [[0.0, 0.2], [0.2, 0.8], [0.8, 1.0]], dtype=np.float32,
    )
    assert_allclose(expected_bin_coord_points, result[3].points)
    assert_allclose(expected_bin_coord_bounds, result[3].bounds)


def test_cbp_one_non_monotonic_bin_pair(
    default_obs_counts,
    default_fcst_counts,
    probability_bin_coord,
    expected_enforced_monotonic,
):
    """Test one bin pair is combined, if one bin pair is non-monotonic."""
    obs_count = np.array([0, 250, 500, 1000, 750], dtype=np.float32)
    forecast_probability_sum = default_obs_counts
    result = Plugin()._combine_bin_pair(
        obs_count, forecast_probability_sum, default_fcst_counts, probability_bin_coord,
    )
    assert_array_equal(result[:3], expected_enforced_monotonic)
    expected_bin_coord_points = np.array([0.1, 0.3, 0.5, 0.8], dtype=np.float32)
    expected_bin_coord_bounds = np.array(
        [[0.0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 1.0]], dtype=np.float32,
    )
    assert_allclose(expected_bin_coord_points, result[3].points)
    assert_allclose(expected_bin_coord_bounds, result[3].bounds)


def test_cbp_two_non_monotonic_bin_pairs(
    default_obs_counts,
    default_fcst_counts,
    probability_bin_coord,
    expected_enforced_monotonic,
):
    """Test one bin pair is combined, if two bin pairs are non-monotonic.
    As only a single bin pair is combined, the resulting observation
    count will still yield a non-monotonic observation frequency."""
    obs_count = np.array([0, 750, 500, 1000, 750], dtype=np.float32)
    forecast_probability_sum = default_obs_counts
    expected_enforced_monotonic[0][1] = 750  # Amend observation count
    result = Plugin()._combine_bin_pair(
        obs_count, forecast_probability_sum, default_fcst_counts, probability_bin_coord,
    )
    assert_array_equal(result[:3], expected_enforced_monotonic)
    expected_bin_coord_points = np.array([0.1, 0.3, 0.5, 0.8], dtype=np.float32)
    expected_bin_coord_bounds = np.array(
        [[0.0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 1.0]], dtype=np.float32,
    )
    assert_allclose(expected_bin_coord_points, result[3].points)
    assert_allclose(expected_bin_coord_bounds, result[3].bounds)


def test_acof_monotonic(default_fcst_counts):
    """Test no change to observation frequency, if already monotonic."""
    obs_count = np.array([0, 0, 250, 500, 750], dtype=np.float32)
    result = Plugin()._assume_constant_observation_frequency(
        obs_count, default_fcst_counts,
    )
    assert_array_equal(result.data, obs_count)


def test_acof_non_monotonic_equal_forecast_count(default_fcst_counts):
    """Test enforcement of monotonicity for observation frequency."""
    obs_count = np.array([0, 750, 500, 1000, 750], dtype=np.float32)
    expected_result = np.array([0, 750, 750, 1000, 1000], dtype=np.float32)
    result = Plugin()._assume_constant_observation_frequency(
        obs_count, default_fcst_counts,
    )
    assert_array_equal(result.data, expected_result)


def test_acof_non_monotonic_lower_forecast_count_on_left():
    """Test enforcement of monotonicity for observation frequency."""
    obs_count = np.array([0, 750, 500, 1000, 750], dtype=np.float32)
    forecast_count = np.array([500, 1000, 1000, 1000, 1000], dtype=np.float32)
    expected_result = np.array([0, 500, 500, 750, 750], dtype=np.float32)
    result = Plugin()._assume_constant_observation_frequency(obs_count, forecast_count)
    assert_array_equal(result.data, expected_result)


def test_acof_non_monotonic_higher_forecast_count_on_left():
    """Test enforcement of monotonicity for observation frequency."""
    obs_count = np.array([0, 750, 500, 1000, 75], dtype=np.float32)
    forecast_count = np.array([1000, 1000, 1000, 1000, 100], dtype=np.float32)
    expected_result = np.array([0, 750, 750, 1000, 100], dtype=np.float32)
    result = Plugin()._assume_constant_observation_frequency(obs_count, forecast_count)
    assert_array_equal(result.data, expected_result)


def test_emcam_combine_undersampled_bins_monotonic(reliability_table_slice):
    """Test expected values are returned when a bin is below the minimum
    forecast count when the observed frequency is monotonic."""

    expected_data = np.array(
        [[0, 250, 425, 1000], [0, 250, 425, 1000], [1000, 1000, 600, 1000]]
    )
    expected_bin_coord_points = np.array([0.1, 0.3, 0.6, 0.9], dtype=np.float32)
    expected_bin_coord_bounds = np.array(
        [[0.0, 0.2], [0.2, 0.4], [0.4, 0.8], [0.8, 1.0]], dtype=np.float32,
    )
    reliability_table_slice.data = np.array(
        [
            [0, 250, 50, 375, 1000],  # Observation count
            [0, 250, 50, 375, 1000],  # Sum of forecast probability
            [1000, 1000, 100, 500, 1000],  # Forecast count
        ],
        dtype=np.float32,
    )
    result = Plugin()._enforce_min_count_and_montonicity(reliability_table_slice.copy())
    assert_array_equal(result.data, expected_data)
    assert_allclose(result.coord("probability_bin").points, expected_bin_coord_points)
    assert_allclose(result.coord("probability_bin").bounds, expected_bin_coord_bounds)


def test_emcam_combine_undersampled_bins_non_monotonic(reliability_table_slice):
    """Test expected values are returned when a bin is below the minimum
    forecast count when the observed frequency is non-monotonic."""

    expected_data = np.array([[1000, 425, 1000], [1000, 425, 1000], [2000, 600, 1000]])
    expected_bin_coord_points = np.array([0.2, 0.6, 0.9], dtype=np.float32)
    expected_bin_coord_bounds = np.array(
        [[0.0, 0.4], [0.4, 0.8], [0.8, 1.0]], dtype=np.float32,
    )
    reliability_table_slice.data = np.array(
        [
            [750, 250, 50, 375, 1000],  # Observation count
            [750, 250, 50, 375, 1000],  # Sum of forecast probability
            [1000, 1000, 100, 500, 1000],  # Forecast count
        ],
        dtype=np.float32,
    )

    result = Plugin()._enforce_min_count_and_montonicity(reliability_table_slice.copy())
    assert_array_equal(result.data, expected_data)
    assert_allclose(result.coord("probability_bin").points, expected_bin_coord_points)
    assert_allclose(result.coord("probability_bin").bounds, expected_bin_coord_bounds)


def test_emcam_highest_bin_non_monotonic(reliability_table_slice):
    """Test expected values are returned where the highest observation
    count bin is non-monotonic."""

    expected_data = np.array(
        [[0, 250, 500, 1750], [0, 250, 500, 1750], [1000, 1000, 1000, 2000]]
    )
    expected_bin_coord_points = np.array([0.1, 0.3, 0.5, 0.8], dtype=np.float32)
    expected_bin_coord_bounds = np.array(
        [[0.0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 1.0]], dtype=np.float32,
    )
    reliability_table_slice.data = np.array(
        [
            [0, 250, 500, 1000, 750],  # Observation count
            [0, 250, 500, 750, 1000],  # Sum of forecast probability
            [1000, 1000, 1000, 1000, 1000],  # Forecast count
        ]
    )

    result = Plugin()._enforce_min_count_and_montonicity(reliability_table_slice.copy())
    assert_array_equal(result.data, expected_data)
    assert_allclose(result.coord("probability_bin").points, expected_bin_coord_points)
    assert_allclose(result.coord("probability_bin").bounds, expected_bin_coord_bounds)


def test_emcam_central_bin_non_monotonic(reliability_table_slice):
    """Test expected values are returned where a central observation
    count bin is non-monotonic."""
    expected_data = np.array(
        [[0, 750, 750, 1000], [0, 750, 750, 1000], [1000, 2000, 1000, 1000]]
    )
    expected_bin_coord_points = np.array([0.1, 0.4, 0.7, 0.9], dtype=np.float32)
    expected_bin_coord_bounds = np.array(
        [[0.0, 0.2], [0.2, 0.6], [0.6, 0.8], [0.8, 1.0]], dtype=np.float32,
    )
    reliability_table_slice.data = np.array(
        [
            [0, 500, 250, 750, 1000],  # Observation count
            [0, 250, 500, 750, 1000],  # Sum of forecast probability
            [1000, 1000, 1000, 1000, 1000],  # Forecast count
        ]
    )

    result = Plugin()._enforce_min_count_and_montonicity(reliability_table_slice.copy())
    assert_array_equal(result.data, expected_data)
    assert_allclose(result.coord("probability_bin").points, expected_bin_coord_points)
    assert_allclose(result.coord("probability_bin").bounds, expected_bin_coord_bounds)


def test_emcam_upper_bins_non_monotonic(reliability_table_slice):
    """Test expected values are returned where the upper observation
    count bins are non-monotonic."""
    expected_data = np.array(
        [[0, 375, 375, 750], [0, 250, 500, 1750], [1000, 1000, 1000, 2000]]
    )
    expected_bin_coord_points = np.array([0.1, 0.3, 0.5, 0.8], dtype=np.float32)
    expected_bin_coord_bounds = np.array(
        [[0.0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 1.0]], dtype=np.float32,
    )
    reliability_table_slice.data = np.array(
        [
            [0, 1000, 750, 500, 250],  # Observation count
            [0, 250, 500, 750, 1000],  # Sum of forecast probability
            [1000, 1000, 1000, 1000, 1000],  # Forecast count
        ]
    )

    result = Plugin()._enforce_min_count_and_montonicity(reliability_table_slice.copy())
    assert_array_equal(result.data, expected_data)
    assert_allclose(result.coord("probability_bin").points, expected_bin_coord_points)
    assert_allclose(result.coord("probability_bin").bounds, expected_bin_coord_bounds)


def test_emcam_lowest_bin_non_monotonic(reliability_table_slice):
    """Test expected values are returned where the lowest observation
    count bin is non-monotonic."""
    expected_data = np.array(
        [[1000, 500, 500, 750], [250, 500, 750, 1000], [2000, 1000, 1000, 1000]]
    )

    expected_bin_coord_points = np.array([0.2, 0.5, 0.7, 0.9], dtype=np.float32)

    expected_bin_coord_bounds = np.array(
        [[0.0, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0]], dtype=np.float32,
    )

    reliability_table_slice.data = np.array(
        [
            [1000, 0, 250, 500, 750],  # Observation count
            [0, 250, 500, 750, 1000],  # Sum of forecast probability
            [1000, 1000, 1000, 1000, 1000],  # Forecast count
        ]
    )
    result = Plugin()._enforce_min_count_and_montonicity(reliability_table_slice.copy())
    assert_array_equal(result.data, expected_data)
    assert_allclose(result.coord("probability_bin").points, expected_bin_coord_points)
    assert_allclose(result.coord("probability_bin").bounds, expected_bin_coord_bounds)


def test_process_no_change_agg(reliability_table_agg):
    """Test with no changes required to preserve monotonicity."""
    result = Plugin().process(reliability_table_agg.copy())
    assert_array_equal(result[0].data, reliability_table_agg[0].data)
    assert result[0].coords() == reliability_table_agg[0].coords()
    assert_array_equal(result[1].data, reliability_table_agg[1].data)
    assert result[1].coords() == reliability_table_agg[1].coords()


def test_process_no_change_point(create_rel_tables_point):
    """Test with no changes required to preserve monotonicity. Parameterized
    using `create_rel_tables` fixture."""
    rel_table = create_rel_tables_point.table
    result = Plugin(point_by_point=True).process(rel_table.copy())

    assert len(result) == 18
    expected = rel_table.data[create_rel_tables_point.indices0]
    assert all([np.array_equal(cube.data, expected) for cube in result[:9]])
    expected = rel_table.data[create_rel_tables_point.indices2]
    assert all([np.array_equal(cube.data, expected) for cube in result[9:]])

    coords_exclude = ["latitude", "longitude", "spot_index", "wmo_id"]
    coords_table = [c for c in rel_table[0].coords() if c.name() not in coords_exclude]
    # Ensure coords are in the same order
    coords_result = [result[0].coord(c.name()) for c in coords_table]
    assert coords_table == coords_result


def test_process_undersampled_non_monotonic_point(create_rel_tables_point):
    """Test expected values are returned when one slice contains a bin that is
    below the minimum forecast count, whilst the observed frequency is
    non-monotonic. Test that remaining data, which requires no change, is not
    changed. Parameterized using `create_rel_tables` fixture."""
    expected_data = np.array([[1000, 425, 1000], [1000, 425, 1000], [2000, 600, 1000]])
    expected_bin_coord_points = np.array([0.2, 0.6, 0.9], dtype=np.float32)
    expected_bin_coord_bounds = np.array(
        [[0.0, 0.4], [0.4, 0.8], [0.8, 1.0]], dtype=np.float32,
    )
    rel_table = create_rel_tables_point.table
    rel_table.data[create_rel_tables_point.indices0] = np.array(
        [
            [750, 250, 50, 375, 1000],  # Observation count
            [750, 250, 50, 375, 1000],  # Sum of forecast probability
            [1000, 1000, 100, 500, 1000],  # Forecast count
        ]
    )

    result = Plugin(point_by_point=True).process(rel_table.copy())
    assert_array_equal(result[0].data, expected_data)
    assert_allclose(
        result[0].coord("probability_bin").points, expected_bin_coord_points
    )
    assert_allclose(
        result[0].coord("probability_bin").bounds, expected_bin_coord_bounds
    )
    # Check the unchanged data remains unchanged
    expected = rel_table.data[create_rel_tables_point.indices1]
    assert all([np.array_equal(cube.data, expected) for cube in result[1:9]])
    expected = rel_table.data[create_rel_tables_point.indices2]
    assert all([np.array_equal(cube.data, expected) for cube in result[9:]])
