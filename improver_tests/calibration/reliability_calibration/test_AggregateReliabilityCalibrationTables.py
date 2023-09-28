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
"""Unit tests for the AggregateReliabilityCalibrationTables plugin."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from improver.calibration.reliability_calibration import (
    AggregateReliabilityCalibrationTables as Plugin,
)


def test_frt_coord_valid_bounds(reliability_cube, different_frt):
    """Test that no exception is raised if the input cubes have forecast
    reference time bounds that do not overlap."""
    plugin = Plugin()
    plugin._check_frt_coord([reliability_cube, different_frt])


def test_frt_coord_matching_bounds(reliability_cube, different_frt):
    """Test that no exception is raised in the input cubes if a cube
    contains matching bounds."""
    lower_bound = reliability_cube.coord("forecast_reference_time").bounds[0][0]
    reliability_cube.coord("forecast_reference_time").bounds = [
        [lower_bound, lower_bound]
    ]
    plugin = Plugin()
    plugin._check_frt_coord([reliability_cube, different_frt])


def test_frt_coord_invalid_bounds(reliability_cube, overlapping_frt):
    """Test that an exception is raised if the input cubes have forecast
    reference time bounds that overlap."""

    plugin = Plugin()
    msg = "Reliability calibration tables have overlapping"
    with pytest.raises(ValueError, match=msg):
        plugin._check_frt_coord([reliability_cube, overlapping_frt])


def test_process_aggregating_multiple_cubes(
    reliability_cube, different_frt, expected_table
):
    """Test of aggregating two cubes without any additional coordinate
    collapsing."""
    frt = "forecast_reference_time"
    expected_points = different_frt.coord(frt).points
    expected_bounds = [
        [
            reliability_cube.coord(frt).bounds[0][0],
            different_frt.coord(frt).bounds[-1][1],
        ]
    ]
    plugin = Plugin()
    result = plugin.process([reliability_cube, different_frt])
    assert_array_equal(result.data, expected_table * 2)
    assert_array_equal(result.shape, (3, 5, 3, 3))
    assert_array_equal(result.coord(frt).points, expected_points)
    assert_array_equal(result.coord(frt).bounds, expected_bounds)


def test_process_aggregating_cubes_with_overlapping_frt(
    reliability_cube, overlapping_frt
):
    """Test that attempting to aggregate reliability calibration tables
    with overlapping forecast reference time bounds raises an exception.
    The presence of overlapping forecast reference time bounds indicates
    that the same forecast data has contributed to both tables, thus
    aggregating them would double count these contributions."""

    plugin = Plugin()
    msg = "Reliability calibration tables have overlapping"
    with pytest.raises(ValueError, match=msg):
        plugin.process([reliability_cube, overlapping_frt])


def test_process_aggregating_over_single_cube_coordinates(
    reliability_cube, lat_lon_collapse
):
    """Test of aggregating over coordinates of a single cube. In this
    instance the latitude and longitude coordinates are collapsed."""

    frt = "forecast_reference_time"
    expected_points = reliability_cube.coord(frt).points
    expected_bounds = reliability_cube.coord(frt).bounds

    plugin = Plugin()
    result = plugin.process([reliability_cube], coordinates=["latitude", "longitude"])
    assert_array_equal(result.data, lat_lon_collapse)
    assert_array_equal(result.coord(frt).points, expected_points)
    assert_array_equal(result.coord(frt).bounds, expected_bounds)


def test_process_aggregating_over_cubes_and_coordinates(
    reliability_cube, different_frt, lat_lon_collapse
):
    """Test of aggregating over coordinates and cubes in a single call. In
    this instance the latitude and longitude coordinates are collapsed and
    the values from two input cube combined."""

    frt = "forecast_reference_time"
    expected_points = different_frt.coord(frt).points
    expected_bounds = [
        [
            reliability_cube.coord(frt).bounds[0][0],
            different_frt.coord(frt).bounds[-1][1],
        ]
    ]

    plugin = Plugin()
    result = plugin.process(
        [reliability_cube, different_frt],
        coordinates=["latitude", "longitude"],
    )
    assert_array_equal(result.data, lat_lon_collapse * 2)
    assert_array_equal(result.coord(frt).points, expected_points)
    assert_array_equal(result.coord(frt).bounds, expected_bounds)


def test_process_aggregating_over_masked_cubes_and_coordinates(
    masked_different_frt, masked_reliability_cube
):
    """Test of aggregating over coordinates and cubes in a single call
    using a masked reliability table. In this instance the latitude and
    longitude coordinates are collapsed and the values from two input cube
    combined."""

    frt = "forecast_reference_time"
    expected_points = masked_different_frt.coord(frt).points
    expected_bounds = [
        [
            masked_reliability_cube.coord(frt).bounds[0][0],
            masked_different_frt.coord(frt).bounds[-1][1],
        ]
    ]
    expected_result = np.array(
        [
            [0.0, 0.0, 2.0, 4.0, 2.0],
            [0.0, 0.625, 2.625, 3.25, 2.0],
            [0.0, 3.0, 5.0, 4.0, 2.0],
        ]
    )

    plugin = Plugin()
    result = plugin.process(
        [masked_reliability_cube, masked_different_frt],
        coordinates=["latitude", "longitude"],
    )
    assert isinstance(result.data, np.ma.MaskedArray)
    assert_array_equal(result.data, expected_result)
    assert_array_equal(result.coord(frt).points, expected_points)
    assert_array_equal(result.coord(frt).bounds, expected_bounds)


def test_single_cube(reliability_cube):
    """Test the plugin returns an unaltered cube if only one is passed in
    and no coordinates are given."""

    plugin = Plugin()
    expected = reliability_cube.copy()
    result = plugin.process([reliability_cube])
    assert result == expected
