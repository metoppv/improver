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
"""Unit tests for the AggregateReliabilityCalibrationTables plugin."""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from improver.calibration.reliability_calibration import (
    AggregateReliabilityCalibrationTables as Plugin)

from improver.calibration.reliability_calibration import (
    ConstructReliabilityCalibrationTables as CalPlugin)
from improver_tests.calibration.reliability_calibration.\
    test_ConstructReliabilityCalibrationTables import Test_Setup


class Test_Aggregation(Test_Setup):

    """Test class for the Test_AggregateReliabilityCalibrationTables tests,
    setting up cubes to use as inputs."""

    def setUp(self):
        """Create reliability calibration tables for testing."""

        super().setUp()
        reliability_cube_format = CalPlugin()._create_reliability_table_cube(
            self.forecasts, self.expected_threshold_coord)
        self.reliability_cube = reliability_cube_format.copy(
            data=self.expected_table)
        self.different_frt = self.reliability_cube.copy()
        new_frt = self.different_frt.coord('forecast_reference_time')
        new_frt.points = new_frt.points + 48*3600
        new_frt.bounds = new_frt.bounds + 48*3600

        self.overlapping_frt = self.reliability_cube.copy()
        new_frt = self.overlapping_frt.coord('forecast_reference_time')
        new_frt.points = new_frt.points + 6*3600
        new_frt.bounds = new_frt.bounds + 6*3600

        self.lat_lon_collapse = np.array([[0., 0., 1., 2., 1.],
                                          [0., 0.375, 1.5, 1.625, 1.],
                                          [1., 2., 3., 2., 1.]])


class Test__repr__(unittest.TestCase):

    """Test the __repr__ method."""

    def test_basic(self):
        """Test repr is as expected."""
        self.assertEqual(
            str(Plugin()), '<AggregateReliabilityCalibrationTables>')


class Test__check_frt_coord(Test_Aggregation):

    """Test the _check_frt_coord method."""

    def test_valid_bounds(self):
        """Test that no exception is raised if the input cubes have forecast
        reference time bounds that do not overlap."""

        plugin = Plugin()
        plugin._check_frt_coord([self.reliability_cube, self.different_frt])

    def test_invalid_bounds(self):
        """Test that an exception is raised if the input cubes have forecast
        reference time bounds that overlap."""

        plugin = Plugin()
        msg = "Reliability calibration tables have overlapping"
        with self.assertRaisesRegex(ValueError, msg):
            plugin._check_frt_coord([self.reliability_cube,
                                     self.overlapping_frt])


class Test_process(Test_Aggregation):

    """Test the process method."""

    def test_aggregating_multiple_cubes(self):
        """Test of aggregating two cubes without any additional coordinate
        collapsing."""

        frt = 'forecast_reference_time'
        expected_points = self.different_frt.coord(frt).points
        expected_bounds = [[self.reliability_cube.coord(frt).bounds[0][0],
                            self.different_frt.coord(frt).bounds[-1][1]]]

        plugin = Plugin()
        result = plugin.process([self.reliability_cube, self.different_frt])
        assert_array_equal(result.data, self.expected_table * 2)
        assert_array_equal(result.shape, (3, 5, 3, 3))
        self.assertEqual(result.coord(frt).points, expected_points)
        assert_array_equal(result.coord(frt).bounds, expected_bounds)

    def test_aggregating_cubes_with_overlapping_frt(self):
        """Test that attempting to aggregate reliability calibration tables
        with overlapping forecast reference time bounds raises an exception.
        The presence of overlapping forecast reference time bounds indicates
        that the same forecast data has contributed to both tables, thus
        aggregating them would double count these contributions."""

        plugin = Plugin()
        msg = "Reliability calibration tables have overlapping"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process([self.reliability_cube, self.overlapping_frt])

    def test_aggregating_over_single_cube_coordinates(self):
        """Test of aggregating over coordinates of a single cube. In this
        instance the latitude and longitude coordinates are collapsed."""

        frt = 'forecast_reference_time'
        expected_points = self.reliability_cube.coord(frt).points
        expected_bounds = self.reliability_cube.coord(frt).bounds

        plugin = Plugin()
        result = plugin.process([self.reliability_cube],
                                coordinates=['latitude', 'longitude'])
        assert_array_equal(result.data, self.lat_lon_collapse)
        self.assertEqual(result.coord(frt).points, expected_points)
        assert_array_equal(result.coord(frt).bounds, expected_bounds)

    def test_aggregating_over_cubes_and_coordinates(self):
        """Test of aggregating over coordinates and cubes in a single call. In
        this instance the latitude and longitude coordinates are collapsed and
        the values from two input cube combined."""

        frt = 'forecast_reference_time'
        expected_points = self.different_frt.coord(frt).points
        expected_bounds = [[self.reliability_cube.coord(frt).bounds[0][0],
                            self.different_frt.coord(frt).bounds[-1][1]]]

        plugin = Plugin()
        result = plugin.process([self.reliability_cube, self.different_frt],
                                coordinates=['latitude', 'longitude'])
        assert_array_equal(result.data, self.lat_lon_collapse * 2)
        self.assertEqual(result.coord(frt).points, expected_points)
        assert_array_equal(result.coord(frt).bounds, expected_bounds)

    def test_single_cube(self):
        """Test the plugin returns an unaltered cube if only one is passed in
        and no coordinates are given."""

        plugin = Plugin()

        expected = self.reliability_cube.copy()
        result = plugin.process([self.reliability_cube])

        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
