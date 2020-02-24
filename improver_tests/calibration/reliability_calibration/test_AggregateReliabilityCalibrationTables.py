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

import types
import numpy as np
from numpy.testing import assert_array_equal
import iris
from iris.exceptions import MergeError

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
            self.forecast_1, self.expected_threshold_coord)
        self.reliability_cube = reliability_cube_format.copy(
            data=self.expected_table)
        self.lat_lon_collapse = np.array([[0., 0., 1., 2., 1.],
                                          [0., 0.375, 1.5, 1.625, 1.],
                                          [1., 2., 3., 2., 1.]])


class Test__init__(unittest.TestCase):

    """Test the __init__ method."""

    def test_basic(self):
        """Test that the init sets up the required variables."""
        plugin = Plugin()
        self.assertEqual(plugin.cube_index_coord, 'cube_coord')
        self.assertIsInstance(plugin.merge_coord, types.LambdaType)
        self.assertIsInstance(plugin.merge_coord(0), iris.coords.DimCoord)


class Test__repr__(unittest.TestCase):

    """Test the __repr__ method."""

    def test_basic(self):
        """Test repr is as expected."""
        self.assertEqual(
            str(Plugin()), '<AggregateReliabilityCalibrationTables>')


class Test__construct_single_cube(Test_Aggregation):

    """Test the _construct_single_cube method."""

    def test_construction_of_cube(self):
        """Test that the method returns a single cube when given multiple
        input cubes. The length of the leading coordinate of the returned
        cube will be equal in length to the number of cubes passed in."""

        plugin = Plugin()
        cube_index_coord = plugin.cube_index_coord

        result = plugin._construct_single_cube([self.reliability_cube,
                                                self.reliability_cube])
        self.assertIsInstance(result.coord(cube_index_coord),
                              iris.coords.DimCoord)
        assert_array_equal(result.coord(cube_index_coord).points, [0, 1])

    def test_unmatched_cubes(self):
        """Test that an exception is raised when cubes that cannot be merged
        are provided. This exception is raised by iris."""

        plugin = Plugin()
        second_cube = self.reliability_cube.copy()
        second_cube.add_aux_coord(
            iris.coords.AuxCoord([0], long_name='unmatched', units=1))

        msg = "failed to merge into a single cube"
        with self.assertRaisesRegex(MergeError, msg):
            plugin._construct_single_cube([self.reliability_cube, second_cube])


class Test_process(Test_Aggregation):

    """Test the process method."""

    def test_aggregating_multiple_cubes(self):
        """Test of aggregating two cubes without any additional coordinate
        collapsing."""

        plugin = Plugin()
        result = plugin.process([self.reliability_cube, self.reliability_cube])

        assert_array_equal(result.data, self.expected_table * 2)
        assert_array_equal(result.shape, (3, 5, 3, 3))

    def test_aggregating_over_single_cube_coordinates(self):
        """Test of aggregating over coordinates of a single cube. In this
        instance the latitude and longitude coordinates are collapsed."""

        plugin = Plugin()
        result = plugin.process([self.reliability_cube],
                                coordinates=['latitude', 'longitude'])
        assert_array_equal(result.data, self.lat_lon_collapse)

    def test_aggregating_over_cubes_and_coordinates(self):
        """Test of aggregating over coordinates of a single cube. In this
        instance the latitude and longitude coordinates are collapsed."""

        plugin = Plugin()
        result = plugin.process([self.reliability_cube, self.reliability_cube],
                                coordinates=['latitude', 'longitude'])
        assert_array_equal(result.data, self.lat_lon_collapse * 2)

    def test_single_cube(self):
        """Test the plugin returns an unaltered cube if only one is passed in
        and no coordinates are given."""

        plugin = Plugin()

        expected = self.reliability_cube.copy()
        result = plugin.process([self.reliability_cube])

        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
