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
"""Unit tests for GenerateTimeLaggedEnsemble plugin."""

import unittest
from datetime import datetime as dt

import iris
import numpy as np
from iris.tests import IrisTest

from improver.utilities.time_lagging import GenerateTimeLaggedEnsemble

from ..set_up_test_cubes import set_up_variable_cube


class Test_process(IrisTest):

    """Test interpolation of cubes to intermediate times using the plugin."""

    def setUp(self):
        """Set up the test inputs."""
        # Create a template cube. This cube has a forecast_reference_time of
        # 20151123T0400Z, a forecast period of T+4 and a
        # validity time of 2015-11-23 07:00:00
        data = np.ones((3, 5, 5), dtype=np.float32)
        self.input_cube = set_up_variable_cube(
            data, time=dt(2015, 11, 23, 7), frt=dt(2015, 11, 23, 3))

        # Create a second cube from a later forecast with a different set of
        # realizations.
        self.input_cube2 = set_up_variable_cube(
            data, time=dt(2015, 11, 23, 7), frt=dt(2015, 11, 23, 4),
            realizations=np.array([3, 4, 5], dtype=np.int32))

        # Put the two cubes in a cubelist ready to use in the plugin.
        self.input_cubelist = iris.cube.CubeList(
            [self.input_cube, self.input_cube2])

        # Set up expected forecast time outputs (from most recent forecast)
        self.expected_fp = self.input_cube2.coord("forecast_period").points[0]
        self.expected_frt = (
            self.input_cube2.coord("forecast_reference_time").points[0])

    def test_return_type(self):
        """Test that an iris cube is returned."""
        result = GenerateTimeLaggedEnsemble().process(self.input_cubelist)
        self.assertIsInstance(result, iris.cube.Cube)

    def test_basic(self):
        """Test that the expected metadata is correct after a simple test"""
        result = GenerateTimeLaggedEnsemble().process(
            self.input_cubelist)
        expected_realizations = [0, 1, 2, 3, 4, 5]
        self.assertArrayAlmostEqual(
            result.coord("forecast_period").points, self.expected_fp)
        self.assertArrayAlmostEqual(
            result.coord("forecast_reference_time").points, self.expected_frt)
        self.assertArrayAlmostEqual(
            result.coord("realization").points, expected_realizations)
        self.assertEqual(
            result.coord("realization").dtype, np.int32)

    def test_realizations(self):
        """Test that the expected metadata is correct with a different
           realizations"""
        self.input_cube2.coord("realization").points = np.array(
            [6, 7, 8], dtype=np.int32)
        result = GenerateTimeLaggedEnsemble().process(
            self.input_cubelist)
        expected_realizations = [0, 1, 2, 6, 7, 8]
        self.assertArrayAlmostEqual(
            result.coord("forecast_period").points, self.expected_fp)
        self.assertArrayAlmostEqual(
            result.coord("forecast_reference_time").points, self.expected_frt)
        self.assertArrayAlmostEqual(
            result.coord("realization").points, expected_realizations)
        self.assertEqual(
            result.coord("realization").dtype, np.int32)

    def test_duplicate_realizations(self):
        """Test that the expected metadata is correct with different
           realizations and that realizations are renumbered if a
           duplicate is found"""
        self.input_cube2.coord("realization").points = np.array([0, 7, 8])
        result = GenerateTimeLaggedEnsemble().process(
            self.input_cubelist)
        expected_realizations = [0, 1, 2, 3, 4, 5]
        self.assertArrayAlmostEqual(
            result.coord("forecast_period").points, self.expected_fp)
        self.assertArrayAlmostEqual(
            result.coord("forecast_reference_time").points, self.expected_frt)
        self.assertArrayAlmostEqual(
            result.coord("realization").points, expected_realizations)
        self.assertEqual(
            result.coord("realization").dtype, np.int32)

    def test_duplicate_realizations_more_input_cubes(self):
        """Test that the expected metadata is correct with different
           realizations and that realizations are renumbered if a
           duplicate is found, with 3 input cubes."""
        self.input_cube2.coord("realization").points = np.array([6, 7, 8])
        input_cube3 = self.input_cube2.copy()
        input_cube3.coord("forecast_reference_time").points = np.array(
            input_cube3.coord("forecast_reference_time").points[0] + 3600)
        input_cube3.coord("forecast_period").points = np.array(
            input_cube3.coord("forecast_period").points[0] - 3600)
        input_cube3.coord("realization").points = np.array([7, 8, 9])
        input_cubelist = iris.cube.CubeList(
            [self.input_cube, self.input_cube2, input_cube3])
        result = GenerateTimeLaggedEnsemble().process(
            input_cubelist)
        expected_forecast_period = self.expected_fp - 3600
        expected_forecast_ref_time = self.expected_frt + 3600
        expected_realizations = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.assertArrayAlmostEqual(
            result.coord("forecast_period").points, expected_forecast_period)
        self.assertArrayAlmostEqual(
            result.coord("forecast_reference_time").points,
            expected_forecast_ref_time)
        self.assertArrayAlmostEqual(
            result.coord("realization").points, expected_realizations)
        self.assertEqual(
            result.coord("realization").dtype, np.int32)

    def test_attributes(self):
        """Test what happens if input cubes have different attributes"""
        self.input_cube.attributes = {'institution': 'Met Office',
                                      'history': 'Process 1'}
        self.input_cube2.attributes = {'institution': 'Met Office',
                                       'history': 'Process 2'}
        result = GenerateTimeLaggedEnsemble().process(
            self.input_cubelist)
        expected_attributes = {'institution': 'Met Office'}
        self.assertEqual(result.attributes, expected_attributes)

    def test_single_cube(self):
        """Test only one input cube returns cube unchanged"""
        input_cubelist = iris.cube.CubeList([self.input_cube])
        expected_cube = self.input_cube.copy()
        result = GenerateTimeLaggedEnsemble().process(input_cubelist)
        self.assertEqual(result, expected_cube)

    def test_non_monotonic_realizations(self):
        """Test handling of case where realization coordinates cannot be
        directly concatenated into a monotonic coordinate"""
        data = 275.*np.ones((3, 3, 3), dtype=np.float32)
        cycletime = dt(2019, 6, 24, 9)
        cube1 = set_up_variable_cube(
            data, realizations=np.array([15, 16, 17], dtype=np.int32),
            time=cycletime, frt=dt(2019, 6, 24, 8))
        cube2 = set_up_variable_cube(
            data, realizations=np.array([0, 18, 19], dtype=np.int32),
            time=cycletime, frt=cycletime)

        expected_cube = set_up_variable_cube(
            275.*np.ones((6, 3, 3), dtype=np.float32),
            realizations=np.array([0, 15, 16, 17, 18, 19], dtype=np.int32),
            time=cycletime, frt=cycletime)

        input_cubelist = iris.cube.CubeList([cube1, cube2])
        result = GenerateTimeLaggedEnsemble().process(input_cubelist)
        self.assertEqual(result, expected_cube)
        self.assertEqual(
            result.coord("realization").dtype, np.int32)


if __name__ == '__main__':
    unittest.main()
