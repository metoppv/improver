#!/usr/bin/env python
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
"""Module with tests for the ExtendRadarMask plugin."""

import unittest

import iris
import numpy as np
from iris.tests import IrisTest

from improver.nowcasting.utilities import ExtendRadarMask

from ...set_up_test_cubes import set_up_variable_cube


class Test__init_(IrisTest):
    """Test the _init_ method"""

    def test_basic(self):
        """Test initialisation of class"""
        plugin = ExtendRadarMask()
        self.assertSequenceEqual(plugin.coverage_valid, [1, 2])


class Test_process(IrisTest):
    """Test the process method"""

    def setUp(self):
        """Set up some input cubes"""
        rainrate_data = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.4, 0.3, 0.0],
            [0.0, 0.2, 0.6, 0.7, 0.6],
            [0.0, 0.0, 0.4, 0.5, 0.4],
            [0.0, 0.0, 0.1, 0.2, 0.3]], dtype=np.float32)

        rainrate_mask = np.array([
            [True, True, True, True, True],
            [True, False, False, False, False],
            [True, False, False, False, False],
            [True, False, False, False, False],
            [True, False, False, False, False]], dtype=bool)

        rainrate_data = np.ma.MaskedArray(rainrate_data, mask=rainrate_mask)

        self.rainrate = set_up_variable_cube(
            rainrate_data, name='lwe_precipitation_rate', units='mm h-1',
            spatial_grid='equalarea')

        coverage_data = np.array([[0, 0, 0, 0, 0],
                                  [0, 2, 1, 1, 3],
                                  [0, 1, 1, 1, 1],
                                  [0, 2, 1, 1, 1],
                                  [0, 3, 1, 1, 1]], dtype=np.int32)

        self.coverage = set_up_variable_cube(
            coverage_data, name='radar_coverage', units='1',
            spatial_grid='equalarea')

        self.expected_mask = np.array([
            [True, True, True, True, True],
            [True, False, False, False, True],
            [True, False, False, False, False],
            [True, False, False, False, False],
            [True, True, False, False, False]], dtype=bool)

    def test_basic(self):
        """Test processing outputs a cube of precipitation rates"""
        result = ExtendRadarMask().process(self.rainrate, self.coverage)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), self.rainrate.name())

    def test_values(self):
        """Test output cube has expected mask and underlying data is
        unchanged"""
        result = ExtendRadarMask().process(self.rainrate, self.coverage)
        self.assertArrayEqual(result.data.mask, self.expected_mask)
        self.assertArrayEqual(result.data.data, self.rainrate.data.data)

    def test_inputs_unmodified(self):
        """Test the rainrate cube is not modified in place"""
        reference = self.rainrate.copy()
        _ = ExtendRadarMask().process(self.rainrate, self.coverage)
        self.assertEqual(reference, self.rainrate)

    def test_coords_unmatched_error(self):
        """Test error is raised if coordinates do not match"""
        x_points = self.rainrate.coord(axis='x').points
        self.rainrate.coord(axis='x').points = x_points + 100.
        msg = 'Rain rate and coverage composites unmatched'
        with self.assertRaisesRegex(ValueError, msg):
            _ = ExtendRadarMask().process(self.rainrate, self.coverage)


if __name__ == '__main__':
    unittest.main()
