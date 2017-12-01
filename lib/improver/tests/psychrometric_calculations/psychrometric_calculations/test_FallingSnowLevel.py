# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
"""Unit tests for psychrometric_calculations FallingSnowLevel."""

import unittest

import numpy as np

from cf_units import Unit
import iris
from iris.tests import IrisTest

from improver.psychrometric_calculations.psychrometric_calculations import (
    FallingSnowLevel)
from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import set_up_cube
from improver.tests.utilities.test_mathematical_operations import (
    set_up_height_cube)


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(FallingSnowLevel())
        msg = ('<FallingSnowLevel: '
               'precision:0.005, falling_level_threshold:90.0>')
        self.assertEqual(result, msg)


class Test_find_falling_level(IrisTest):

    """Test the find_falling_level method."""

    def setUp(self):
        """Set up arrays."""
        self.wb_int_data = np.array([[[80.0, 80.0], [70.0, 50.0]],
                                     [[90.0, 100.0], [80.0, 60.0]],
                                     [[100.0, 110.0], [90.0, 100.0]]])

        self.orog_data = np.array([[0.0, 0.0], [5.0, 3.0]])
        self.height_points = np.array([5.0, 10.0, 20.0])

    def test_basic(self):
        """Test method returns an array with correct data"""
        plugin = FallingSnowLevel()
        expected = np.array([[10.0, 7.5], [25.0, 20.5]])
        result = plugin.find_falling_level(
            self.wb_int_data, self.orog_data, self.height_points)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, expected)

    def test_outside_range(self):
        """Test method returns an nan if data outside range"""
        plugin = FallingSnowLevel()
        wb_int_data = self.wb_int_data
        wb_int_data[2, 1, 1] = 70.0
        result = plugin.find_falling_level(
            wb_int_data, self.orog_data, self.height_points)
        self.assertTrue(np.isnan(result[1, 1]))


class Test_fill_in_missing_data(IrisTest):

    """Test the fill_in_missing_data method."""

    def test_basic(self):
        """Test method returns an array with correct data"""
        plugin = FallingSnowLevel()
        snow_level_data = np.array([[1.0, 1.0, 2.0],
                                    [1.0, np.nan, 2.0],
                                    [1.0, 2.0, 2.0]])
        expected = np.array([[1.0, 1.0, 2.0],
                             [1.0, 1.5, 2.0],
                             [1.0, 2.0, 2.0]])
        result = plugin.fill_in_missing_data(snow_level_data)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, expected)


class Test_process(IrisTest):

    """Test the FallingSnowLevel processing works"""

    def setUp(self):
        """Set up cubes."""

        temp_vals = [278.0, 280.0, 285.0, 286.0]
        pressure_vals = [93856.0, 95034.0, 96216.0, 97410.0]

        data = np.ones((2, 1, 3, 3))
        relh_data = np.ones((2, 1, 3, 3)) * 0.65

        temperature = set_up_cube(data, 'air_temperature', 'K',
                                  realizations=np.array([0, 1]))
        relative_humidity = set_up_cube(relh_data,
                                        'relative_humidity', '%',
                                        realizations=np.array([0, 1]))
        pressure = set_up_cube(data, 'air_pressure', 'Pa',
                               realizations=np.array([0, 1]))
        self.height_points = np.array([5., 195., 200.])
        self.temperature_cube = set_up_height_cube(
            self.height_points, cube=temperature)
        self.relative_humidity_cube = (
            set_up_height_cube(self.height_points, cube=relative_humidity))
        self.pressure_cube = set_up_height_cube(
            self.height_points, cube=pressure)
        for i in range(0, 3):
            self.temperature_cube.data[i, ::] = temp_vals[i+1]
            self.pressure_cube.data[i, ::] = pressure_vals[i+1]
            # Add hole in middle of data.
            self.temperature_cube.data[i, :, :, 1, 1] = temp_vals[i]
            self.pressure_cube.data[i, :, :, 1, 1] = pressure_vals[i]

        self.orog = iris.cube.Cube(np.zeros((3, 3)),
                                   standard_name='surface_altitude', units='m')
        self.orog.add_dim_coord(
            iris.coords.DimCoord(np.linspace(-45.0, 45.0, 3),
                                 'latitude', units='degrees'), 0)
        self.orog.add_dim_coord(iris.coords.DimCoord(np.linspace(120, 180, 3),
                                                     'longitude',
                                                     units='degrees'), 1)

    def test_basic(self):
        """Test that process returns a cube with the right name and units."""
        result = FallingSnowLevel().process(
            self.temperature_cube, self.relative_humidity_cube,
            self.pressure_cube, self.orog)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), "falling_snow_level_asl")
        self.assertEqual(result.units, Unit('m'))

    def test_data(self):
        """Test that the falling snow level process returns a cube
        containing the expected data."""
        expected = np.ones((2, 3, 3)) * 65.88732723
        result = FallingSnowLevel().process(
            self.temperature_cube, self.relative_humidity_cube,
            self.pressure_cube, self.orog)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)


if __name__ == '__main__':
    unittest.main()
