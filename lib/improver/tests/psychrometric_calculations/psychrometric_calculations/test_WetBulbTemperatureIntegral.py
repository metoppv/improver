# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
"""Unit tests for psychrometric_calculations WetBulbTemperatureIntegral."""

import unittest

import numpy as np

from cf_units import Unit
import iris
from iris.tests import IrisTest

from improver.psychrometric_calculations.psychrometric_calculations import (
    WetBulbTemperatureIntegral)
from improver.tests.psychrometric_calculations.psychrometric_calculations.\
    test_WetBulbTemperature import set_up_cubes_for_wet_bulb_temperature
from improver.tests.utilities.test_mathematical_operations import (
    set_up_height_cube)


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(WetBulbTemperatureIntegral())
        msg = ('<WetBulbTemperatureIntegral: <WetBulbTemperature: '
               'precision: 0.005>, <Integration: coord_name_to_integrate: '
               'height, start_point: None, end_point: None, '
               'direction_of_integration: negative>>')
        self.assertEqual(result, msg)


class Test_process(IrisTest):

    """Test the calculation of the wet bulb temperature integral from
    temperature, pressure, and relative humidity information using the
    process function. Integration is calculated in the vertical."""

    def setUp(self):
        """Set up cubes."""
        temperature, pressure, relative_humidity, _ = (
            set_up_cubes_for_wet_bulb_temperature())
        self.height_points = np.array([5., 10., 20.])
        self.temperature_cube = set_up_height_cube(
            self.height_points, cube=temperature)
        self.relative_humidity_cube = (
            set_up_height_cube(self.height_points, cube=relative_humidity))
        self.pressure_cube = set_up_height_cube(
            self.height_points, cube=pressure)

    def test_basic(self):
        """Test that the wet bulb temperature integral returns a cube
        with the expected name."""
        wb_temp, wb_temp_int = WetBulbTemperatureIntegral().process(
            self.temperature_cube, self.relative_humidity_cube,
            self.pressure_cube)
        self.assertIsInstance(wb_temp_int, iris.cube.Cube)
        self.assertEqual(wb_temp_int.name(), "wet_bulb_temperature_integral")
        self.assertEqual(wb_temp_int.units, Unit('K m'))
        self.assertIsInstance(wb_temp, iris.cube.Cube)
        self.assertEqual(wb_temp.name(), "wet_bulb_temperature")
        self.assertEqual(wb_temp.units, Unit('celsius'))

    def test_data(self):
        """Test that the wet bulb temperature integral returns a cube
        containing the expected data."""
        expected_wb_int = np.array(
            [[0.0, 0.0, 608.106507],
             [0.0, 0.0, 912.159761]])
        expected_wb_temp = np.array(
            [[-90., -13.26694545, 60.81065074],
             [-90., -13.26694545, 60.81065074],
             [-90., -13.26694545, 60.81065074]])
        wb_temp, wb_temp_int = WetBulbTemperatureIntegral().process(
            self.temperature_cube, self.relative_humidity_cube,
            self.pressure_cube)
        self.assertIsInstance(wb_temp, iris.cube.Cube)
        self.assertIsInstance(wb_temp_int, iris.cube.Cube)
        self.assertArrayAlmostEqual(wb_temp_int.data, expected_wb_int)
        self.assertArrayAlmostEqual(wb_temp.data, expected_wb_temp)


if __name__ == '__main__':
    unittest.main()
