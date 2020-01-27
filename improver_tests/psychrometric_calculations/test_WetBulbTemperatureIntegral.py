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
"""Unit tests for psychrometric_calculations WetBulbTemperatureIntegral."""

import unittest

import iris
import numpy as np
from cf_units import Unit
from iris.tests import IrisTest

from improver.psychrometric_calculations.psychrometric_calculations import (
    WetBulbTemperatureIntegral)

from ..set_up_test_cubes import add_coordinate, set_up_variable_cube


class Test_process(IrisTest):

    """Test the calculation of the wet bulb temperature integral from
    temperature, pressure, and relative humidity information using the
    process function. Integration is calculated in the vertical.

    The wet bulb temperature calculated at each level, and the difference in
    height between the levels are used to calculate an integrated wet bulb
    temperature. This is used to ascertain the degree of melting that initially
    frozen precipitation undergoes.
    """

    def setUp(self):
        """Set up temperature, pressure, and relative humidity cubes that
        contain multiple height levels; in this case the values of these
        diagnostics are identical on each level."""
        super().setUp()

        self.height_points = np.array([5., 10., 20.])
        height_attribute = {"positive": "up"}

        data = np.array(
            [[-88.15, -13.266943, 60.81063]], dtype=np.float32)
        self.wet_bulb_temperature = set_up_variable_cube(
            data, name='wet_bulb_temperature', units='Celsius')
        self.wet_bulb_temperature = add_coordinate(
            self.wet_bulb_temperature, self.height_points, 'height',
            coord_units='m', attributes=height_attribute)

    def test_basic(self):
        """Test that the wet bulb temperature integral returns a cube
        with the expected name."""
        wb_temp_int = WetBulbTemperatureIntegral().process(
            self.wet_bulb_temperature)
        self.assertIsInstance(wb_temp_int, iris.cube.Cube)
        self.assertEqual(wb_temp_int.name(), "wet_bulb_temperature_integral")
        self.assertEqual(str(wb_temp_int.units), 'K m')

    def test_data(self):
        """Test that the wet bulb temperature integral returns a cube
        containing the expected data."""
        expected_wb_int = np.array(
            [[[0.0, 0.0, 608.1063]],
             [[0.0, 0.0, 912.1595]]], dtype=np.float32)
        wb_temp_int = WetBulbTemperatureIntegral().process(
            self.wet_bulb_temperature)
        self.assertIsInstance(wb_temp_int, iris.cube.Cube)
        self.assertArrayAlmostEqual(wb_temp_int.data, expected_wb_int)


if __name__ == '__main__':
    unittest.main()
