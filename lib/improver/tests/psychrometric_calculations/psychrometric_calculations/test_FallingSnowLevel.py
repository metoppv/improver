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
from improver.tests.psychrometric_calculations.psychrometric_calculations.\
    test_WetBulbTemperature import set_up_cubes_for_wet_bulb_temperature
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


class Test_process(IrisTest):

    """Test the FallingSnowLevel processing works"""

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
        self.orog = self.temperature_cube[0]
        self.orog = iris.util.new_axis(self.orog)

    def test_basic(self):
        """Test that process returns a cube with the right name and units."""
        result = FallingSnowLevel().process(
            self.temperature_cube, self.relative_humidity_cube,
            self.pressure_cube, self.orog)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), "falling_snow_level_asl")
        self.assertEqual(result.units, Unit('m'))

    def test_data(self):
        """Test that the wet bulb temperature integral returns a cube
        containing the expected data."""
        expected = np.array(
            [0.0, 0.0, 0.0])
        result = FallingSnowLevel().process(
            self.temperature_cube, self.relative_humidity_cube,
            self.pressure_cube, self.orog)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)


if __name__ == '__main__':
    unittest.main()
