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
"""Unit tests for the feels_like_temperature.WindChill plugin."""

import unittest

import numpy as np
from iris.tests import IrisTest

from improver.feels_like_temperature import calculate_wind_chill
from improver.tests.set_up_test_cubes import set_up_variable_cube


class Test_calculate_wind_chill(IrisTest):
    """Test the wind chill function."""

    def setUp(self):
        """Creates cubes to input"""

        temperature = np.array([[226.15, 237.4, 248.65]], dtype=np.float32)
        self.temperature_cube = set_up_variable_cube(temperature)

        wind_speed = np.array([[0., 7.5, 15.]], dtype=np.float32)
        self.wind_speed_cube = set_up_variable_cube(
            wind_speed, name="wind_speed", units="m s-1")

    def test_wind_chill_values(self):
        """Test output values when from the wind chill equation."""

        # use a temperature less than 10 degrees C.
        self.temperature_cube.data = np.full((1, 3), 274.85)
        self.wind_speed_cube.data = np.full((1, 3), 3)
        expected_result = np.full((1, 3), 271.674652, dtype=np.float32)
        result = calculate_wind_chill(
            self.temperature_cube, self.wind_speed_cube)
        self.assertArrayAlmostEqual(result.data, expected_result)

    def test_name_and_units(self):
        """Test correct outputs for name and units."""
        expected_name = "wind_chill"
        expected_units = 'K'
        result = calculate_wind_chill(
            self.temperature_cube, self.wind_speed_cube)
        self.assertEqual(result.name(), expected_name)
        self.assertEqual(result.units, expected_units)

    def test_different_units(self):
        """Test that values are correct from input cubes with
        different units"""

        self.temperature_cube.convert_units('fahrenheit')
        self.wind_speed_cube.convert_units('knots')

        data = np.array(
            [[257.05949633, 220.76791229, 231.12778024]])
        # convert to fahrenheit
        expected_result = (data * (9.0/5.0) - 459.67).astype(np.float32)
        result = calculate_wind_chill(
            self.temperature_cube, self.wind_speed_cube)
        self.assertArrayAlmostEqual(result.data, expected_result)

    def test_unit_conversion(self):
        """Tests that input cubes have the same units at the end of the
        function as they do at input"""

        self.temperature_cube.convert_units('fahrenheit')
        self.wind_speed_cube.convert_units('knots')

        calculate_wind_chill(
            self.temperature_cube, self.wind_speed_cube)

        temp_units = self.temperature_cube.units
        wind_speed_units = self.wind_speed_cube.units

        self.assertEqual(temp_units, 'fahrenheit')
        self.assertEqual(wind_speed_units, 'knots')


if __name__ == '__main__':
    unittest.main()
