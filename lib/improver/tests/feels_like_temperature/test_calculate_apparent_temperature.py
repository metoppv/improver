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
"""Unit tests for the feels_like_temperature.ApparentTemperature plugin."""

import unittest

import numpy as np
from iris.tests import IrisTest

from improver.feels_like_temperature import calculate_apparent_temperature
from improver.tests.set_up_test_cubes import set_up_variable_cube


class Test_calculate_apparent_temperature(IrisTest):
    """Test the apparent temperature function."""

    def setUp(self):
        """Create cubes to input."""
        temperature = np.array([[293.65, 304.9, 316.15]], dtype=np.float32)
        self.temperature_cube = set_up_variable_cube(temperature)

        wind_speed = np.array([[0., 7.5, 15.]], dtype=np.float32)
        self.wind_speed_cube = set_up_variable_cube(
            wind_speed, name="wind_speed", units="m s-1")

        pressure = np.array([[99998., 101248., 102498.]], dtype=np.float32)
        self.pressure_cube = set_up_variable_cube(
            pressure, name="air_pressure", units="Pa")

        relh = np.array([[0., 0.075, 0.15]], dtype=np.float32)
        self.relative_humidity_cube = set_up_variable_cube(
            relh, name="relative_humidity", units="1")

    def test_apparent_temperature_values(self):
        """Test output values from apparent temperature equation."""

        # use a temperature greater than 20 degress C.
        self.temperature_cube.data = np.full((1, 3), 295.15)
        self.wind_speed_cube.data = np.full((1, 3), 5)
        expected_result = np.array(
            [[290.07998657, 290.47833252, 290.8767395]],
            dtype=np.float32
        )
        result = calculate_apparent_temperature(
            self.temperature_cube, self.wind_speed_cube,
            self.relative_humidity_cube, self.pressure_cube)
        self.assertArrayAlmostEqual(result.data, expected_result)

    def test_name_and_units(self):
        """Test correct outputs for name and units."""

        expected_name = "apparent_temperature"
        expected_units = 'K'
        result = calculate_apparent_temperature(
            self.temperature_cube, self.wind_speed_cube,
            self.relative_humidity_cube, self.pressure_cube)
        self.assertEqual(result.name(), expected_name)
        self.assertEqual(result.units, expected_units)

    def test_different_units(self):
        """Test that values are correct from input cubes with
        different units"""

        self.temperature_cube.convert_units('fahrenheit')
        self.wind_speed_cube.convert_units('knots')
        self.relative_humidity_cube.convert_units('%')
        self.pressure_cube.convert_units('hPa')

        data = np.array(
            [[291.77001953, 299.30181885, 308.02746582]])
        # convert to fahrenheit
        expected_result = (data * (9.0/5.0) - 459.67).astype(np.float32)
        result = calculate_apparent_temperature(
            self.temperature_cube, self.wind_speed_cube,
            self.relative_humidity_cube, self.pressure_cube)
        self.assertArrayAlmostEqual(result.data, expected_result, decimal=4)

    def test_unit_conversion(self):
        """Tests that input cubes have the same units at the end of the
        function as they do at input"""

        self.temperature_cube.convert_units('fahrenheit')
        self.wind_speed_cube.convert_units('knots')
        self.relative_humidity_cube.convert_units('%')
        self.pressure_cube.convert_units('hPa')

        calculate_apparent_temperature(
            self.temperature_cube, self.wind_speed_cube,
            self.relative_humidity_cube, self.pressure_cube)

        temp_units = self.temperature_cube.units
        wind_speed_units = self.wind_speed_cube.units
        relative_humidity_units = self.relative_humidity_cube.units
        pressure_units = self.pressure_cube.units

        self.assertEqual(temp_units, 'fahrenheit')
        self.assertEqual(wind_speed_units, 'knots')
        self.assertEqual(relative_humidity_units, '%')
        self.assertEqual(pressure_units, 'hPa')


if __name__ == '__main__':
    unittest.main()
