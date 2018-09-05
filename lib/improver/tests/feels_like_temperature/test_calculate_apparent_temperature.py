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
"""Unit tests for the feels_like_temperature.ApparentTemperature plugin."""

import unittest
import numpy as np
from iris.tests import IrisTest

from improver.feels_like_temperature import calculate_apparent_temperature
from improver.tests.ensemble_calibration.ensemble_calibration. \
    helper_functions import (set_up_temperature_cube, set_up_wind_speed_cube,
                             set_up_cube)


class Test_calculate_apparent_temperature(IrisTest):
    """Test the apparent temperature function."""

    def setUp(self):
        """Create cubes to input."""

        self.temperature_cube = set_up_temperature_cube()[0, :, 2]
        self.wind_speed_cube = set_up_wind_speed_cube()[0, :, 0]
        # create cube with metadata and values suitable for pressure.
        pressure_data = (
            np.tile(np.linspace(100000, 110000, 9), 3).reshape(3, 1, 3, 3))
        pressure_data[0] -= 2
        pressure_data[1] += 2
        pressure_data[2] += 4
        self.pressure_cube = set_up_cube(
            pressure_data, "air_pressure", "Pa")[0, :, 0]

        # create cube with metadata and values suitable for relative humidity.
        relative_humidity_data = (
            np.tile(np.linspace(0, 0.6, 9), 3).reshape(3, 1, 3, 3))
        relative_humidity_data[0] += 0
        relative_humidity_data[1] += 0.2
        relative_humidity_data[2] += 0.4
        self.relative_humidity_cube = set_up_cube(
            relative_humidity_data, "relative_humidity", "1")[0, :, 0]

    def test_apparent_temperature_values(self):
        """Test output values from apparent temperature equation."""

        # use a temperature greater than 20 degress C.
        self.temperature_cube.data = np.full((1, 3), 295.15)
        self.wind_speed_cube.data = np.full((1, 3), 5.)
        expected_result = (
            [[290.07999999999998, 290.47834089999998, 290.87672928000001]])
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
            [[291.76999999999998, 299.30183324000001, 308.02746487999997]])
        # convert to fahrenheit
        expected_result = data * (9.0/5.0) - 459.67
        result = calculate_apparent_temperature(
            self.temperature_cube, self.wind_speed_cube,
            self.relative_humidity_cube, self.pressure_cube)
        self.assertArrayAlmostEqual(result.data, expected_result)

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
