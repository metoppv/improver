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
"""Unit tests for psychrometric_calculations WetBulbTemperature"""

import unittest

import iris
import numpy as np
from cf_units import Unit
from iris.cube import Cube
from iris.tests import IrisTest

from improver.psychrometric_calculations.psychrometric_calculations import (
    WetBulbTemperature)
from improver.tests.set_up_test_cubes import set_up_variable_cube
from improver.utilities.warnings_handler import ManageWarnings


class Test_WetBulbTemperature(IrisTest):
    """Test class for the WetBulbTemperature tests, setting up cubes."""

    def setUp(self):
        """Set up the initial conditions for tests."""
        data = np.array([[185.0, 260.65, 338.15]], dtype=np.float32)
        self.temperature = set_up_variable_cube(data)
        data = np.array([[60., 70., 80.]], dtype=np.float32)
        self.relative_humidity = set_up_variable_cube(
            data, name='relative_humidity', units='%')
        data = np.array([[1.E5, 9.9E4, 9.8E4]], dtype=np.float32)
        self.pressure = set_up_variable_cube(
            data, name='air_pressure', units='Pa')
        data = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        self.mixing_ratio = set_up_variable_cube(
            data, name='humidity_mixing_ratio', units='1')


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(WetBulbTemperature())
        msg = '<WetBulbTemperature: precision: 0.005>'
        self.assertEqual(result, msg)


class Test_check_range(Test_WetBulbTemperature):

    """Test function that checks temperatures fall within a suitable range."""

    @ManageWarnings(record=True)
    def test_basic(self, warning_list=None):
        """Basic test that a warning is raised if temperatures fall outside the
        allowed range."""

        WetBulbTemperature.check_range(self.temperature,
                                       270., 360.)
        warning_msg = "Wet bulb temperatures are"
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))


class Test_lookup_svp(Test_WetBulbTemperature):

    """Test the lookup of saturated vapour pressures."""

    def test_values(self):
        """Basic extraction of some SVP values from the lookup table."""
        self.temperature.data[0, 1] = 260.56833
        expected = [[1.350531e-02, 2.06000274e+02, 2.501530e+04]]
        result = WetBulbTemperature().lookup_svp(self.temperature)
        self.assertArrayAlmostEqual(result.data, expected)
        self.assertEqual(result.units, Unit('Pa'))

    @ManageWarnings(record=True)
    def test_beyond_table_bounds(self, warning_list=None):
        """Extracting SVP values from the lookup table with temperatures beyond
        its valid range. Should return the nearest end of the table."""
        self.temperature.data[0, 0] = 150.
        self.temperature.data[0, 2] = 400.
        expected = [[9.664590e-03, 2.075279e+02, 2.501530e+04]]
        result = WetBulbTemperature().lookup_svp(self.temperature)
        warning_msg = "Wet bulb temperatures are"
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertArrayAlmostEqual(result.data, expected)
        self.assertEqual(result.units, Unit('Pa'))


class Test_pressure_correct_svp(Test_WetBulbTemperature):

    """Test the conversion of saturated vapour pressures in a pure water
    vapour system into SVPs in air."""

    def test_values(self):
        """Basic pressure correction of water vapour SVPs to give SVPs in
        air."""
        svp = self.pressure.copy(data=[[197.41815, 474.1368, 999.5001]])
        expected = [[199.226956, 476.293096, 1006.391004]]
        result = WetBulbTemperature().pressure_correct_svp(
            svp, self.temperature, self.pressure)

        self.assertArrayAlmostEqual(result.data, expected)
        self.assertEqual(result.units, Unit('Pa'))


class Test__calculate_mixing_ratio(Test_WetBulbTemperature):

    """Test the calculation of the specific mixing ratio from temperature,
    and pressure information using the SVP."""

    def test_values(self):
        """Basic mixing ratio calculation."""

        expected = [[6.06744631e-08, 1.31079322e-03, 1.77063149e-01]]
        result = WetBulbTemperature()._calculate_mixing_ratio(
            self.temperature, self.pressure)

        self.assertArrayAlmostEqual(result.data, expected)
        self.assertEqual(result.units, Unit('1'))


class Test_calculate_wet_bulb_temperature(Test_WetBulbTemperature):

    """Test the calculation of wet bulb temperatures from temperature,
    pressure, and relative humidity information."""

    def test_cube_metadata(self):
        """Check metadata of returned cube."""

        result = WetBulbTemperature().calculate_wet_bulb_temperature(
            self.temperature, self.relative_humidity, self.pressure)

        self.assertIsInstance(result, Cube)
        self.assertEqual(result.units, Unit('K'))
        self.assertEqual(result.name(), 'wet_bulb_temperature')

    def test_values(self):
        """Basic wet bulb temperature calculation."""

        expected = np.array([[185.0, 259.88306, 333.96063]], dtype=np.float32)
        result = WetBulbTemperature().calculate_wet_bulb_temperature(
            self.temperature, self.relative_humidity, self.pressure)

        self.assertArrayAlmostEqual(result.data, expected)
        self.assertEqual(result.units, Unit('K'))

    def test_different_units(self):
        """Basic wet bulb temperature calculation with a unit conversion
        required."""

        self.temperature.convert_units('celsius')
        self.relative_humidity.convert_units('1')
        self.pressure.convert_units('kPa')

        expected = np.array([[185.0, 259.88306, 333.96063]], dtype=np.float32)
        result = WetBulbTemperature().calculate_wet_bulb_temperature(
            self.temperature, self.relative_humidity, self.pressure)

        self.assertArrayAlmostEqual(result.data, expected)
        self.assertEqual(result.units, Unit('K'))


class Test_process(Test_WetBulbTemperature):

    """Test the calculation of wet bulb temperatures from temperature,
    pressure, and relative humidity information using the process function."""

    @staticmethod
    def _make_multi_level(cube, time_promote=False):
        """
        Take the input cube data and duplicate it to make a two height level
        cube for testing multi-level data.

        Args:
            cube (iris.cube.Cube):
                Cube to be made multi-level.
            time_promote (bool):
                Option to promote a scalar time coordinate to a dimension.
        """
        height1 = iris.coords.DimCoord([10], 'height', units='m')
        height1.attributes['positive'] = 'up'
        height2 = iris.coords.DimCoord([20], 'height', units='m')
        height2.attributes['positive'] = 'up'
        cube1 = cube.copy()
        cube2 = cube.copy()
        cube1.add_aux_coord(height1)
        cube2.add_aux_coord(height2)
        new_cube = iris.cube.CubeList([cube1, cube2]).merge_cube()
        if time_promote:
            new_cube = iris.util.new_axis(new_cube, 'time')
        return new_cube

    def test_cube_metadata(self):
        """Check metadata of returned cube."""

        result = WetBulbTemperature().process(
            self.temperature, self.relative_humidity, self.pressure)

        self.assertIsInstance(result, Cube)
        self.assertEqual(result.units, Unit('K'))
        self.assertEqual(result.name(), 'wet_bulb_temperature')

    def test_values_single_level(self):
        """Basic wet bulb temperature calculation as if calling the
        calculate_wet_bulb_temperature function directly with single
        level data."""

        expected = np.array([[185.0, 259.88306, 333.96063]], dtype=np.float32)
        result = WetBulbTemperature().process(
            self.temperature, self.relative_humidity, self.pressure)

        self.assertArrayAlmostEqual(result.data, expected)
        self.assertEqual(result.units, Unit('K'))

    def test_values_multi_level(self):
        """Basic wet bulb temperature calculation using multi-level
        data."""

        temperature = self._make_multi_level(self.temperature)
        relative_humidity = self._make_multi_level(self.relative_humidity)
        pressure = self._make_multi_level(self.pressure)
        expected = np.array([[185.0, 259.88306, 333.96063]], dtype=np.float32)

        result = WetBulbTemperature().process(
            temperature, relative_humidity, pressure)

        self.assertArrayAlmostEqual(result.data[0], expected)
        self.assertArrayAlmostEqual(result.data[1], expected)
        self.assertEqual(result.units, Unit('K'))
        self.assertArrayEqual(result.coord('height').points, [10, 20])

    def test_different_level_types(self):
        """Check an exception is raised if trying to work with data on a mix of
        height and pressure levels."""

        temperature = self._make_multi_level(self.temperature)
        relative_humidity = self._make_multi_level(self.relative_humidity)
        pressure = self._make_multi_level(self.pressure)
        temperature.coord('height').rename('pressure')

        msg = 'WetBulbTemperature: Cubes have differing'
        with self.assertRaisesRegex(ValueError, msg):
            WetBulbTemperature().process(
                temperature, relative_humidity, pressure)

    def test_cube_multi_level(self):
        """Check the cube is returned with expected formatting after the data
        has been sliced and reconstructed."""

        temperature = self._make_multi_level(self.temperature,
                                             time_promote=True)
        relative_humidity = self._make_multi_level(self.relative_humidity,
                                                   time_promote=True)
        pressure = self._make_multi_level(self.pressure,
                                          time_promote=True)

        result = WetBulbTemperature().process(
            temperature, relative_humidity, pressure)

        self.assertEqual(result.coord_dims('time')[0], 0)
        self.assertEqual(result.coord_dims('height')[0], 1)


if __name__ == '__main__':
    unittest.main()
