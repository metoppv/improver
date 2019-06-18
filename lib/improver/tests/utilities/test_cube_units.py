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
"""Unit tests for the cube_units utility."""

import unittest
from datetime import datetime
import numpy as np

from iris.tests import IrisTest

from improver.utilities import cube_units
from improver.tests.set_up_test_cubes import set_up_variable_cube


class Test_enforce_coordinate_units_and_dtypes(IrisTest):

    """Test the enforcement of coordinate units and data types."""

    def setUp(self):
        """Set up a cube to test."""
        original_units = {
            "time": {
                "unit": "seconds since 1970-01-01 00:00:00",
                "dtype": np.int64},
            "forecast_reference_time": {
                "unit": "seconds since 1970-01-01 00:00:00",
                "dtype": np.int64},
            "forecast_period": {
                "unit": "seconds",
                "dtype": np.int32},
            "projection_x_coordinate": {
                "unit": "m",
                "dtype": np.float32}
        }

        cube_units.DEFAULT_UNITS = original_units
        self.plugin = cube_units.enforce_coordinate_units_and_dtypes
        self.cube = set_up_variable_cube(np.ones((5, 5), dtype=np.float32),
                                         spatial_grid='equalarea')

        self.cube_non_integer_intervals = set_up_variable_cube(
            np.ones((5, 5), dtype=np.float32), spatial_grid='equalarea',
            time=datetime(2017, 11, 10, 4, 30))
        crd = self.cube_non_integer_intervals.coord('projection_x_coordinate')
        crd.points = crd.points * 1.0005

    def test_time_coordinate_to_hours_valid(self):
        """Test that a cube with a validity time on the hour can be converted
        to integer hours."""

        target_units = "hours since 1970-01-01 00:00:00"
        coord = 'time'
        cube = self.cube
        cube_units.DEFAULT_UNITS[coord]['unit'] = target_units
        expected = 419524

        self.plugin([cube], [coord])

        self.assertEqual(cube.coord(coord).points[0], expected)
        self.assertEqual(cube.coord(coord).units, target_units)
        self.assertIsInstance(cube.coord(coord).points[0], np.int64)

    def test_time_coordinate_to_hours_invalid(self):
        """Test that a cube with a validity time on the half hour cannot be
        converted to integer hours."""

        target_units = "hours since 1970-01-01 00:00:00"
        coord = 'time'
        cube = self.cube_non_integer_intervals
        cube_units.DEFAULT_UNITS[coord]['unit'] = target_units

        msg = ('Data type of coordinate "time" could not be'
               ' enforced without losing significant precision.')
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin([cube], [coord])

    def test_time_coordinate_to_invalid_units(self):
        """Test that a cube with time coordinate cannot be converted to an
        incompatible units, e.g. metres."""

        target_units = "m"
        coord = 'time'
        cube = self.cube
        cube_units.DEFAULT_UNITS[coord]['unit'] = target_units

        msg = 'time units cannot be converted to "m"'
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin([cube], [coord])

    def test_time_coordinate_to_hours_float(self):
        """Test that a cube with a validity time on the half hour can be
        converted to float hours."""

        target_units = "hours since 1970-01-01 00:00:00"
        coord = 'time'
        cube = self.cube_non_integer_intervals
        cube_units.DEFAULT_UNITS[coord]['unit'] = target_units
        cube_units.DEFAULT_UNITS[coord]['dtype'] = np.float64
        expected = 419524.5

        self.plugin([cube], [coord])

        self.assertEqual(cube.coord(coord).points[0], expected)
        self.assertEqual(cube.coord(coord).units, target_units)
        self.assertIsInstance(cube.coord(coord).points[0], np.float64)

    def test_time_coordinate_hours_to_seconds_integer(self):
        """Test that a cube with a validity time in units of hours can be
        converted to integer seconds."""

        target_units = "seconds since 1970-01-01 00:00:00"
        coord = 'time'
        cube = self.cube.copy()
        cube.coord('time').convert_units(target_units)
        cube.coord('forecast_reference_time').convert_units(
            target_units)
        cube.coord('forecast_period').convert_units("hours")

        cube_units.DEFAULT_UNITS[coord]['unit'] = target_units
        expected = 1510286400

        self.plugin([cube], [coord])

        self.assertEqual(cube.coord(coord).points[0], expected)
        self.assertEqual(cube.coord(coord).units, target_units)
        self.assertIsInstance(cube.coord(coord).points[0], np.int64)

    def test_spatial_coordinate_to_km_valid(self):
        """Test that a cube with a grid at km intervals expressed in metres can
        be converted to integer kilometres."""

        target_units = "km"
        coord = 'projection_x_coordinate'
        cube = self.cube
        cube_units.DEFAULT_UNITS[coord]['unit'] = target_units
        cube_units.DEFAULT_UNITS[coord]['dtype'] = np.int32
        expected = [-400, -200, 0, 200, 400]

        self.plugin([self.cube], [coord])

        self.assertArrayEqual(cube.coord(coord).points, expected)
        self.assertEqual(cube.coord(coord).units, target_units)
        self.assertIsInstance(cube.coord(coord).points[0], np.int32)

    def test_spatial_coordinate_to_km_invalid(self):
        """Test that a cube with a grid at intervals that fall between whole
        kilometers cannot be converted to integer kilometres."""

        target_units = "km"
        coord = 'projection_x_coordinate'
        cube = self.cube_non_integer_intervals
        cube_units.DEFAULT_UNITS[coord]['unit'] = target_units
        cube_units.DEFAULT_UNITS[coord]['dtype'] = np.int32

        msg = ('Data type of coordinate "projection_x_coordinate" could not be'
               ' enforced without losing significant precision.')
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin([cube], [coord])

    def test_spatial_coordinate_to_invalid_units(self):
        """Test that a cube with spatial coordinates in metres cannot be
        converted to an incompatible unit, e.g. seconds."""

        target_units = "seconds"
        coord = 'projection_x_coordinate'
        cube = self.cube
        cube_units.DEFAULT_UNITS[coord]['unit'] = target_units

        msg = 'projection_x_coordinate units cannot be converted to "seconds"'
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin([cube], [coord])

    def test_spatial_coordinate_to_km_float(self):
        """Test that a cube with a grid at intervals that fall between whole
        kilometers can be converted to float kilometres."""

        target_units = "km"
        coord = 'projection_x_coordinate'
        cube = self.cube_non_integer_intervals
        cube_units.DEFAULT_UNITS[coord]['unit'] = target_units
        cube_units.DEFAULT_UNITS[coord]['dtype'] = np.float32
        expected = np.array([-400.2, -200.1, 0., 200.1, 400.2],
                            dtype=np.float32)

        self.plugin([cube], [coord])

        self.assertArrayEqual(cube.coord(coord).points, expected)
        self.assertEqual(cube.coord(coord).units, target_units)
        self.assertIsInstance(cube.coord(coord).points[0], np.float32)

    def test_spatial_coordinate_km_to_m_integer(self):
        """Test that a cube with a grid at intervals that fall between whole
        kilometers can be converted to integer metres."""

        target_units = "m"
        coord = 'projection_x_coordinate'
        cube = self.cube_non_integer_intervals.copy()
        cube.coord('projection_x_coordinate').convert_units('km')

        cube_units.DEFAULT_UNITS[coord]['unit'] = target_units
        cube_units.DEFAULT_UNITS[coord]['dtype'] = np.int64
        expected = np.array([-400200, -200100, 0., 200100, 400200],
                            dtype=np.int64)

        self.plugin([cube], [coord])

        self.assertArrayEqual(cube.coord(coord).points, expected)
        self.assertEqual(cube.coord(coord).units, target_units)
        self.assertIsInstance(cube.coord(coord).points[0], np.int64)

    def test_unavailable_coordinate(self):
        """Test application of the function to a coordinate for which the
        default units and data type are not defined, resulting in an
        exception."""

        coord = 'number_of_fish'
        cube = self.cube

        msg = 'Coordinate number_of_fish not defined in units.py'
        with self.assertRaisesRegex(KeyError, msg):
            self.plugin([cube], [coord])

    def test_return_changes_as_copy(self):
        """Test that using the inplace=False keyword arg results in the input
        cube remaining unchanged, and a new modified cube being returned."""

        target_units = "hours since 1970-01-01 00:00:00"
        coord = 'time'
        cube = self.cube.copy()
        cube_units.DEFAULT_UNITS[coord]['unit'] = target_units
        expected = 419524

        result, = self.plugin([cube], [coord], inplace=False)

        self.assertEqual(cube.coord(coord).points[0],
                         self.cube.coord(coord).points[0],)
        self.assertEqual(cube.coord(coord).units, self.cube.coord(coord).units)

        self.assertEqual(result.coord(coord).points[0], expected)
        self.assertEqual(result.coord(coord).units, target_units)
        self.assertIsInstance(result.coord(coord).points[0], np.int64)

    def test_multiple_cubes(self):
        """Test that when a cube list is provided, all the cubes are modified
        as expected."""

        target_units = "hours since 1970-01-01 00:00:00"
        coords = ['time', 'forecast_reference_time']
        cubes = [self.cube, self.cube_non_integer_intervals]

        for coord in coords:
            cube_units.DEFAULT_UNITS[coord]['unit'] = target_units
            cube_units.DEFAULT_UNITS[coord]['dtype'] = np.float64

        expected = {'time': [419524, 419524.5],
                    'forecast_reference_time': [419520, 419520]}

        self.plugin(cubes, coords)

        for coord in coords:
            for index in range(2):
                self.assertEqual(cubes[index].coord(coord).points[0],
                                 expected[coord][index])
                self.assertEqual(cubes[index].coord(coord).units, target_units)
                self.assertIsInstance(cubes[index].coord(coord).points[0],
                                      np.float64)


class Test_enforce_diagnostic_units_and_dtypes(IrisTest):

    """Test the enforcement of diagnostic units and data types."""

    def setUp(self):
        """Set up a cube to test."""
        original_units = {
            "air_temperature": {
                "unit": "K",
                "dtype": np.float32},
        }

        cube_units.DEFAULT_UNITS = original_units
        self.plugin = cube_units.enforce_diagnostic_units_and_dtypes
        self.cube = set_up_variable_cube(np.ones((5, 5), dtype=np.float32),
                                         spatial_grid='equalarea')

        self.cube_non_integer_intervals = set_up_variable_cube(
            np.ones((5, 5), dtype=np.float32), spatial_grid='equalarea')
        self.cube_non_integer_intervals.data *= 1.5

    def test_temperature_to_integer_kelvin_valid(self):
        """Test that a cube with temperatures at whole kelvin intervals can
        be converted to integer kelvin. Precision checking is invoked here to
        ensure the change of data type does not result in a loss of
        information."""

        target_units = "kelvin"
        diagnostic = "air_temperature"
        cube = self.cube
        cube_units.DEFAULT_UNITS[diagnostic]['unit'] = target_units
        cube_units.DEFAULT_UNITS[diagnostic]['dtype'] = np.int32
        expected = np.ones((5, 5), dtype=np.int32)

        self.plugin([cube])

        self.assertArrayEqual(cube.data, expected)
        self.assertEqual(cube.units, target_units)
        self.assertEqual(cube.data.dtype, np.int32)

    def test_temperature_to_integer_kelvin_invalid(self):
        """Test that a cube with temperatures not at whole kelvin intervals
        cannot be converted to integer kelvin. Precision checking is invoked
        here to ensure the change of data type does not result in a loss of
        information; in this case it does and raises an exception."""

        target_units = "kelvin"
        diagnostic = "air_temperature"
        cube = self.cube_non_integer_intervals
        cube_units.DEFAULT_UNITS[diagnostic]['unit'] = target_units
        cube_units.DEFAULT_UNITS[diagnostic]['dtype'] = np.int32

        msg = ('Data type of diagnostic "air_temperature" could not be'
               ' enforced without losing significant precision.')
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin([cube])

    def test_temperature_to_invalid_units(self):
        """Test that a cube with temperatures in kelvin cannot be converted
        so incompatible units, e.g. metres."""

        target_units = "m"
        diagnostic = "air_temperature"
        cube = self.cube
        cube_units.DEFAULT_UNITS[diagnostic]['unit'] = target_units

        msg = ('Data type of diagnostic "air_temperature" could not be'
               ' enforced without losing significant precision.')
        msg = 'air_temperature units cannot be converted to "m"'
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin([cube])

    def test_unavailable_diagnostic(self):
        """Test application of the function to a cube for which the default
        units and data type are not defined, resulting in an exception."""

        cube = self.cube
        cube.rename('number_of_fish')

        msg = 'Diagnostic number_of_fish not defined in units.py'
        with self.assertRaisesRegex(KeyError, msg):
            self.plugin([cube])

    def test_return_changes_as_copy(self):
        """Test that using the inplace=False keyword arg results in the input
        cube remaining unchanged, and a new modified cube being returned. In
        this test the temperature are converted to Celsius to make the change
        clear."""

        target_units = "celsius"
        diagnostic = "air_temperature"
        cube = self.cube.copy()
        cube_units.DEFAULT_UNITS[diagnostic]['unit'] = target_units
        cube_units.DEFAULT_UNITS[diagnostic]['dtype'] = np.float32
        expected = np.full((5, 5), -272.15, dtype=np.float32)

        result, = self.plugin([cube], inplace=False)

        self.assertArrayEqual(cube.data, self.cube.data)
        self.assertEqual(cube.units, self.cube.units)

        self.assertArrayEqual(result.data, expected)
        self.assertEqual(result.units, target_units)
        self.assertEqual(result.data.dtype, np.float32)

    def test_temperature_to_float_celsius_valid(self):
        """Test that a cube with temperatures at whole kelvin intervals stored
        as integers can be converted to float Celsius, changing units and data
        type, whilst checking precision is not lost."""

        target_units = "celsius"
        diagnostic = "air_temperature"
        cube = self.cube.copy(data=self.cube.data.astype(np.int))
        cube_units.DEFAULT_UNITS[diagnostic]['unit'] = target_units
        cube_units.DEFAULT_UNITS[diagnostic]['dtype'] = np.float32
        expected = np.full((5, 5), -272.15, dtype=np.float32)

        self.plugin([cube])

        self.assertArrayEqual(cube.data, expected)
        self.assertEqual(cube.units, target_units)
        self.assertEqual(cube.data.dtype, np.float32)

    def test_multiple_cubes(self):
        """Test that when a cube list is provided, all the cubes are modified
        as expected."""

        target_units = "kelvin"
        diagnostic = "air_temperature"
        cubes = [self.cube, self.cube_non_integer_intervals]

        cube_units.DEFAULT_UNITS[diagnostic]['unit'] = target_units
        cube_units.DEFAULT_UNITS[diagnostic]['dtype'] = np.float64

        expected = [np.ones((5, 5), dtype=np.float64),
                    np.full((5, 5), 1.5, dtype=np.float64)]

        self.plugin(cubes)

        for index in range(2):
            self.assertArrayEqual(cubes[index].data, expected[index])
            self.assertEqual(cubes[index].units, target_units)
            self.assertEqual(cubes[index].data.dtype, np.float64)


class Test_check_precision_loss(IrisTest):

    """Test the check_precision_loss function behaves as expected."""
    def setUp(self):
        """Make an instance of the plugin that is to be tested."""
        self.plugin = cube_units.check_precision_loss

    def test_non_lossy_float_to_integer(self):
        """Test that the function returns true when whole numbers in float type
        are checked for loss upon conversion to integers. This means that the
        conversion can go ahead without loss."""

        data = np.full((3, 3), 1, dtype=np.float64)
        dtype = np.int32

        result = self.plugin(dtype, data)

        self.assertTrue(result)

    def test_lossy_float_to_integer(self):
        """Test that the function returns false when non-whole numbers in float
        type are checked for loss upon conversion to integers. This means that
        the conversion cannot go ahead without loss."""

        data = np.full((3, 3), 1.5, dtype=np.float64)
        dtype = np.int32

        result = self.plugin(dtype, data)

        self.assertFalse(result)

    def test_lossy_float_to_integer_low_precision(self):
        """Test that the function returns True when non-whole numbers in float
        type are checked for loss upon conversion to integers but the
        fractional component is smaller than the given precision. This means
        that the conversion can go ahead without significant loss."""

        data = np.full((3, 3), 1.005, dtype=np.float64)
        dtype = np.int32

        result = self.plugin(dtype, data, precision=2)

        self.assertTrue(result)

    def test_lossy_float_to_float(self):
        """Test that the function returns true even when a float conversion
        will result in the loss of some precision. The function is not
        designed to prevent loss when converting floats to floats. In this
        test the conversion will result in the values becoming 1.0"""

        data = np.full((3, 3), 1.00000005, dtype=np.float64)
        dtype = np.float32

        result = self.plugin(dtype, data)

        self.assertTrue(result)

    def test_non_lossy_int_to_int_down(self):
        """Test that the function returns true if an integer type is changed
        to a lower precision integer type but no information is lost."""

        data = np.full((3, 3), 1234567891, dtype=np.int64)
        dtype = np.int32

        result = self.plugin(dtype, data)

        self.assertTrue(result)

    def test_non_lossy_int_to_int_up(self):
        """Test that the function returns true if an integer type is changed
        to a higher precision integer type but no information is lost."""

        data = np.full((3, 3), 1234567891, dtype=np.int32)
        dtype = np.int64

        result = self.plugin(dtype, data)

        self.assertTrue(result)

    def test_lossy_int_to_int(self):
        """Test that the function returns false if an integer type is changed
        to a lower precision integer type that results in the loss of
        information."""

        data = np.full((3, 3), 12345678910, dtype=np.int64)
        dtype = np.int32

        result = self.plugin(dtype, data)

        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
