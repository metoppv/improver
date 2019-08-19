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

import iris
from iris.tests import IrisTest

import improver.units
from improver.utilities import cube_units
from improver.tests.set_up_test_cubes import (
    set_up_variable_cube, set_up_probability_cube, set_up_percentile_cube)


class Test_enforce_units_and_dtypes(IrisTest):
    """Test checking with option of enforcement or failure"""

    def setUp(self):
        """Set up some conformant cubes of air_temperature to test"""
        data = 275*np.ones((3, 3), dtype=np.float32)
        self.data_cube = set_up_variable_cube(data, spatial_grid='equalarea')

        data = np.ones((3, 3, 3), dtype=np.float32)
        thresholds = np.array([272, 273, 274], dtype=np.float32)
        self.probability_cube = set_up_probability_cube(data, thresholds)

        data = np.array([274*np.ones((3, 3), dtype=np.float32),
                         275*np.ones((3, 3), dtype=np.float32),
                         276*np.ones((3, 3), dtype=np.float32)])
        percentiles = np.array([25, 50, 75], np.float32)
        self.percentile_cube = set_up_percentile_cube(data, percentiles)
        # set to real source here, to test consistency of setup functions
        # with up-to-date metadata standard
        cube_units.DEFAULT_UNITS = improver.units.DEFAULT_UNITS

    def test_basic(self):
        """Test function returns a CubeList"""
        cubelist = [self.data_cube]
        result = cube_units.enforce_units_and_dtypes(cubelist)
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_cube_input(self):
        """Test function behaves sensibly with a single cube"""
        result = cube_units.enforce_units_and_dtypes(self.data_cube)
        self.assertIsInstance(result, iris.cube.CubeList)
        self.assertArrayAlmostEqual(result[0].data, self.data_cube.data)
        self.assertEqual(result[0].metadata, self.data_cube.metadata)

    def test_conformant_cubes(self):
        """Test conformant data, percentile and probability cubes are all
        passed when enforce=False (ie set to fail on non-conformance)"""
        cubelist = [
            self.data_cube, self.probability_cube, self.percentile_cube]
        result = cube_units.enforce_units_and_dtypes(cubelist, enforce=False)
        self.assertIsInstance(result, iris.cube.CubeList)
        for cube, ref in zip(result, cubelist):
            self.assertArrayAlmostEqual(cube.data, ref.data)
            self.assertEqual(cube.metadata, ref.metadata)

    def test_data_units_enforce(self):
        """Test units are changed on the returned cube and the input cube is
        unmodified"""
        self.data_cube.convert_units('Fahrenheit')
        result, = cube_units.enforce_units_and_dtypes(self.data_cube)
        self.assertEqual(result.units, 'K')
        self.assertEqual(self.data_cube.units, 'Fahrenheit')

    def test_coord_units_enforce(self):
        """Test coordinate units are enforced and the input cube is
        unmodified"""
        test_coord = 'projection_x_coordinate'
        self.data_cube.coord(test_coord).convert_units('km')
        result, = cube_units.enforce_units_and_dtypes(self.data_cube)
        self.assertEqual(self.data_cube.coord(test_coord).units, 'km')
        self.assertEqual(result.coord(test_coord).units, 'm')

    def test_data_units_fail(self):
        """Test error is raised when enforce=False"""
        self.data_cube.convert_units('Fahrenheit')
        msg = "Units Fahrenheit of air_temperature cube do not conform"
        with self.assertRaisesRegex(ValueError, msg):
            cube_units.enforce_units_and_dtypes(self.data_cube, enforce=False)

    def test_coord_units_fail(self):
        """Test error is raised when enforce=False"""
        self.probability_cube.coord(
            'air_temperature').convert_units('Fahrenheit')
        msg = "Units Fahrenheit of coordinate air_temperature on probability_"
        with self.assertRaisesRegex(ValueError, msg):
            cube_units.enforce_units_and_dtypes(
                self.probability_cube, enforce=False)

    def test_data_datatype_enforce(self):
        """Test dataset datatypes are enforced"""
        self.data_cube.data = self.data_cube.data.astype(np.float64)
        result, = cube_units.enforce_units_and_dtypes(self.data_cube)
        self.assertEqual(result.dtype, np.float32)

    def test_coord_datatype_enforce(self):
        """Test coordinate datatypes are enforced (using substring processing)
        """
        test_coord = 'forecast_reference_time'
        self.data_cube.coord(test_coord).points = (
             self.data_cube.coord(test_coord).points.astype(np.float64))
        result, = cube_units.enforce_units_and_dtypes(self.data_cube)
        self.assertEqual(result.coord(test_coord).dtype, np.int64)

    def test_data_datatype_fail(self):
        """Test error is raised when enforce=False"""
        self.percentile_cube.data = (
            self.percentile_cube.data.astype(np.float64))
        msg = "of air_temperature cube does not conform"
        with self.assertRaisesRegex(ValueError, msg):
            cube_units.enforce_units_and_dtypes(
                self.percentile_cube, enforce=False)

    def test_coord_datatype_fail(self):
        """Test error is raised when enforce=False"""
        self.percentile_cube.coord('percentile').points = (
            self.percentile_cube.coord('percentile').points.astype(np.int32))
        msg = "of coordinate percentile on air_temperature cube"
        with self.assertRaisesRegex(ValueError, msg):
            cube_units.enforce_units_and_dtypes(
                self.percentile_cube, enforce=False)

    def test_coordinates_correctly_identified(self):
        """Test all coordinates in a heterogeneous cube list are identified and
        corrected"""
        self.percentile_cube.coord('percentile').points = (
            self.percentile_cube.coord('percentile').points.astype(np.int32))
        self.probability_cube.coord(
            'air_temperature').convert_units('Fahrenheit')
        result = cube_units.enforce_units_and_dtypes(
            [self.percentile_cube, self.probability_cube])
        self.assertEqual(result[0].coord('percentile').dtype, np.float32)
        self.assertEqual(result[1].coord('air_temperature').units, 'K')

    def test_subset_of_coordinates(self):
        """Test function can enforce on a selected subset of coordinates and
        leave all others unchanged"""
        self.percentile_cube.coord('percentile').points = (
            self.percentile_cube.coord('percentile').points.astype(np.int32))
        self.percentile_cube.coord('time').convert_units(
            'hours since 1970-01-01 00:00:00')
        self.probability_cube.coord(
            'air_temperature').convert_units('Fahrenheit')
        self.probability_cube.coord('forecast_period').convert_units('h')

        result = cube_units.enforce_units_and_dtypes(
            [self.percentile_cube, self.probability_cube],
            coords=["time", "forecast_period"])
        self.assertEqual(result[0].coord('percentile').dtype, np.int32)
        self.assertEqual(
            result[0].coord('time').units, 'seconds since 1970-01-01 00:00:00')
        self.assertEqual(
            result[1].coord('air_temperature').units, 'Fahrenheit')
        self.assertEqual(result[1].coord('forecast_period').units, 's')


class Test__find_dict_key(IrisTest):
    """Test method to find suitable substring keys in dictionary"""

    @staticmethod
    def setUp():
        """Redirect to dummy dictionary"""
        cube_units.DEFAULT_UNITS = {
            "time": {
                "unit": "seconds since 1970-01-01 00:00:00",
                "dtype": np.int64},
            "percentile": {"unit": "%"},
            "probability": {"unit": "1"},
            "temperature": {"unit": "K"},
            "rainfall": {"unit": "m s-1"},
            "rate": {"unit": "m s-1"}
        }

    def test_match(self):
        """Test correct identification of single substring match"""
        result = cube_units._find_dict_key("air_temperature", "")
        self.assertEqual(result, "temperature")

    def test_probability_match(self):
        """Test the correct substring is returned for an IMPROVER-style
        probability cube name that matches multiple substrings"""
        result = cube_units._find_dict_key(
            "probability_of_air_temperature_above_threshold", "")
        self.assertEqual(result, "probability")

    def test_no_matches_error(self):
        """Test a KeyError is raised if there is no matching substring"""
        with self.assertRaises(KeyError):
            cube_units._find_dict_key("snowfall", "")

    def test_multiple_matches_error(self):
        """Test a KeyError is raised if there are multiple matching substrings
        """
        with self.assertRaises(KeyError):
            cube_units._find_dict_key("rainfall_rate", "")


class Test__check_units_and_dtypes(IrisTest):
    """Test method to check object conformance"""

    def setUp(self):
        """Set up test cube"""
        self.cube = set_up_variable_cube(
            data=275.*np.ones((3, 3), dtype=np.float32),
            spatial_grid='equalarea')
        self.coord = self.cube.coord('projection_x_coordinate')

    def test_pass_cube(self):
        """Test return value for compliant cube"""
        result = cube_units._check_units_and_dtype(self.cube, 'K', np.float32)
        self.assertTrue(result)

    def test_fail_cube(self):
        """Test return value for non-compliant cube"""
        result = cube_units._check_units_and_dtype(
            self.cube, 'degC', np.float32)
        self.assertFalse(result)

    def test_pass_coord(self):
        """Test return value for compliant coordinate"""
        result = cube_units._check_units_and_dtype(
            self.coord, 'm', np.float32)
        self.assertTrue(result)

    def test_fail_coord(self):
        """Test return value for non-compliant coordinate"""
        result = cube_units._check_units_and_dtype(self.coord, 'm', np.int32)
        self.assertFalse(result)


class Test__enforce_coordinate_units_and_dtypes(IrisTest):

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
        self.plugin = cube_units._enforce_coordinate_units_and_dtypes
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
        cube.coord('time').convert_units("hours since 1970-01-01 00:00:00")
        cube.coord('forecast_reference_time').convert_units(
            "hours since 1970-01-01 00:00:00")
        cube.coord('forecast_period').convert_units("hours")

        cube_units.DEFAULT_UNITS[coord]['unit'] = target_units
        expected = 1510286400

        self.plugin([cube], [coord])

        self.assertEqual(cube.coord(coord).points[0], expected)
        self.assertEqual(cube.coord(coord).units, target_units)
        self.assertIsInstance(cube.coord(coord).points[0], np.int64)

    def test_basic_non_time_coordinate(self):
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

    def test_unavailable_coordinate(self):
        """Test application of the function to a coordinate for which the
        default units and data type are not defined, resulting in an
        exception."""

        coord = 'number_of_fish'
        cube = self.cube

        msg = "'number_of_fish' not defined in units.py"
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


class Test__enforce_diagnostic_units_and_dtypes(IrisTest):

    """Test the enforcement of diagnostic units and data types."""

    def setUp(self):
        """Set up a cube to test."""
        original_units = {
            "air_temperature": {
                "unit": "K",
                "dtype": np.float32},
        }

        cube_units.DEFAULT_UNITS = original_units
        self.plugin = cube_units._enforce_diagnostic_units_and_dtypes
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

        msg = "'number_of_fish' not defined in units.py"
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
