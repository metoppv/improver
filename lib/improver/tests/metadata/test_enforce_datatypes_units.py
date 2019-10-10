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
"""Unit tests for the improver.metadata.enforce_datatypes_units module."""

import unittest
from datetime import datetime

import iris
import numpy as np
from iris.coords import AuxCoord
from iris.tests import IrisTest

import improver.metadata.enforce_datatypes_units as enforce
from improver.tests.set_up_test_cubes import (
    set_up_variable_cube, set_up_probability_cube, set_up_percentile_cube)


class Test_check_cube_not_float64(IrisTest):

    """Test whether a cube contains any float64 values."""

    def setUp(self):
        """Set up a cube to test."""
        self.cube = set_up_variable_cube(np.ones((5, 5), dtype=np.float32),
                                         spatial_grid='equalarea')

    def test_float32_ok(self):
        """Test a cube that should pass."""
        enforce.check_cube_not_float64(self.cube)

    def test_float64_cube_data(self):
        """Test a failure of a cube with 64 bit data."""
        self.cube.data = self.cube.data.astype(np.float64)
        msg = "64 bit cube not allowed"
        with self.assertRaisesRegex(TypeError, msg):
            enforce.check_cube_not_float64(self.cube)

    def test_float64_cube_data_with_fix(self):
        """Test a cube with 64 bit data is converted to 32 bit data."""
        self.cube.data = self.cube.data.astype(np.float64)
        expected_cube = self.cube.copy()
        expected_cube.data = expected_cube.data.astype(np.float64)
        enforce.check_cube_not_float64(self.cube, fix=True)
        self.assertEqual(self.cube, expected_cube)

    def test_float64_cube_coord_points(self):
        """Test a failure of a cube with 64 bit coord points."""
        self.cube.coord("projection_x_coordinate").points = (
            self.cube.coord("projection_x_coordinate").points.astype(
                np.float64)
        )
        msg = "64 bit coord points not allowed"
        with self.assertRaisesRegex(TypeError, msg):
            enforce.check_cube_not_float64(self.cube)

    def test_float64_cube_coord_points_with_fix(self):
        """Test a failure of a cube with 64 bit coord points."""
        self.cube.coord("projection_x_coordinate").points = (
            self.cube.coord("projection_x_coordinate").points.astype(
                np.float64))
        expected_cube = self.cube.copy()
        expected_cube.coord("projection_x_coordinate").points = (
            expected_cube.coord("projection_x_coordinate").points.astype(
                np.float64))
        expected_coord = expected_cube.coord("projection_x_coordinate")
        enforce.check_cube_not_float64(self.cube, fix=True)
        self.assertEqual(self.cube, expected_cube)
        self.assertEqual(
            self.cube.coord("projection_x_coordinate"), expected_coord)

    def test_float64_cube_coord_bounds(self):
        """Test a failure of a cube with 64 bit coord bounds."""
        x_coord = self.cube.coord("projection_x_coordinate")
        # Default np.array for float input is np.float64.
        x_coord.bounds = (
            np.array([(point - 10., point + 10.) for point in x_coord.points])
        )
        msg = "64 bit coord bounds not allowed"
        with self.assertRaisesRegex(TypeError, msg):
            enforce.check_cube_not_float64(self.cube)

    def test_float64_cube_coord_bounds_with_fix(self):
        """Test a failure of a cube with 64 bit coord bounds."""
        x_coord = self.cube.coord("projection_x_coordinate")
        # Default np.array for float input is np.float64.
        x_coord.bounds = (
            np.array([(point - 10., point + 10.) for point in x_coord.points])
        )
        expected_cube = self.cube.copy()
        expected_cube.coord("projection_x_coordinate").points = (
            expected_cube.coord("projection_x_coordinate").points.astype(
                np.float64))
        expected_coord = expected_cube.coord("projection_x_coordinate")
        enforce.check_cube_not_float64(self.cube, fix=True)
        self.assertEqual(self.cube, expected_cube)
        self.assertEqual(
            self.cube.coord("projection_x_coordinate"), expected_coord)

    def test_float64_cube_time_coord_points_ok(self):
        """Test a pass of a cube with 64 bit time coord points."""
        self.cube.coord("time").points = (
             self.cube.coord("time").points.astype(np.float64))
        enforce.check_cube_not_float64(self.cube)

    def test_float64_cube_forecast_ref_time_coord_points_ok(self):
        """Test a pass of a cube with 64 bit fcast ref time coord points."""
        frt_coord = iris.coords.AuxCoord(
            [np.float64(min(self.cube.coord("time").points))],
            standard_name="forecast_reference_time")
        self.cube.add_aux_coord(frt_coord)
        enforce.check_cube_not_float64(self.cube)


class Test_check_datatypes(IrisTest):
    """Test datatype checking with option of enforcement or failure"""

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

    def test_basic(self):
        """Test function returns a Cube"""
        result = enforce.check_datatypes(self.data_cube)
        self.assertIsInstance(result, iris.cube.Cube)

    def test_conformant_cubes(self):
        """Test conformant data, percentile and probability cubes are all
        passed when enforce=False (ie set to fail on non-conformance)"""
        cubelist = [
            self.data_cube, self.probability_cube, self.percentile_cube]
        for cube in cubelist:
            result = enforce.check_datatypes(cube, enforce=False)
            self.assertIsInstance(result, iris.cube.Cube)
            self.assertArrayAlmostEqual(result.data, cube.data)
            self.assertEqual(result.metadata, cube.metadata)

    def test_data_datatype_enforce(self):
        """Test dataset datatypes are enforced"""
        self.data_cube.data = self.data_cube.data.astype(np.float64)
        result = enforce.check_datatypes(self.data_cube, enforce=True)
        self.assertEqual(result.dtype, np.float32)
        # check input is unchanged
        self.assertEqual(self.data_cube.dtype, np.float64)

    def test_coord_datatype_enforce(self):
        """Test coordinate datatypes are enforced (using substring processing)
        """
        test_coord = 'forecast_reference_time'
        self.data_cube.coord(test_coord).points = (
             self.data_cube.coord(test_coord).points.astype(np.float64))
        result = enforce.check_datatypes(self.data_cube, enforce=True)
        self.assertEqual(result.coord(test_coord).dtype, np.int64)
        # check input is unchanged
        self.assertEqual(self.data_cube.coord(test_coord).dtype, np.float64)

    def test_string_coord(self):
        """Test string coordinate does not throw an error"""
        self.data_cube.add_aux_coord(
            AuxCoord(["ukv"], long_name="model", units="no_unit"))
        enforce.check_datatypes(self.data_cube, enforce=False)

    def test_data_datatype_fail(self):
        """Test error is raised when enforce=False"""
        self.percentile_cube.data = (
            self.percentile_cube.data.astype(np.float64))
        msg = "does not conform"
        with self.assertRaisesRegex(ValueError, msg):
            enforce.check_datatypes(
                self.percentile_cube, enforce=False)

    def test_coord_datatype_fail(self):
        """Test error is raised when enforce=False"""
        self.percentile_cube.coord('percentile').points = (
            self.percentile_cube.coord('percentile').points.astype(np.int32))
        msg = "does not conform"
        with self.assertRaisesRegex(ValueError, msg):
            enforce.check_datatypes(
                self.percentile_cube, enforce=False)

    def test_subset_of_coordinates(self):
        """Test function can enforce on a selected subset of coordinates and
        leave all others unchanged"""
        self.percentile_cube.coord('percentile').points = (
            self.percentile_cube.coord('percentile').points.astype(np.int32))
        self.percentile_cube.coord('forecast_period').points = (
            self.percentile_cube.coord('forecast_period').points.astype(
                np.float32))
        result = enforce.check_datatypes(
            self.percentile_cube, coords=["forecast_period"], enforce=True)
        # unselected coordinate unchanged
        self.assertEqual(result.coord('percentile').dtype, np.int32)
        # selected coordinate changed
        self.assertEqual(result.coord('forecast_period').dtype, np.int32)

    def test_multiple_errors(self):
        """Test a list of errors is correctly caught and re-raised"""
        self.percentile_cube.coord('percentile').points = (
            self.percentile_cube.coord('percentile').points.astype(np.int32))
        self.percentile_cube.coord('forecast_period').points = (
            self.percentile_cube.coord('forecast_period').points.astype(
                np.float32))
        msg = ("The following errors were raised during processing:\n"
               "percentile datatype int32 does not conform to expected "
               "standard \\(\\<class 'numpy.float32'\\>\\)\n"
               "forecast_period datatype float32 does not conform to expected "
               "standard \\(\\<class 'numpy.int32'\\>\\)\n")
        with self.assertRaisesRegex(ValueError, msg):
            enforce.check_datatypes(
                self.percentile_cube, enforce=False)


class Test__check_units_and_dtype(IrisTest):
    """Test method to check object conformance"""

    def setUp(self):
        """Set up test cube"""
        self.cube = set_up_variable_cube(
            data=275.*np.ones((3, 3), dtype=np.float32),
            spatial_grid='equalarea')
        self.coord = self.cube.coord('projection_x_coordinate')

    def test_pass_cube(self):
        """Test return value for compliant cube"""
        result = enforce._check_units_and_dtype(self.cube, 'K', np.float32)
        self.assertTrue(result)

    def test_fail_cube(self):
        """Test return value for non-compliant cube"""
        result = enforce._check_units_and_dtype(
            self.cube, 'degC', np.float32)
        self.assertFalse(result)

    def test_pass_coord(self):
        """Test return value for compliant coordinate"""
        result = enforce._check_units_and_dtype(
            self.coord, 'm', np.float32)
        self.assertTrue(result)

    def test_fail_coord(self):
        """Test return value for non-compliant coordinate"""
        result = enforce._check_units_and_dtype(self.coord, 'm', np.int32)
        self.assertFalse(result)


class Test__convert_coordinate_dtype(IrisTest):
    """Test method to convert coordinate datatypes"""

    def setUp(self):
        """Set up cubes for testing"""
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
        coord = self.cube.coord('time')
        expected = 419524

        coord.convert_units(target_units)
        enforce._convert_coordinate_dtype(coord, np.int64)

        self.assertEqual(coord.points[0], expected)
        self.assertEqual(coord.units, target_units)
        self.assertIsInstance(coord.points[0], np.int64)

    def test_time_coordinate_to_hours_invalid(self):
        """Test that a cube with a validity time on the half hour cannot be
        converted to integer hours."""
        target_units = "hours since 1970-01-01 00:00:00"
        coord = self.cube_non_integer_intervals.coord('time')
        coord.convert_units(target_units)

        msg = ('Data type of coordinate "time" could not be'
               ' enforced without losing significant precision.')
        with self.assertRaisesRegex(ValueError, msg):
            enforce._convert_coordinate_dtype(coord, np.int64)

    def test_time_coordinate_to_hours_float(self):
        """Test that a cube with a validity time on the half hour can be
        converted to float hours."""
        target_units = "hours since 1970-01-01 00:00:00"
        coord = self.cube_non_integer_intervals.coord('time')
        expected = 419524.5

        coord.convert_units(target_units)
        enforce._convert_coordinate_dtype(coord, np.float64)

        self.assertEqual(coord.points[0], expected)
        self.assertEqual(coord.units, target_units)
        self.assertIsInstance(coord.points[0], np.float64)


class Test__convert_diagnostic_dtype(IrisTest):
    """Test method to convert diagnostic (cube.data) datatypes"""

    def setUp(self):
        """Set up cubes for testing"""
        self.cube = set_up_variable_cube(np.ones((5, 5), dtype=np.float32),
                                         spatial_grid='equalarea')
        self.cube_non_integer_intervals = set_up_variable_cube(
            np.ones((5, 5), dtype=np.float32), spatial_grid='equalarea')
        self.cube_non_integer_intervals.data *= 1.5

    def test_temperature_to_integer_kelvin_valid(self):
        """Test that a cube with temperatures at whole kelvin intervals can
        be converted to integer kelvin"""
        expected = np.ones((5, 5), dtype=np.int32)
        enforce._convert_diagnostic_dtype(self.cube, np.int32)
        self.assertArrayEqual(self.cube.data, expected)
        self.assertEqual(self.cube.data.dtype, np.int32)

    def test_temperature_to_integer_kelvin_invalid(self):
        """Test that a cube with temperatures not at whole kelvin intervals
        cannot be converted to integer kelvin"""
        msg = ('Data type of diagnostic "air_temperature" could not be'
               ' enforced without losing significant precision.')
        with self.assertRaisesRegex(ValueError, msg):
            enforce._convert_diagnostic_dtype(
                self.cube_non_integer_intervals, np.int32)


class Test_check_precision_loss(IrisTest):

    """Test the check_precision_loss function behaves as expected."""
    def setUp(self):
        """Make an instance of the plugin that is to be tested."""
        self.plugin = enforce.check_precision_loss

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
