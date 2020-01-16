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
"""Unit tests for the improver.metadata.check_datatypes module."""

import unittest

import numpy as np
from iris.coords import AuxCoord
from iris.tests import IrisTest

from improver.metadata.check_datatypes import (
    _check_units_and_dtype, _construct_object_list, check_cube_not_float64,
    check_datatypes, check_time_coordinate_metadata)

from ..set_up_test_cubes import (
    set_up_percentile_cube, set_up_probability_cube, set_up_variable_cube)


class Test_check_cube_not_float64(IrisTest):

    """Test whether a cube contains any float64 values."""

    def setUp(self):
        """Set up a test cube with the following data and coordinates, which
        comply with the IMPROVER datatypes standard:

        +-------------------------+-------------+
        | Name                    | Datatype    |
        +=========================+=============+
        | data (air_temperature)  | np.float32  |
        +-------------------------+-------------+
        | projection_x_coordinate | np.float32  |
        +-------------------------+-------------+
        | projection_y_coordinate | np.float32  |
        +-------------------------+-------------+
        | time                    | np.int64    |
        +-------------------------+-------------+
        | forecast_reference_time | np.int64    |
        +-------------------------+-------------+
        | forecast_period         | np.int32    |
        +-------------------------+-------------+
        """
        self.cube = set_up_variable_cube(280*np.ones((5, 5), dtype=np.float32),
                                         spatial_grid='equalarea')

    def test_success(self):
        """Test a cube that should pass does not throw an error."""
        check_cube_not_float64(self.cube)

    def test_float64_cube_data(self):
        """Test a failure of a cube with 64 bit data."""
        self.cube.data = self.cube.data.astype(np.float64)
        msg = "64 bit cube not allowed"
        with self.assertRaisesRegex(TypeError, msg):
            check_cube_not_float64(self.cube)

    def test_float64_cube_data_with_fix(self):
        """Test a cube with 64 bit data is converted to 32 bit data."""
        self.cube.data = self.cube.data.astype(np.float64)
        check_cube_not_float64(self.cube, fix=True)
        self.assertEqual(self.cube.data.dtype, np.float32)

    def test_float64_cube_coord_points(self):
        """Test a failure of a cube with 64 bit coord points."""
        self.cube.coord("projection_x_coordinate").points = (
            self.cube.coord("projection_x_coordinate").points.astype(
                np.float64)
        )
        msg = "64 bit coord points not allowed"
        with self.assertRaisesRegex(TypeError, msg):
            check_cube_not_float64(self.cube)

    def test_float64_cube_coord_points_with_fix(self):
        """Test a failure of a cube with 64 bit coord points."""
        self.cube.coord("projection_x_coordinate").points = (
            self.cube.coord("projection_x_coordinate").points.astype(
                np.float64))
        check_cube_not_float64(self.cube, fix=True)
        coord = self.cube.coord("projection_x_coordinate")
        self.assertEqual(coord.points.dtype, np.float32)

    def test_float64_cube_coord_bounds(self):
        """Test a failure of a cube with 64 bit coord bounds."""
        x_coord = self.cube.coord("projection_x_coordinate")
        # Default np.array for float input is np.float64.
        x_coord.bounds = (
            np.array([(point - 10., point + 10.) for point in x_coord.points])
        )
        msg = "64 bit coord bounds not allowed"
        with self.assertRaisesRegex(TypeError, msg):
            check_cube_not_float64(self.cube)

    def test_float64_cube_coord_bounds_with_fix(self):
        """Test a failure of a cube with 64 bit coord bounds."""
        x_coord = self.cube.coord("projection_x_coordinate")
        # Default np.array for float input is np.float64.
        x_coord.bounds = (
            np.array([(point - 10., point + 10.) for point in x_coord.points])
        )
        check_cube_not_float64(self.cube, fix=True)
        coord = self.cube.coord("projection_x_coordinate")
        self.assertEqual(coord.points.dtype, np.float32)
        self.assertEqual(coord.bounds.dtype, np.float32)


class Test__construct_object_list(IrisTest):
    """Test the private _construct_object_list method"""

    def setUp(self):
        """Make a template cube"""
        self.cube = set_up_variable_cube(
            278*np.ones((3, 4, 4), dtype=np.float32))

    def test_basic(self):
        """Test it works for all coordinates"""
        expected_result = {
            self.cube, self.cube.coord('realization'),
            self.cube.coord('latitude'), self.cube.coord('longitude'),
            self.cube.coord('time'), self.cube.coord('forecast_period'),
            self.cube.coord('forecast_reference_time')}
        result = _construct_object_list(self.cube, None)
        self.assertSetEqual(set(result), expected_result)

    def test_subset(self):
        """Test it works on a subset and ignores any missing coordinates"""
        expected_result = {
            self.cube, self.cube.coord('realization'), self.cube.coord('time')}
        result = _construct_object_list(
            self.cube, ['realization', 'time', 'kittens'])
        self.assertSetEqual(set(result), expected_result)


class Test_check_datatypes(IrisTest):
    """Test datatype checking"""

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

    def test_conformant_cubes(self):
        """Test conformant data, percentile and probability cubes all pass
        (no error is thrown)"""
        cubelist = [
            self.data_cube, self.probability_cube, self.percentile_cube]
        for cube in cubelist:
            check_datatypes(cube)

    def test_string_coord(self):
        """Test string coordinate does not throw an error"""
        self.data_cube.add_aux_coord(
            AuxCoord(["ukv"], long_name="model", units="no_unit"))
        check_datatypes(self.data_cube)

    def test_data_datatype_fail(self):
        """Test error is raised for 64-bit data"""
        self.percentile_cube.data = (
            self.percentile_cube.data.astype(np.float64))
        msg = "does not conform"
        with self.assertRaisesRegex(ValueError, msg):
            check_datatypes(self.percentile_cube)

    def test_coord_datatype_fail(self):
        """Test error is raised for 64-bit coordinate"""
        self.percentile_cube.coord('percentile').points = (
            self.percentile_cube.coord('percentile').points.astype(np.float64))
        msg = "does not conform"
        with self.assertRaisesRegex(ValueError, msg):
            check_datatypes(self.percentile_cube)

    def test_coord_bounds_datatype_fail(self):
        """Test error is raised for a coordinate whose bounds datatype is
        incorrect, but points are correct"""
        time_bounds = np.array(
            [self.data_cube.coord("time").points[0] - 3600,
             self.data_cube.coord("time").points[0] + 3600], dtype=np.int32)
        self.data_cube.coord("time").bounds = [time_bounds]
        msg = "does not conform"
        with self.assertRaisesRegex(ValueError, msg):
            check_datatypes(self.data_cube)

    def test_subset_of_coordinates(self):
        """Test function can check a selected subset of coordinates and
        ignore others"""
        self.percentile_cube.coord('percentile').points = (
            self.percentile_cube.coord('percentile').points.astype(np.float64))
        check_datatypes(
            self.percentile_cube, coords=["forecast_period"])
        self.assertEqual(
            self.percentile_cube.coord('percentile').dtype, np.float64)

    def test_multiple_errors(self):
        """Test a list of errors is correctly caught and re-raised"""
        self.percentile_cube.coord('percentile').points = (
            self.percentile_cube.coord('percentile').points.astype(np.float64))
        self.percentile_cube.coord('forecast_period').points = (
            self.percentile_cube.coord('forecast_period').points.astype(
                np.int64))
        msg = ("percentile datatype float64 does not conform to expected "
               "standard \\(\\<class 'numpy.float32'\\>\\)\n"
               "forecast_period datatype int64 does not conform to expected "
               "standard \\(\\<class 'numpy.int32'\\>\\)\n")
        with self.assertRaisesRegex(ValueError, msg):
            check_datatypes(self.percentile_cube)


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
        result = _check_units_and_dtype(self.cube, 'K', np.float32)
        self.assertTrue(result)

    def test_fail_cube(self):
        """Test return value for non-compliant cube"""
        result = _check_units_and_dtype(
            self.cube, 'degC', np.float32)
        self.assertFalse(result)

    def test_pass_coord(self):
        """Test return value for compliant coordinate"""
        result = _check_units_and_dtype(
            self.coord, 'm', np.float32)
        self.assertTrue(result)

    def test_fail_coord(self):
        """Test return value for non-compliant coordinate"""
        result = _check_units_and_dtype(self.coord, 'm', np.int32)
        self.assertFalse(result)


class Test_check_time_coordinate_metadata(IrisTest):
    """Test check_time_coordinate_metatadata function"""

    def setUp(self):
        """Set up a test cube"""
        self.cube = set_up_variable_cube(278*np.ones((4, 4), dtype=np.float32))

    def test_basic(self):
        """Test success"""
        check_time_coordinate_metadata(self.cube)

    def test_fails_wrong_datatype(self):
        """Test failure if any coordinate datatype is wrong"""
        self.cube.coord("time").points = (
            self.cube.coord("time").points.astype(np.float64))
        msg = 'Coordinate time does not match required standard'
        with self.assertRaisesRegex(ValueError, msg):
            check_time_coordinate_metadata(self.cube)

    def test_fails_wrong_units(self):
        """Test failure if any coordinate unit is wrong"""
        self.cube.coord("forecast_period").convert_units("hours")
        msg = 'Coordinate forecast_period does not match required standard'
        with self.assertRaisesRegex(ValueError, msg):
            check_time_coordinate_metadata(self.cube)


if __name__ == '__main__':
    unittest.main()
