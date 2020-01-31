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
from iris.tests import IrisTest
from iris.cube import CubeList

from improver.metadata.check_datatypes import (
    check_mandatory_standards, check_units)

from ..set_up_test_cubes import (
    set_up_percentile_cube, set_up_probability_cube, set_up_variable_cube)


class Test_check_mandatory_standards(IrisTest):

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
        self.cube = set_up_variable_cube(280*np.ones((3, 3), dtype=np.float32),
                                         spatial_grid='equalarea')

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
        (no error is thrown and cube is not changed)"""
        cubelist = [
            self.cube, self.probability_cube, self.percentile_cube]
        for cube in cubelist:
            result = cube.copy()
            check_mandatory_standards(result)
            self.assertStringEqual(CubeList([cube]).xml(checksum=True),
                                   CubeList([result]).xml(checksum=True))

    def test_int64_cube_data(self):
        """Test conformant data with a cube with 64 bit integer data."""
        self.cube.data = self.cube.data.astype(np.int64)
        check_mandatory_standards(self.cube)

    def test_float64_cube_data(self):
        """Test a failure of a cube with 64 bit data."""
        self.cube.data = self.cube.data.astype(np.float64)
        msg = ("does not have required dtype.\n"
               "Expected: float32, Actual: float64")
        with self.assertRaisesRegex(ValueError, msg):
            check_mandatory_standards(self.cube)

    def test_float64_cube_coord_points(self):
        """Test a failure of a cube with 64 bit coord points."""
        self.cube.coord("projection_x_coordinate").points = (
            self.cube.coord("projection_x_coordinate").points.astype(
                np.float64)
        )
        msg = ("does not have required dtype.\n"
               "Expected: float32, Actual \\(points\\): float64")
        with self.assertRaisesRegex(ValueError, msg):
            check_mandatory_standards(self.cube)

    def test_float64_cube_coord_bounds(self):
        """Test a failure of a cube with 64 bit coord bounds."""
        x_coord = self.cube.coord("projection_x_coordinate")
        x_coord.bounds = (
            np.array([(point - 10., point + 10.) for point in x_coord.points],
                     dtype=np.float64)
        )
        msg = ("does not have required dtype.\n"
               "Expected: float32, "
               "Actual \\(points\\): float32, "
               "Actual \\(bounds\\): float64")
        with self.assertRaisesRegex(ValueError, msg):
            check_mandatory_standards(self.cube)

    def test_multiple_errors(self):
        """Test a list of errors is correctly caught and re-raised"""
        self.percentile_cube.coord('percentile').points = (
            self.percentile_cube.coord('percentile').points.astype(np.float64))
        self.percentile_cube.coord('forecast_period').points = (
            self.percentile_cube.coord('forecast_period').points.astype(
                np.int64))
        msg = ("percentile of type .*DimCoord.* "
               "does not have required dtype.\n"
               "Expected: float32, Actual \\(points\\): float64\n"
               "forecast_period of type .*DimCoord.* "
               "does not have required dtype.\n"
               "Expected: int32, Actual \\(points\\): int64")
        with self.assertRaisesRegex(ValueError, msg):
            check_mandatory_standards(self.percentile_cube)


class Test_check_units(IrisTest):
    """Test method to check object units"""

    def setUp(self):
        """Set up test cube"""
        self.cube = set_up_variable_cube(
            data=275.*np.ones((3, 3), dtype=np.float32),
            spatial_grid='equalarea')
        self.coord = self.cube.coord('forecast_period')

    def test_pass_cube(self):
        """Test return value for cube (no requirement on units)"""
        input_cube = self.cube.copy()
        result = check_units(input_cube)
        self.assertTrue(result)
        self.assertStringEqual(CubeList([self.cube]).xml(checksum=True),
                               CubeList([input_cube]).xml(checksum=True))

    def test_pass_coord(self):
        """Test return value for time coordinate with correct units"""
        result = check_units(self.coord)
        self.assertTrue(result)

    def test_fail_coord(self):
        """Test return value for time coordinate with wrong units"""
        self.coord.convert_units('minutes')
        result = check_units(self.coord)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
