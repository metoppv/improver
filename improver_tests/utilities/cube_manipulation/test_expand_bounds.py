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
"""
Unit tests for the function "cube_manipulation.expand_bounds".
"""
import unittest
from datetime import datetime as dt

import iris
import numpy as np
from cf_units import date2num
from iris.tests import IrisTest

from improver.utilities.cube_manipulation import expand_bounds

from ...set_up_test_cubes import set_up_variable_cube

TIME_UNIT = 'seconds since 1970-01-01 00:00:00'
CALENDAR = 'gregorian'


class Test_expand_bounds(IrisTest):

    """Test expand_bounds function"""

    def setUp(self):
        """Set up a cubelist for testing"""

        data = 275.5*np.ones((3, 3), dtype=np.float32)
        frt = dt(2015, 11, 19, 0)
        time_points = [dt(2015, 11, 19, 1), dt(2015, 11, 19, 3)]
        time_bounds = [[dt(2015, 11, 19, 0), dt(2015, 11, 19, 2)],
                       [dt(2015, 11, 19, 1), dt(2015, 11, 19, 3)]]

        self.cubelist = iris.cube.CubeList([])
        for tpoint, tbounds in zip(time_points, time_bounds):
            cube = set_up_variable_cube(
                data, frt=frt, time=tpoint, time_bounds=tbounds)
            self.cubelist.append(cube)

        self.expected_bounds_seconds = [
            date2num(dt(2015, 11, 19, 0), TIME_UNIT,
                     CALENDAR).astype(np.int64),
            date2num(dt(2015, 11, 19, 3), TIME_UNIT,
                     CALENDAR).astype(np.int64)]

        self.expected_bounds_hours = [
            date2num(dt(2015, 11, 19, 0), 'hours since 1970-01-01 00:00:00',
                     CALENDAR),
            date2num(dt(2015, 11, 19, 3), 'hours since 1970-01-01 00:00:00',
                     CALENDAR)]

    def test_basic_time_mid(self):
        """Test that expand_bound produces sensible bounds
        when given arg 'mid' for times in seconds"""
        time_point = np.around(date2num(dt(2015, 11, 19, 1, 30), TIME_UNIT,
                                        CALENDAR)).astype(np.int64)
        expected_result = iris.coords.DimCoord(
            [time_point], bounds=self.expected_bounds_seconds,
            standard_name='time', units=TIME_UNIT)
        result = expand_bounds(
            self.cubelist[0], self.cubelist, ['time'], use_midpoint=True)
        self.assertEqual(result.coord('time'), expected_result)
        self.assertEqual(result.coord('time').dtype, np.int64)

    def test_time_mid_data_precision(self):
        """Test that expand_bound does not escalate precision when input is
        of dtype int32"""
        expected_result = iris.coords.DimCoord(
            np.array([5400], dtype=np.int32),
            bounds=np.array([0, 10800], dtype=np.int32),
            standard_name='forecast_period', units='seconds')
        result = expand_bounds(
            self.cubelist[0], self.cubelist, ['forecast_period'],
            use_midpoint=True)
        self.assertEqual(result.coord('forecast_period'), expected_result)
        self.assertEqual(result.coord('forecast_period').dtype, np.int32)

    def test_float_time_mid(self):
        """Test that expand_bound produces sensible bounds
        when given arg 'mid' for times in hours"""
        time_unit = 'hours since 1970-01-01 00:00:00'
        for cube in self.cubelist:
            cube.coord("time").convert_units(time_unit)
        time_point = date2num(dt(2015, 11, 19, 1, 30), time_unit, CALENDAR)
        expected_result = iris.coords.DimCoord(
            [time_point], bounds=self.expected_bounds_hours,
            standard_name='time', units=time_unit)
        result = expand_bounds(
            self.cubelist[0], self.cubelist, ['time'], use_midpoint=True)
        self.assertEqual(result.coord('time'), expected_result)
        self.assertEqual(result.coord('time').dtype, np.float32)

    def test_basic_time_upper(self):
        """Test that expand_bound produces sensible bounds
        when given arg 'upper'"""
        time_point = np.around(date2num(dt(2015, 11, 19, 3), TIME_UNIT,
                                        CALENDAR)).astype(np.int64)
        expected_result = iris.coords.DimCoord(
            [time_point], bounds=self.expected_bounds_seconds,
            standard_name='time', units=TIME_UNIT)
        result = expand_bounds(self.cubelist[0], self.cubelist, ['time'])
        self.assertEqual(result.coord('time'), expected_result)

    def test_multiple_coordinate_expanded(self):
        """Test that expand_bound produces sensible bounds when more than one
        coordinate is operated on, in this case expanding both the time and
        forecast period coordinates."""
        time_point = np.around(date2num(dt(2015, 11, 19, 3), TIME_UNIT,
                                        CALENDAR)).astype(np.int64)
        expected_result_time = iris.coords.DimCoord(
            [time_point], bounds=self.expected_bounds_seconds,
            standard_name='time', units=TIME_UNIT)
        expected_result_fp = iris.coords.DimCoord(
            [10800], bounds=[0, 10800],
            standard_name='forecast_period', units='seconds')

        result = expand_bounds(self.cubelist[0], self.cubelist,
                               ['time', 'forecast_period'])
        self.assertEqual(result.coord('time'), expected_result_time)
        self.assertEqual(result.coord('forecast_period'), expected_result_fp)
        self.assertEqual(result.coord('time').dtype, np.int64)

    def test_basic_no_time_bounds(self):
        """Test that it creates appropriate bounds if there are no time bounds
        """
        for cube in self.cubelist:
            cube.coord('time').bounds = None

        time_point = np.around(date2num(dt(2015, 11, 19, 2), TIME_UNIT,
                                        CALENDAR)).astype(np.int64)
        time_bounds = [
            np.around(date2num(dt(2015, 11, 19, 1), TIME_UNIT,
                               CALENDAR)).astype(np.int64),
            np.around(date2num(dt(2015, 11, 19, 3), TIME_UNIT,
                               CALENDAR)).astype(np.int64)]
        expected_result = iris.coords.DimCoord(
            time_point, bounds=time_bounds,
            standard_name='time', units=TIME_UNIT)

        result = expand_bounds(self.cubelist[0], self.cubelist, ['time'],
                               use_midpoint=True)
        self.assertEqual(result.coord('time'), expected_result)

    def test_fails_with_multi_point_coord(self):
        """Test that if an error is raised if a coordinate with more than
        one point is given"""
        emsg = 'the expand bounds function should only be used on a'
        with self.assertRaisesRegex(ValueError, emsg):
            expand_bounds(
                self.cubelist[0], self.cubelist, ['latitude'])


if __name__ == '__main__':
    unittest.main()
