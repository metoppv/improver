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
""" Unit tests for DayNightMask class """

import unittest

import cf_units as unit
import iris
import numpy as np
from iris.tests import IrisTest

from improver.utilities.solar import DayNightMask

from ...nbhood.nbhood.test_BaseNeighbourhoodProcessing import (
    set_up_cube, set_up_cube_lat_long)


class Test__init__(IrisTest):
    """ Test initialisation of the DayNightMask class """

    def test_basic_init(self):
        """ Test Initiation of DayNightMask Object"""
        plugin = DayNightMask()
        self.assertEqual(plugin.day, 1)
        self.assertEqual(plugin.night, 0)


class Test__repr__(IrisTest):
    """ Test string representation """

    def test_basic_repr(self):
        """ Test Representation string of DayNightMask Object"""
        expected = '<DayNightMask : Day = 1, Night = 0>'
        result = str(DayNightMask())
        self.assertEqual(result, expected)


class Test__create_daynight_mask(IrisTest):
    """ Test string representation """

    def setUp(self):
        """Set up the cube for testing."""
        self.cube = set_up_cube()

    def test_basic_daynight_mask(self):
        """ Test this create a blank mask cube"""
        result = DayNightMask()._create_daynight_mask(self.cube)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.long_name, 'day_night_mask')
        self.assertEqual(result.units, unit.Unit('1'))
        self.assertEqual(result.data.min(), DayNightMask().night)
        self.assertEqual(result.data.max(), DayNightMask().night)


class Test__daynight_lat_lon_cube(IrisTest):
    """ Test string representation """

    def setUp(self):
        """Set up the cube for testing."""
        cube = set_up_cube_lat_long()
        lon_points = np.linspace(-8, 7, 16)
        lon_points_360 = np.linspace(345, 360, 16)
        lat_points = np.linspace(49, 64, 16)
        cube.coord('latitude').points = lat_points
        cube.coord('longitude').points = lon_points
        self.mask_cube = DayNightMask()._create_daynight_mask(cube)[0]
        cube_360 = cube.copy()
        cube_360.coord('longitude').points = lon_points_360
        self.mask_cube_360 = DayNightMask()._create_daynight_mask(cube_360)[0]

    def test_basic_lat_lon_cube(self):
        """ Test this create a blank mask cube"""
        day_of_year = 10
        utc_hour = 12.0
        expected_result = np.ones((16, 16))
        result = DayNightMask()._daynight_lat_lon_cube(self.mask_cube,
                                                       day_of_year,
                                                       utc_hour)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.data, expected_result)

    def test_basic_lat_lon_cube_360(self):
        """ Test this still works with 360 data"""
        day_of_year = 10
        utc_hour = 0.0
        expected_result = np.zeros((16, 16))
        result = DayNightMask()._daynight_lat_lon_cube(self.mask_cube_360,
                                                       day_of_year,
                                                       utc_hour)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.data, expected_result)


class Test_process(IrisTest):

    """Test DayNight Mask."""

    def setUp(self):
        """Set up the cubes for testing."""
        self.cube = set_up_cube()
        x_points = np.linspace(-30000, 0, 16)
        self.cube.coord('projection_x_coordinate').points = x_points
        dtval = self.cube.coord('time').points[0]
        self.cube.coord('time').points = np.array(dtval + 7.5 + 24.0)
        # Lat lon cube
        self.cube_lat_lon = set_up_cube_lat_long()
        lon_points = np.linspace(-8, 7, 16)
        lat_points = np.linspace(49, 64, 16)
        self.cube_lat_lon.coord('latitude').points = lat_points
        self.cube_lat_lon.coord('longitude').points = lon_points
        dt = self.cube_lat_lon.coord('time').points[0]
        self.cube_lat_lon.coord('time').points[0] = dt + 7.5 + 24.0
        self.cube_lat_lon_360 = self.cube_lat_lon.copy()
        lon_points_360 = np.linspace(345, 360, 16)
        self.cube_lat_lon_360.coord('longitude').points = lon_points_360

    def test_basic_standard_grid_ccrs(self):
        """Test day_night mask with standard_grid_ccrs projection."""
        result = DayNightMask().process(self.cube)
        expected_result = np.array([[
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])
        self.assertArrayEqual(result.data, expected_result)

    def test_basic_lat_lon(self):
        """Test day_night mask with lat lon data."""
        result = DayNightMask().process(self.cube_lat_lon)
        expected_result = np.array([[
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])
        self.assertArrayEqual(result.data, expected_result)

    def test_basic_lat_lon_360(self):
        """Test day_night mask with lat lon data 360 data."""
        result = DayNightMask().process(self.cube_lat_lon_360)
        expected_result = np.array([[
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])
        self.assertArrayEqual(result.data, expected_result)


if __name__ == '__main__':
    unittest.main()
