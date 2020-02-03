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
"""Unit tests for weather code utilities."""

import datetime
import os
import pathlib
import unittest
from tempfile import mkdtemp

import iris
import numpy as np
from cf_units import Unit, date2num
from iris.coords import DimCoord
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest

from improver.grids import ELLIPSOID, STANDARD_GRID_CCRS
from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf
from improver.wxcode.utilities import (
    WX_DICT, weather_code_attributes, expand_nested_lists,
    interrogate_decision_tree, update_daynight)

from ...calibration.ensemble_calibration.helper_functions import set_up_cube


def datetime_to_numdateval(year=2018, month=9, day=12, hour=5, minutes=43):
    """
    Convert date and time to a numdateval for use in a cube

    Args:
        year (int):
           require year, default is 2018
        month (int):
           require year, default is 9
        day (int):
           require year, default is 12
        hour (int):
           require year, default is 5
        minutes (int):
           require year, default is 43

    Default values should be roughly sunrise in Exeter.

    Returns:
        float:
           date and time as a value relative to time_origin
    """

    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"
    dateval = datetime.datetime(year, month, day, hour, minutes)
    numdateval = date2num(dateval, time_origin, calendar)
    return numdateval


def _core_wxcube(time_points, num_grid_points):
    """
    Set up a wxcube with unnamed spatial dimensions

    Args:
        time_points (numpy.ndarray):
            Array of time points
        num_grid_points (int):
            Side length of square spatial grid

    Returns:
        iris.cube.Cube:
            cube of weather codes set to 1
            data shape (time_points, num_grid_points, num_grid_points)
    """
    if time_points is None:
        time_points = np.array([datetime_to_numdateval()])

    data = np.ones((len(time_points),
                    num_grid_points,
                    num_grid_points))

    cube = Cube(data, long_name="weather_code",
                units="1", attributes=weather_code_attributes())

    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"
    tunit = Unit(time_origin, calendar)

    cube.add_dim_coord(DimCoord(time_points,
                                standard_name="time", units=tunit), 0)
    return cube


def set_up_wxcube(time_points=None):
    """
    Set up a wxcube

    Args:
        time_points (numpy.ndarray):
            Array of time points

    Returns:
        iris.cube.Cube:
            cube of weather codes set to 1
            data shape (time_points, 16, 16)
            grid covering 0 to 30km west of origin and
            0 to 30km north of origin. origin = 54.9N 2.5W
    """
    num_grid_points = 16
    cube = _core_wxcube(time_points, num_grid_points)

    step_size = 2000
    y_points = np.arange(0, step_size*num_grid_points, step_size)
    cube.add_dim_coord(
        DimCoord(
            y_points,
            'projection_y_coordinate',
            units='m',
            coord_system=STANDARD_GRID_CCRS
        ),
        1
    )

    x_points = np.linspace(-30000, 0, num_grid_points)
    cube.add_dim_coord(
        DimCoord(
            x_points,
            'projection_x_coordinate',
            units='m',
            coord_system=STANDARD_GRID_CCRS
        ),
        2
    )

    return cube


def set_up_wxcube_lat_lon(time_points=None):

    """
    Set up a lat-lon wxcube

    Args:
        time_points (numpy.ndarray):
            Array of time points

    Returns:
        iris.cube.Cube:
            cube of weather codes set to 1
            data shape (time_points, 16, 16)
            grid covering 8W to 7E, 49N to 64N
    """
    num_grid_points = 16
    cube = _core_wxcube(time_points, num_grid_points)

    lon_points = np.linspace(-8, 7, num_grid_points)
    lat_points = np.linspace(49, 64, num_grid_points)

    cube.add_dim_coord(
        DimCoord(lat_points,
                 'latitude',
                 units='degrees',
                 coord_system=ELLIPSOID),
        1
    )
    cube.add_dim_coord(
        DimCoord(lon_points,
                 'longitude',
                 units='degrees',
                 coord_system=ELLIPSOID),
        2
    )
    return cube


class Test_wx_dict(IrisTest):
    """ Test WX_DICT set correctly """

    def test_wxcode_values(self):
        """Check wxcode values are set correctly."""
        self.assertEqual(WX_DICT[0], 'Clear_Night')
        self.assertEqual(WX_DICT[1], 'Sunny_Day')
        self.assertEqual(WX_DICT[2], 'Partly_Cloudy_Night')
        self.assertEqual(WX_DICT[3], 'Partly_Cloudy_Day')
        self.assertEqual(WX_DICT[4], 'Dust')
        self.assertEqual(WX_DICT[5], 'Mist')
        self.assertEqual(WX_DICT[6], 'Fog')
        self.assertEqual(WX_DICT[7], 'Cloudy')
        self.assertEqual(WX_DICT[8], 'Overcast')
        self.assertEqual(WX_DICT[9], 'Light_Shower_Night')
        self.assertEqual(WX_DICT[10], 'Light_Shower_Day')
        self.assertEqual(WX_DICT[11], 'Drizzle')
        self.assertEqual(WX_DICT[12], 'Light_Rain')
        self.assertEqual(WX_DICT[13], 'Heavy_Shower_Night')
        self.assertEqual(WX_DICT[14], 'Heavy_Shower_Day')
        self.assertEqual(WX_DICT[15], 'Heavy_Rain')
        self.assertEqual(WX_DICT[16], 'Sleet_Shower_Night')
        self.assertEqual(WX_DICT[17], 'Sleet_Shower_Day')
        self.assertEqual(WX_DICT[18], 'Sleet')
        self.assertEqual(WX_DICT[19], 'Hail_Shower_Night')
        self.assertEqual(WX_DICT[20], 'Hail_Shower_Day')
        self.assertEqual(WX_DICT[21], 'Hail')
        self.assertEqual(WX_DICT[22], 'Light_Snow_Shower_Night')
        self.assertEqual(WX_DICT[23], 'Light_Snow_Shower_Day')
        self.assertEqual(WX_DICT[24], 'Light_Snow')
        self.assertEqual(WX_DICT[25], 'Heavy_Snow_Shower_Night')
        self.assertEqual(WX_DICT[26], 'Heavy_Snow_Shower_Day')
        self.assertEqual(WX_DICT[27], 'Heavy_Snow')
        self.assertEqual(WX_DICT[28], 'Thunder_Shower_Night')
        self.assertEqual(WX_DICT[29], 'Thunder_Shower_Day')
        self.assertEqual(WX_DICT[30], 'Thunder')


class Test_weather_code_attributes(IrisTest):
    """ Test weather_code_attributes is working correctly """

    def setUp(self):
        """Set up cube """
        data = np.array(
            [0, 1, 5, 11, 20, 5, 9, 10, 4, 2, 0, 1, 29, 30, 1, 5, 6, 6],
            dtype=np.int32
        ).reshape((2, 1, 3, 3))
        self.cube = set_up_cube(data, 'weather_code', '1',
                                realizations=np.array([0, 1], dtype=np.int32))
        self.wxcode = np.array(list(WX_DICT.keys()))
        self.wxmeaning = " ".join(WX_DICT.values())
        self.data_directory = mkdtemp()
        self.nc_file = self.data_directory + '/wxcode.nc'
        pathlib.Path(self.nc_file).touch(exist_ok=True)

    def tearDown(self):
        """Remove temporary directories created for testing."""
        os.remove(self.nc_file)
        os.rmdir(self.data_directory)

    def test_values(self):
        """Test attribute values are correctly set."""
        result = weather_code_attributes()
        self.assertArrayEqual(result['weather_code'], self.wxcode)
        self.assertEqual(result['weather_code_meaning'], self.wxmeaning)

    def test_metadata_saves(self):
        """Test that the metadata saves as NetCDF correctly."""
        self.cube.attributes.update(weather_code_attributes())
        save_netcdf(self.cube, self.nc_file)
        result = load_cube(self.nc_file)
        self.assertArrayEqual(result.attributes['weather_code'], self.wxcode)
        self.assertEqual(result.attributes['weather_code_meaning'],
                         self.wxmeaning)


class Test_expand_nested_lists(IrisTest):
    """ Test expand_nested_lists is working correctly """

    def setUp(self):
        """ Set up dictionary for testing """
        self.dictionary = {'list': ['a', 'a'],
                           'list_of_lists': [['a', 'a'], ['a', 'a']]}

    def test_basic(self):
        """Test that the expand_nested_lists returns a list."""
        result = expand_nested_lists(self.dictionary, 'list')
        self.assertIsInstance(result, list)

    def test_simple_list(self):
        """Testexpand_nested_lists returns a expanded list if given a list."""
        result = expand_nested_lists(self.dictionary, 'list')
        for val in result:
            self.assertEqual(val, 'a')

    def test_list_of_lists(self):
        """Returns a expanded list if given a list of lists."""
        result = expand_nested_lists(self.dictionary, 'list_of_lists')
        for val in result:
            self.assertEqual(val, 'a')


class Test_update_daynight(IrisTest):
    """Test updating weather cube depending on whether it is day or night"""

    def setUp(self):
        """Set up for update_daynight class"""
        self.wxcode = np.array(list(WX_DICT.keys()))
        self.wxmeaning = " ".join(WX_DICT.values())
        self.cube_data = np.array([[
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
            [17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17],
            [18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18],
            [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
            [23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23],
            [26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26],
            [27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27],
            [29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29]]])

    def test_basic(self):
        """Test that the function returns a weather code cube."""
        cube = set_up_wxcube()
        result = update_daynight(cube)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), 'weather_code')
        self.assertEqual(result.units, Unit("1"))
        self.assertArrayEqual(result.attributes['weather_code'], self.wxcode)
        self.assertEqual(result.attributes['weather_code_meaning'],
                         self.wxmeaning)

    def test_raise_error_no_time_coordinate(self):
        """Test that the function raises an error if no time coordinate."""
        cube = set_up_wxcube()
        cube.coord('time').rename('nottime')
        msg = "cube must have time coordinate"
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            update_daynight(cube)

    def test_wxcode_updated(self):
        """Test Correct wxcodes returned for cube."""
        cube = set_up_wxcube()
        cube.data = self.cube_data
        # Only 1,3,10, 14, 17, 20, 23, 26 and 29 change from day to night
        expected_result = np.array([[
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10],
            [13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14],
            [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
            [16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17],
            [18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18],
            [19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20],
            [22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23],
            [25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26],
            [27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27],
            [28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29]]])
        result = update_daynight(cube)
        self.assertArrayEqual(result.data, expected_result)

    def test_wxcode_time_as_attribute(self):
        """ Test code works if time is an attribute not a dimension """
        cube = set_up_wxcube()
        cube.data = self.cube_data
        cube = iris.util.squeeze(cube)
        expected_result = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10],
            [13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14],
            [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
            [16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17],
            [18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18],
            [19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20],
            [22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23],
            [25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26],
            [27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27],
            [28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29]])
        result = update_daynight(cube)

        self.assertArrayEqual(result.data, expected_result)
        self.assertEqual(result.data.shape, (16, 16))

    def test_wxcode_time_different_seconds(self):
        """ Test code works if time coordinate has a difference in the number
        of seconds, which should round to the same time in hours and minutes.
        This was raised by changes to cftime which altered its precision."""
        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        dateval = datetime.datetime(2018, 9, 12, 5, 42, 59)
        numdateval = date2num(dateval, time_origin, calendar)
        time_points = [numdateval]

        cube = set_up_wxcube(time_points=time_points)
        cube.data = self.cube_data
        cube = iris.util.squeeze(cube)
        expected_result = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10],
            [13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14],
            [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
            [16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17],
            [18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18],
            [19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20],
            [22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23],
            [25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26],
            [27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27],
            [28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29]])
        result = update_daynight(cube)

        self.assertArrayEqual(result.data, expected_result)
        self.assertEqual(result.data.shape, (16, 16))

    def test_wxcode_time_as_array(self):
        """ Test code works if time is an array of dimension > 1 """
        num1 = datetime_to_numdateval(year=2018, month=9, day=12, hour=5,
                                      minutes=0)
        num2 = datetime_to_numdateval(year=2018, month=9, day=12, hour=6,
                                      minutes=0)
        num3 = datetime_to_numdateval(year=2018, month=9, day=12, hour=7,
                                      minutes=0)
        cube = set_up_wxcube(time_points=[num1, num2, num3])
        expected_result = np.ones((3, 16, 16))
        expected_result[0, :, :] = 0
        result = update_daynight(cube)
        self.assertArrayEqual(result.data, expected_result)

    def test_basic_lat_lon(self):
        """Test that the function returns a weather code lat lon cube.."""
        cube = set_up_wxcube_lat_lon()
        result = update_daynight(cube)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), 'weather_code')
        self.assertEqual(result.units, Unit("1"))
        self.assertArrayEqual(result.attributes['weather_code'], self.wxcode)
        self.assertEqual(result.attributes['weather_code_meaning'],
                         self.wxmeaning)

    def test_wxcode_updated_on_latlon(self):
        """Test Correct wxcodes returned for lat lon cube."""
        cube = set_up_wxcube_lat_lon()
        cube.data = self.cube_data

        expected_result = np.array([[
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            [13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
            [16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17],
            [18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18],
            [19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
            [22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23],
            [25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26],
            [27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27],
            [28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29]]])
        result = update_daynight(cube)
        self.assertArrayEqual(result.data, expected_result)


class Test_interrogate_decision_tree(IrisTest):
    """Test the function for generating extended help."""

    def test_return_type(self):
        """Test that the function returns a string."""
        result = interrogate_decision_tree('global')
        self.assertIsInstance(result, str)

    def test_raises_exception(self):
        """Test the function raises an exception for an unknown weather symbol
        tree name."""
        msg = "Unknown decision tree name provided."
        with self.assertRaisesRegex(ValueError, msg):
            interrogate_decision_tree('kittens')


if __name__ == '__main__':
    unittest.main()
