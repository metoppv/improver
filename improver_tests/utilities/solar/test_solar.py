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
"""Unit tests for the solar calculations in solar.py """

import unittest

import numpy as np
from iris.tests import IrisTest

from improver.utilities.solar import (
    calc_solar_declination, calc_solar_elevation, calc_solar_hour_angle,
    daynight_terminator)


class Test_calc_solar_declination(IrisTest):
    """Test Solar declination."""

    def test_basic_solar_declination(self):
        """Test the calc of declination for different days of the year"""
        day_of_year = [1, 10, 100, 365]
        expected_result = [-23.1223537018, -22.1987731152,
                           7.20726681123, -23.2078460336]
        for i, dayval in enumerate(day_of_year):
            result = calc_solar_declination(dayval)
            self.assertIsInstance(result, float)
            self.assertAlmostEqual(result, expected_result[i])

    def test_solar_dec_raises_exception(self):
        """Test an exception is raised if day of year out of range"""
        day_of_year = -1
        msg = 'Day of the year must be between 0 and 365'
        with self.assertRaisesRegex(ValueError, msg):
            calc_solar_declination(day_of_year)


class Test_calc_solar_hour_angle(IrisTest):
    """Test Calculation of the Solar Hour angle."""

    def setUp(self):
        """Set up the longitudes."""
        self.longitudes = np.array([0.0, 10.0, -10.0, 180.0, -179.0])
        self.day_of_year = 10
        self.utc_hour = 12.0

    def test_basic_solar_hour_angle(self):
        """Test the calculation of solar hour_angle. Single Value"""
        result = calc_solar_hour_angle(0.0, 10, 0.0)
        expected_result = -181.783274102
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, expected_result)

    def test_basic_solar_hour_angle_array(self):
        """Test the calc of solar hour_angle for an array of longitudes"""
        result = calc_solar_hour_angle(self.longitudes, self.day_of_year,
                                       self.utc_hour)
        expected_result = np.array([-1.7832741, 8.2167259, -11.7832741,
                                    178.2167259, -180.7832741])
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_result)

    def test_basic_solar_hour_angle_array_360(self):
        """Test the calc of solar hour_angle for longitudes > 180"""
        longitudes = np.array([0.0, 10.0, 350.0, 180.0, 181.0])
        result = calc_solar_hour_angle(longitudes, self.day_of_year,
                                       self.utc_hour)
        expected_result = np.array([-1.7832741, 8.2167259, 360.0-11.7832741,
                                    178.2167259, 360.0-180.7832741])
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_result)

    def test_solar_hour_raises_exception_day_of_year(self):
        """Test an exception is raised if day of year out of range"""
        day_of_year = 367
        msg = 'Day of the year must be between 0 and 365'
        with self.assertRaisesRegex(ValueError, msg):
            calc_solar_hour_angle(self.longitudes, day_of_year,
                                  self.utc_hour)

    def test_solar_hour_raises_exception_hour(self):
        """Test an exception is raised if hour out of range"""
        utc_hour = -10.0
        msg = 'Hour must be between 0 and 24.0'
        with self.assertRaisesRegex(ValueError, msg):
            calc_solar_hour_angle(self.longitudes, self.day_of_year, utc_hour)


class Test_calc_solar_elevation(IrisTest):
    """Test Calculation of the Solar Elevation."""

    def setUp(self):
        """Set up the longitudes."""
        self.latitudes = np.array([50.0, 50.0, 50.0])
        self.longitudes = np.array([-5.0, 0.0, 5.0])
        self.day_of_year = 10
        self.utc_hour = 8.0

    def test_basic_solar_elevation(self):
        """Test the solar elevation for a single point over several hours."""
        expected_results = [-0.460611756793, 6.78261282655,
                            1.37746106416, -6.75237871867]
        for i, hour in enumerate([8.0, 9.0, 16.0, 17.0]):
            result = calc_solar_elevation(50.0, 0.0, 10, hour)
            self.assertIsInstance(result, float)
            self.assertAlmostEqual(result, expected_results[i])

    def test_basic_solar_elevation_array(self):
        """Test the solar elevation for an array of lats and lons."""
        expected_array = np.array([-3.1423043, -0.46061176, 2.09728301])
        result = calc_solar_elevation(self.latitudes, self.longitudes,
                                      self.day_of_year, self.utc_hour)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_array)

    def test_basic_solar_elevation_array_360(self):
        """Test the solar elevation for lons > 180."""
        longitudes = np.array([355.0, 0.0, 5.0])
        expected_array = np.array([-3.1423043, -0.46061176, 2.09728301])
        result = calc_solar_elevation(self.latitudes, longitudes,
                                      self.day_of_year, self.utc_hour)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_array)

    def test_solar_elevation_raises_exception_lat(self):
        """Test an exception is raised if latitudes out of range"""
        latitudes = np.array([-150.0, 50.0, 50.0])
        msg = 'Latitudes must be between -90.0 and 90.0'
        with self.assertRaisesRegex(ValueError, msg):
            calc_solar_elevation(latitudes, self.longitudes,
                                 self.day_of_year, self.utc_hour)

    def test_solar_elevation_raises_exception_day_of_year(self):
        """Test an exception is raised if day of year out of range"""
        day_of_year = 367
        msg = 'Day of the year must be between 0 and 365'
        with self.assertRaisesRegex(ValueError, msg):
            calc_solar_elevation(self.latitudes, self.longitudes,
                                 day_of_year, self.utc_hour)

    def test_solar_elevation_raises_exception_hour(self):
        """Test an exception is raised if hour out of range"""
        utc_hour = -10.0
        msg = 'Hour must be between 0 and 24.0'
        with self.assertRaisesRegex(ValueError, msg):
            calc_solar_elevation(self.latitudes, self.longitudes,
                                 self.day_of_year, utc_hour)

    def test_sine_solar_elevation(self):
        """Test the solar elevation with return_sine equal true."""
        expected_results = [-0.00803911,  0.11810263,  0.02403892,
                            -0.11757863]
        for i, hour in enumerate([8.0, 9.0, 16.0, 17.0]):
            result = calc_solar_elevation(50.0, 0.0, 10, hour,
                                          return_sine=True)
            self.assertIsInstance(result, float)
            self.assertAlmostEqual(result, expected_results[i])


class Test_daynight_terminator(IrisTest):

    """Test DayNight terminator."""

    def setUp(self):
        """Set up the longitudes."""
        self.longitudes = np.linspace(-180.0, 180.0, 21)
        self.day_of_year = 10
        self.utc_hour = 12.0

    def test_basic_winter(self):
        """Test we get the terminator in winter."""
        result = daynight_terminator(self.longitudes, 10, 12.0)
        expected_lats = np.array([-67.79151577, -66.97565648, -63.73454167,
                                  -56.33476507, -39.67331783, -4.36090124,
                                  34.38678745, 54.03238506, 62.69165892,
                                  66.55543647, 67.79151577, 66.97565648,
                                  63.73454167, 56.33476507, 39.67331783,
                                  4.36090124, -34.38678745, -54.03238506,
                                  -62.69165892, -66.55543647, -67.79151577])
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_lats)

    def test_basic_spring(self):
        """Test we get the terminator in spring."""
        result = daynight_terminator(self.longitudes, 100, 0.0)
        expected_lats = np.array([-82.7926115, -82.44008918, -81.15273499,
                                  -77.9520655, -68.09955141, -2.64493534,
                                  67.37718951, 77.7625883, 81.07852035,
                                  82.41166928, 82.7926115, 82.44008918,
                                  81.15273499, 77.95206547, 68.09955141,
                                  2.64493534, -67.37718951, -77.7625883,
                                  -81.07852035, -82.41166928, -82.7926115])
        self.assertArrayAlmostEqual(result, expected_lats)

    def test_basic_winter_360(self):
        """Test we get the terminator in winter with lon > 180."""
        longitudes = np.linspace(0.0, 360.0, 21)
        result = daynight_terminator(longitudes, 10, 12.0)
        expected_lats = np.array([67.79151577, 66.97565648,
                                  63.73454167, 56.33476507, 39.67331783,
                                  4.36090124, -34.38678745, -54.03238506,
                                  -62.69165892, -66.55543647, -67.79151577,
                                  -66.97565648, -63.73454167,
                                  -56.33476507, -39.67331783, -4.36090124,
                                  34.38678745, 54.03238506, 62.69165892,
                                  66.55543647, 67.79151577])
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_lats)

    def test_basic_sprint_360(self):
        """Test we get the terminator in spring with lon > 180."""
        longitudes = np.linspace(0.0, 360.0, 21)
        result = daynight_terminator(longitudes, 100, 0.0)
        expected_lats = np.array([82.7926115, 82.44008918,
                                  81.15273499, 77.95206547, 68.09955141,
                                  2.64493534, -67.37718951, -77.7625883,
                                  -81.07852035, -82.41166928, -82.7926115,
                                  -82.44008918, -81.15273499,
                                  -77.9520655, -68.09955141, -2.64493534,
                                  67.37718951, 77.7625883, 81.07852035,
                                  82.41166928, 82.7926115])
        self.assertArrayAlmostEqual(result, expected_lats)

    def test_daynight_terminator_raises_exception_day_of_year(self):
        """Test an exception is raised if day of year out of range"""
        day_of_year = 367
        msg = 'Day of the year must be between 0 and 365'
        with self.assertRaisesRegex(ValueError, msg):
            daynight_terminator(self.longitudes,
                                day_of_year, self.utc_hour)

    def test_daynight_terminator_raises_exception_hour(self):
        """Test an exception is raised if hour out of range"""
        utc_hour = -10.0
        msg = 'Hour must be between 0 and 24.0'
        with self.assertRaisesRegex(ValueError, msg):
            daynight_terminator(self.longitudes,
                                self.day_of_year, utc_hour)


if __name__ == '__main__':
    unittest.main()
