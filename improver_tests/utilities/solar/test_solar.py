# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the solar calculations in solar.py """

import unittest
from datetime import datetime, timezone

import numpy as np
from iris.tests import IrisTest

from improver.utilities.solar import (
    calc_solar_declination,
    calc_solar_elevation,
    calc_solar_hour_angle,
    calc_solar_time,
    daynight_terminator,
    get_day_of_year,
    get_hour_of_day,
)


class Test_get_day_of_year(IrisTest):
    """Test day of year extraction."""

    def test_get_day_of_year(self):
        datetimes = [
            datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2020, 2, 29, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2021, 3, 1, 0, 0, 50, tzinfo=timezone.utc),
            datetime(2020, 3, 1, 0, 0, 29, tzinfo=timezone.utc),
            datetime(2021, 12, 31, 23, 58, 59, tzinfo=timezone.utc),
            datetime(2020, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
        ]

        expected_result = np.array([0, 59, 59, 60, 364, 365], dtype=int)
        result = [get_day_of_year(dt) for dt in datetimes]
        self.assertArrayEqual(result, expected_result)


class Test_get_hour_of_day(IrisTest):
    """Test utc hour extraction."""

    def test_get_hour_of_day(self):
        datetimes = [
            datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2020, 2, 29, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2021, 3, 1, 0, 0, 50, tzinfo=timezone.utc),
            datetime(2020, 3, 1, 0, 0, 29, tzinfo=timezone.utc),
            datetime(2021, 12, 31, 23, 58, 59, tzinfo=timezone.utc),
            datetime(2020, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
        ]

        expected_result = np.array([0.0, 1.0, 1.0 / 60.0, 0.0, 24.0 - 1.0 / 60.0, 24.0])
        result = [get_hour_of_day(dt) for dt in datetimes]
        self.assertArrayEqual(result, expected_result)


class Test_calc_solar_declination(IrisTest):
    """Test Solar declination."""

    def test_basic_solar_declination(self):
        """Test the calc of declination for different days of the year"""
        day_of_year = [1, 10, 100, 365]
        expected_result = [
            -23.1223537018,
            -22.1987731152,
            7.20726681123,
            -23.2078460336,
        ]
        for i, dayval in enumerate(day_of_year):
            result = calc_solar_declination(dayval)
            self.assertIsInstance(result, float)
            self.assertAlmostEqual(result, expected_result[i])

    def test_solar_dec_raises_exception(self):
        """Test an exception is raised if day of year out of range"""
        day_of_year = -1
        msg = "Day of the year must be between 0 and 365"
        with self.assertRaisesRegex(ValueError, msg):
            calc_solar_declination(day_of_year)


class Test_calc_solar_time(IrisTest):
    """Test Calculation of the Local Solar Time."""

    def setUp(self):
        """Set up the longitudes."""
        self.longitudes = np.array([0.0, 10.0, -10.0, 180.0, -179.0])
        self.day_of_year = 10
        self.utc_hour = 12.0

    def test_basic_solar_time(self):
        """Test the calculation of local solar time. Single Value."""
        result = calc_solar_time(0.0, 10, 0.0)
        expected_result = -0.1188849
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, expected_result)

    def test_basic_solar_time_array(self):
        """Test the calc of solar time for an array of longitudes"""
        result = calc_solar_time(self.longitudes, self.day_of_year, self.utc_hour)
        expected_result = np.array(
            [11.8811151, 12.5477817, 11.2144484, 23.8811151, -0.0522183]
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_result)

    def test_basic_solar_time_normalised(self):
        """Test the calc of solar time for an array of longitudes with solar time values
        normalised to be on the interval 0-24."""
        result = calc_solar_time(
            self.longitudes, self.day_of_year, 12.0, normalise=True
        )
        expected_result = np.array(
            [11.8811151, 12.5477817, 11.2144484, 23.8811151, 23.9477817]
        )
        self.assertArrayAlmostEqual(result, expected_result)

        result = calc_solar_time(
            self.longitudes, self.day_of_year, 14.0, normalise=True
        )
        expected_result = np.array(
            [13.8811151, 14.5477817, 13.2144484, 1.8811151, 1.9477817]
        )
        self.assertArrayAlmostEqual(result, expected_result)

    def test_solar_time_raises_exception_day_of_year(self):
        """Test an exception is raised if day of year out of range"""
        day_of_year = 367
        msg = "Day of the year must be between 0 and 365"
        with self.assertRaisesRegex(ValueError, msg):
            calc_solar_hour_angle(self.longitudes, day_of_year, self.utc_hour)

    def test_solar_time_raises_exception_hour(self):
        """Test an exception is raised if hour out of range"""
        utc_hour = -10.0
        msg = "Hour must be between 0 and 24.0"
        with self.assertRaisesRegex(ValueError, msg):
            calc_solar_hour_angle(self.longitudes, self.day_of_year, utc_hour)


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
        result = calc_solar_hour_angle(self.longitudes, self.day_of_year, self.utc_hour)
        expected_result = np.array(
            [-1.7832741, 8.2167259, -11.7832741, 178.2167259, -180.7832741]
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_result)

    def test_basic_solar_hour_angle_array_360(self):
        """Test the calc of solar hour_angle for longitudes > 180"""
        longitudes = np.array([0.0, 10.0, 350.0, 180.0, 181.0])
        result = calc_solar_hour_angle(longitudes, self.day_of_year, self.utc_hour)
        expected_result = np.array(
            [
                -1.7832741,
                8.2167259,
                360.0 - 11.7832741,
                178.2167259,
                360.0 - 180.7832741,
            ]
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_result)


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
        expected_results = [
            -0.460611756793,
            6.78261282655,
            1.37746106416,
            -6.75237871867,
        ]
        for i, hour in enumerate([8.0, 9.0, 16.0, 17.0]):
            result = calc_solar_elevation(50.0, 0.0, 10, hour)
            self.assertIsInstance(result, float)
            self.assertAlmostEqual(result, expected_results[i])

    def test_basic_solar_elevation_array(self):
        """Test the solar elevation for an array of lats and lons."""
        expected_array = np.array([-3.1423043, -0.46061176, 2.09728301])
        result = calc_solar_elevation(
            self.latitudes, self.longitudes, self.day_of_year, self.utc_hour
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_array)

    def test_basic_solar_elevation_array_360(self):
        """Test the solar elevation for lons > 180."""
        longitudes = np.array([355.0, 0.0, 5.0])
        expected_array = np.array([-3.1423043, -0.46061176, 2.09728301])
        result = calc_solar_elevation(
            self.latitudes, longitudes, self.day_of_year, self.utc_hour
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_array)

    def test_solar_elevation_raises_exception_lat(self):
        """Test an exception is raised if latitudes out of range"""
        latitudes = np.array([-150.0, 50.0, 50.0])
        msg = "Latitudes must be between -90.0 and 90.0"
        with self.assertRaisesRegex(ValueError, msg):
            calc_solar_elevation(
                latitudes, self.longitudes, self.day_of_year, self.utc_hour
            )

    def test_solar_elevation_raises_exception_day_of_year(self):
        """Test an exception is raised if day of year out of range"""
        day_of_year = 367
        msg = "Day of the year must be between 0 and 365"
        with self.assertRaisesRegex(ValueError, msg):
            calc_solar_elevation(
                self.latitudes, self.longitudes, day_of_year, self.utc_hour
            )

    def test_solar_elevation_raises_exception_hour(self):
        """Test an exception is raised if hour out of range"""
        utc_hour = -10.0
        msg = "Hour must be between 0 and 24.0"
        with self.assertRaisesRegex(ValueError, msg):
            calc_solar_elevation(
                self.latitudes, self.longitudes, self.day_of_year, utc_hour
            )

    def test_sine_solar_elevation(self):
        """Test the solar elevation with return_sine equal true."""
        expected_results = [-0.00803911, 0.11810263, 0.02403892, -0.11757863]
        for i, hour in enumerate([8.0, 9.0, 16.0, 17.0]):
            result = calc_solar_elevation(50.0, 0.0, 10, hour, return_sine=True)
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
        expected_lats = np.array(
            [
                -67.79151577,
                -66.97565648,
                -63.73454167,
                -56.33476507,
                -39.67331783,
                -4.36090124,
                34.38678745,
                54.03238506,
                62.69165892,
                66.55543647,
                67.79151577,
                66.97565648,
                63.73454167,
                56.33476507,
                39.67331783,
                4.36090124,
                -34.38678745,
                -54.03238506,
                -62.69165892,
                -66.55543647,
                -67.79151577,
            ]
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_lats)

    def test_basic_spring(self):
        """Test we get the terminator in spring."""
        result = daynight_terminator(self.longitudes, 100, 0.0)
        expected_lats = np.array(
            [
                -82.7926115,
                -82.44008918,
                -81.15273499,
                -77.9520655,
                -68.09955141,
                -2.64493534,
                67.37718951,
                77.7625883,
                81.07852035,
                82.41166928,
                82.7926115,
                82.44008918,
                81.15273499,
                77.95206547,
                68.09955141,
                2.64493534,
                -67.37718951,
                -77.7625883,
                -81.07852035,
                -82.41166928,
                -82.7926115,
            ]
        )
        self.assertArrayAlmostEqual(result, expected_lats)

    def test_basic_winter_360(self):
        """Test we get the terminator in winter with lon > 180."""
        longitudes = np.linspace(0.0, 360.0, 21)
        result = daynight_terminator(longitudes, 10, 12.0)
        expected_lats = np.array(
            [
                67.79151577,
                66.97565648,
                63.73454167,
                56.33476507,
                39.67331783,
                4.36090124,
                -34.38678745,
                -54.03238506,
                -62.69165892,
                -66.55543647,
                -67.79151577,
                -66.97565648,
                -63.73454167,
                -56.33476507,
                -39.67331783,
                -4.36090124,
                34.38678745,
                54.03238506,
                62.69165892,
                66.55543647,
                67.79151577,
            ]
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_lats)

    def test_basic_sprint_360(self):
        """Test we get the terminator in spring with lon > 180."""
        longitudes = np.linspace(0.0, 360.0, 21)
        result = daynight_terminator(longitudes, 100, 0.0)
        expected_lats = np.array(
            [
                82.7926115,
                82.44008918,
                81.15273499,
                77.95206547,
                68.09955141,
                2.64493534,
                -67.37718951,
                -77.7625883,
                -81.07852035,
                -82.41166928,
                -82.7926115,
                -82.44008918,
                -81.15273499,
                -77.9520655,
                -68.09955141,
                -2.64493534,
                67.37718951,
                77.7625883,
                81.07852035,
                82.41166928,
                82.7926115,
            ]
        )
        self.assertArrayAlmostEqual(result, expected_lats)

    def test_daynight_terminator_raises_exception_day_of_year(self):
        """Test an exception is raised if day of year out of range"""
        day_of_year = 367
        msg = "Day of the year must be between 0 and 365"
        with self.assertRaisesRegex(ValueError, msg):
            daynight_terminator(self.longitudes, day_of_year, self.utc_hour)

    def test_daynight_terminator_raises_exception_hour(self):
        """Test an exception is raised if hour out of range"""
        utc_hour = -10.0
        msg = "Hour must be between 0 and 24.0"
        with self.assertRaisesRegex(ValueError, msg):
            daynight_terminator(self.longitudes, self.day_of_year, utc_hour)


if __name__ == "__main__":
    unittest.main()
