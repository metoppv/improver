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
"""Unit tests for file name generation."""

import unittest
from datetime import datetime

import numpy as np
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest

from improver.tests.set_up_test_cubes import set_up_variable_cube
from improver.utilities.filename import generate_file_name


class Test_generate_file_name(IrisTest):
    """Test the generate_file_name function"""

    def setUp(self):
        """Set up dummy cube"""
        data = np.zeros((3, 3), dtype=np.float32)
        self.cube15m = set_up_variable_cube(
            data, name="air_temperature", units="degreesC",
            spatial_grid="equalarea", time=datetime(2015, 11, 19, 0, 30),
            frt=datetime(2015, 11, 19, 0, 15),
            time_bounds=[datetime(2015, 11, 19, 0, 15),
                         datetime(2015, 11, 19, 0, 30)])

        self.cube1h = set_up_variable_cube(
            data, name="air_temperature", units="degreesC",
            spatial_grid="equalarea", time=datetime(2015, 11, 19, 1, 15),
            frt=datetime(2015, 11, 19, 0, 15),
            time_bounds=[datetime(2015, 11, 19, 0, 15),
                         datetime(2015, 11, 19, 1, 15)])

        self.cube1h15m = set_up_variable_cube(
            data, name="air_temperature", units="degreesC",
            spatial_grid="equalarea", time=datetime(2015, 11, 19, 1, 30),
            frt=datetime(2015, 11, 19, 0, 15),
            time_bounds=[datetime(2015, 11, 19, 0, 15),
                         datetime(2015, 11, 19, 1, 30)])

    def test_basic(self):
        """Test basic file name generation"""
        name = generate_file_name(self.cube15m)
        self.assertIsInstance(name, str)
        self.assertEqual(name, "20151119T0030Z-PT0000H15M-air_temperature.nc")

    def test_input_cube_unmodified(self):
        """Test the function does not modify the input cube"""
        reference_cube = self.cube15m.copy()
        generate_file_name(self.cube15m)
        self.assertArrayAlmostEqual(self.cube15m.data, reference_cube.data)
        self.assertEqual(self.cube15m.metadata, reference_cube.metadata)

    def test_longer_lead_time(self):
        """Test with lead time > 1 hr"""
        self.cube15m.coord("forecast_period").points = (
            np.array([75*60], dtype=np.int32))
        name = generate_file_name(self.cube15m)
        self.assertEqual(name, "20151119T0030Z-PT0001H15M-air_temperature.nc")

    def test_missing_lead_time(self):
        """Test with missing lead time"""
        self.cube15m.remove_coord("forecast_period")
        name = generate_file_name(self.cube15m)
        self.assertEqual(name, "20151119T0030Z-PT0000H00M-air_temperature.nc")

    def test_missing_time(self):
        """Test error is raised if "time" coordinate is missing"""
        self.cube15m.remove_coord("time")
        with self.assertRaises(CoordinateNotFoundError):
            generate_file_name(self.cube15m)

    def test_funny_cube_name(self):
        """Test cube names are correctly parsed to remove spaces, brackets,
        slashes, parentheses and uppercase letters"""
        self.cube15m.rename("Rainfall rate / (Composite)")
        name = generate_file_name(self.cube15m)
        self.assertEqual(
            name, "20151119T0030Z-PT0000H15M-rainfall_rate_composite.nc")

    def test_parameter_name(self):
        """Test basic file name generation"""
        name = generate_file_name(
            self.cube15m, parameter='another_temperature')
        self.assertEqual(
            name, "20151119T0030Z-PT0000H15M-another_temperature.nc")

    def test_time_period_in_hours_from_forecast_period(self):
        """Test including a period within the filename when the period
        is in hours and deduced from the forecast_period coordinate."""
        self.cube1h.coord("time").bounds = None
        name = generate_file_name(self.cube1h, include_period=True)
        self.assertIsInstance(name, str)
        self.assertEqual(
            name, "20151119T0115Z-PT0001H00M-air_temperature-PT01H.nc")

    def test_time_period_in_hours_from_time(self):
        """Test including a period within the filename when the period is in
        hours and deduced from the time coordinate."""
        self.cube1h.coord("forecast_period").bounds = None
        name = generate_file_name(self.cube1h, include_period=True)
        self.assertIsInstance(name, str)
        self.assertEqual(
            name, "20151119T0115Z-PT0001H00M-air_temperature-PT01H.nc")

    def test_time_period_in_minutes_from_forecast_period(self):
        """Test including a period within the filename when the period
        is in minutes and deduced from the forecast_period coordinate."""
        self.cube15m.coord("time").bounds = None
        name = generate_file_name(self.cube15m, include_period=True)
        self.assertIsInstance(name, str)
        self.assertEqual(
            name, "20151119T0030Z-PT0000H15M-air_temperature-PT15M.nc")

    def test_time_period_in_minutes_from_time(self):
        """Test including a period within the filename when the period is in
        minutes and deduced from the time coordinate."""
        self.cube15m.coord("forecast_period").bounds = None
        name = generate_file_name(self.cube15m, include_period=True)
        self.assertIsInstance(name, str)
        self.assertEqual(
            name, "20151119T0030Z-PT0000H15M-air_temperature-PT15M.nc")

    def test_no_bounds_exception(self):
        """Test that an exception is raised if the forecast_period and time
        coordinates provided do not have bounds."""
        self.cube1h15m.coord("forecast_period").bounds = None
        self.cube1h15m.coord("time").bounds = None
        msg = "Neither the forecast_period coordinate"
        with self.assertRaisesRegex(ValueError, msg):
            generate_file_name(self.cube1h15m, include_period=True)

    def test_hours_and_minutes_exception(self):
        """Test that an exception is raised if the difference between the
        bounds is greater than 1 hour and not equal to a whole hour."""
        msg = "If the difference between the bounds of the"
        with self.assertRaisesRegex(ValueError, msg):
            generate_file_name(self.cube1h15m, include_period=True)


if __name__ == '__main__':
    unittest.main()
