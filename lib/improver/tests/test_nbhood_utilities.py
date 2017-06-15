# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
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
"""Unit tests for the nbhood.Utilities plugin."""


import unittest

from cf_units import Unit

from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest
import numpy as np

from improver.nbhood import Utilities
from improver.tests.helper_functions_ensemble_calibration import (
    add_forecast_reference_time_and_forecast_period)
from improver.tests.test_nbhood_neighbourhoodprocessing import (
    set_up_cube, set_up_cube_lat_long)


class Test_find_required_lead_times(IrisTest):

    """Test determining of the lead times present within the input cube."""

    def test_basic(self):
        """Test that a numpy array is returned."""
        cube = add_forecast_reference_time_and_forecast_period(set_up_cube())
        result = Utilities.find_required_lead_times(cube)
        self.assertIsInstance(result, np.ndarray)

    def test_check_coordinate(self):
        """
        Test that the data within the numpy array is as expected, when
        the input cube has a forecast_period coordinate.
        """
        cube = add_forecast_reference_time_and_forecast_period(set_up_cube())
        expected_result = cube.coord("forecast_period").points
        result = Utilities.find_required_lead_times(cube)
        self.assertArrayAlmostEqual(result, expected_result)

    def test_check_coordinate_without_forecast_period(self):
        """
        Test that the data within the numpy array is as expected, when
        the input cube has a time coordinate and a forecast_reference_time
        coordinate.
        """
        cube = add_forecast_reference_time_and_forecast_period(set_up_cube())
        cube.remove_coord("forecast_period")
        expected_result = (
            cube.coord("time").points -
            cube.coord("forecast_reference_time").points)
        result = Utilities.find_required_lead_times(cube)
        self.assertArrayAlmostEqual(result, expected_result)

    def test_check_forecast_period_unit_conversion(self):
        """
        Test that the data within the numpy array is as expected, when
        the input cube has a forecast_period coordinate with units
        other than the desired units of hours.
        """
        cube = add_forecast_reference_time_and_forecast_period(set_up_cube())
        expected_result = cube.coord("forecast_period").points.copy()
        cube.coord("forecast_period").convert_units("seconds")
        result = Utilities.find_required_lead_times(cube)
        self.assertArrayAlmostEqual(result, expected_result)

    def test_check_time_unit_conversion(self):
        """
        Test that the data within the numpy array is as expected, when
        the input cube has a time coordinate with units
        other than the desired units of hours since 1970-01-01 00:00:00.
        """
        cube = add_forecast_reference_time_and_forecast_period(set_up_cube())
        expected_result = cube.coord("forecast_period").points.copy()
        cube.coord("time").convert_units("seconds since 1970-01-01 00:00:00")
        result = Utilities.find_required_lead_times(cube)
        self.assertArrayAlmostEqual(result, expected_result)

    def test_check_forecast_period_unit_conversion_exception(self):
        """
        Test that an exception is raised, when the input cube has a
        forecast_period coordinate with units that can not be converted
        into hours.
        """
        cube = add_forecast_reference_time_and_forecast_period(set_up_cube())
        cube.coord("forecast_period").units = Unit("Celsius")
        msg = "For forecast_period"
        with self.assertRaisesRegexp(ValueError, msg):
            Utilities.find_required_lead_times(cube)

    def test_check_forecast_reference_time_unit_conversion_exception(self):
        """
        Test that an exception is raised, when the input cube has a
        forecast_reference_time coordinate with units that can not be
        converted into hours.
        """
        cube = add_forecast_reference_time_and_forecast_period(set_up_cube())
        cube.remove_coord("forecast_period")
        cube.coord("forecast_reference_time").units = Unit("Celsius")
        msg = "For time/forecast_reference_time"
        with self.assertRaisesRegexp(ValueError, msg):
            Utilities.find_required_lead_times(cube)

    def test_exception_raised(self):
        """
        Test that a CoordinateNotFoundError exception is raised if the
        forecast_period, or the time and forecast_reference_time,
        are not present.
        """
        cube = set_up_cube()
        msg = "The forecast period coordinate is not available"
        with self.assertRaisesRegexp(CoordinateNotFoundError, msg):
            Utilities.find_required_lead_times(cube)


if __name__ == '__main__':
    unittest.main()
