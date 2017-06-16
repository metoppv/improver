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


class Test_get_neighbourhood_width_in_grid_cells(IrisTest):

    """Test conversion of kernel radius in kilometres to grid cells."""

    RADIUS_IN_KM = 6.1
    MAX_RADIUS_IN_GRID_CELLS = 500

    def test_basic_radius_to_grid_cells(self):
        """Test the radius in km to grid cell conversion."""
        cube = set_up_cube()
        result = Utilities().get_neighbourhood_width_in_grid_cells(
            cube, self.RADIUS_IN_KM, self.MAX_RADIUS_IN_GRID_CELLS)
        self.assertEqual(result, (3, 3))

    def test_basic_radius_to_grid_cells_different_max_radius(self):
        """
        Test the radius in km to grid cell conversion for an alternative
        max radius in grid cells.
        """
        cube = set_up_cube()
        max_radius_in_grid_cells = 50
        result = Utilities().get_neighbourhood_width_in_grid_cells(
            cube, self.RADIUS_IN_KM, max_radius_in_grid_cells)
        self.assertEqual(result, (3, 3))

    def test_basic_radius_to_grid_cells_km_grid(self):
        """Test the radius-to-grid-cell conversion, grid in km."""
        cube = set_up_cube()
        cube.coord("projection_x_coordinate").convert_units("kilometres")
        cube.coord("projection_y_coordinate").convert_units("kilometres")
        result = Utilities().get_neighbourhood_width_in_grid_cells(
            cube, self.RADIUS_IN_KM, self.MAX_RADIUS_IN_GRID_CELLS)
        self.assertEqual(result, (3, 3))

    def test_single_point_lat_long(self):
        """Test behaviour for a single grid cell on lat long grid."""
        cube = set_up_cube_lat_long()
        msg = "Invalid grid: projection_x/y coords required"
        with self.assertRaisesRegexp(ValueError, msg):
            Utilities().get_neighbourhood_width_in_grid_cells(
                cube, self.RADIUS_IN_KM, self.MAX_RADIUS_IN_GRID_CELLS)

    def test_single_point_range_negative(self):
        """Test behaviour with a non-zero point with negative range."""
        cube = set_up_cube()
        radius_in_km = -1.0 * self.RADIUS_IN_KM
        msg = "radius of -6.1 km gives a negative cell extent"
        with self.assertRaisesRegexp(ValueError, msg):
            Utilities().get_neighbourhood_width_in_grid_cells(
                cube, radius_in_km, self.MAX_RADIUS_IN_GRID_CELLS)

    def test_single_point_range_0(self):
        """Test behaviour with a non-zero point with zero range."""
        cube = set_up_cube()
        radius_in_km = 0.005
        msg = "radius of 0.005 km gives zero cell extent"
        with self.assertRaisesRegexp(ValueError, msg):
            Utilities().get_neighbourhood_width_in_grid_cells(
                cube, radius_in_km, self.MAX_RADIUS_IN_GRID_CELLS)

    def test_single_point_range_lots(self):
        """Test behaviour with a non-zero point with unhandleable range."""
        cube = set_up_cube()
        radius_in_km = 500000.0
        msg = "radius of 500000.0 km exceeds maximum grid cell extent"
        with self.assertRaisesRegexp(ValueError, msg):
            Utilities().get_neighbourhood_width_in_grid_cells(
                cube, radius_in_km, self.MAX_RADIUS_IN_GRID_CELLS)


if __name__ == '__main__':
    unittest.main()
