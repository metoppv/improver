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
"""Unit tests for psychrometric_calculations FallingSnowLevel."""
import unittest

import iris
import numpy as np
from cf_units import Unit
from iris.tests import IrisTest

from improver.psychrometric_calculations.psychrometric_calculations import (
    FallingSnowLevel)
from improver.tests.set_up_test_cubes import (set_up_variable_cube,
                                              add_coordinate)


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(FallingSnowLevel())
        msg = ('<FallingSnowLevel: '
               'precision:0.005, falling_level_threshold:90.0,'
               ' grid_point_radius: 2>')
        self.assertEqual(result, msg)


class Test_find_falling_level(IrisTest):

    """Test the find_falling_level method."""

    def setUp(self):
        """Set up arrays."""
        self.wb_int_data = np.array([[[80.0, 80.0], [70.0, 50.0]],
                                     [[90.0, 100.0], [80.0, 60.0]],
                                     [[100.0, 110.0], [90.0, 100.0]]])

        self.orog_data = np.array([[0.0, 0.0], [5.0, 3.0]])
        self.height_points = np.array([5.0, 10.0, 20.0])

    def test_basic(self):
        """Test method returns an array with correct data"""
        plugin = FallingSnowLevel()
        expected = np.array([[10.0, 7.5], [25.0, 20.5]])
        result = plugin.find_falling_level(
            self.wb_int_data, self.orog_data, self.height_points)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, expected)

    def test_outside_range(self):
        """Test method returns an nan if data outside range"""
        plugin = FallingSnowLevel()
        wb_int_data = self.wb_int_data
        wb_int_data[2, 1, 1] = 70.0
        result = plugin.find_falling_level(
            wb_int_data, self.orog_data, self.height_points)
        self.assertTrue(np.isnan(result[1, 1]))


class Test_fill_in_high_snow_falling_levels(IrisTest):

    """Test the fill_in_high_snow_falling_levels method."""

    def setUp(self):
        """ Set up arrays for testing."""
        self.snow_level_data = np.array([[1.0, 1.0, 2.0],
                                         [1.0, np.nan, 2.0],
                                         [1.0, 2.0, 2.0]])
        self.snow_data_no_interp = np.array([[np.nan, np.nan, np.nan],
                                             [1.0, np.nan, 2.0],
                                             [1.0, 2.0, np.nan]])
        self.orog = np.ones((3, 3))
        self.highest_wb_int = np.ones((3, 3))
        self.highest_height = 300.0

    def test_basic(self):
        """Test fills in missing data with orography + highest height"""
        plugin = FallingSnowLevel()
        self.highest_wb_int[1, 1] = 100.0
        expected = np.array([[1.0, 1.0, 2.0],
                             [1.0, 301.0, 2.0],
                             [1.0, 2.0, 2.0]])
        plugin.fill_in_high_snow_falling_levels(
            self.snow_level_data, self.orog, self.highest_wb_int,
            self.highest_height)
        self.assertArrayEqual(self.snow_level_data, expected)

    def test_no_fill_if_conditions_not_met(self):
        """Test it doesn't fill in NaN if the heighest wet bulb integral value
           is less than the threshold."""
        plugin = FallingSnowLevel()
        expected = np.array([[1.0, 1.0, 2.0],
                             [1.0, np.nan, 2.0],
                             [1.0, 2.0, 2.0]])
        plugin.fill_in_high_snow_falling_levels(
            self.snow_level_data, self.orog, self.highest_wb_int,
            self.highest_height)
        self.assertArrayEqual(self.snow_level_data, expected)


class Test_linear_wet_bulb_fit(IrisTest):

    """Test the linear_wet_bulb_fit method."""

    def setUp(self):
        """
        Set up arrays for testing.

        Set up a wet bulb temperature array with a linear trend near sea
        level. Some of the straight line fits of wet bulb temperature will
        cross the height axis above zero and some below.
        """
        data = np.ones((5, 3, 3))*-0.8
        self.heights = np.array([5, 10, 20, 30, 50])
        for i in range(5):
            data[i] = data[i]*self.heights[i]
        data[:, :, 0] = data[:, :, 0]-10
        data[:, :, 2] = data[:, :, 2]+20
        self.wet_bulb_temperature = data
        self.sea_points = np.array([[True, True, True],
                                    [False, False, False],
                                    [True, True, True]])
        self.expected_gradients = np.array([[-0.8, -0.8, -0.8],
                                            [0.0, 0.0, 0.0],
                                            [-0.8, -0.8, -0.8]])
        self.expected_intercepts = np.array([[-10, 0.0, 20.0],
                                             [0.0, 0.0, 0.0],
                                             [-10, 0.0, 20.0]])

    def test_basic(self):
        """Test we find the correct gradient and intercepts for simple case"""
        plugin = FallingSnowLevel()

        gradients, intercepts = plugin.linear_wet_bulb_fit(
            self.wet_bulb_temperature, self.heights, self.sea_points)
        self.assertArrayAlmostEqual(self.expected_gradients, gradients)
        self.assertArrayAlmostEqual(self.expected_intercepts, intercepts)

    def test_land_points(self):
        """Test it returns arrays of zeros if points are land."""
        plugin = FallingSnowLevel()
        sea_points = np.ones((3, 3))*False
        gradients, intercepts = plugin.linear_wet_bulb_fit(
            self.wet_bulb_temperature, self.heights, sea_points)
        self.assertArrayAlmostEqual(np.zeros((3, 3)), gradients)
        self.assertArrayAlmostEqual(np.zeros((3, 3)), intercepts)


class Test_find_extrapolated_falling_level(IrisTest):

    """Test the find_extrapolated_falling_level method."""

    def setUp(self):
        """
        Set up arrays for testing.
        Set up a wet bulb temperature array with a linear trend near sea
        level. Some of the straight line fits of wet bulb temperature will
        cross the height axis above zero and some below.
        """
        self.snow_falling_level = np.ones((3, 3))*np.nan
        self.max_wb_integral = np.array([[0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0],
                                         [10.0, 10.0, 10.0]])
        self.sea_points = np.array([[True, True, True],
                                    [False, False, False],
                                    [True, True, True]])
        self.gradients = np.array([[-0.8, -0.8, -0.8],
                                   [0.0, 0.0, 0.0],
                                   [-0.8, -0.8, -0.8]])
        self.intercepts = np.array([[-10, 0.0, 20.0],
                                    [0.0, 0.0, 0.0],
                                    [-10, 0.0, 20.0]])
        self.expected_snow_falling_level = np.array(
            [[-27.5, -15.0, -4.154759],
             [np.nan, np.nan, np.nan],
             [-26.642136, -14.142136, -3.722813]])

    def test_basic(self):
        """Test we fill in the correct snow falling levels for a simple case"""
        plugin = FallingSnowLevel()

        plugin.find_extrapolated_falling_level(
            self.max_wb_integral, self.gradients, self.intercepts,
            self.snow_falling_level, self.sea_points)
        self.assertArrayAlmostEqual(self.expected_snow_falling_level,
                                    self.snow_falling_level)

    def test_gradients_zero(self):
        """Test we do nothing if all gradients are zero"""
        plugin = FallingSnowLevel()
        gradients = np.zeros((3, 3))
        plugin.find_extrapolated_falling_level(
            self.max_wb_integral, gradients, self.intercepts,
            self.snow_falling_level, self.sea_points)
        expected_snow_falling_level = np.ones((3, 3))*np.nan
        self.assertArrayAlmostEqual(expected_snow_falling_level,
                                    self.snow_falling_level)


class Test_fill_sea_points(IrisTest):

    """Test the fill_in_sea_points method."""

    def setUp(self):
        """ Set up arrays for testing."""
        self.snow_falling_level = np.ones((3, 3))*np.nan
        self.max_wb_integral = np.array([[0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0],
                                         [10.0, 10.0, 10.0]])

        self.land_sea = np.array([[0, 0, 0],
                                  [1, 1, 1],
                                  [0, 0, 0]])
        data = np.ones((5, 3, 3))*-0.8
        self.heights = np.array([5, 10, 20, 30, 50])
        for i in range(5):
            data[i] = data[i]*self.heights[i]
        data[:, :, 0] = data[:, :, 0] - 10
        data[:, :, 2] = data[:, :, 2] + 20
        self.wet_bulb_temperature = data
        self.expected_snow_falling_level = np.array(
            [[-27.5, -15.0, -4.154759],
             [np.nan, np.nan, np.nan],
             [-26.642136, -14.142136, -3.722813]])

    def test_basic(self):
        """Test it fills in the points it's meant to."""
        plugin = FallingSnowLevel()
        plugin.fill_in_sea_points(self.snow_falling_level, self.land_sea,
                                  self.max_wb_integral,
                                  self.wet_bulb_temperature, self.heights)
        self.assertArrayAlmostEqual(self.snow_falling_level.data,
                                    self.expected_snow_falling_level)

    def test_no_sea(self):
        """Test it only fills in sea points, and ignores a land point"""
        plugin = FallingSnowLevel()
        expected = np.ones((3, 3))*np.nan
        land_sea = np.ones((3, 3))
        plugin.fill_in_sea_points(self.snow_falling_level, land_sea,
                                  self.max_wb_integral,
                                  self.wet_bulb_temperature, self.heights)
        self.assertArrayAlmostEqual(self.snow_falling_level.data, expected)

    def test_all_above_threshold(self):
        """Test it doesn't change points that are all above the threshold"""
        plugin = FallingSnowLevel()
        self.max_wb_integral[0, 1] = 100
        self.snow_falling_level[0, 1] = 100
        self.expected_snow_falling_level[0, 1] = 100
        plugin.fill_in_sea_points(self.snow_falling_level, self.land_sea,
                                  self.max_wb_integral,
                                  self.wet_bulb_temperature, self.heights)
        self.assertArrayAlmostEqual(self.snow_falling_level.data,
                                    self.expected_snow_falling_level)


class Test_fill_in_by_horizontal_interpolation(IrisTest):
    """Test the fill_in_by_horizontal_interpolation method"""
    def setUp(self):
        """ Set up arrays for testing."""
        self.snow_level_data = np.array([[1.0, 1.0, 2.0],
                                        [1.0, np.nan, 2.0],
                                        [1.0, 2.0, 2.0]])
        self.orog_data = np.array([[6.0, 6.0, 6.0],
                                   [6.0, 7.0, 6.0],
                                   [6.0, 6.0, 6.0]])
        self.max_in_nbhood_orog = np.array([[7.0, 7.0, 7.0],
                                            [7.0, 7.0, 7.0],
                                            [7.0, 7.0, 7.0]])
        self.plugin = FallingSnowLevel()

    def test_basic(self):
        """Test when all the points around the missing data are the same."""
        snow_level_data = np.ones((3, 3))
        snow_level_data[1, 1] = np.nan
        expected = np.array([[1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0]])
        snow_level_updated = self.plugin.fill_in_by_horizontal_interpolation(
            snow_level_data, self.max_in_nbhood_orog, self.orog_data)
        self.assertArrayEqual(snow_level_updated, expected)

    def test_not_enough_points_to_fill(self):
        """Test when there are not enough points to fill the gaps.
           This raises a QhullError if there are less than 3 points available
           to use for the interpolation. The QhullError is different to the one
           raised by test_badly_arranged_valid_data"""
        snow_level_data = np.array([[np.nan, 1, np.nan],
                                    [np.nan, np.nan, np.nan],
                                    [np.nan, 1, np.nan]])
        expected = np.array([[1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0]])
        snow_level_updated = self.plugin.fill_in_by_horizontal_interpolation(
            snow_level_data, self.max_in_nbhood_orog, self.orog_data)
        self.assertArrayEqual(snow_level_updated, expected)

    def test_badly_arranged_valid_data(self):
        """Test when there are enough points but they aren't arranged in a
           suitable way to allow horizontal interpolation. This raises a
           QhullError that we want to ignore and use nearest neighbour
           interpolation instead. This QhullError is different to the one
           raised by test_not_enough_points_to_fill."""
        snow_level_data = np.array([[np.nan, 1, np.nan],
                                    [np.nan, 1, np.nan],
                                    [np.nan, 1, np.nan]])
        expected = np.array([[1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0]])
        snow_level_updated = self.plugin.fill_in_by_horizontal_interpolation(
            snow_level_data, self.max_in_nbhood_orog, self.orog_data)
        self.assertArrayEqual(snow_level_updated, expected)

    def test_different_data(self):
        """Test when the points around the missing data have different
           values."""
        expected = np.array([[1.0, 1.0, 2.0],
                             [1.0, 1.5, 2.0],
                             [1.0, 2.0, 2.0]])
        snow_level_updated = self.plugin.fill_in_by_horizontal_interpolation(
            self.snow_level_data, self.max_in_nbhood_orog, self.orog_data)
        self.assertArrayEqual(snow_level_updated, expected)

    def test_lots_missing(self):
        """Test when there's an extra missing value at the corner
           of the grid. This point can't be filled in by linear interpolation,
           but is instead filled by nearest neighbour extrapolation."""
        self.snow_level_data[2, 2] = np.nan
        expected = np.array([[1.0, 1.0, 2.0],
                             [1.0, 1.5, 2.0],
                             [1.0, 2.0, 2.0]])
        snow_level_updated = self.plugin.fill_in_by_horizontal_interpolation(
            self.snow_level_data, self.max_in_nbhood_orog, self.orog_data)
        self.assertArrayEqual(snow_level_updated, expected)

    def test_all_above_max_orography(self):
        """Test that nothing is filled in if all the snow falling levels are
           above the maximum orography"""
        max_in_nbhood_orog = np.zeros((3, 3))
        orography = np.zeros((3, 3))
        expected = np.array([[1.0, 1.0, 2.0],
                             [1.0, np.nan, 2.0],
                             [1.0, 2.0, 2.0]])
        snow_level_updated = self.plugin.fill_in_by_horizontal_interpolation(
            self.snow_level_data, max_in_nbhood_orog, orography)
        self.assertArrayEqual(snow_level_updated, expected)

    def test_set_to_orography(self):
        """Test when the linear interpolation gives values that are higher
           than the orography the snow falling level is set back to the
           orography"""
        snow_falling_level = np.array([[10.0, np.nan, np.nan, np.nan, 20.0],
                                       [10.0, np.nan, np.nan, np.nan, 20.0],
                                       [10.0, np.nan, np.nan, np.nan, 20.0],
                                       [10.0, np.nan, np.nan, np.nan, 20.0],
                                       [10.0, np.nan, np.nan, np.nan, 20.0]])

        orography = np.array([[0.0, 30.0, 12.0, 30.0, 0.0],
                              [0.0, 30.0, 12.0, 30.0, 0.0],
                              [0.0, 30.0, 12.0, 30.0, 0.0],
                              [0.0, 30.0, 12.0, 30.0, 0.0],
                              [0.0, 30.0, 12.0, 30.0, 0.0]])

        max_in_nbhood_orog = np.ones((5, 5))*30.0
        expected = np.array([[10.0, 12.5, 12.0, 17.5, 20.0],
                             [10.0, 12.5, 12.0, 17.5, 20.0],
                             [10.0, 12.5, 12.0, 17.5, 20.0],
                             [10.0, 12.5, 12.0, 17.5, 20.0],
                             [10.0, 12.5, 12.0, 17.5, 20.0]])
        snow_level_updated = self.plugin.fill_in_by_horizontal_interpolation(
            snow_falling_level, max_in_nbhood_orog, orography)
        self.assertArrayEqual(snow_level_updated, expected)


class Test_find_max_in_nbhood_orography(IrisTest):

    """Test the find_max_in_nbhood_orography method"""

    def setUp(self):
        """Set up a cube with x and y coordinates"""
        data = np.array([[0, 10, 20, 5, 0],
                         [0, 50, 20, 5, 0],
                         [0, 80, 90, 0, 0],
                         [0, 20, 5, 10, 0],
                         [0, 5, 10, 10, 0]])
        self.cube = iris.cube.Cube(data, standard_name="air_temperature",
                                   units="celsius")
        self.cube.add_dim_coord(
            iris.coords.DimCoord(np.linspace(2000.0, 10000.0, 5),
                                 'projection_x_coordinate', units='m'), 0)
        self.cube.add_dim_coord(
            iris.coords.DimCoord(np.linspace(2000.0, 10000.0, 5),
                                 "projection_y_coordinate", units='m'), 1)
        self.expected_data = ([[50, 50, 50, 20, 5],
                               [80, 90, 90, 90, 5],
                               [80, 90, 90, 90, 10],
                               [80, 90, 90, 90, 10],
                               [20, 20, 20, 10, 10]])

    def test_basic(self):
        """Test the function does what it's meant to in a simple case."""
        plugin = FallingSnowLevel(grid_point_radius=1)
        result = plugin.find_max_in_nbhood_orography(self.cube)
        self.assertArrayAlmostEqual(result.data, self.expected_data)


class Test_process(IrisTest):

    """Test the FallingSnowLevel processing works"""

    def setUp(self):
        """Set up orography and land-sea mask cubes. Also create temperature,
        pressure, and relative humidity cubes that contain multiple height
        levels."""

        data = np.ones((3, 3), dtype=np.float32)
        relh_data = np.ones((3, 3), dtype=np.float32) * 0.65

        self.height_points = [5., 195., 200.]
        height_attribute = {"positive": "up"}

        self.orog = set_up_variable_cube(
            data, name='surface_altitude', units='m', spatial_grid='equalarea')
        self.land_sea = set_up_variable_cube(
            data, name='land_binary_mask', units=1, spatial_grid='equalarea')

        temperature = set_up_variable_cube(data, spatial_grid='equalarea')
        temperature = add_coordinate(temperature, [0, 1], 'realization')
        self.temperature_cube = add_coordinate(
            temperature, self.height_points, 'height', coord_units='m',
            attributes=height_attribute)

        relative_humidity = set_up_variable_cube(
            relh_data, name='relative_humidity', units='%',
            spatial_grid='equalarea')
        relative_humidity = add_coordinate(
            relative_humidity, [0, 1], 'realization')
        self.relative_humidity_cube = add_coordinate(
            relative_humidity, self.height_points, 'height', coord_units='m',
            attributes=height_attribute)

        pressure = set_up_variable_cube(
            data, name='air_pressure', units='Pa', spatial_grid='equalarea')
        pressure = add_coordinate(pressure, [0, 1], 'realization')
        self.pressure_cube = add_coordinate(
            pressure, self.height_points, 'height', coord_units='m',
            attributes=height_attribute)

        # Assign different temperatures and pressures to each height.
        temp_vals = [278.0, 280.0, 285.0, 286.0]
        pressure_vals = [93856.0, 95034.0, 96216.0, 97410.0]
        for i in range(0, 3):
            self.temperature_cube.data[i, ::] = temp_vals[i+1]
            self.pressure_cube.data[i, ::] = pressure_vals[i+1]
            # Add hole in middle of data.
            self.temperature_cube.data[i, :, 1, 1] = temp_vals[i]
            self.pressure_cube.data[i, :, 1, 1] = pressure_vals[i]

    def test_basic(self):
        """Test that process returns a cube with the right name and units."""
        self.orog.data[1, 1] = 100.0
        result = FallingSnowLevel().process(
            self.temperature_cube, self.relative_humidity_cube,
            self.pressure_cube, self.orog, self.land_sea)
        expected = np.ones((2, 3, 3), dtype=np.float32) * 66.88566
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), "falling_snow_level_asl")
        self.assertEqual(result.units, Unit('m'))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_data(self):
        """Test that the falling snow level process returns a cube
        containing the expected data when points at sea-level."""
        expected = np.ones((2, 3, 3), dtype=np.float32) * 65.88566
        orog = self.orog
        orog.data = orog.data * 0.0
        orog.data[1, 1] = 100.0
        land_sea = self.land_sea
        land_sea = land_sea * 0.0
        result = FallingSnowLevel().process(
            self.temperature_cube, self.relative_humidity_cube,
            self.pressure_cube, orog, land_sea)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)


if __name__ == '__main__':
    unittest.main()
