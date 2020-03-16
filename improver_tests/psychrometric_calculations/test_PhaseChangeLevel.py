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
"""Unit tests for psychrometric_calculations PhaseChangeLevel."""

import unittest

import iris
import numpy as np
from cf_units import Unit
from iris.cube import CubeList
from iris.tests import IrisTest

from improver.psychrometric_calculations.psychrometric_calculations import (
    PhaseChangeLevel)
from improver.utilities.cube_manipulation import sort_coord_in_cube

from ..set_up_test_cubes import add_coordinate, set_up_variable_cube


class Test__init__(IrisTest):

    """Test the init method."""

    def test_snow_sleet(self):
        """Test that the __init__ method configures the plugin as expected
        for the snow-sleet phase change."""

        phase_change = 'snow-sleet'
        plugin = PhaseChangeLevel(phase_change, grid_point_radius=3)

        self.assertEqual(plugin.falling_level_threshold, 90.)
        self.assertEqual(plugin.phase_change_name, 'snow_falling')
        self.assertEqual(plugin.grid_point_radius, 3)

    def test_sleet_rain(self):
        """Test that the __init__ method configures the plugin as expected
        for the sleet_rain phase change."""

        phase_change = 'sleet-rain'
        plugin = PhaseChangeLevel(phase_change, grid_point_radius=3)

        self.assertEqual(plugin.falling_level_threshold, 202.5)
        self.assertEqual(plugin.phase_change_name, 'rain_falling')
        self.assertEqual(plugin.grid_point_radius, 3)

    def test_unknown_phase_change(self):
        """Test that the __init__ method raised an exception for an unknown
        phase change argument."""

        phase_change = 'kittens-puppies'
        msg = ("Unknown phase change 'kittens-puppies' requested.\n"
               "Available options are: snow-sleet, sleet-rain")

        with self.assertRaisesRegex(ValueError, msg):
            PhaseChangeLevel(phase_change)


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(PhaseChangeLevel(phase_change='snow-sleet'))
        msg = ('<PhaseChangeLevel: '
               'falling_level_threshold:90.0,'
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
        plugin = PhaseChangeLevel(phase_change='snow-sleet')
        expected = np.array([[10.0, 7.5], [25.0, 20.5]])
        result = plugin.find_falling_level(
            self.wb_int_data, self.orog_data, self.height_points)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayEqual(result, expected)

    def test_outside_range(self):
        """Test method returns an nan if data outside range"""
        plugin = PhaseChangeLevel(phase_change='snow-sleet')
        wb_int_data = self.wb_int_data
        wb_int_data[2, 1, 1] = 70.0
        result = plugin.find_falling_level(
            wb_int_data, self.orog_data, self.height_points)
        self.assertTrue(np.isnan(result[1, 1]))


class Test_fill_in_high_phase_change_falling_levels(IrisTest):

    """Test the fill_in_high_phase_change_falling_levels method."""

    def setUp(self):
        """ Set up arrays for testing."""
        self.phase_change_level_data = np.array([[1.0, 1.0, 2.0],
                                                 [1.0, np.nan, 2.0],
                                                 [1.0, 2.0, 2.0]])
        self.phase_change_data_no_interp = np.array([[np.nan, np.nan, np.nan],
                                                     [1.0, np.nan, 2.0],
                                                     [1.0, 2.0, np.nan]])
        self.orog = np.ones((3, 3))
        self.highest_wb_int = np.ones((3, 3))
        self.highest_height = 300.0

    def test_basic(self):
        """Test fills in missing data with orography + highest height"""
        plugin = PhaseChangeLevel(phase_change='snow-sleet')
        self.highest_wb_int[1, 1] = 100.0
        expected = np.array([[1.0, 1.0, 2.0],
                             [1.0, 301.0, 2.0],
                             [1.0, 2.0, 2.0]])
        plugin.fill_in_high_phase_change_falling_levels(
            self.phase_change_level_data, self.orog, self.highest_wb_int,
            self.highest_height)
        self.assertArrayEqual(self.phase_change_level_data, expected)

    def test_no_fill_if_conditions_not_met(self):
        """Test it doesn't fill in NaN if the heighest wet bulb integral value
           is less than the threshold."""
        plugin = PhaseChangeLevel(phase_change='snow-sleet')
        expected = np.array([[1.0, 1.0, 2.0],
                             [1.0, np.nan, 2.0],
                             [1.0, 2.0, 2.0]])
        plugin.fill_in_high_phase_change_falling_levels(
            self.phase_change_level_data, self.orog, self.highest_wb_int,
            self.highest_height)
        self.assertArrayEqual(self.phase_change_level_data, expected)


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
        plugin = PhaseChangeLevel(phase_change='snow-sleet')

        gradients, intercepts = plugin.linear_wet_bulb_fit(
            self.wet_bulb_temperature, self.heights, self.sea_points)
        self.assertArrayAlmostEqual(self.expected_gradients, gradients)
        self.assertArrayAlmostEqual(self.expected_intercepts, intercepts)

    def test_land_points(self):
        """Test it returns arrays of zeros if points are land."""
        plugin = PhaseChangeLevel(phase_change='snow-sleet')
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
        self.phase_change_level = np.ones((3, 3))*np.nan
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
        self.expected_phase_change_level = np.array(
            [[-27.5, -15.0, -4.154759],
             [np.nan, np.nan, np.nan],
             [-26.642136, -14.142136, -3.722813]])

    def test_basic(self):
        """Test we fill in the correct snow falling levels for a simple case"""
        plugin = PhaseChangeLevel(phase_change='snow-sleet')

        plugin.find_extrapolated_falling_level(
            self.max_wb_integral, self.gradients, self.intercepts,
            self.phase_change_level, self.sea_points)
        self.assertArrayAlmostEqual(self.expected_phase_change_level,
                                    self.phase_change_level)

    def test_gradients_zero(self):
        """Test we do nothing if all gradients are zero"""
        plugin = PhaseChangeLevel(phase_change='snow-sleet')
        gradients = np.zeros((3, 3))
        plugin.find_extrapolated_falling_level(
            self.max_wb_integral, gradients, self.intercepts,
            self.phase_change_level, self.sea_points)
        expected_phase_change_level = np.ones((3, 3))*np.nan
        self.assertArrayAlmostEqual(expected_phase_change_level,
                                    self.phase_change_level)


class Test_fill_sea_points(IrisTest):

    """Test the fill_in_sea_points method."""

    def setUp(self):
        """ Set up arrays for testing."""
        self.phase_change_level = np.ones((3, 3))*np.nan
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
        self.expected_phase_change_level = np.array(
            [[-27.5, -15.0, -4.154759],
             [np.nan, np.nan, np.nan],
             [-26.642136, -14.142136, -3.722813]])

    def test_basic(self):
        """Test it fills in the points it's meant to."""
        plugin = PhaseChangeLevel(phase_change='snow-sleet')
        plugin.fill_in_sea_points(self.phase_change_level, self.land_sea,
                                  self.max_wb_integral,
                                  self.wet_bulb_temperature, self.heights)
        self.assertArrayAlmostEqual(self.phase_change_level.data,
                                    self.expected_phase_change_level)

    def test_no_sea(self):
        """Test it only fills in sea points, and ignores a land point"""
        plugin = PhaseChangeLevel(phase_change='snow-sleet')
        expected = np.ones((3, 3))*np.nan
        land_sea = np.ones((3, 3))
        plugin.fill_in_sea_points(self.phase_change_level, land_sea,
                                  self.max_wb_integral,
                                  self.wet_bulb_temperature, self.heights)
        self.assertArrayAlmostEqual(self.phase_change_level.data, expected)

    def test_all_above_threshold(self):
        """Test it doesn't change points that are all above the threshold"""
        plugin = PhaseChangeLevel(phase_change='snow-sleet')
        self.max_wb_integral[0, 1] = 100
        self.phase_change_level[0, 1] = 100
        self.expected_phase_change_level[0, 1] = 100
        plugin.fill_in_sea_points(self.phase_change_level, self.land_sea,
                                  self.max_wb_integral,
                                  self.wet_bulb_temperature, self.heights)
        self.assertArrayAlmostEqual(self.phase_change_level.data,
                                    self.expected_phase_change_level)


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
        plugin = PhaseChangeLevel(phase_change='snow-sleet',
                                  grid_point_radius=1)
        result = plugin.find_max_in_nbhood_orography(self.cube)
        self.assertArrayAlmostEqual(result.data, self.expected_data)


class Test_process(IrisTest):

    """Test the PhaseChangeLevel processing works"""

    def setUp(self):
        """Set up orography and land-sea mask cubes. Also create temperature,
        pressure, and relative humidity cubes that contain multiple height
        levels."""

        data = np.ones((3, 3), dtype=np.float32)

        self.orog = set_up_variable_cube(
            data, name='surface_altitude', units='m', spatial_grid='equalarea')
        self.land_sea = set_up_variable_cube(
            data, name='land_binary_mask', units=1, spatial_grid='equalarea')

        wbt_0 = np.array([[271.46216, 271.46216, 271.46216],
                          [271.46216, 270.20343, 271.46216],
                          [271.46216, 271.46216, 271.46216]])
        wbt_1 = np.array([[274.4207, 274.4207, 274.4207],
                          [274.4207, 271.46216, 274.4207],
                          [274.4207, 274.4207, 274.4207]])
        wbt_2 = np.array([[275.0666, 275.0666, 275.0666],
                          [275.0666, 274.4207, 275.0666],
                          [275.0666, 275.0666, 275.0666]])
        wbt_data = np.array(
            [np.broadcast_to(wbt_0, (3, 3, 3)),
             np.broadcast_to(wbt_1, (3, 3, 3)),
             np.broadcast_to(wbt_2, (3, 3, 3))], dtype=np.float32)

        # Note the values below are ordered at [5, 195] m.
        wbti_0 = np.array([[128.68324, 128.68324, 128.68324],
                           [128.68324, 3.176712, 128.68324],
                           [128.68324, 128.68324, 128.68324]])
        wbti_1 = np.array([[7.9681854, 7.9681854, 7.9681854],
                           [7.9681854, 3.176712, 7.9681854],
                           [7.9681854, 7.9681854, 7.9681854]])
        wbti_data = np.array(
            [np.broadcast_to(wbti_0, (3, 3, 3)),
             np.broadcast_to(wbti_1, (3, 3, 3))], dtype=np.float32)

        height_points = [5., 195., 200.]
        height_attribute = {"positive": "up"}

        wet_bulb_temperature = set_up_variable_cube(
            data, spatial_grid='equalarea', name='wet_bulb_temperature')
        wet_bulb_temperature = add_coordinate(
            wet_bulb_temperature, [0, 1, 2], 'realization')
        self.wet_bulb_temperature_cube = add_coordinate(
            wet_bulb_temperature, height_points, 'height',
            coord_units='m', attributes=height_attribute)
        self.wet_bulb_temperature_cube.data = wbt_data

        # Note that the iris cubelist merge_cube operation sorts the coordinate
        # being merged into ascending order. The cube created below is thus
        # in the incorrect height order, i.e. [5, 195] instead of [195, 5].
        # There is a function in the the PhaseChangeLevel plugin that ensures
        # the height coordinate is in descending order. This is tested here by
        # creating test cubes with both orders.

        height_attribute = {"positive": "down"}

        wet_bulb_integral = set_up_variable_cube(
            data, spatial_grid='equalarea',
            name='wet_bulb_temperature_integral', units='K m',)
        wet_bulb_integral = add_coordinate(
            wet_bulb_integral, [0, 1, 2], 'realization')
        self.wet_bulb_integral_cube_inverted = add_coordinate(
            wet_bulb_integral, height_points[0:2], 'height',
            coord_units='m', attributes=height_attribute)
        self.wet_bulb_integral_cube_inverted.data = wbti_data
        self.wet_bulb_integral_cube = sort_coord_in_cube(
            self.wet_bulb_integral_cube_inverted, 'height', descending=True)

        self.expected_snow_sleet = (
            np.ones((3, 3, 3), dtype=np.float32) * 66.88566)

    def test_snow_sleet_phase_change(self):
        """Test that process returns a cube with the right name, units and
        values. In this instance the phase change is from snow to sleet. The
        returned level is consistent across the field, despite a high point
        that sits above the snow falling level."""
        self.orog.data[1, 1] = 100.0
        result = PhaseChangeLevel(phase_change='snow-sleet').process(
            CubeList([self.wet_bulb_temperature_cube,
                      self.wet_bulb_integral_cube,
                      self.orog, self.land_sea])
            )
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), "altitude_of_snow_falling_level")
        self.assertEqual(result.units, Unit('m'))
        self.assertArrayAlmostEqual(result.data, self.expected_snow_sleet)

    def test_snow_sleet_phase_change_reorder_cubes(self):
        """Same test as test_snow_sleet_phase_change but the cubes are in a
        different order"""
        self.orog.data[1, 1] = 100.0
        result = PhaseChangeLevel(phase_change='snow-sleet').process(
            CubeList([self.wet_bulb_integral_cube,
                      self.wet_bulb_temperature_cube,
                      self.orog, self.land_sea])
            )
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), "altitude_of_snow_falling_level")
        self.assertEqual(result.units, Unit('m'))
        self.assertArrayAlmostEqual(result.data, self.expected_snow_sleet)

    def test_sleet_rain_phase_change(self):
        """Test that process returns a cube with the right name, units and
        values. In this instance the phase change is from sleet to rain. Note
        that the wet bulb temperature integral values are doubled such that the
        rain threshold is reached above the surface. The returned level is
        consistent across the field, despite a high point that sits above the
        rain falling level."""
        self.orog.data[1, 1] = 100.0
        self.wet_bulb_integral_cube.data *= 2.
        result = PhaseChangeLevel(phase_change='sleet-rain').process(
            CubeList([self.wet_bulb_temperature_cube,
                      self.wet_bulb_integral_cube,
                      self.orog, self.land_sea]))
        expected = np.ones((3, 3, 3), dtype=np.float32) * 49.178673
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), "altitude_of_rain_falling_level")
        self.assertEqual(result.units, Unit('m'))
        self.assertArrayAlmostEqual(result.data, expected)

    def test_inverted_input_cube(self):
        """Test that the phase change level process returns a cube
        containing the expected data when the height coordinate is in
        ascending order rather than the expected descending order."""
        self.orog.data[1, 1] = 100.0
        result = PhaseChangeLevel(phase_change='snow-sleet').process(
            CubeList([self.wet_bulb_temperature_cube,
                      self.wet_bulb_integral_cube,
                      self.orog, self.land_sea]))
        self.assertArrayAlmostEqual(result.data, self.expected_snow_sleet)

    def test_interpolation_from_sea_points(self):
        """Test that the phase change level process returns a cube
        containing the expected data. In this case there is a single
        non-sea-level point in the orography. The snow falling level is below
        the surface of the sea, so for the single high point falling level is
        interpolated from the surrounding sea-level points."""
        orog = self.orog
        orog.data = orog.data * 0.0
        orog.data[1, 1] = 100.0
        land_sea = self.land_sea
        land_sea.data[1, 1] = 1
        result = PhaseChangeLevel(phase_change='snow-sleet').process(
            CubeList([self.wet_bulb_temperature_cube,
                      self.wet_bulb_integral_cube,
                      orog, land_sea]))
        expected = np.ones((3, 3, 3), dtype=np.float32) * 65.88566
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_too_many_cubes(self):
        """Tests that an error is raised if there are too many cubes."""
        msg = "Expected 4"
        with self.assertRaisesRegex(ValueError, msg):
            PhaseChangeLevel(phase_change='snow-sleet').process(
                CubeList([self.wet_bulb_temperature_cube,
                          self.wet_bulb_integral_cube,
                          self.orog, self.land_sea, self.orog]))

    def test_empty_cube_list(self):
        """Tests that an error is raised if there is an empty list."""
        msg = "Expected 4"
        with self.assertRaisesRegex(ValueError, msg):
            PhaseChangeLevel(phase_change='snow-sleet').process(
                CubeList([]))


if __name__ == '__main__':
    unittest.main()
