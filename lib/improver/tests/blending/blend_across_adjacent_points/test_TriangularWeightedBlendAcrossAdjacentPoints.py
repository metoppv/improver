# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
"""Unit tests for the
   weighted_blend.TriangularWeightedBlendAcrossAdjacentPoints plugin."""

import unittest

from cf_units import Unit

from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests import IrisTest
from iris.exceptions import CoordinateNotFoundError

import numpy as np

from improver.blending.blend_across_adjacent_points import \
    TriangularWeightedBlendAcrossAdjacentPoints


def set_up_cube():
    """A helper function to set up input cubes for unit tests.
       The cube has latitude, longitude and time dimensions"""
    data = np.zeros((2, 2, 2))

    orig_cube = Cube(data, units="m",
                     standard_name="lwe_thickness_of_precipitation_amount")
    orig_cube.add_dim_coord(DimCoord(np.linspace(-45.0, 45.0, 2),
                                     'latitude', units='degrees'), 1)
    orig_cube.add_dim_coord(DimCoord(np.linspace(120, 180, 2), 'longitude',
                                     units='degrees'), 2)
    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"
    tunit = Unit(time_origin, calendar)
    orig_cube.add_dim_coord(DimCoord([402192.5, 402193.5],
                                     "time", units=tunit), 0)
    orig_cube.add_aux_coord(DimCoord([0, 1],
                                     "forecast_period", units="hours"), 0)
    return orig_cube


class Test__repr__(IrisTest):

    """Test the __repr__ method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        width = 3.0
        result = str(TriangularWeightedBlendAcrossAdjacentPoints(
            'time', width, 'hours', 'weighted_mean'))
        msg = ('<TriangularWeightedBlendAcrossAdjacentPoints:'
               ' coord = time, width = 3.00,'
               ' parameter_units = hours, mode = weighted_mean>')
        self.assertEqual(result, msg)


class Test__init__(IrisTest):

    """Test the __init__ method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        width = 3.0
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'time', width, 'hours', 'weighted_mean')
        expected_coord = "time"
        expected_width = 3.0
        expected_parameter_units = "hours"
        self.assertEqual(plugin.coord, expected_coord)
        self.assertEqual(plugin.width, expected_width)
        self.assertEqual(plugin.parameter_units, expected_parameter_units)

    def test_raises_expression(self):
        """Test that the __init__ raises the right error"""
        message = ("weighting_mode: no_mode is not recognised, "
                   "must be either weighted_maximum or weighted_mean")
        with self.assertRaisesRegex(ValueError, message):
            TriangularWeightedBlendAcrossAdjacentPoints(
                'time', 3.0, 'hours', 'no_mode')


class Test_correct_collapsed_coordinates(IrisTest):

    """Test the correct_collapsed_coordinates method"""

    def setUp(self):
        """Set up a test orig_cube, new_cube and plugin instance."""
        self.orig_cube = set_up_cube()
        new_cube = set_up_cube()
        new_cube.remove_coord('longitude')
        new_cube.add_dim_coord(DimCoord(np.linspace(100, 160, 2), 'longitude',
                                        units='degrees'), 2)
        new_cube.remove_coord('forecast_period')
        new_cube.add_aux_coord(DimCoord([5, 6],
                                        "forecast_period", units="hours"), 0)
        self.new_cube = new_cube
        self.plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'time', 1.0, 'hours', 'weighted_mean')

    def test_no_change_to_new_cube(self):
        """Test it does nothing when nothing to correct"""
        input_new_cube = self.new_cube.copy()
        self.plugin.correct_collapsed_coordinates(self.orig_cube,
                                                  input_new_cube, ['latitude'])
        self.assertEqual(input_new_cube, self.new_cube)

    def test_change_one_coord(self):
        """Test it changes only one coordinate"""
        input_new_cube = self.new_cube.copy()
        self.plugin.correct_collapsed_coordinates(self.orig_cube,
                                                  input_new_cube,
                                                  ['longitude'])
        self.assertEqual(input_new_cube.coord('longitude'),
                         self.orig_cube.coord('longitude'))
        self.assertEqual(input_new_cube.coord('latitude'),
                         self.new_cube.coord('latitude'))
        self.assertEqual(input_new_cube.coord('forecast_period'),
                         self.new_cube.coord('forecast_period'))

    def test_change_two_coord(self):
        """Test it corrects multiple coordinate"""
        input_new_cube = self.new_cube.copy()
        self.plugin.correct_collapsed_coordinates(
            self.orig_cube, input_new_cube, ['longitude', 'forecast_period'])
        self.assertEqual(input_new_cube.coord('longitude'),
                         self.orig_cube.coord('longitude'))
        self.assertEqual(input_new_cube.coord('latitude'),
                         self.new_cube.coord('latitude'))
        self.assertEqual(input_new_cube.coord('forecast_period'),
                         self.orig_cube.coord('forecast_period'))

    def test_bounds_corrected(self):
        """Test it corrects bounds"""
        input_new_cube = self.new_cube.copy()
        input_orig_cube = self.orig_cube.copy()
        input_orig_cube.remove_coord('forecast_period')
        input_orig_cube.add_aux_coord(DimCoord(
            [5, 6], "forecast_period", bounds=[[4.5, 5.5], [5.5, 6.5]],
            units="hours"), 0)
        self.plugin.correct_collapsed_coordinates(
            input_orig_cube, input_new_cube, ['forecast_period'])
        self.assertEqual(input_new_cube.coord('forecast_period'),
                         input_orig_cube.coord('forecast_period'))

    def test_wrong_size_coords(self):
        """Test it raises an error when new_cube and old_cube have
           different length coordinates"""
        data = np.zeros((2, 2, 2))
        orig_cube = Cube(data, units="m",
                         standard_name="lwe_thickness_of_precipitation_amount")
        orig_cube.add_dim_coord(DimCoord([0, 1], "forecast_period",
                                         units="hours"), 0)
        data = np.zeros((3, 2, 2))
        new_cube = Cube(data, units="m",
                        standard_name="lwe_thickness_of_precipitation_amount")
        new_cube.add_dim_coord(DimCoord([0, 1, 2], "forecast_period",
                                        units="hours"), 0)

        # r added in front of error message string to make this a raw string
        # and avoid 'anomalous backslash in string' codacy and travis errors.
        message = r"Require data with shape \(3,\), got \(2,\)\."
        with self.assertRaisesRegex(ValueError, message):
            self.plugin.correct_collapsed_coordinates(orig_cube, new_cube,
                                                      ['forecast_period'])

    def test_exception_when_coord_not_found(self):
        """Test that an exception is raised by Iris when we try to correct
           a coordinate that doesn't exist."""
        self.new_cube.remove_coord('forecast_period')
        message = "Expected to find exactly 1 .* coordinate, but found none."
        with self.assertRaisesRegex(CoordinateNotFoundError, message):
            self.plugin.correct_collapsed_coordinates(self.orig_cube,
                                                      self.new_cube,
                                                      ['forecast_period'])


class Test_process(IrisTest):
    """Test the process method."""

    def setUp(self):
        """Set up a test cube."""
        self.cube = set_up_cube()
        data = np.zeros((2, 2, 2))
        data[0][:][:] = 1.0
        data[1][:][:] = 2.0
        self.cube.data = data

    def test_basic_triangle_width_1(self):
        """Test that the plugin produces sensible results when the width
           of the triangle is 1. This is equivalent to no blending."""
        width = 1.0
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'forecast_period', width, 'hours', 'weighted_mean')
        result = plugin.process(self.cube)
        self.assertEqual(self.cube.coord('forecast_period'),
                         result.coord('forecast_period'))
        self.assertEqual(self.cube.coord('time'), result.coord('time'))
        self.assertArrayEqual(self.cube.data, result.data)

    def test_basic_triangle_width_2(self):
        """Test that the plugin produces sensible results when the width
           of the triangle is 2 and there is some blending."""
        width = 2.0
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'forecast_period', width, 'hours', 'weighted_mean')
        result = plugin.process(self.cube)
        expected_data = np.array([[[1.333333, 1.333333],
                                   [1.333333, 1.333333]],
                                  [[1.666666, 1.666666],
                                   [1.666666, 1.666666]]])
        self.assertEqual(self.cube.coord('forecast_period'),
                         result.coord('forecast_period'))
        self.assertEqual(self.cube.coord('time'), result.coord('time'))
        self.assertArrayAlmostEqual(expected_data, result.data)

    def test_basic_triangle_width_1_max_mode(self):
        """Test that the plugin produces sensible results when the width
           of the triangle is 1. This is equivalent to no blending. This time
           use the weighted_maximum mode"""
        width = 1.0
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'forecast_period', width, 'hours', 'weighted_maximum')
        result = plugin.process(self.cube)
        self.assertEqual(self.cube.coord('forecast_period'),
                         result.coord('forecast_period'))
        self.assertEqual(self.cube.coord('time'), result.coord('time'))
        self.assertArrayEqual(self.cube.data, result.data)

    def test_basic_triangle_width_2_max_mode(self):
        """Test that the plugin produces sensible results when the width
           of the triangle is 2 and there is some blending. This time
           use the weighted_maximum mode"""
        width = 2.0
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'forecast_period', width, 'hours', 'weighted_maximum')
        result = plugin.process(self.cube)
        expected_data = np.array([[[0.6666666, 0.6666666],
                                   [0.6666666, 0.6666666]],
                                  [[1.3333333, 1.3333333],
                                   [1.3333333, 1.3333333]]])
        self.assertEqual(self.cube.coord('forecast_period'),
                         result.coord('forecast_period'))
        self.assertEqual(self.cube.coord('time'), result.coord('time'))
        self.assertArrayAlmostEqual(expected_data, result.data)


if __name__ == '__main__':
    unittest.main()
