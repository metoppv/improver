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

import iris
from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests import IrisTest
from iris.exceptions import CoordinateNotFoundError

import numpy as np

from improver.blending.blend_across_adjacent_points import \
    TriangularWeightedBlendAcrossAdjacentPoints
from improver.tests.blending.weights.helper_functions import (
    cubes_for_triangular_weighted_blend_tests)
from improver.utilities.warnings_handler import ManageWarnings
from improver.utilities.cube_metadata import add_coord
from improver.utilities.cube_manipulation import concatenate_cubes


class Test__repr__(IrisTest):

    """Test the __repr__ method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        width = 3.0
        forecast_period = 1
        result = str(TriangularWeightedBlendAcrossAdjacentPoints(
            'time', forecast_period, 'hours', width, 'weighted_mean'))
        msg = ('<TriangularWeightedBlendAcrossAdjacentPoints:'
               ' coord = time, central_point = 1.00, '
               'parameter_units = hours, width = 3.00, mode = weighted_mean>')
        self.assertEqual(result, msg)


class Test__init__(IrisTest):

    """Test the __init__ method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        width = 3.0
        forecast_period = 1
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'time', forecast_period, 'hours', width, 'weighted_mean')
        expected_coord = 'time'
        expected_width = 3.0
        expected_parameter_units = 'hours'
        self.assertEqual(plugin.coord, expected_coord)
        self.assertEqual(plugin.width, expected_width)
        self.assertEqual(plugin.parameter_units, expected_parameter_units)

    def test_raises_expression(self):
        """Test that the __init__ raises the right error"""
        message = ("weighting_mode: no_mode is not recognised, "
                   "must be either weighted_maximum or weighted_mean")
        with self.assertRaisesRegex(ValueError, message):
            TriangularWeightedBlendAcrossAdjacentPoints(
                'time', 1, 'hours', 3.0, 'no_mode')


class Test__find_central_point(IrisTest):
    """Test the _find_central_point."""

    def setUp(self):
        """Set up a test cube."""
        self.cube, self.central_cube, self.forecast_period = (
            cubes_for_triangular_weighted_blend_tests())
        self.width = 1.0

    def test_central_point_available(self):
        """Test that the central point is available within the input cube."""
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'forecast_period', self.forecast_period, 'hours', self.width,
            'weighted_mean')
        central_cube = plugin._find_central_point(self.cube)
        self.assertEqual(self.central_cube.coord('forecast_period'),
                         central_cube.coord('forecast_period'))
        self.assertEqual(self.central_cube.coord('time'),
                         central_cube.coord('time'))
        self.assertArrayEqual(self.central_cube.data, central_cube.data)

    def test_central_point_not_available(self):
        """Test that the central point is not available within the
           input cube."""
        forecast_period = 2
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'forecast_period', forecast_period, 'hours', self.width,
            'weighted_mean')
        msg = 'The central point of'
        with self.assertRaisesRegex(ValueError, msg):
            plugin._find_central_point(self.cube)


class Test_process(IrisTest):
    """Test the process method."""

    def setUp(self):
        """Set up a test cube."""
        self.cube, self.central_cube, self.forecast_period = (
            cubes_for_triangular_weighted_blend_tests())

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_triangle_width_1(self):
        """Test that the plugin produces sensible results when the width
           of the triangle is 1. This is equivalent to no blending."""
        width = 1.0
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'forecast_period', self.forecast_period, 'hours', width,
            'weighted_mean')
        result = plugin.process(self.cube)
        self.assertEqual(self.central_cube.coord('forecast_period'),
                         result.coord('forecast_period'))
        self.assertEqual(self.central_cube.coord('time'), result.coord('time'))
        self.assertArrayEqual(self.central_cube.data, result.data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_triangle_width_2(self):
        """Test that the plugin produces sensible results when the width
           of the triangle is 2 and there is some blending."""
        width = 2.0
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'forecast_period', self.forecast_period, 'hours', width,
            'weighted_mean')
        result = plugin.process(self.cube)
        expected_data = np.array([[1.333333, 1.333333],
                                  [1.333333, 1.333333]])
        self.assertEqual(self.central_cube.coord('forecast_period'),
                         result.coord('forecast_period'))
        self.assertEqual(self.central_cube.coord('time'), result.coord('time'))
        self.assertArrayAlmostEqual(expected_data, result.data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_triangle_width_1_max_mode(self):
        """Test that the plugin produces sensible results when the width
           of the triangle is 1. This is equivalent to no blending. This time
           use the weighted_maximum mode"""
        width = 1.0
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'forecast_period', self.forecast_period, 'hours', width,
            'weighted_maximum')
        result = plugin.process(self.cube)
        self.assertEqual(self.central_cube.coord('forecast_period'),
                         result.coord('forecast_period'))
        self.assertEqual(self.central_cube.coord('time'), result.coord('time'))
        self.assertArrayEqual(self.central_cube.data, result.data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_basic_triangle_width_2_max_mode(self):
        """Test that the plugin produces sensible results when the width
           of the triangle is 2 and there is some blending. This time
           use the weighted_maximum mode"""
        width = 2.0
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'forecast_period', self.forecast_period, 'hours', width,
            'weighted_maximum')
        result = plugin.process(self.cube)
        expected_data = np.array([[0.6666666, 0.6666666],
                                  [0.6666666, 0.6666666]])
        self.assertEqual(self.central_cube.coord('forecast_period'),
                         result.coord('forecast_period'))
        self.assertEqual(self.central_cube.coord('time'), result.coord('time'))
        self.assertArrayAlmostEqual(expected_data, result.data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_central_point_not_in_allowed_range(self):
        """Test that an exception is generated when the central cube is not
           within the allowed range."""
        width = 1.0
        forecast_period = 2
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'forecast_period', forecast_period, 'hours', width,
            'weighted_mean')
        msg = "The central point of"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(self.cube)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_alternative_parameter_units(self):
        """Test that the plugin produces sensible results when the width
           of the triangle is 7200 seconds. """
        forecast_period = 0
        width = 7200.0
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'forecast_period', forecast_period, 'seconds', width,
            'weighted_mean')
        result = plugin.process(self.cube)
        expected_data = np.array([[1.333333, 1.333333],
                                  [1.333333, 1.333333]])
        self.assertEqual(self.central_cube.coord('forecast_period'),
                         result.coord('forecast_period'))
        self.assertEqual(self.central_cube.coord('time'), result.coord('time'))
        self.assertArrayAlmostEqual(expected_data, result.data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_input_cube_no_change(self):
        """Test that the plugin does not change the origonal input cube."""

        # Add threshold axis to standard input cube.
        changes = {'points': [0.5], 'units': '1'}
        cube_with_thresh = add_coord(self.cube.copy(), 'threshold', changes)
        original_cube = cube_with_thresh.copy()

        width = 2.0
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'forecast_period', self.forecast_period, 'hours', width,
            'weighted_mean')
        _ = plugin.process(cube_with_thresh)

        # Test that the input cube is unchanged by the function.
        self.assertEqual(cube_with_thresh, original_cube)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_works_one_thresh(self):
        """Test that the plugin retains the single threshold from the input
           cube."""

        # Creates a cube containing the expected outputs.
        fill_value = 1 + 1/3.0
        data = np.full((2, 2), fill_value)

        # Take a slice of the time coordinate.
        expected_cube = self.cube[0].copy(data)

        # Add threshold axis to expected output cube.
        changes = {'points': [0.5], 'units': '1'}
        expected_cube = add_coord(expected_cube, 'threshold', changes)

        # Add threshold axis to standard input cube.
        cube_with_thresh = add_coord(self.cube.copy(), 'threshold', changes)

        width = 2.0
        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'forecast_period', self.forecast_period, 'hours', width,
            'weighted_mean')
        result = plugin.process(cube_with_thresh)

        # Test that the result cube retains threshold co-ordinates
        # from origonal cube.
        print("self.cube = ", self.cube)
        print("self.cube = ", self.cube.coord("time"))
        print("result = ", result)
        print("expected = ", expected_cube)
        self.assertEqual(expected_cube.coord('threshold'),
                         result.coord('threshold'))
        self.assertArrayEqual(expected_cube.data, result.data)
        self.assertEqual(expected_cube, result)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate."])
    def test_works_two_thresh(self):
        """Test that the plugin works with a cube that contains multiple
           thresholds."""
        width = 2.0

        changes = {'points': [0.25], 'units': '1'}
        cube_with_thresh1 = add_coord(self.cube.copy(), 'threshold', changes)

        changes = {'points': [0.5], 'units': '1'}
        cube_with_thresh2 = add_coord(self.cube.copy(), 'threshold', changes)

        changes = {'points': [0.75], 'units': '1'}
        cube_with_thresh3 = add_coord(self.cube.copy(), 'threshold', changes)

        cubelist = iris.cube.CubeList([cube_with_thresh1, cube_with_thresh2,
                                       cube_with_thresh3])

        thresh_cubes = concatenate_cubes(cubelist,
                                         coords_to_slice_over='threshold')

        plugin = TriangularWeightedBlendAcrossAdjacentPoints(
            'forecast_period', self.forecast_period, 'hours', width,
            'weighted_mean')
        result = plugin.process(thresh_cubes)

        # Test that the result cube retains threshold co-ordinates
        # from origonal cube.
        self.assertEqual(thresh_cubes.coord('threshold'),
                         result.coord('threshold'))


if __name__ == '__main__':
    unittest.main()
