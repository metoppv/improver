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
"""Tests for the improver.metadata.probabilistic module"""

import unittest

import iris
import numpy as np
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest

from improver.metadata.probabilistic import (
    extract_diagnostic_name, find_percentile_coordinate,
    find_threshold_coordinate, in_vicinity_name_format)

from ..metadata.test_amend import create_cube_with_threshold
from ..set_up_test_cubes import set_up_probability_cube
from ..wind_calculations.wind_gust_diagnostic.test_WindGustDiagnostic import (
    create_cube_with_percentile_coord)


class Test_in_vicinity_name_format(unittest.TestCase):
    """Test that the 'in_vicinity' above/below threshold probability
    cube naming function produces the correctly formatted names."""

    def setUp(self):
        """Set up test cube"""
        self.cube = create_cube_with_threshold()
        self.cube.long_name = 'probability_of_X_rate_above_threshold'

    def test_in_vicinity_name_format(self):
        """Test that 'in_vicinity' is added correctly to the name for both
        above and below threshold cases"""
        correct_name_above = (
            'probability_of_X_rate_in_vicinity_above_threshold')
        new_name_above = in_vicinity_name_format(self.cube.name())
        self.cube.rename('probability_of_X_below_threshold')
        correct_name_below = (
            'probability_of_X_in_vicinity_below_threshold')
        new_name_below = in_vicinity_name_format(self.cube.name())
        self.assertEqual(new_name_above, correct_name_above)
        self.assertEqual(new_name_below, correct_name_below)

    def test_between_thresholds(self):
        """Test for "between_thresholds" suffix"""
        self.cube.rename('probability_of_visibility_between_thresholds')
        correct_name = (
            'probability_of_visibility_in_vicinity_between_thresholds')
        new_name = in_vicinity_name_format(self.cube.name())
        self.assertEqual(new_name, correct_name)

    def test_no_above_below_threshold(self):
        """Test the case of name without above/below_threshold is handled
        correctly"""
        self.cube.rename('probability_of_X')
        correct_name_no_threshold = (
            'probability_of_X_in_vicinity')
        new_name_no_threshold = in_vicinity_name_format(self.cube.name())
        self.assertEqual(new_name_no_threshold, correct_name_no_threshold)

    def test_in_vicinity_already_exists(self):
        """Test the case of 'in_vicinity' already existing in the cube name"""
        self.cube.rename('probability_of_X_in_vicinity')
        result = in_vicinity_name_format(self.cube.name())
        self.assertEqual(result, 'probability_of_X_in_vicinity')


class Test_extract_diagnostic_name(unittest.TestCase):
    """Test utility to extract diagnostic name from probability cube name"""

    def test_basic(self):
        """Test correct name is returned from a standard (above threshold)
        probability field"""
        result = extract_diagnostic_name(
            'probability_of_air_temperature_above_threshold')
        self.assertEqual(result, 'air_temperature')

    def test_below_threshold(self):
        """Test correct name is returned from a probability below threshold"""
        result = extract_diagnostic_name(
            'probability_of_air_temperature_below_threshold')
        self.assertEqual(result, 'air_temperature')

    def test_between_thresholds(self):
        """Test correct name is returned from a probability between thresholds
        """
        result = extract_diagnostic_name(
            'probability_of_visibility_in_air_between_thresholds')
        self.assertEqual(result, 'visibility_in_air')

    def test_in_vicinity(self):
        """Test correct name is returned from an "in vicinity" probability.
        Name "cloud_height" is used in this test to illustrate why suffix
        cannot be removed with "rstrip"."""
        diagnostic = 'cloud_height'
        result = extract_diagnostic_name(
            'probability_of_{}_in_vicinity_above_threshold'.format(diagnostic))
        self.assertEqual(result, diagnostic)

    def test_error_not_probability(self):
        """Test exception if input is not a probability cube name"""
        with self.assertRaises(ValueError):
            extract_diagnostic_name('lwe_precipitation_rate')


class Test_find_threshold_coordinate(IrisTest):
    """Test the find_threshold_coordinate function"""

    def setUp(self):
        """Set up test probability cubes with old and new threshold coordinate
        naming conventions"""
        data = np.ones((3, 3, 3), dtype=np.float32)
        self.threshold_points = np.array([276, 277, 278], dtype=np.float32)
        cube = set_up_probability_cube(data, self.threshold_points)

        self.cube_new = cube.copy()
        self.cube_old = cube.copy()
        self.cube_old.coord("air_temperature").rename("threshold")

    def test_basic(self):
        """Test function returns an iris.coords.Coord"""
        threshold_coord = find_threshold_coordinate(self.cube_new)
        self.assertIsInstance(threshold_coord, iris.coords.Coord)

    def test_old_convention(self):
        """Test function recognises threshold coordinate with name "threshold"
        """
        threshold_coord = find_threshold_coordinate(self.cube_old)
        self.assertEqual(threshold_coord.name(), "threshold")
        self.assertArrayAlmostEqual(
            threshold_coord.points, self.threshold_points)

    def test_new_convention(self):
        """Test function recognises threshold coordinate with standard
        diagnostic name and "threshold" as var_name"""
        threshold_coord = find_threshold_coordinate(self.cube_new)
        self.assertEqual(threshold_coord.name(), "air_temperature")
        self.assertEqual(threshold_coord.var_name, "threshold")
        self.assertArrayAlmostEqual(
            threshold_coord.points, self.threshold_points)

    def test_fails_if_not_cube(self):
        """Test error if given a non-cube argument"""
        msg = "Expecting data to be an instance of iris.cube.Cube"
        with self.assertRaisesRegex(TypeError, msg):
            find_threshold_coordinate([self.cube_new])

    def test_fails_if_no_threshold_coord(self):
        """Test error if no threshold coordinate is present"""
        self.cube_new.coord("air_temperature").var_name = None
        msg = "No threshold coord found"
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            find_threshold_coordinate(self.cube_new)


class Test_find_percentile_coordinate(IrisTest):

    """Test whether the cube has a percentile coordinate."""

    def setUp(self):
        """Create a wind-speed and wind-gust cube with percentile coord."""
        data = np.zeros((2, 2, 2, 2))
        self.wg_perc = 50.0
        self.ws_perc = 95.0
        gust = "wind_speed_of_gust"
        self.cube_wg = (
            create_cube_with_percentile_coord(
                data=data,
                perc_values=[self.wg_perc, 90.0],
                perc_name='percentile',
                standard_name=gust))

    def test_basic(self):
        """Test that the function returns a Coord."""
        perc_coord = find_percentile_coordinate(self.cube_wg)
        self.assertIsInstance(perc_coord, iris.coords.Coord)
        self.assertEqual(perc_coord.name(), "percentile")

    def test_fails_if_data_is_not_cube(self):
        """Test it raises a Type Error if cube is not a cube."""
        msg = ('Expecting data to be an instance of '
               'iris.cube.Cube but is'
               ' {}.'.format(type(self.wg_perc)))
        with self.assertRaisesRegex(TypeError, msg):
            find_percentile_coordinate(self.wg_perc)

    def test_fails_if_no_perc_coord(self):
        """Test it raises an Error if there is no percentile coord."""
        msg = ('No percentile coord found on')
        cube = self.cube_wg
        cube.remove_coord("percentile")
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            find_percentile_coordinate(cube)

    def test_fails_if_too_many_perc_coord(self):
        """Test it raises a Value Error if there are too many perc coords."""
        msg = ('Too many percentile coords found')
        cube = self.cube_wg
        new_perc_coord = (
            iris.coords.AuxCoord(1,
                                 long_name='percentile',
                                 units='no_unit'))
        cube.add_aux_coord(new_perc_coord)
        with self.assertRaisesRegex(ValueError, msg):
            find_percentile_coordinate(cube)


if __name__ == '__main__':
    unittest.main()
