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

from improver.metadata.probabilistic import (
    in_vicinity_name_format, extract_diagnostic_name)
from improver.tests.metadata.test_amend import create_cube_with_threshold


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


if __name__ == '__main__':
    unittest.main()
