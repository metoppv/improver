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
"""Unit tests for Weather Symbols class."""

import unittest

import numpy as np


import iris
from iris.cube import Cube
from iris.tests import IrisTest
from cf_units import Unit

from improver.wxcode.weather_symbols import WeatherSymbols
from improver.wxcode.wxcode_utilities import WX_DICT
from improver.tests.ensemble_calibration.ensemble_calibration. \
    helper_functions import set_up_probability_above_threshold_cube


iris.FUTURE.netcdf_promote = True


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(WeatherSymbols())
        msg = '<WeatherSymbols>'
        self.assertEqual(result, msg)


class Test_check_input_cubes(IrisTest):

    """Test the check_input_cubes method."""

    def test_basic(self):
        """Test that the invert_condition method returns a string."""
        plugin = WeatherSymbols()
        tree = plugin.queries
        print tree[tree.keys()[0]]
        result = plugin.invert_condition(tree[tree.keys()[0]])
        self.assertIsInstance(result, str)

# class Test_invert_condition(IrisTest):

#    """Test the invert condition method."""

#    def test_basic(self):
#        """Test that the invert_condition method returns a string."""
#        plugin = WeatherSymbols()
#        tree = plugin.queries
#        print tree[tree.keys()[0]]
#        result = plugin.invert_condition(tree[tree.keys()[0]])
#        self.assertIsInstance(result, str)


# class Test_construct_condition(IrisTest):

#    """Test the construct condition method."""

#    def test_basic(self):
#        """Test that the construct_condition method returns a string."""
#        plugin = WeatherSymbols()
#        tree = plugin.queries
#        print tree[tree.keys()[0]]
#        result = plugin.invert_condition(tree[tree.keys()[0]])
#        self.assertIsInstance(result, str)

# class Test_format_condition_chain(IrisTest):

#    """Test the format_condition_chain method."""

#    def test_basic(self):
#        """Test that the format_condition_change method returns a string."""
#        plugin = WeatherSymbols()
#        tree = plugin.queries
#        print tree[tree.keys()[0]]
#        result = plugin.invert_condition(tree[tree.keys()[0]])
#        self.assertIsInstance(result, str)


# class Test_create_condition_chain(IrisTest):

#    """Test the create_condition_chain method."""

#    def test_basic(self):
#        """Test create_condition_change method returns a list of strings."""
#        plugin = WeatherSymbols()
#        tree = plugin.queries
#        print tree[tree.keys()[0]]
#        result = plugin.invert_condition(tree[tree.keys()[0]])
#        self.assertIsInstance(result, list)
#        self.assertIsInstance(result[0], str)


# class Test_construct_extract_constraint(IrisTest):
#
    """Test the construct_extract_constraint method ."""

#    def test_basic(self):
#        """Test construct_extract_constraint method returns a iris.Constraint.
#            or list of iris.Constraint"""
#        plugin = WeatherSymbols()
#        tree = plugin.queries
#        print tree[tree.keys()[0]]
#        result = plugin.invert_condition(tree[tree.keys()[0]])
#        if isinstance(result, list)
#            self.assertIsInstance(result[0], iris.Constraint)
#        else:
#            self.assertIsInstance(result, iris.Constraint)


# class Test_find_all_routes(IrisTest):
#
#    """Test the find_all_routes method ."""
#
#    def test_basic(self):
#        """Test construct_extract_constraint method returns a iris.Constraint.
#            or list of iris.Constraint"""
#        plugin = WeatherSymbols()
#        tree = plugin.queries
#        print tree[tree.keys()[0]]
#        result = plugin.invert_condition(tree[tree.keys()[0]])
#        if isinstance(result, list)
#            self.assertIsInstance(result[0], iris.Constraint)
#        else:
#            self.assertIsInstance(result, iris.Constraint)

class Test_create_symbol_cube(IrisTest):

    """Test the create_symbol_cube method ."""

    def setUp(self):
        """Set up cube """
        data = np.array([0.1, 0.3, 0.4, 0.2, 0.6, 0.7, 0.4, 0.2, 0.1,
                         0.2, 0.2, 0.5, 0.1, 0.3, 0.9, 0.8, 0.5, 0.3,
                         0.6, 0.3, 0.5, 0.6, 0.8, 0.2,
                         0.8, 0.1, 0.2]).reshape(3, 1, 3, 3)
        self.cube = set_up_probability_above_threshold_cube(data,
                                                            'air_temperature',
                                                            'K')
        self.wxcode = np.array(WX_DICT.keys())
        self.wxmeaning = " ".join(WX_DICT.values())

    def test_basic(self):
        """Test construct_extract_constraint method returns a iris.Constraint.
            or list of iris.Constraint"""
        plugin = WeatherSymbols()

        result = plugin.create_symbol_cube(self.cube[0])
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.attributes['weather_code'], self.wxcode)
        self.assertEqual(result.attributes['weather_code_meaning'],
                         self.wxmeaning)


# class Test_process(IrisTest):

#    """Test the find_all_routes method ."""

#    def test_basic(self):
#        """Test construct_extract_constraint method returns a iris.Constraint.
#            or list of iris.Constraint"""
#        plugin = WeatherSymbols()
#        tree = plugin.queries
#        print tree[tree.keys()[0]]
#        result = plugin.invert_condition(tree[tree.keys()[0]])

#        self.assertIsInstance(result[0], iris.cube.Cube)


if __name__ == '__main__':
    unittest.main()
