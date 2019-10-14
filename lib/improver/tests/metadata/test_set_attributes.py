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
"""Tests for the improver.metadata.set_attributes module"""

import unittest
import numpy as np

from improver.metadata.set_attributes import (
    set_product_attributes, _match_title, update_spot_title_attribute)
from improver.tests.set_up_test_cubes import set_up_variable_cube
from improver.utilities.warnings_handler import ManageWarnings


class Test_set_product_attributes(unittest.TestCase):
    """Test the set_product_attributes function"""

    def setUp(self):
        """Set up a dummy cube"""
        attributes = {
            "source": "Met Office Unified Model",
            "title": "MOGREPS-UK Temperature Forecast "
                     "on UK 2 km Standard Grid"}
        self.cube = set_up_variable_cube(
            278*np.ones((5, 5), dtype=np.float32), attributes=attributes)

        self.blend_descriptor = "IMPROVER Post-Processed Multi-Model Blend"
        self.nowcast_descriptor = "MONOW Extrapolation Nowcast"
        self.grid_descriptor = " on UK 2 km Standard Grid"

    def test_blend(self):
        """Test blended attributes are correctly set"""
        expected_title = self.blend_descriptor + self.grid_descriptor
        result = set_product_attributes(self.cube, "multi-model blend")
        attrs = result.attributes
        self.assertEqual(attrs["source"], "IMPROVER")
        self.assertEqual(attrs["title"], expected_title)
        self.assertEqual(attrs["institution"], "Met Office")

    def test_nowcast(self):
        """Test nowcast attributes are correctly set"""
        expected_title = self.nowcast_descriptor + self.grid_descriptor
        result = set_product_attributes(self.cube, "nowcast")
        attrs = result.attributes
        self.assertEqual(attrs["source"], "IMPROVER")
        self.assertEqual(attrs["title"], expected_title)
        self.assertEqual(attrs["institution"], "Met Office")

    def test_no_grid_descriptor(self):
        """Test blended cube only has grid descriptor where appropriate"""
        self.cube.attributes["title"] = "MOGREPS-UK Temperature Forecast"
        result = set_product_attributes(self.cube, "multi-model blend")
        self.assertEqual(result.attributes["title"], self.blend_descriptor)

    def test_error_invalid_product(self):
        """Test error when product is not recognised"""
        msg = "product 'unknown product' not available"
        with self.assertRaisesRegex(ValueError, msg):
            set_product_attributes(self.cube, "unknown product")


class Test__match_title(unittest.TestCase):
    """Test the _match_title function"""

    def test_basic(self):
        """Test pattern matching is working as expected"""
        expected_field = "MOGREPS-UK Temperature Forecast"
        expected_grid = "UK 2 km Standard Grid"
        original_title = (
            "MOGREPS-UK Temperature Forecast on UK 2 km Standard Grid")
        result = _match_title(original_title)
        self.assertEqual(result.group("field"), expected_field)
        self.assertEqual(result.group("grid"), expected_grid)


class Test_update_spot_title_attribute(unittest.TestCase):
    """Test the update_spot_title_attribute function"""

    def setUp(self):
        """Set up a test cube (spot data not required)"""
        attributes = {
            "title": "MOGREPS-UK Temperature Forecast "
                     "on UK 2 km Standard Grid"}
        self.cube = set_up_variable_cube(
            278*np.ones((2, 2), dtype=np.float32), attributes=attributes)
        self.expected_title = "MOGREPS-UK Temperature Forecast UK Spot Values"

    def test_basic(self):
        """Test title is correctly updated in place"""
        update_spot_title_attribute(self.cube)
        self.assertEqual(self.cube.attributes["title"], self.expected_title)

    def test_global_grid(self):
        """Test function responds correctly to a non-UK grid"""
        self.cube.attributes["title"] = (
            "MOGREPS-G Temperature Forecast on Global Grid")
        expected_title = "MOGREPS-G Temperature Forecast Spot Values"
        update_spot_title_attribute(self.cube)
        self.assertEqual(self.cube.attributes["title"], expected_title)

    def test_other_grid(self):
        """Test function responds correctly to an unknown grid"""
        self.cube.attributes["title"] = (
            "MOGREPS-UK Temperature Forecast on Other Grid")
        expected_title = "MOGREPS-UK Temperature Forecast Spot Values"
        update_spot_title_attribute(self.cube)
        self.assertEqual(self.cube.attributes["title"], expected_title)

    def test_other_uk_grid(self):
        """Test function responds correctly to an unknown UK grid"""
        self.cube.attributes["title"] = (
            "MOGREPS-UK Temperature Forecast on Other UK Grid")
        update_spot_title_attribute(self.cube)
        self.assertEqual(self.cube.attributes["title"], self.expected_title)

    def test_no_title(self):
        """Test no change is made to an input cube with no title"""
        self.cube.attributes.pop("title")
        update_spot_title_attribute(self.cube)
        self.assertNotIn("title", self.cube.attributes.keys())

    def test_title_already_spot(self):
        """Test no change is made to an input cube where the title contains
        a spot-data-descriptive substring"""
        self.cube.attributes["title"] = "Spot Values"
        update_spot_title_attribute(self.cube)
        self.assertEqual(self.cube.attributes["title"], "Spot Values")

    @ManageWarnings(record=True)
    def test_unexpected_title(self, warning_list=None):
        """Test a warning is raised if we don't recognise the input form"""
        self.cube.attributes["title"] = "kitten farm"
        warning_msg = "'title' attribute does not match expected pattern"
        update_spot_title_attribute(self.cube)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertEqual(self.cube.attributes["title"], "kitten farm")


if __name__ == '__main__':
    unittest.main()
