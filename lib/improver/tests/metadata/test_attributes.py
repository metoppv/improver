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
"""
Tests for the improver.metadata.attributes module
"""

import unittest
import numpy as np

from improver.metadata.attributes import (
    set_product_attributes, update_spot_title_attribute)
from improver.tests.set_up_test_cubes import set_up_variable_cube


class Test_set_product_attributes(unittest.TestCase):
    """Test the set_product_attributes function"""

    def setUp(self):
        """Set up a dummy cube"""
        attributes = {
            "source": "Met Office Unified Model",
            "title": "MOGREPS-UK Temperature Forecast "
                     "on UK 2 km Standard Grid"}
        self.cube = set_up_variable_cube(
            278*np.ones((5, 5), dtype=np.float32),
            attributes=attributes)

        self.blend_descriptor = "IMPROVER Post-Processed Multi-Model Blend"
        self.nowcast_descriptor = "MONOW Extrapolation Nowcast"
        self.grid_descriptor = " on UK 2 km Standard Grid"

    def test_blend(self):
        """Test cube is modified in place"""
        expected_title = self.blend_descriptor + self.grid_descriptor
        set_product_attributes(self.cube, "multi-model blend")
        attrs = self.cube.attributes
        self.assertEqual(attrs["source"], "IMPROVER")
        self.assertEqual(attrs["title"], expected_title)
        self.assertEqual(attrs["institution"], "Met Office")

    def test_nowcast(self):
        """Test nowcast attributes"""
        expected_title = self.nowcast_descriptor + self.grid_descriptor
        set_product_attributes(self.cube, "nowcast")
        attrs = self.cube.attributes
        self.assertEqual(attrs["source"], "IMPROVER")
        self.assertEqual(attrs["title"], expected_title)
        self.assertEqual(attrs["institution"], "Met Office")        

    def test_no_grid_descriptor(self):
        """Test blended cube only has grid descriptor where appropriate"""
        self.cube.attributes["title"] = "MOGREPS-UK Temperature Forecast"
        set_product_attributes(self.cube, "multi-model blend")
        self.assertEqual(self.cube.attributes["title"], self.blend_descriptor)

    def test_error_invalid_product(self):
        """Test error when product is not recognised"""
        msg = "product 'unknown product' not available"
        with self.assertRaisesRegex(ValueError, msg):
            set_product_attributes(self.cube, "unknown product")




if __name__ == '__main__':
    unittest.main()
