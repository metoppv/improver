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
Unit tests for the function "cube_manipulation.get_filtered_attributes".
"""

import unittest

import numpy as np
from iris.tests import IrisTest

from improver.utilities.cube_manipulation import get_filtered_attributes

from ...set_up_test_cubes import set_up_variable_cube


class Test_get_filtered_attributes(IrisTest):

    """Test the get_filtered_attributes function."""

    def setUp(self):
        """Use temperature cube to test with."""

        self.attributes = {
            'mosg__grid_domain': 'uk',
            'mosg__grid_type': 'standard',
            'mosg__grid_version': '1.2.0',
            'mosg__model_configuration': 'uk_det'}

        data = np.arange(25).reshape(5, 5).astype(np.float32)
        self.cube = set_up_variable_cube(data,
                                         attributes=self.attributes,
                                         spatial_grid="equalarea")

    def test_no_filter(self):
        """Test a case in which all the attributes of the cube passed in are
        returned as no filter is specified."""
        result = get_filtered_attributes(self.cube)
        self.assertEqual(result, self.cube.attributes)

    def test_all_matches(self):
        """Test a case in which the cube passed in contains attributes that
        all partially match the attribute_filter string."""
        attribute_filter = 'mosg'
        result = get_filtered_attributes(self.cube,
                                         attribute_filter=attribute_filter)
        self.assertEqual(result, self.attributes)

    def test_subset_of_matches(self):
        """Test a case in which the cube passed in contains some attributes
        that partially match the attribute_filter string, and some that do
        not."""
        attribute_filter = 'mosg__grid'
        expected = self.attributes
        expected.pop('mosg__model_configuration')
        result = get_filtered_attributes(self.cube,
                                         attribute_filter=attribute_filter)
        self.assertEqual(result, expected)

    def test_without_matches(self):
        """Test a case in which the cube passed in does not contain any
        attributes that partially match the expected string."""
        attribute_filter = 'test'
        result = get_filtered_attributes(self.cube,
                                         attribute_filter=attribute_filter)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
