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
"""Unit tests for the generate_ancillary._make_mask_cube function."""

import unittest
from iris.tests import IrisTest
from iris.coords import DimCoord
import numpy as np

from improver.generate_ancillaries.generate_ancillary import _make_mask_cube


class TestMakeCube(IrisTest):
    """Test private function to make cube from generated mask"""

    def setUp(self):
        """setting up test data"""
        premask = np.array([[0., 3., 2.],
                            [0.5, 0., 1.5],
                            [0.2, 0., 0]])
        self.mask = np.ma.masked_where(premask > 1., premask)
        self.key = 'test key'
        self.x_coord = DimCoord([1, 2, 3], long_name='longitude')
        self.y_coord = DimCoord([1, 2, 3], long_name='latitude')
        self.coords = [self.x_coord, self.y_coord]
        self.upper = 100.
        self.lower = 0.

    def test_nobounds(self):
        """test creating cube with neither upper nor lower threshold set"""
        result = _make_mask_cube(self.mask, self.key, self.coords)
        self.assertEqual(result.coord('longitude'), self.x_coord)
        self.assertEqual(result.coord('latitude'), self.y_coord)
        self.assertArrayEqual(result.data, self.mask)
        self.assertEqual(result.attributes['Topographical Type'],
                         self.key.title())

    def test_upperbound(self):
        """test creating cube with upper threshold only set"""
        result = _make_mask_cube(self.mask, self.key, self.coords,
                                 upper_threshold=self.upper)
        self.assertEqual(result.coords('topographic_bound_upper')[0].points,
                         self.upper)

    def test_bothbounds(self):
        """test creating cube with both thresholds set"""
        result = _make_mask_cube(self.mask, self.key, self.coords,
                                 upper_threshold=self.upper,
                                 lower_threshold=self.lower)
        self.assertEqual(result.coords('topographic_bound_upper')[0].points,
                         self.upper)
        self.assertEqual(result.coords('topographic_bound_lower')[0].points,
                         self.lower)


if __name__ == "__main__":
    unittest.main()
