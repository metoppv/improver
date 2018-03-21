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
"""
Unit tests for the OrographicAlphas utility.

"""

import unittest

import warnings
import numpy as np
from iris.coords import DimCoord
from iris.tests import IrisTest
from iris.cube import Cube
from cf_units import Unit

from improver.utilities.ancillary_creation import OrographicAlphas


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(OrographicAlphas())
        msg = ('<OrographicAlphas: min_alpha: {}; max_alpha: {}; '
               'coefficient: {}; power: {}; intercept: {}; '
               'invert_alphas: {}>'.format(0.0, 1.0, 1, 1, 0, True))
        self.assertEqual(result, msg)


class Test_functions(IrisTest):

    """ Test that the OrographicAlphas functions works as expected. """

    def setUp(self):
        """ set up a cube with dimensions 3 x 3"""

        self.plugin = OrographicAlphas()
        data = np.array([[1., 5., 10.],
                         [3., 4., 7.],
                         [0., 2., 1.]])
        cube = Cube(data, "precipitation_amount", units="kg m^-2 s^-1")
        cube.add_dim_coord(DimCoord(np.linspace(0.0, 4.0, 3), 'latitude',
                                    units='m'), 0)
        cube.add_dim_coord(DimCoord(np.linspace(0.0, 4.0, 3), 'longitude',
                                    units='m'), 1)
        self.cube = cube

    def test_normalise(self):
        """
        Test the basic function of the normalise_cube, using the
        standard max and min alphas.
        """
        cubelist = [self.cube, self.cube]
        result = self.plugin.normalise_cube(cubelist)
        expected = np.array([[0.1, 0.5, 1.0],
                             [0.3, 0.4, 0.7],
                             [0.0, 0.2, 0.1]])
        self.assertArrayEqual(result[0].data, expected)
        self.assertArrayEqual(result[1].data, expected)

    def test_maxmin(self):
        """
        Tests the function of the normalise cube, using a max
        and min value for alpha.
        """
        cubelist = [self.cube, self.cube]
        result = self.plugin.normalise_cube(cubelist, 3, 5)
        expected = np.array([[3.2, 4.0, 5.0],
                             [3.6, 3.8, 4.4],
                             [3.0, 3.4, 3.2]])
        self.assertArrayEqual(result[0].data, expected)
        self.assertArrayEqual(result[1].data, expected)

    def test_gradient(self):
        """ Tests that it correctly calculates the gradient"""
        result = self.plugin.difference_to_gradient(self.cube, self.cube)
        expected = np.array([[0.5, 2.5, 5.0],
                            [1.5, 2., 3.5],
                            [0., 1.0, 0.5]])
        self.assertArrayEqual(result[0].data, expected)
        self.assertArrayEqual(result[1].data, expected)

    def test_process(self):
        """
        Tests that the final processing step gets the
        right values.
        """
        result = self.plugin.process(self.cube)
        expected_x = np.array([[0.53333333, 0.4, 0.26666667],
                              [1.0, 0.73333333, 0.46666667],
                              [0.66666667, 0.8, 0.93333333]])
        expected_y = np.array([[0.8, 0.93333333, 0.8],
                              [0.66666667, 0.8, 0.4],
                              [0.53333333, 0.66666667, 0.]])
        self.assertArrayAlmostEqual(result[0].data, expected_x)
        self.assertArrayAlmostEqual(result[1].data, expected_y)


if __name__ == '__main__':
    unittest.main()
