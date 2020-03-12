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
"""Unit tests for the nbhood.NeighbourhoodProcessing plugin."""


import unittest

import numpy as np
from iris.cube import Cube
from iris.tests import IrisTest

from improver.nbhood.nbhood import NeighbourhoodProcessing as NBHood

from .test_BaseNeighbourhoodProcessing import set_up_cube


class Test__init__(IrisTest):

    """Test the __init__ method of NeighbourhoodProcessing."""

    def test_neighbourhood_method_exists(self):
        """Test that no exception is raised if the requested neighbourhood
         method exists."""
        neighbourhood_method = 'circular'
        radii = 10000
        result = NBHood(neighbourhood_method, radii)
        msg = ('<CircularNeighbourhood: weighted_mode: True, '
               'sum_or_fraction: fraction>')
        self.assertEqual(str(result.neighbourhood_method), msg)

    def test_neighbourhood_method_does_not_exist(self):
        """Test that desired error message is raised, if the neighbourhood
        method does not exist."""
        neighbourhood_method = 'nonsense'
        radii = 10000
        msg = 'The neighbourhood_method requested: '
        with self.assertRaisesRegex(KeyError, msg):
            NBHood(neighbourhood_method, radii)


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(NBHood("circular", 10000))
        msg = ('<BaseNeighbourhoodProcessing: neighbourhood_method: '
               '<CircularNeighbourhood: weighted_mode: True, '
               'sum_or_fraction: fraction>; '
               'radii: 10000.0; lead_times: None>')
        self.assertEqual(result, msg)


class Test_process(IrisTest):

    """Test the process method."""

    def setUp(self):
        """Set up a cube."""
        self.cube = set_up_cube(
            zero_point_indices=((0, 0, 2, 2),), num_grid_points=5)

    def test_weighted_mode_is_true(self):
        """Test that the circular neighbourhood processing is successful, if
        the weighted mode is True."""
        expected = np.array(
            [[[[1., 1., 1., 1., 1.],
               [1., 0.91666667, 0.875, 0.91666667, 1.],
               [1., 0.875, 0.83333333, 0.875, 1.],
               [1., 0.91666667, 0.875, 0.91666667, 1.],
               [1., 1., 1., 1., 1.]]]])
        neighbourhood_method = 'circular'
        radii = 4000
        result = NBHood(neighbourhood_method, radii)(self.cube)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_weighted_mode_is_false(self):
        """Test that the circular neighbourhood processing is successful, if
        the weighted mode is False."""
        expected = np.array(
            [[[[1., 1., 0.92307692, 1., 1.],
               [1., 0.92307692, 0.92307692, 0.92307692, 1.],
               [0.92307692, 0.92307692, 0.92307692, 0.92307692, 0.92307692],
               [1., 0.92307692, 0.92307692, 0.92307692, 1.],
               [1., 1., 0.92307692, 1., 1.]]]])
        neighbourhood_method = 'circular'
        radii = 4000
        result = NBHood(neighbourhood_method, radii,
                        weighted_mode=False)(self.cube)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_square_neighbourhood(self):
        """Test that the square neighbourhood processing is successful."""
        expected = np.array(
            [[[[1., 1., 1., 1., 1.],
               [1., 0.88888889, 0.88888889, 0.88888889, 1.],
               [1., 0.88888889, 0.88888889, 0.88888889, 1.],
               [1., 0.88888889, 0.88888889, 0.88888889, 1.],
               [1., 1., 1., 1., 1.]]]])
        neighbourhood_method = 'square'
        radii = 2000
        result = NBHood(neighbourhood_method, radii)(self.cube)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, expected)


if __name__ == '__main__':
    unittest.main()
