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
Unit tests for the OrographicSmoothingCoefficients utility.

"""

import unittest

import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests import IrisTest

from improver.utilities.ancillary_creation import (
    OrographicSmoothingCoefficients)
from improver.utilities.spatial import DifferenceBetweenAdjacentGridSquares


def set_up_cube():
    """Set up dummy cube for tests"""
    data = np.array([[1., 5., 10.],
                     [3., 4., 7.],
                     [0., 2., 1.]])
    cube = Cube(data, "precipitation_amount", units="kg m^-2 s^-1")
    cube.add_dim_coord(DimCoord(np.linspace(0.0, 4.0, 3),
                                'projection_y_coordinate',
                                units='m'), 0)
    cube.add_dim_coord(DimCoord(np.linspace(0.0, 4.0, 3),
                                'projection_x_coordinate',
                                units='m'), 1)
    return cube


class Test__init__(IrisTest):
    """Test the init method."""

    def test_basic(self):
        """Test default attribute initialisation"""
        result = OrographicSmoothingCoefficients()
        self.assertEqual(result.min_smoothing_coefficient, 0.)
        self.assertEqual(result.max_smoothing_coefficient, 1.)
        self.assertEqual(result.coefficient, 1.)
        self.assertEqual(result.power, 1.)


class Test__repr__(IrisTest):
    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(OrographicSmoothingCoefficients())
        msg = ('<OrographicSmoothingCoefficients: min_smoothing_coefficient: '
               '{}; max_smoothing_coefficient: {}; coefficient: {}; power: {}'
               '>'.format(
                   0.0, 1.0, 1, 1))
        self.assertEqual(result, msg)


class Test_scale_smoothing_coefficients(IrisTest):
    """Class to test the scale_smoothing_coefficients function"""

    def setUp(self):
        """Set up cube & plugin"""
        self.plugin = OrographicSmoothingCoefficients()
        cube = set_up_cube()
        self.cubelist = [cube, cube]

    def test_basic(self):
        """
        Test the basic function of scale_smoothing_coefficients, using the
        standard max and min smoothing_coefficients.
        """
        result = self.plugin.scale_smoothing_coefficients(self.cubelist)
        expected = np.array([[0.1, 0.5, 1.0],
                             [0.3, 0.4, 0.7],
                             [0.0, 0.2, 0.1]])
        self.assertArrayAlmostEqual(result[0].data, expected)
        self.assertArrayAlmostEqual(result[1].data, expected)

    def test_maxmin(self):
        """
        Tests the function of scale_smoothing_coefficients, using a max
        and min value for smoothing_coefficient.
        """
        result = self.plugin.scale_smoothing_coefficients(
            self.cubelist, 0.3, 0.5)
        expected = np.array([[0.32, 0.40, 0.50],
                             [0.36, 0.38, 0.44],
                             [0.30, 0.34, 0.32]])
        self.assertArrayAlmostEqual(result[0].data, expected)
        self.assertArrayAlmostEqual(result[1].data, expected)


class Test_unnormalised_smoothing_coefficients(IrisTest):
    """Class to test the basic smoothing_coefficients function"""

    def setUp(self):
        """Set up cube & plugin"""
        self.plugin = OrographicSmoothingCoefficients(
            coefficient=0.5, power=2.)
        self.cube = set_up_cube()

    def test_basic(self):
        """Test data are as expected"""
        expected = np.array([[1.53125, 2.53125, 3.78125],
                             [0., 0.5, 2.],
                             [1.53125, 0.03125, 0.78125]])
        gradient_x, _ = \
            DifferenceBetweenAdjacentGridSquares(gradient=True).process(
                self.cube)
        smoothing_coefficient_x = (
            self.plugin.unnormalised_smoothing_coefficients(gradient_x))
        self.assertArrayAlmostEqual(smoothing_coefficient_x.data, expected)


class Test_gradient_to_smoothing_coefficient(IrisTest):

    """Class to test smoothing_coefficients data and metadata output"""

    def setUp(self):
        """Set up cube & plugin"""
        self.plugin = OrographicSmoothingCoefficients(
            min_smoothing_coefficient=0.5, max_smoothing_coefficient=0.3)
        self.cube = set_up_cube()
        self.gradient_x, self.gradient_y = \
            DifferenceBetweenAdjacentGridSquares(gradient=True).process(
                self.cube)

    def test_basic(self):
        """Test basic version of gradient to smoothing_coefficient"""

        expected = np.array([[0.40666667, 0.38, 0.35333333],
                             [0.5, 0.44666667, 0.39333333],
                             [0.40666667, 0.48666667, 0.43333333]])

        result = self.plugin.gradient_to_smoothing_coefficient(
            self.gradient_x, self.gradient_y)
        self.assertEqual(result[0].name(), 'smoothing_coefficient_x')
        self.assertArrayAlmostEqual(result[0].data, expected)
        self.assertNotIn('forecast_period', [coord.name()
                         for coord in result[0].coords()])
        self.assertNotIn('forecast_time', [coord.name()
                         for coord in result[0].coords()])


class Test_process(IrisTest):
    """Class to test end-to-end smoothing_coefficients creation"""

    def setUp(self):
        """Set up cube & plugin"""
        self.plugin = OrographicSmoothingCoefficients(
            min_smoothing_coefficient=1., max_smoothing_coefficient=0.)
        self.cube = set_up_cube()

    def test_basic(self):
        """Tests that the final processing step gets the right values."""
        result = self.plugin.process(self.cube)

        expected_x = np.array([[0.53333333, 0.4, 0.26666667],
                               [1., 0.73333333, 0.46666667],
                               [0.53333333, 0.93333333, 0.66666667]])

        expected_y = np.array([[0.4, 0.93333333, 0.8],
                               [0.93333333, 0.8, 0.4],
                               [0.26666667, 0.66666667, 0.]])

        self.assertArrayAlmostEqual(result[0].data, expected_x)
        self.assertArrayAlmostEqual(result[1].data, expected_y)

    def test_list_error(self):
        """Test that a list of orography input cubes raises a value error"""
        with self.assertRaises(ValueError):
            self.plugin.process([self.cube, self.cube])


if __name__ == '__main__':
    unittest.main()
