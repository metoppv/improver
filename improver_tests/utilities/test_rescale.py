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
"""Unit tests for the rescale function from rescale.py."""

import unittest
from datetime import datetime

import numpy as np
from iris.tests import IrisTest

from improver.utilities.rescale import apply_double_scaling, rescale

from ..set_up_test_cubes import set_up_variable_cube


class Test_rescale(IrisTest):

    """Test the utilities.rescale rescale function."""

    def setUp(self):
        """Create a cube of ones with a single zero point."""
        self.cube = set_up_variable_cube(
            np.ones((1, 16, 16), dtype=np.float32))
        self.cube.data[0, 7, 7] = 0.

    def test_basic(self):
        """Test that the method returns the expected array type"""
        result = rescale(self.cube.data)
        self.assertIsInstance(result, np.ndarray)

    def test_zero_range_input(self):
        """Test that the method returns the expected error"""
        msg = "Cannot rescale a zero input range"
        with self.assertRaisesRegex(ValueError, msg):
            rescale(self.cube.data, data_range=[0, 0])

    def test_zero_range_output(self):
        """Test that the method returns the expected error"""
        msg = "Cannot rescale a zero output range"
        with self.assertRaisesRegex(ValueError, msg):
            rescale(self.cube.data, scale_range=[4, 4])

    def test_rescaling_inrange(self):
        """Test that the method returns the expected values when in range"""
        expected = self.cube.data.copy()
        expected[...] = 110.
        expected[0, 7, 7] = 100.
        result = rescale(self.cube.data, data_range=(0., 1.),
                         scale_range=(100., 110.))
        self.assertArrayAlmostEqual(result, expected)

    def test_rescaling_outrange(self):
        """Test that the method gives the expected values when out of range"""
        expected = self.cube.data.copy()
        expected[...] = 108.
        expected[0, 7, 7] = 98.
        result = rescale(self.cube.data, data_range=(0.2, 1.2),
                         scale_range=(100., 110.))
        self.assertArrayAlmostEqual(result, expected)

    def test_clip(self):
        """Test that the method clips values when out of range"""
        expected = self.cube.data.copy()
        expected[...] = 108.
        expected[0, 7, 7] = 100.
        result = rescale(self.cube.data, data_range=(0.2, 1.2),
                         scale_range=(100., 110.), clip=True)
        self.assertArrayAlmostEqual(result, expected)


class Test_apply_double_scaling(IrisTest):

    """Test the apply_double_scaling method."""

    def setUp(self):
        """Create cubes with a single zero prob(precip) point.
        The cubes look like this:
        precipitation_amount / (kg m^-2)
        Dimension coordinates:
            projection_y_coordinate: 4;
            projection_x_coordinate: 4;
        Scalar coordinates:
            time: 2015-11-23 03:00:00
            forecast_reference_time: 2015-11-23 03:00:00
            forecast_period (on time coord): 0.0 hours
        Data:
            self.cube_a:
                All points contain float(1.)
            self.cube_b:
                All points contain float(1.)
        """
        self.cube_a = set_up_variable_cube(
            np.ones((4, 4), dtype=np.float32),
            time=datetime(2015, 11, 23, 3, 0),
            frt=datetime(2015, 11, 23, 3, 0))

        self.cube_b = set_up_variable_cube(
            np.ones((4, 4), dtype=np.float32),
            time=datetime(2015, 11, 23, 3, 0),
            frt=datetime(2015, 11, 23, 3, 0))

        self.thr_a = (0.1, 0.5, 0.8)
        self.thr_b = (0.0, 0.5, 0.9)

    def test_basic(self):
        """Test that the method returns the expected cube type"""
        result = apply_double_scaling(self.cube_a,
                                      self.cube_b,
                                      self.thr_a,
                                      self.thr_b)
        self.assertIsInstance(result, np.ndarray)

    def test_input(self):
        """Test that the method does not modify the input cubes."""
        cube_a = self.cube_a.copy()
        cube_b = self.cube_b.copy()
        apply_double_scaling(self.cube_a,
                             self.cube_b,
                             self.thr_a,
                             self.thr_b)
        self.assertArrayAlmostEqual(cube_a.data, self.cube_a.data)
        self.assertArrayAlmostEqual(cube_b.data, self.cube_b.data)

    def test_values_default(self):
        """Test that the method returns the expected data values with default
        minimum function"""
        # Create an array of correct shape and fill with expected value
        expected = np.full_like(self.cube_a.data, 0.9)
        # Row zero should be changed to all-zeroes
        expected[0, :] = [0., 0., 0., 0.]
        # Row one should be like cube_a but with most values reduced to 0.5
        expected[1, :] = [0.0, 0.4, 0.5, 0.5]
        # Row two should be like cube_a but with late values limited to 0.9
        expected[2, :] = [0.0, 0.4, 0.8, 0.9]
        self.cube_a.data[0, :] = [0., 0., 0., 0.]
        self.cube_a.data[1, :] = [0.5, 0.5, 0.5, 0.5]
        self.cube_a.data[2, :] = [1., 1., 1., 1.]
        self.cube_b.data[0, :] = np.arange(0., 1.6, 0.4)
        self.cube_b.data[1, :] = np.arange(0., 1.6, 0.4)
        self.cube_b.data[2, :] = np.arange(0., 1.6, 0.4)
        result = apply_double_scaling(self.cube_a,
                                      self.cube_b,
                                      self.thr_a,
                                      self.thr_b)
        self.assertArrayAlmostEqual(result, expected)

    def test_values_max(self):
        """Test that the method returns the expected data values with max
        function"""
        expected = self.cube_a.data.copy()
        # Row zero should be unchanged from ltng_cube
        expected[0, :] = np.arange(0., 1.6, 0.4)
        # Row one should be like cube_a but with early values raised to 0.5
        expected[1, :] = [0.5, 0.5, 0.8, 1.2]
        # Row two should be like cube_a but with most values raised to 0.9
        expected[2, :] = [0.9, 0.9, 0.9, 1.2]
        self.cube_a.data[0, :] = [0., 0., 0., 0.]
        self.cube_a.data[1, :] = [0.5, 0.5, 0.5, 0.5]
        self.cube_a.data[2, :] = [1., 1., 1., 1.]
        self.cube_b.data[0, :] = np.arange(0., 1.6, 0.4)
        self.cube_b.data[1, :] = np.arange(0., 1.6, 0.4)
        self.cube_b.data[2, :] = np.arange(0., 1.6, 0.4)
        result = apply_double_scaling(self.cube_a,
                                      self.cube_b,
                                      self.thr_a,
                                      self.thr_b,
                                      combine_function=np.maximum)
        self.assertArrayAlmostEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
