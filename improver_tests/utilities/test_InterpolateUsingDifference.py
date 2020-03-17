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
"""Unit tests for the InterpolateUsingDifference plugin."""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from improver.utilities.interpolation import InterpolateUsingDifference
from ..set_up_test_cubes import set_up_variable_cube


class Test_repr(unittest.TestCase):

    """Test the InterpolateUsingDifference __repr__ method."""

    def test_basic(self):
        """Test expected string representation is returned."""
        self.assertEqual(str(InterpolateUsingDifference()),
                         "<InterpolateUsingDifference>")


class Test_process(unittest.TestCase):

    """Test the InterpolateUsingDifference process method."""

    def setUp(self):
        """ Set up arrays for testing."""
        snow_sleet = np.array([[5.0, 5.0, 5.0],
                               [10., 10., 10.],
                               [5.0, 5.0, 5.0]], dtype=np.float32)
        sleet_rain = np.array([[4.0, 4.0, 4.0],
                               [np.nan, np.nan, np.nan],
                               [3.0, 3.0, 3.0]], dtype=np.float32)
        sleet_rain = np.ma.masked_invalid(sleet_rain)
        limit_data = np.array([[4.0, 4.0, 4.0],
                               [10., 8., 6.],
                               [4.0, 4.0, 4.0]], dtype=np.float32)

        self.snow_sleet = set_up_variable_cube(
            snow_sleet, name="altitude_of_snow_falling_level", units='m',
            spatial_grid='equalarea')
        self.sleet_rain = set_up_variable_cube(
            sleet_rain, name="altitude_of_rain_falling_level", units='m',
            spatial_grid='equalarea')
        self.limit = set_up_variable_cube(
            limit_data, name="surface_altitude", units='m',
            spatial_grid='equalarea')

    def test_unlimited(self):
        """Test interpolation to complete an incomplete field using a reference
        field. No limit is imposed upon the returned interpolated values."""

        expected = np.array([[4.0, 4.0, 4.0],
                             [8.5, 8.5, 8.5],
                             [3.0, 3.0, 3.0]], dtype=np.float32)

        result = InterpolateUsingDifference().process(self.sleet_rain,
                                                      self.snow_sleet)

        assert_array_equal(result.data, expected)
        self.assertEqual(result.coords(), self.sleet_rain.coords())
        self.assertEqual(result.metadata, self.sleet_rain.metadata)

    def test_maximum_limited(self):
        """Test interpolation to complete an incomplete field using a reference
        field. A limit is imposed upon the returned interpolated values,
        forcing these values to the maximum limit if they exceed it."""

        expected = np.array([[4.0, 4.0, 4.0],
                             [8.5, 8.0, 6.0],
                             [3.0, 3.0, 3.0]], dtype=np.float32)

        result = InterpolateUsingDifference().process(
            self.sleet_rain, self.snow_sleet, limit=self.limit,
            limit_as_maximum=True)

        assert_array_equal(result.data, expected)
        self.assertEqual(result.coords(), self.sleet_rain.coords())
        self.assertEqual(result.metadata, self.sleet_rain.metadata)

    def test_minimum_limited(self):
        """Test interpolation to complete an incomplete field using a reference
        field. A limit is imposed upon the returned interpolated values,
        forcing these values to the minimum limit if they are below it."""

        expected = np.array([[4.0, 4.0, 4.0],
                             [10., 8.5, 8.5],
                             [3.0, 3.0, 3.0]], dtype=np.float32)

        result = InterpolateUsingDifference().process(
            self.sleet_rain, self.snow_sleet, limit=self.limit,
            limit_as_maximum=False)

        assert_array_equal(result.data, expected)
        self.assertEqual(result.coords(), self.sleet_rain.coords())
        self.assertEqual(result.metadata, self.sleet_rain.metadata)

    def test_incomplete_reference(self):
        """Test an exception is raised if the reference field is incomplete."""

        self.snow_sleet.data[1, 1] = np.nan
        msg = "The reference field contains np.nan data"
        with self.assertRaisesRegex(ValueError, msg):
            InterpolateUsingDifference().process(
                self.sleet_rain, self.snow_sleet, limit=self.limit,
                limit_as_maximum=False)

    def test_crossing_values(self):
        """Test interpolation when the reference field and field to be
        completed by interpolation cross one another. In the absence of any
        limit it should be possible to return an interpolated field of values
        that pass through the reference field in an expected way. In another
        case we apply the reference field as a lower bound to the interpolated
        values."""

        snow_sleet = np.array([[15., 15., 15.],
                               [10., 10., 10.],
                               [8.0, 8.0, 8.0]], dtype=np.float32)

        sleet_rain = np.array([[5.0, 5.0, 5.0],
                               [np.nan, np.nan, np.nan],
                               [15., 15., 15.]], dtype=np.float32)
        sleet_rain = np.ma.masked_invalid(sleet_rain)

        self.snow_sleet.data = snow_sleet
        self.sleet_rain.data = sleet_rain

        expected_unlimited = np.array([[5.0, 5.0, 5.0],
                                       [8.5, 8.5, 8.5],
                                       [15., 15., 15.]], dtype=np.float32)
        expected_limited = np.array([[5.0, 5.0, 5.0],
                                     [10., 10., 10.],
                                     [15., 15., 15.]], dtype=np.float32)

        result_unlimited = InterpolateUsingDifference().process(
            self.sleet_rain, self.snow_sleet)

        result_limited = InterpolateUsingDifference().process(
            self.sleet_rain, self.snow_sleet, limit=self.snow_sleet,
            limit_as_maximum=False)

        assert_array_equal(result_unlimited.data, expected_unlimited)
        assert_array_equal(result_limited.data, expected_limited)


if __name__ == '__main__':
    unittest.main()
