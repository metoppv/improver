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
"""Unit tests for the functions within interpolation.py"""

import unittest

import numpy as np
from iris.tests import IrisTest

from improver.utilities.interpolation import interpolate_missing_data


class Test_interpolate_missing_data(IrisTest):
    """Test the interpolate_missing_data method"""
    def setUp(self):
        """ Set up arrays for testing."""
        self.data = np.array([[1.0, 1.0, 2.0],
                              [1.0, np.nan, 2.0],
                              [1.0, 2.0, 2.0]])
        self.limit_data = np.full((3, 3), 3.)
        self.valid_data = np.full((3, 3), True)

        self.data_for_limit_test = (
            np.array([[10.0, np.nan, np.nan, np.nan, 20.0],
                      [10.0, np.nan, np.nan, np.nan, 20.0],
                      [10.0, np.nan, np.nan, np.nan, 20.0],
                      [10.0, np.nan, np.nan, np.nan, 20.0],
                      [10.0, np.nan, np.nan, np.nan, 20.0]]))
        self.limit_for_limit_test = (
            np.array([[0.0, 30.0, 12.0, 30.0, 0.0],
                      [0.0, 30.0, 12.0, 30.0, 0.0],
                      [0.0, 30.0, 12.0, 30.0, 0.0],
                      [0.0, 30.0, 12.0, 30.0, 0.0],
                      [0.0, 30.0, 12.0, 30.0, 0.0]]))

        self.valid_data_for_limit_test = np.full((5, 5), True)
        self.valid_data_for_limit_test[:, 1:4] = False

    def test_basic_linear(self):
        """Test when all the points around the missing data are the same."""
        data = np.ones((3, 3))
        data[1, 1] = np.nan
        expected = np.array([[1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0]])

        data_updated = interpolate_missing_data(data)

        self.assertArrayEqual(data_updated, expected)

    def test_basic_nearest(self):
        """Test when all the points around the missing data are the same."""
        data = np.ones((3, 3))
        data[1, 1] = np.nan
        expected = np.array([[1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0]])

        data_updated = interpolate_missing_data(data, method='nearest')

        self.assertArrayEqual(data_updated, expected)

    def test_different_data_for_linear_interpolation(self):
        """Test result when linearly interpolating using points around the
        missing data with different values."""
        expected = np.array([[1.0, 1.0, 2.0],
                             [1.0, 1.5, 2.0],
                             [1.0, 2.0, 2.0]])

        data_updated = interpolate_missing_data(self.data)

        self.assertArrayEqual(data_updated, expected)

    def test_different_data_for_nearest_neighbour(self):
        """Test result when using nearest neighbour using points around the
        missing data with different values."""
        expected = np.array([[1.0, 1.0, 2.0],
                             [1.0, 1.0, 2.0],
                             [1.0, 2.0, 2.0]])

        data_updated = interpolate_missing_data(self.data, method='nearest')

        self.assertArrayEqual(data_updated, expected)

    def test_too_few_points_to_linearly_interpolate(self):
        """Test that when there are not enough points to fill the gaps using
        linear interpolation we recover the input data. This occurs if there
        are less than 3 points available to use for the interpolation."""
        data = np.array([[np.nan, 1, np.nan],
                         [np.nan, np.nan, np.nan],
                         [np.nan, 1, np.nan]])

        data_updated = interpolate_missing_data(data.copy())

        self.assertArrayEqual(data_updated, data)

    def test_nearest_neighbour_with_few_points(self):
        """Test that when there are not enough points to fill the gaps using
        linear interpolation (above test), we can still use nearest neighbour
        filling."""
        data = np.array([[np.nan, 1, np.nan],
                         [np.nan, np.nan, np.nan],
                         [np.nan, 1, np.nan]])
        expected = np.ones((3, 3))

        data_updated = interpolate_missing_data(data, method='nearest')

        self.assertArrayEqual(data_updated, expected)

    def test_badly_arranged_valid_data_for_linear_interpolation(self):
        """Test when there are enough points but they aren't arranged in a
        suitable way to allow linear interpolation. This QhullError raised
        in this case is different to the one raised by
        test_too_few_points_to_linearly_interpolate."""
        data = np.array([[np.nan, 1, np.nan],
                         [np.nan, 1, np.nan],
                         [np.nan, 1, np.nan]])

        data_updated = interpolate_missing_data(data.copy())

        self.assertArrayEqual(data_updated, data)

    def test_nearest_neighbour_with_badly_arranged_valid_data(self):
        """Test that when there are enough points but unsuitably arrange to
        fill the gaps using linear interpolation (above test), we can still
        use nearest neighbour filling."""
        data = np.array([[np.nan, 1, np.nan],
                         [np.nan, 1, np.nan],
                         [np.nan, 1, np.nan]])
        expected = np.ones((3, 3))

        data_updated = interpolate_missing_data(data, method='nearest')

        self.assertArrayEqual(data_updated, expected)

    def test_missing_corner_point_linear_interpolation(self):
        """Test when there's an extra missing value at the corner of the grid.
        This point can't be filled in by linear interpolation, and will remain
        unfilled."""
        self.data[2, 2] = np.nan
        expected = np.array([[1.0, 1.0, 2.0],
                             [1.0, 1.5, 2.0],
                             [1.0, 2.0, np.nan]])

        data_updated = interpolate_missing_data(self.data)

        self.assertArrayEqual(data_updated, expected)

    def test_data_maked_as_invalid(self):
        """Test that marking some of the edge data as invalid with a mask
        results in an appropriately changed result."""

        expected = np.array([[1.0, 1.0, 2.0],
                             [1.0, 4./3., 2.0],
                             [1.0, 2.0, 2.0]])

        self.valid_data[2, 1] = False
        self.valid_data[1, 2] = False
        data_updated = interpolate_missing_data(
            self.data, valid_points=self.valid_data)

        self.assertArrayAlmostEqual(data_updated, expected)

    def test_all_data_marked_as_invalid(self):
        """Test that nothing is filled in if none of the data points are marked
        as valid points."""
        expected = np.array([[1.0, 1.0, 2.0],
                             [1.0, np.nan, 2.0],
                             [1.0, 2.0, 2.0]])

        data_updated = interpolate_missing_data(
            self.data, valid_points=~self.valid_data)

        self.assertArrayEqual(data_updated, expected)

    def test_set_to_limit_as_maximum(self):
        """Test that when the linear interpolation gives values that are higher
        than the limit values the returned data is set back to the limit values
        in those positions. This uses the default behaviour where the limit is
        the maximum allowed value."""

        expected = np.array([[10.0, 12.5, 12.0, 17.5, 20.0],
                             [10.0, 12.5, 12.0, 17.5, 20.0],
                             [10.0, 12.5, 12.0, 17.5, 20.0],
                             [10.0, 12.5, 12.0, 17.5, 20.0],
                             [10.0, 12.5, 12.0, 17.5, 20.0]])
        data_updated = interpolate_missing_data(
                self.data_for_limit_test,
                valid_points=self.valid_data_for_limit_test,
                limit=self.limit_for_limit_test)
        self.assertArrayEqual(data_updated, expected)

    def test_set_to_limit_as_minimum(self):
        """Test that when the linear interpolation gives values that are lower
        than the limit values the returned data is set back to the limit values
        in those positions. This tests the option of using the limit values as
        minimums."""

        expected = np.array([[10.0, 30., 15.0, 30., 20.0],
                             [10.0, 30., 15.0, 30., 20.0],
                             [10.0, 30., 15.0, 30., 20.0],
                             [10.0, 30., 15.0, 30., 20.0],
                             [10.0, 30., 15.0, 30., 20.0]])
        data_updated = interpolate_missing_data(
                self.data_for_limit_test,
                valid_points=self.valid_data_for_limit_test,
                limit=self.limit_for_limit_test,
                limit_as_maximum=False)
        self.assertArrayEqual(data_updated, expected)


if __name__ == '__main__':
    unittest.main()
