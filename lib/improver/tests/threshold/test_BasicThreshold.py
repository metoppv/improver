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
"""Unit tests for the threshold.BasicThreshold plugin."""


import unittest

import numpy as np
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests import IrisTest

from improver.threshold import BasicThreshold as Threshold


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_single_threshold(self):
        """Test that the __repr__ returns the expected string."""
        threshold = [0.6]
        fuzzy_bounds = [(0.6, 0.6)]
        below_thresh_ok = False
        result = str(Threshold(threshold,
                               below_thresh_ok=below_thresh_ok))
        msg = ('<BasicThreshold: thresholds {}, '
               'fuzzy_bounds {}, '
               'below_thresh_ok: {}>'.format(
                   threshold, fuzzy_bounds, below_thresh_ok))
        self.assertEqual(result, msg)

    def test_multiple_thresholds(self):
        """Test that the __repr__ returns the expected string."""
        threshold = [0.6, 0.8]
        fuzzy_bounds = [(0.6, 0.6), (0.8, 0.8)]
        below_thresh_ok = False
        result = str(Threshold(threshold,
                               below_thresh_ok=below_thresh_ok))
        msg = ('<BasicThreshold: thresholds {}, '
               'fuzzy_bounds {}, '
               'below_thresh_ok: {}>'.format(
                   threshold, fuzzy_bounds, below_thresh_ok))
        self.assertEqual(result, msg)

    def test_below_fuzzy_threshold(self):
        """Test that the __repr__ returns the expected string."""
        threshold = 0.6
        fuzzy_factor = 0.2
        fuzzy_bounds = [(0.12, 1.08)]
        below_thresh_ok = True
        result = str(Threshold(threshold,
                               fuzzy_factor=fuzzy_factor,
                               below_thresh_ok=below_thresh_ok))
        msg = ('<BasicThreshold: thresholds [{}], '
               'fuzzy_bounds {}, '
               'below_thresh_ok: {}>'.format(
                   threshold, fuzzy_bounds, below_thresh_ok))
        self.assertEqual(result, msg)

    def test_fuzzy_bounds_scalar(self):
        """Test that the __repr__ returns the expected string."""
        threshold = 0.6
        fuzzy_bounds = (0.4, 0.8)
        below_thresh_ok = False
        result = str(Threshold(threshold,
                               fuzzy_bounds=fuzzy_bounds,
                               below_thresh_ok=below_thresh_ok))
        msg = ('<BasicThreshold: thresholds [{}], '
               'fuzzy_bounds [{}], '
               'below_thresh_ok: {}>'.format(
                   threshold, fuzzy_bounds, below_thresh_ok))
        self.assertEqual(result, msg)

    def test_fuzzy_bounds_list(self):
        """Test that the __repr__ returns the expected string."""
        threshold = [0.6, 2.0]
        fuzzy_bounds = [(0.4, 0.8), (1.8, 2.1)]
        below_thresh_ok = False
        result = str(Threshold(threshold,
                               fuzzy_bounds=fuzzy_bounds,
                               below_thresh_ok=below_thresh_ok))
        msg = ('<BasicThreshold: thresholds {}, '
               'fuzzy_bounds {}, '
               'below_thresh_ok: {}>'.format(
                   threshold, fuzzy_bounds, below_thresh_ok))
        self.assertEqual(result, msg)


class Test_process(IrisTest):

    """Test the thresholding plugin."""

    def setUp(self):
        """Create a cube with a single non-zero point."""
        self.fuzzy_factor = 0.5
        data = np.zeros((1, 5, 5))
        data[0][2][2] = 0.5  # ~2 mm/hr
        cube = Cube(data, standard_name="precipitation_amount",
                    units="kg m^-2 s^-1")
        cube.add_dim_coord(DimCoord(np.linspace(-45.0, 45.0, 5), 'latitude',
                                    units='degrees'), 1)
        cube.add_dim_coord(DimCoord(np.linspace(120, 180, 5), 'longitude',
                                    units='degrees'), 2)
        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube.add_dim_coord(DimCoord([402192.5],
                                    "time", units=tunit), 0)
        self.cube = cube

    def test_basic(self):
        """Test that the plugin returns an iris.cube.Cube."""
        fuzzy_factor = 0.95
        threshold = 0.1
        plugin = Threshold(threshold, fuzzy_factor=fuzzy_factor)
        result = plugin.process(self.cube)
        self.assertIsInstance(result, Cube)

    def test_metadata_changes(self):
        """Test the metadata altering functionality"""
        # Copy the cube as the cube.data is used as the basis for comparison.
        cube = self.cube.copy()
        plugin = Threshold(0.1)
        result = plugin.process(cube)
        # The single 0.5-valued point => 1.0, so cheat by * 2.0 vs orig data.
        name = "probability_of_{}"
        expected_name = name.format(self.cube.name())
        expected_attribute = "above"
        expected_units = 1
        expected_coord = DimCoord(0.1,
                                  long_name='threshold',
                                  units=self.cube.units)
        self.assertEqual(result.name(), expected_name)
        self.assertEqual(result.attributes['relative_to_threshold'],
                         expected_attribute)
        self.assertEqual(result.units, expected_units)
        self.assertEqual(result.coord('threshold'),
                         expected_coord)

    def test_threshold_dimension_added(self):
        """Test that a threshold dimension coordinate is added."""
        plugin = Threshold(0.1)
        result = plugin.process(self.cube)
        expected_coord = DimCoord([0.1], long_name='threshold',
                                  units=self.cube.units)
        self.assertEqual(result.coord('threshold'), expected_coord)

    def test_threshold(self):
        """Test the basic threshold functionality."""
        # Copy the cube as the cube.data is used as the basis for comparison.
        cube = self.cube.copy()
        fuzzy_factor = 0.95
        plugin = Threshold(0.1, fuzzy_factor=fuzzy_factor)
        result = plugin.process(cube)
        # The single 0.5-valued point => 1.0, so cheat by * 2.0 vs orig data.
        expected_result_array = (self.cube.data * 2.0).reshape(1, 1, 5, 5)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_above_threshold_without_fuzzy_factor(self):
        """Test if the fixed threshold is below the value in the data."""
        # Copy the cube as the cube.data is used as the basis for comparison.
        cube = self.cube.copy()
        plugin = Threshold(0.1)
        result = plugin.process(cube)
        expected_result_array = self.cube.data.reshape(1, 1, 5, 5)
        expected_result_array[0][0][2][2] = 1.0
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_below_threshold_without_fuzzy_factor(self):
        """Test if the fixed threshold is above the value in the data."""
        plugin = Threshold(0.6)
        result = plugin.process(self.cube)
        expected_result_array = np.zeros_like(self.cube.data).reshape(
            1, 1, 5, 5)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_fuzzy(self):
        """Test when a point is in the fuzzy threshold area."""
        plugin = Threshold(0.6, fuzzy_factor=self.fuzzy_factor)
        result = plugin.process(self.cube)
        expected_result_array = np.zeros_like(self.cube.data).reshape(
            1, 1, 5, 5)
        expected_result_array[0][0][2][2] = 1.0/3.0
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_fuzzybounds(self):
        """Test when a point is in the fuzzy threshold area."""
        bounds = (0.6 * self.fuzzy_factor, 0.6 * (2. - self.fuzzy_factor))
        plugin = Threshold(0.6, fuzzy_bounds=bounds)
        result = plugin.process(self.cube)
        expected_result_array = np.zeros_like(self.cube.data).reshape(
            1, 1, 5, 5)
        expected_result_array[0][0][2][2] = 1.0/3.0
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_boundingzero(self):
        """Test fuzzy threshold of zero."""
        bounds = (-1.0, 1.0)
        plugin = Threshold(0.0, fuzzy_bounds=bounds)
        result = plugin.process(self.cube)
        expected_result_array = np.full_like(
            self.cube.data, fill_value=0.5).reshape(1, 1, 5, 5)
        expected_result_array[0][0][2][2] = 0.75
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_boundingzero_above(self):
        """Test fuzzy threshold of zero."""
        bounds = (-0.1, 0.1)
        plugin = Threshold(0.0, fuzzy_bounds=bounds)
        result = plugin.process(self.cube)
        expected_result_array = np.full_like(
            self.cube.data, fill_value=0.5).reshape(1, 1, 5, 5)
        expected_result_array[0][0][2][2] = 1.
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_boundingbelowzero(self):
        """Test fuzzy threshold of below-zero."""
        bounds = (-1.0, 1.0)
        plugin = Threshold(0.0, fuzzy_bounds=bounds, below_thresh_ok=True)
        result = plugin.process(self.cube)
        expected_result_array = np.full_like(
            self.cube.data, fill_value=0.5).reshape(1, 1, 5, 5)
        expected_result_array[0][0][2][2] = 0.25
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_assymetric_bounds_below(self):
        """Test when a point is below assymmetric fuzzy threshold area."""
        bounds = (0.51, 0.9)
        plugin = Threshold(0.6, fuzzy_bounds=bounds)
        result = plugin.process(self.cube)
        expected_result_array = np.zeros_like(self.cube.data).reshape(
            1, 1, 5, 5)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_assymetric_bounds_lower(self):
        """Test when a point is in lower assymmetric fuzzy threshold area."""
        bounds = (0.4, 0.9)
        plugin = Threshold(0.6, fuzzy_bounds=bounds)
        result = plugin.process(self.cube)
        expected_result_array = np.zeros_like(self.cube.data).reshape(
            1, 1, 5, 5)
        expected_result_array[0][0][2][2] = 0.25
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_assymetric_bounds_middle(self):
        """Test when a point is on the threshold with assymmetric fuzzy
        bounds."""
        bounds = (0.4, 0.9)
        plugin = Threshold(0.5, fuzzy_bounds=bounds)
        result = plugin.process(self.cube)
        expected_result_array = np.zeros_like(self.cube.data).reshape(
            1, 1, 5, 5)
        expected_result_array[0][0][2][2] = 0.5
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_assymetric_bounds_upper(self):
        """Test when a point is in upper assymmetric fuzzy threshold area."""
        bounds = (0.0, 0.6)
        plugin = Threshold(0.4, fuzzy_bounds=bounds)
        result = plugin.process(self.cube)
        expected_result_array = np.zeros_like(self.cube.data).reshape(
            1, 1, 5, 5)
        expected_result_array[0][0][2][2] = 0.75
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_assymetric_bounds_above(self):
        """Test when a point is above assymmetric fuzzy threshold area."""
        bounds = (0.0, 0.45)
        plugin = Threshold(0.4, fuzzy_bounds=bounds)
        result = plugin.process(self.cube)
        expected_result_array = np.zeros_like(self.cube.data).reshape(
            1, 1, 5, 5)
        expected_result_array[0][0][2][2] = 1.
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_assymetric_bounds_upper_below(self):
        """Test when a point is in upper assymmetric fuzzy threshold area
        and below-threshold is requested."""
        bounds = (0.0, 0.6)
        plugin = Threshold(0.4, fuzzy_bounds=bounds, below_thresh_ok=True)
        result = plugin.process(self.cube)
        expected_result_array = np.ones_like(self.cube.data).reshape(
            1, 1, 5, 5)
        expected_result_array[0][0][2][2] = 0.25
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_fuzzy_miss(self):
        """Test when a point is not within the fuzzy threshold area."""
        plugin = Threshold(2.0, fuzzy_factor=self.fuzzy_factor)
        result = plugin.process(self.cube)
        expected_result_array = np.zeros_like(self.cube.data).reshape(
            1, 1, 5, 5)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_fuzzy_miss_high_threshold(self):
        """Test when a point is not within the fuzzy high threshold area."""
        plugin = Threshold(3.0, fuzzy_factor=self.fuzzy_factor)
        result = plugin.process(self.cube)
        expected_result_array = np.zeros_like(self.cube.data).reshape(
            1, 1, 5, 5)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_negative(self):
        """Test a point when the threshold is negative."""
        self.cube.data[0][2][2] = -0.75
        plugin = Threshold(
            -1.0, fuzzy_factor=self.fuzzy_factor, below_thresh_ok=True)
        result = plugin.process(self.cube)
        expected_result_array = np.zeros_like(self.cube.data).reshape(
            1, 1, 5, 5)
        expected_result_array[0][0][2][2] = 0.25
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_below(self):
        """Test a point when we are in below-threshold mode."""
        plugin = Threshold(
            0.1, fuzzy_factor=self.fuzzy_factor, below_thresh_ok=True)
        result = plugin.process(self.cube)
        expected_result_array = np.ones_like(self.cube.data).reshape(
            1, 1, 5, 5)
        expected_result_array[0][0][2][2] = 0
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_below_fuzzy(self):
        """Test a point in fuzzy threshold in below-threshold-mode."""
        plugin = Threshold(
            0.6, fuzzy_factor=self.fuzzy_factor, below_thresh_ok=True)
        result = plugin.process(self.cube)
        expected_result_array = np.ones_like(self.cube.data).reshape(
            1, 1, 5, 5)
        expected_result_array[0][0][2][2] = 2.0/3.0
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_multiple_thresholds(self):
        """Test multiple thresholds applied to the cube return a single cube
        with multiple arrays corresponding to each threshold."""
        thresholds = [0.2, 0.4, 0.6]
        plugin = Threshold(thresholds)
        result = plugin.process(self.cube)
        expected_array12 = np.zeros_like(self.cube.data).reshape(1, 1, 5, 5)
        expected_array12[0][0][2][2] = 1.
        expected_array3 = expected_array12 * 0.
        expected_result_array = np.vstack([expected_array12,
                                           expected_array12,
                                           expected_array3])
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_below_fuzzy_miss(self):
        """Test not meeting the threshold in fuzzy below-threshold-mode."""
        plugin = Threshold(
            2.0, fuzzy_factor=self.fuzzy_factor, below_thresh_ok=True)
        result = plugin.process(self.cube)
        expected_result_array = np.ones_like(self.cube.data).reshape(
            1, 1, 5, 5)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_point_nan(self):
        """Test behaviour for a single NaN grid cell."""
        # Need to copy the cube as we're adjusting the data.
        self.cube.data[0][2][2] = np.NAN
        msg = "NaN detected in input cube data"
        plugin = Threshold(
            2.0, fuzzy_factor=self.fuzzy_factor, below_thresh_ok=True)
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.process(self.cube)

    def test_threshold_zero_with_fuzzy_factor(self):
        """Test when a threshold of zero is used with a multiplicative
        fuzzy factor (invalid)."""
        fuzzy_factor = 0.6
        msg = "Invalid threshold with fuzzy factor"
        with self.assertRaisesRegexp(ValueError, msg):
            Threshold(0.0, fuzzy_factor=fuzzy_factor)

    def test_threshold_fuzzy_factor_minus_1(self):
        """Test when a fuzzy factor of minus 1 is given (invalid)."""
        fuzzy_factor = -1.0
        msg = "Invalid fuzzy_factor: must be >0 and <1: -1.0"
        with self.assertRaisesRegexp(ValueError, msg):
            Threshold(0.6, fuzzy_factor=fuzzy_factor)

    def test_threshold_fuzzy_factor_0(self):
        """Test when a fuzzy factor of zero is given (invalid)."""
        fuzzy_factor = 0.0
        msg = "Invalid fuzzy_factor: must be >0 and <1: 0.0"
        with self.assertRaisesRegexp(ValueError, msg):
            Threshold(0.6, fuzzy_factor=fuzzy_factor)

    def test_threshold_fuzzy_factor_1(self):
        """Test when a fuzzy factor of unity is given (invalid)."""
        fuzzy_factor = 1.0
        msg = "Invalid fuzzy_factor: must be >0 and <1: 1.0"
        with self.assertRaisesRegexp(ValueError, msg):
            Threshold(0.6, fuzzy_factor=fuzzy_factor)

    def test_threshold_fuzzy_factor_2(self):
        """Test when a fuzzy factor of 2 is given (invalid)."""
        fuzzy_factor = 2.0
        msg = "Invalid fuzzy_factor: must be >0 and <1: 2.0"
        with self.assertRaisesRegexp(ValueError, msg):
            Threshold(0.6, fuzzy_factor=fuzzy_factor)

    def test_fuzzy_factor_and_fuzzy_bounds(self):
        """Test when fuzzy_factor and fuzzy_bounds both set (ambiguous)."""
        fuzzy_factor = 2.0
        fuzzy_bounds = (0.4, 0.8)
        msg = ("Invalid combination of keywords. Cannot specify "
               "fuzzy_factor and fuzzy_bounds together")
        with self.assertRaisesRegexp(ValueError, msg):
            Threshold(0.6, fuzzy_factor=fuzzy_factor,
                      fuzzy_bounds=fuzzy_bounds)

    def test_invalid_bounds_toofew(self):
        """Test when fuzzy_bounds contains one value (invalid)."""
        threshold = 0.6
        fuzzy_bounds = (0.4, )
        # Regexp matches .* with any string.
        msg = ("Invalid bounds for one threshold: .*. "
               "Expected 2 floats.")
        with self.assertRaisesRegexp(AssertionError, msg):
            Threshold(threshold,
                      fuzzy_bounds=fuzzy_bounds)

    def test_invalid_bounds_toomany(self):
        """Test when fuzzy_bounds contains three values (invalid)."""
        threshold = 0.6
        fuzzy_bounds = (0.4, 0.8, 1.2)
        # Regexp matches .* with any string.
        msg = ("Invalid bounds for one threshold: .*. "
               "Expected 2 floats.")
        with self.assertRaisesRegexp(AssertionError, msg):
            Threshold(threshold,
                      fuzzy_bounds=fuzzy_bounds)

    def test_invalid_upper_bound(self):
        """Test when fuzzy_bounds do not bound threshold (invalid)."""
        threshold = 0.6
        fuzzy_bounds = (0.4, 0.5)
        # Note that back-slashes are necessary to make regexp literal.
        msg = ("Threshold must be within bounds: "
               r"\!\( {} <= {} <= {} \)".format(
                   fuzzy_bounds[0], threshold, fuzzy_bounds[1]))
        with self.assertRaisesRegexp(AssertionError, msg):
            Threshold(threshold,
                      fuzzy_bounds=fuzzy_bounds)

    def test_invalid_lower_bound(self):
        """Test when fuzzy_bounds do not bound threshold (invalid)."""
        threshold = 0.6
        fuzzy_bounds = (0.7, 0.8)
        # Note that back-slashes are necessary to make regexp literal.
        msg = ("Threshold must be within bounds: "
               r"\!\( {} <= {} <= {} \)".format(
                   fuzzy_bounds[0], threshold, fuzzy_bounds[1]))
        with self.assertRaisesRegexp(AssertionError, msg):
            Threshold(threshold,
                      fuzzy_bounds=fuzzy_bounds)


if __name__ == '__main__':
    unittest.main()
