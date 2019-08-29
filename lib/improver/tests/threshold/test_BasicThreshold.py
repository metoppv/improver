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
"""Unit tests for the threshold.BasicThreshold plugin."""


import unittest

import numpy as np
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests import IrisTest

from improver.tests.set_up_test_cubes import set_up_variable_cube
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


class Test__add_threshold_coord(IrisTest):
    """Test the _add_threshold_coord method"""

    def setUp(self):
        """Set up a cube and plugin for testing."""
        self.cube = set_up_variable_cube(np.ones((3, 3), dtype=np.float32))
        self.plugin = Threshold([1])
        self.plugin.threshold_coord_name = self.cube.name()

    def test_basic(self):
        """Test a scalar threshold coordinate is created"""
        result = self.plugin._add_threshold_coord(self.cube, 1)
        self.assertEqual(result.ndim, 3)
        self.assertIn("air_temperature", [coord.standard_name for coord in
                                          result.coords(dim_coords=True)])
        threshold_coord = result.coord("air_temperature")
        self.assertEqual(threshold_coord.var_name, "threshold")
        self.assertEqual(threshold_coord.attributes,
                         {"spp__relative_to_threshold": "above"})
        self.assertAlmostEqual(threshold_coord.points[0], 1)
        self.assertEqual(threshold_coord.units, self.cube.units)

    def test_long_name(self):
        """Test coordinate is created with non-standard diagnostic name"""
        self.cube.rename("sky_temperature")
        self.plugin.threshold_coord_name = self.cube.name()
        result = self.plugin._add_threshold_coord(self.cube, 1)
        self.assertIn("sky_temperature", [coord.long_name for coord in
                                          result.coords(dim_coords=True)])

    def test_value_error(self):
        """Test method catches ValueErrors unrelated to name, by passing it a
        list of values where a scalar is required"""
        with self.assertRaises(ValueError):
            self.plugin._add_threshold_coord(self.cube, [1, 1])


class Test_process(IrisTest):

    """Test the thresholding plugin."""

    def setUp(self):
        """Create a cube with a single non-zero point."""

        latitude = DimCoord(np.linspace(-45.0, 45.0, 5), 'latitude',
                            units='degrees')
        longitude = DimCoord(np.linspace(120, 180, 5), 'longitude',
                             units='degrees')

        self.fuzzy_factor = 0.5
        data = np.zeros((1, 5, 5))
        data[0][2][2] = 0.5  # ~2 mm/hr
        cube = Cube(data, standard_name="precipitation_amount",
                    units="kg m^-2 s^-1")
        cube.add_dim_coord(latitude, 1)
        cube.add_dim_coord(longitude, 2)
        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        cube.add_dim_coord(DimCoord([402192.5],
                                    "time", units=tunit), 0)
        self.cube = cube

        # cube to test unit conversion
        rate_data = np.zeros((5, 5))
        rate_data[2][2] = 1.39e-6  # 5.004 mm/hr
        rate_cube = Cube(rate_data, 'rainfall_rate', units='m s-1',
                         dim_coords_and_dims=[(latitude, 0), (longitude, 1)])
        self.rate_cube = rate_cube

    def test_basic(self):
        """Test that the plugin returns an iris.cube.Cube."""
        fuzzy_factor = 0.95
        threshold = 0.1
        plugin = Threshold(threshold, fuzzy_factor=fuzzy_factor)
        result = plugin.process(self.cube)
        self.assertIsInstance(result, Cube)

    def test_data_precision_preservation(self):
        """Test that the plugin returns an iris.cube.Cube of the same float
        precision as the input cube."""
        threshold = 0.1
        plugin = Threshold(threshold, fuzzy_factor=self.fuzzy_factor)
        f64cube = self.cube.copy(data=self.cube.data.astype(np.float64))
        f32cube = self.cube.copy(data=self.cube.data.astype(np.float32))
        f64result = plugin.process(f64cube)
        f32result = plugin.process(f32cube)
        self.assertEqual(f64cube.dtype, f64result.dtype)
        self.assertEqual(f32cube.dtype, f32result.dtype)

    def test_data_type_change_for_ints(self):
        """Test that the plugin returns an iris.cube.Cube of float32 type
        if the input cube is of int type. This allows fuzzy bounds to be used
        which return fractional values."""
        fuzzy_factor = 5./6.
        threshold = 12
        self.cube.data = np.arange(25).reshape(1, 5, 5)
        plugin = Threshold(threshold, fuzzy_factor=fuzzy_factor)
        result = plugin.process(self.cube)
        expected = np.round(np.arange(0, 1, 1./25.)).reshape(1, 1, 5, 5)
        expected[0, 0, 2, 1:4] = [0.25, 0.5, 0.75]
        self.assertEqual(result.dtype, 'float32')
        self.assertArrayEqual(result.data, expected)

    def test_metadata_changes(self):
        """Test the metadata altering functionality"""
        # Copy the cube as the cube.data is used as the basis for comparison.
        cube = self.cube.copy()
        plugin = Threshold(0.1)
        result = plugin.process(cube)
        # The single 0.5-valued point => 1.0, so cheat by * 2.0 vs orig data.
        name = "probability_of_{}_above_threshold"
        expected_name = name.format(self.cube.name())
        expected_attribute = "above"
        expected_units = 1
        expected_coord = DimCoord(np.array([0.1], dtype=np.float32),
                                  standard_name=self.cube.name(),
                                  var_name='threshold',
                                  units=self.cube.units,
                                  attributes={"spp__relative_to_threshold":
                                              "above"})
        self.assertEqual(result.name(), expected_name)
        self.assertEqual(
            result.coord(var_name="threshold"
                         ).attributes['spp__relative_to_threshold'],
            expected_attribute)
        self.assertEqual(result.units, expected_units)
        self.assertEqual(result.coord(self.cube.name()), expected_coord)

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

    def test_masked_array(self):
        """Test masked array are handled correctly.
        Masked values are preserverd following thresholding."""
        cube = self.cube.copy()
        data = np.zeros((1, 5, 5))
        mask = np.zeros((1, 5, 5))
        data[0][2][2] = 0.5
        data[0][0][0] = -32768.0
        mask[0][0][0] = 1
        masked_data = np.ma.MaskedArray(data, mask=mask)
        cube.data = masked_data
        plugin = Threshold(0.1)
        result = plugin.process(cube)
        expected_result_array = data.reshape(1, 1, 5, 5)
        expected_result_array[0][0][2][2] = 1.0
        self.assertArrayAlmostEqual(result.data.data, expected_result_array)
        self.assertArrayEqual(result.data.mask, mask.reshape(1, 1, 5, 5))

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
        """Test fuzzy threshold of zero where data are above upper-bound."""
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

    def test_threshold_asymmetric_bounds_below(self):
        """Test when a point is below asymmetric fuzzy threshold area."""
        bounds = (0.51, 0.9)
        plugin = Threshold(0.6, fuzzy_bounds=bounds)
        result = plugin.process(self.cube)
        expected_result_array = np.zeros_like(self.cube.data).reshape(
            1, 1, 5, 5)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_asymmetric_bounds_lower(self):
        """Test when a point is in lower asymmetric fuzzy threshold area."""
        bounds = (0.4, 0.9)
        plugin = Threshold(0.6, fuzzy_bounds=bounds)
        result = plugin.process(self.cube)
        expected_result_array = np.zeros_like(self.cube.data).reshape(
            1, 1, 5, 5)
        expected_result_array[0][0][2][2] = 0.25
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_asymmetric_bounds_middle(self):
        """Test when a point is on the threshold with asymmetric fuzzy
        bounds."""
        bounds = (0.4, 0.9)
        plugin = Threshold(0.5, fuzzy_bounds=bounds)
        result = plugin.process(self.cube)
        expected_result_array = np.zeros_like(self.cube.data).reshape(
            1, 1, 5, 5)
        expected_result_array[0][0][2][2] = 0.5
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_asymmetric_bounds_upper(self):
        """Test when a point is in upper asymmetric fuzzy threshold area."""
        bounds = (0.0, 0.6)
        plugin = Threshold(0.4, fuzzy_bounds=bounds)
        result = plugin.process(self.cube)
        expected_result_array = np.zeros_like(self.cube.data).reshape(
            1, 1, 5, 5)
        expected_result_array[0][0][2][2] = 0.75
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_asymmetric_bounds_above(self):
        """Test when a point is above asymmetric fuzzy threshold area."""
        bounds = (0.0, 0.45)
        plugin = Threshold(0.4, fuzzy_bounds=bounds)
        result = plugin.process(self.cube)
        expected_result_array = np.zeros_like(self.cube.data).reshape(
            1, 1, 5, 5)
        expected_result_array[0][0][2][2] = 1.
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_asymmetric_bounds_upper_below(self):
        """Test when a point is in upper asymmetric fuzzy threshold area
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

    def test_threshold_unit_conversion(self):
        """Test data are correctly thresholded when the threshold is given in
        units different from that of the input cube.  In this test two
        thresholds (of 4 and 6 mm/h) are used on a 5x5 cube where the
        central data point value is 1.39e-6 m/s (~ 5 mm/h)."""
        expected_result_array = np.zeros((2, 5, 5))
        expected_result_array[0][2][2] = 1.
        plugin = Threshold([4.0, 6.0], threshold_units='mm h-1')
        result = plugin.process(self.rate_cube)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_unit_conversion_fuzzy_factor(self):
        """Test for sensible fuzzy factor behaviour when units of threshold
        are different from input cube.  A fuzzy factor of 0.75 is equivalent
        to bounds +/- 25% around the threshold in the given units.  So for a
        threshold of 4 (6) mm/h, the thresholded exceedance probabilities
        increase linearly from 0 at 3 (4.5) mm/h to 1 at 5 (7.5) mm/h."""
        expected_result_array = np.zeros((2, 5, 5))
        expected_result_array[0][2][2] = 1.
        expected_result_array[1][2][2] = 0.168
        plugin = Threshold([4.0, 6.0], threshold_units='mm h-1',
                           fuzzy_factor=0.75)
        result = plugin.process(self.rate_cube)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_point_nan(self):
        """Test behaviour for a single NaN grid cell."""
        # Need to copy the cube as we're adjusting the data.
        self.cube.data[0][2][2] = np.NAN
        msg = "NaN detected in input cube data"
        plugin = Threshold(
            2.0, fuzzy_factor=self.fuzzy_factor, below_thresh_ok=True)
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(self.cube)

    def test_threshold_zero_with_fuzzy_factor(self):
        """Test when a threshold of zero is used with a multiplicative
        fuzzy factor (invalid)."""
        fuzzy_factor = 0.6
        msg = "Invalid threshold with fuzzy factor"
        with self.assertRaisesRegex(ValueError, msg):
            Threshold(0.0, fuzzy_factor=fuzzy_factor)

    def test_threshold_fuzzy_factor_minus_1(self):
        """Test when a fuzzy factor of minus 1 is given (invalid)."""
        fuzzy_factor = -1.0
        msg = "Invalid fuzzy_factor: must be >0 and <1: -1.0"
        with self.assertRaisesRegex(ValueError, msg):
            Threshold(0.6, fuzzy_factor=fuzzy_factor)

    def test_threshold_fuzzy_factor_0(self):
        """Test when a fuzzy factor of zero is given (invalid)."""
        fuzzy_factor = 0.0
        msg = "Invalid fuzzy_factor: must be >0 and <1: 0.0"
        with self.assertRaisesRegex(ValueError, msg):
            Threshold(0.6, fuzzy_factor=fuzzy_factor)

    def test_threshold_fuzzy_factor_1(self):
        """Test when a fuzzy factor of unity is given (invalid)."""
        fuzzy_factor = 1.0
        msg = "Invalid fuzzy_factor: must be >0 and <1: 1.0"
        with self.assertRaisesRegex(ValueError, msg):
            Threshold(0.6, fuzzy_factor=fuzzy_factor)

    def test_threshold_fuzzy_factor_2(self):
        """Test when a fuzzy factor of 2 is given (invalid)."""
        fuzzy_factor = 2.0
        msg = "Invalid fuzzy_factor: must be >0 and <1: 2.0"
        with self.assertRaisesRegex(ValueError, msg):
            Threshold(0.6, fuzzy_factor=fuzzy_factor)

    def test_fuzzy_factor_and_fuzzy_bounds(self):
        """Test when fuzzy_factor and fuzzy_bounds both set (ambiguous)."""
        fuzzy_factor = 2.0
        fuzzy_bounds = (0.4, 0.8)
        msg = ("Invalid combination of keywords. Cannot specify "
               "fuzzy_factor and fuzzy_bounds together")
        with self.assertRaisesRegex(ValueError, msg):
            Threshold(0.6, fuzzy_factor=fuzzy_factor,
                      fuzzy_bounds=fuzzy_bounds)

    def test_invalid_bounds_toofew(self):
        """Test when fuzzy_bounds contains one value (invalid)."""
        threshold = 0.6
        fuzzy_bounds = (0.4, )
        # Regexp matches .* with any string.
        msg = ("Invalid bounds for one threshold: .*. "
               "Expected 2 floats.")
        with self.assertRaisesRegex(AssertionError, msg):
            Threshold(threshold,
                      fuzzy_bounds=fuzzy_bounds)

    def test_invalid_bounds_toomany(self):
        """Test when fuzzy_bounds contains three values (invalid)."""
        threshold = 0.6
        fuzzy_bounds = (0.4, 0.8, 1.2)
        # Regexp matches .* with any string.
        msg = ("Invalid bounds for one threshold: .*. "
               "Expected 2 floats.")
        with self.assertRaisesRegex(AssertionError, msg):
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
        with self.assertRaisesRegex(AssertionError, msg):
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
        with self.assertRaisesRegex(AssertionError, msg):
            Threshold(threshold,
                      fuzzy_bounds=fuzzy_bounds)


if __name__ == '__main__':
    unittest.main()
