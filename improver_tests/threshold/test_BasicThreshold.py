# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
from iris.coords import CellMethod, DimCoord
from iris.cube import Cube
from iris.tests import IrisTest

from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)
from improver.threshold import BasicThreshold as Threshold


class Test__add_threshold_coord(IrisTest):
    """Test the _add_threshold_coord method"""

    def setUp(self):
        """Set up a cube and plugin for testing."""
        self.cube = set_up_variable_cube(np.ones((3, 3), dtype=np.float32))
        self.plugin = Threshold([1])
        self.plugin.threshold_coord_name = self.cube.name()

    def test_basic(self):
        """Test a scalar threshold coordinate is created"""
        self.plugin._add_threshold_coord(self.cube, 1)
        self.assertEqual(self.cube.ndim, 2)
        self.assertIn(
            "air_temperature",
            [coord.standard_name for coord in self.cube.coords(dim_coords=False)],
        )
        threshold_coord = self.cube.coord("air_temperature")
        self.assertEqual(threshold_coord.var_name, "threshold")
        self.assertAlmostEqual(threshold_coord.points[0], 1)
        self.assertEqual(threshold_coord.units, self.cube.units)

    def test_long_name(self):
        """Test coordinate is created with non-standard diagnostic name"""
        self.cube.rename("sky_temperature")
        self.plugin.threshold_coord_name = self.cube.name()
        self.plugin._add_threshold_coord(self.cube, 1)
        self.assertIn(
            "sky_temperature",
            [coord.long_name for coord in self.cube.coords(dim_coords=False)],
        )

    def test_value_error(self):
        """Test method catches ValueErrors unrelated to name, by passing it a
        list of values where a scalar is required"""
        with self.assertRaises(ValueError):
            self.plugin._add_threshold_coord(self.cube, [1, 1])


class Test_process(IrisTest):

    """Test the thresholding plugin."""

    def setUp(self):
        """Create a cube with a single non-zero point."""
        attributes = {"title": "UKV Model Forecast"}
        data = np.zeros((5, 5), dtype=np.float32)
        data[2][2] = 0.5  # ~2 mm/hr
        self.cube = set_up_variable_cube(
            data,
            name="precipitation_amount",
            units="kg m^-2 s^-1",
            attributes=attributes,
            standard_grid_metadata="uk_det",
        )

        rate_data = np.zeros((5, 5), dtype=np.float32)
        rate_data[2][2] = 1.39e-6  # 5.004 mm/hr
        self.rate_cube = set_up_variable_cube(
            rate_data,
            name="rainfall_rate",
            units="m s-1",
            attributes=attributes,
            standard_grid_metadata="uk_det",
        )

        self.fuzzy_factor = 0.5
        self.plugin = Threshold(0.1, fuzzy_factor=0.95)

        self.masked_cube = self.cube.copy()
        data = np.zeros((5, 5))
        mask = np.zeros((5, 5))
        data[2][2] = 0.5
        data[0][0] = -32768.0
        mask[0][0] = 1
        self.masked_cube.data = np.ma.MaskedArray(data, mask=mask)

    def test_basic(self):
        """Test that the plugin returns an iris.cube.Cube."""
        result = self.plugin(self.cube)
        self.assertIsInstance(result, Cube)

    def test_title_updated(self):
        """Test title is updated"""
        expected_title = "Post-Processed UKV Model Forecast"
        result = self.plugin(self.cube)
        self.assertEqual(result.attributes["title"], expected_title)

    def test_data_type_change_for_ints(self):
        """Test that the plugin returns an iris.cube.Cube of float32 type
        if the input cube is of int type. This allows fuzzy bounds to be used
        which return fractional values."""
        fuzzy_factor = 5.0 / 6.0
        threshold = 12
        self.cube.data = np.arange(25).reshape((5, 5))
        plugin = Threshold(threshold, fuzzy_factor=fuzzy_factor)
        result = plugin(self.cube)
        expected = np.round(np.arange(0, 1, 1.0 / 25.0)).reshape((5, 5))
        expected[2, 1:4] = [0.25, 0.5, 0.75]
        self.assertEqual(result.dtype, "float32")
        self.assertArrayEqual(result.data, expected)

    def test_metadata_changes(self):
        """Test the metadata altering functionality"""
        plugin = Threshold(0.1)
        result = plugin(self.cube)
        name = "probability_of_{}_above_threshold"
        expected_name = name.format(self.cube.name())
        expected_attribute = "greater_than"
        expected_units = 1
        expected_coord = DimCoord(
            np.array([0.1], dtype=np.float32),
            standard_name=self.cube.name(),
            var_name="threshold",
            units=self.cube.units,
            attributes={"spp__relative_to_threshold": "greater_than"},
        )
        self.assertEqual(result.name(), expected_name)
        self.assertEqual(
            result.coord(var_name="threshold").attributes["spp__relative_to_threshold"],
            expected_attribute,
        )
        self.assertEqual(result.units, expected_units)
        self.assertEqual(result.coord(self.cube.name()), expected_coord)

    def test_threshold(self):
        """Test the basic threshold functionality."""
        expected_result_array = np.zeros_like(self.cube.data)
        expected_result_array[2][2] = 1.0
        result = self.plugin(self.cube)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_above_threshold_without_fuzzy_factor(self):
        """Test if the fixed threshold is below the value in the data."""
        plugin = Threshold(0.1)
        expected_result_array = np.zeros_like(self.cube.data)
        expected_result_array[2][2] = 1.0
        result = plugin(self.cube)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_below_threshold_without_fuzzy_factor(self):
        """Test if the fixed threshold is above the value in the data."""
        plugin = Threshold(0.6)
        result = plugin(self.cube)
        expected_result_array = np.zeros_like(self.cube.data)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_masked_array(self):
        """Test masked array are handled correctly.
        Masked values are preserved following thresholding."""
        plugin = Threshold(0.1)
        result = plugin(self.masked_cube)
        expected_result_array = np.zeros_like(self.masked_cube.data)
        expected_result_array[2][2] = 1.0
        self.assertArrayAlmostEqual(result.data.data, expected_result_array)
        self.assertArrayEqual(result.data.mask, self.masked_cube.data.mask)

    def test_fill_masked(self):
        """Test plugin when masked points are replaced with fill value"""
        plugin = Threshold(0.6, fill_masked=np.inf)
        result = plugin(self.masked_cube)
        expected_result = np.zeros_like(self.masked_cube.data)
        expected_result[0][0] = 1.0
        self.assertArrayEqual(result.data, expected_result)

    def test_threshold_fuzzy(self):
        """Test when a point is in the fuzzy threshold area."""
        plugin = Threshold(0.6, fuzzy_factor=self.fuzzy_factor)
        result = plugin(self.cube)
        expected_result_array = np.zeros_like(self.cube.data)
        expected_result_array[2][2] = 1.0 / 3.0
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_fuzzybounds(self):
        """Test when a point is in the fuzzy threshold area."""
        bounds = (0.6 * self.fuzzy_factor, 0.6 * (2.0 - self.fuzzy_factor))
        plugin = Threshold(0.6, fuzzy_bounds=bounds)
        result = plugin(self.cube)
        expected_result_array = np.zeros_like(self.cube.data)
        expected_result_array[2][2] = 1.0 / 3.0
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_masked_array_fuzzybounds(self):
        """Test masked array are handled correctly when using fuzzy bounds.
        Masked values are preserved following thresholding."""
        bounds = (0.6 * self.fuzzy_factor, 0.6 * (2.0 - self.fuzzy_factor))
        plugin = Threshold(0.6, fuzzy_bounds=bounds)
        result = plugin.process(self.masked_cube)
        expected_result_array = np.zeros_like(self.masked_cube.data)
        expected_result_array[2][2] = 1.0 / 3.0
        self.assertArrayAlmostEqual(result.data.data, expected_result_array)
        self.assertArrayEqual(
            result.data.mask, self.masked_cube.data.mask.reshape((5, 5))
        )

    def test_threshold_boundingzero(self):
        """Test fuzzy threshold of zero."""
        bounds = (-1.0, 1.0)
        plugin = Threshold(0.0, fuzzy_bounds=bounds)
        result = plugin(self.cube)
        expected_result_array = np.full_like(self.cube.data, fill_value=0.5)
        expected_result_array[2][2] = 0.75
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_boundingzero_above(self):
        """Test fuzzy threshold of zero where data are above upper-bound."""
        bounds = (-0.1, 0.1)
        plugin = Threshold(0.0, fuzzy_bounds=bounds)
        result = plugin(self.cube)
        expected_result_array = np.full_like(self.cube.data, fill_value=0.5)
        expected_result_array[2][2] = 1.0
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_boundingbelowzero(self):
        """Test fuzzy threshold of below-zero."""
        bounds = (-1.0, 1.0)
        plugin = Threshold(0.0, fuzzy_bounds=bounds, comparison_operator="<")
        result = plugin(self.cube)
        expected_result_array = np.full_like(self.cube.data, fill_value=0.5)
        expected_result_array[2][2] = 0.25
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_asymmetric_bounds_below(self):
        """Test when a point is below asymmetric fuzzy threshold area."""
        bounds = (0.51, 0.9)
        plugin = Threshold(0.6, fuzzy_bounds=bounds)
        result = plugin(self.cube)
        expected_result_array = np.zeros_like(self.cube.data)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_asymmetric_bounds_lower(self):
        """Test when a point is in lower asymmetric fuzzy threshold area."""
        bounds = (0.4, 0.9)
        plugin = Threshold(0.6, fuzzy_bounds=bounds)
        result = plugin(self.cube)
        expected_result_array = np.zeros_like(self.cube.data)
        expected_result_array[2][2] = 0.25
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_asymmetric_bounds_middle(self):
        """Test when a point is on the threshold with asymmetric fuzzy
        bounds."""
        bounds = (0.4, 0.9)
        plugin = Threshold(0.5, fuzzy_bounds=bounds)
        result = plugin(self.cube)
        expected_result_array = np.zeros_like(self.cube.data)
        expected_result_array[2][2] = 0.5
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_asymmetric_bounds_upper(self):
        """Test when a point is in upper asymmetric fuzzy threshold area."""
        bounds = (0.0, 0.6)
        plugin = Threshold(0.4, fuzzy_bounds=bounds)
        result = plugin(self.cube)
        expected_result_array = np.zeros_like(self.cube.data)
        expected_result_array[2][2] = 0.75
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_asymmetric_bounds_above(self):
        """Test when a point is above asymmetric fuzzy threshold area."""
        bounds = (0.0, 0.45)
        plugin = Threshold(0.4, fuzzy_bounds=bounds)
        result = plugin(self.cube)
        expected_result_array = np.zeros_like(self.cube.data)
        expected_result_array[2][2] = 1.0
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_asymmetric_bounds_upper_below(self):
        """Test when a point is in upper asymmetric fuzzy threshold area
        and below-threshold is requested."""
        bounds = (0.0, 0.6)
        plugin = Threshold(0.4, fuzzy_bounds=bounds, comparison_operator="<")
        result = plugin(self.cube)
        expected_result_array = np.ones_like(self.cube.data)
        expected_result_array[2][2] = 0.25
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_fuzzy_miss(self):
        """Test when a point is not within the fuzzy threshold area."""
        plugin = Threshold(2.0, fuzzy_factor=self.fuzzy_factor)
        result = plugin(self.cube)
        expected_result_array = np.zeros_like(self.cube.data)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_fuzzy_miss_high_threshold(self):
        """Test when a point is not within the fuzzy high threshold area."""
        plugin = Threshold(3.0, fuzzy_factor=self.fuzzy_factor)
        result = plugin(self.cube)
        expected_result_array = np.zeros_like(self.cube.data)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_negative(self):
        """Test a point when the threshold is negative."""
        self.cube.data[2][2] = -0.75
        plugin = Threshold(
            -1.0, fuzzy_factor=self.fuzzy_factor, comparison_operator="<"
        )
        result = plugin(self.cube)
        expected_result_array = np.zeros_like(self.cube.data)
        expected_result_array[2][2] = 0.25
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_below_fuzzy(self):
        """Test a point in fuzzy threshold in below-threshold-mode."""
        plugin = Threshold(0.6, fuzzy_factor=self.fuzzy_factor, comparison_operator="<")
        result = plugin(self.cube)
        expected_result_array = np.ones_like(self.cube.data)
        expected_result_array[2][2] = 2.0 / 3.0
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_below_fuzzy_miss(self):
        """Test not meeting the threshold in fuzzy below-threshold-mode."""
        plugin = Threshold(2.0, fuzzy_factor=self.fuzzy_factor, comparison_operator="<")
        result = plugin(self.cube)
        expected_result_array = np.ones_like(self.cube.data)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_gt(self):
        """Test a point when we are in > threshold mode."""
        plugin = Threshold(0.5)
        name = "probability_of_{}_above_threshold"
        expected_name = name.format(self.cube.name())
        expected_attribute = "greater_than"
        result = plugin(self.cube)
        expected_result_array = np.zeros_like(self.cube.data)
        expected_result_array[2][2] = 0
        self.assertEqual(result.name(), expected_name)
        self.assertEqual(
            result.coord(var_name="threshold").attributes["spp__relative_to_threshold"],
            expected_attribute,
        )
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_ge(self):
        """Test a point when we are in >= threshold mode."""
        plugin = Threshold(0.5, comparison_operator=">=")
        name = "probability_of_{}_above_threshold"
        expected_name = name.format(self.cube.name())
        expected_attribute = "greater_than_or_equal_to"
        result = plugin(self.cube)
        expected_result_array = np.zeros_like(self.cube.data)
        expected_result_array[2][2] = 1
        self.assertEqual(result.name(), expected_name)
        self.assertEqual(
            result.coord(var_name="threshold").attributes["spp__relative_to_threshold"],
            expected_attribute,
        )
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_lt(self):
        """Test a point when we are in < threshold mode."""
        plugin = Threshold(0.5, comparison_operator="<")
        name = "probability_of_{}_below_threshold"
        expected_name = name.format(self.cube.name())
        expected_attribute = "less_than"
        result = plugin(self.cube)
        expected_result_array = np.ones_like(self.cube.data)
        expected_result_array[2][2] = 0
        self.assertEqual(result.name(), expected_name)
        self.assertEqual(
            result.coord(var_name="threshold").attributes["spp__relative_to_threshold"],
            expected_attribute,
        )
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_le(self):
        """Test a point when we are in le threshold mode."""
        plugin = Threshold(0.5, comparison_operator="<=")
        name = "probability_of_{}_below_threshold"
        expected_name = name.format(self.cube.name())
        expected_attribute = "less_than_or_equal_to"
        result = plugin(self.cube)
        expected_result_array = np.ones_like(self.cube.data)
        expected_result_array[2][2] = 1
        self.assertEqual(result.name(), expected_name)
        self.assertEqual(
            result.coord(var_name="threshold").attributes["spp__relative_to_threshold"],
            expected_attribute,
        )
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_multiple_thresholds(self):
        """Test multiple thresholds applied to a multi-realization cube return a
        single cube arrays corresponding to each realization and threshold."""
        multi_realization_cube = add_coordinate(
            self.cube, [0, 1, 2], "realization", dtype=np.int32
        )
        all_zeroes = np.zeros_like(multi_realization_cube.data)
        one_exceed_point = all_zeroes.copy()
        one_exceed_point[:, 2, 2] = 1.0
        expected_result_array = np.array(
            [one_exceed_point, one_exceed_point, all_zeroes]
        )
        # transpose array so that realization is leading coordinate
        expected_result_array = np.transpose(expected_result_array, [1, 0, 2, 3])
        thresholds = [0.2, 0.4, 0.6]
        plugin = Threshold(thresholds)
        result = plugin(multi_realization_cube)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_unit_conversion(self):
        """Test data are correctly thresholded when the threshold is given in
        units different from that of the input cube.  In this test two
        thresholds (of 4 and 6 mm/h) are used on a 5x5 cube where the
        central data point value is 1.39e-6 m/s (~ 5 mm/h)."""
        expected_result_array = np.zeros((2, 5, 5))
        expected_result_array[0][2][2] = 1.0
        plugin = Threshold([4.0, 6.0], threshold_units="mm h-1")
        result = plugin(self.rate_cube)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_unit_conversion_2(self):
        """Test threshold coordinate points after undergoing unit conversion.
        Specifically ensuring that small floating point values have no floating
        point precision errors after the conversion (float equality check with no
        tolerance)."""
        plugin = Threshold([0.03, 0.09, 0.1], threshold_units="mm s-1")
        result = plugin(self.rate_cube)
        self.assertArrayEqual(
            result.coord(var_name="threshold").points,
            np.array([3e-5, 9.0e-05, 1e-4], dtype="float32"),
        )

    def test_threshold_unit_conversion_fuzzy_factor(self):
        """Test for sensible fuzzy factor behaviour when units of threshold
        are different from input cube.  A fuzzy factor of 0.75 is equivalent
        to bounds +/- 25% around the threshold in the given units.  So for a
        threshold of 4 (6) mm/h, the thresholded exceedance probabilities
        increase linearly from 0 at 3 (4.5) mm/h to 1 at 5 (7.5) mm/h."""
        expected_result_array = np.zeros((2, 5, 5))
        expected_result_array[0][2][2] = 1.0
        expected_result_array[1][2][2] = 0.168
        plugin = Threshold([4.0, 6.0], threshold_units="mm h-1", fuzzy_factor=0.75)
        result = plugin(self.rate_cube)
        self.assertArrayAlmostEqual(result.data, expected_result_array)

    def test_threshold_point_nan(self):
        """Test behaviour for a single NaN grid cell."""
        # Need to copy the cube as we're adjusting the data.
        self.cube.data[2][2] = np.NAN
        msg = "NaN detected in input cube data"
        plugin = Threshold(2.0, fuzzy_factor=self.fuzzy_factor, comparison_operator="<")
        with self.assertRaisesRegex(ValueError, msg):
            plugin(self.cube)

    def test_each_threshold_func(self):
        """Test user supplied func is applied on each threshold cube."""
        # Need to copy the cube as we're adjusting the data.
        new_attr = {"new_attribute": "narwhal"}
        plugin = Threshold(
            2.0,
            each_threshold_func=lambda cube: cube.attributes.update(new_attr) or cube,
        )
        result = plugin(self.cube)
        self.assertTrue("new_attribute" in result.attributes)

    def test_cell_method_updates(self):
        """Test plugin adds correct information to cell methods"""
        self.cube.add_cell_method(CellMethod("max", coords="time"))
        plugin = Threshold(2.0, comparison_operator=">")
        result = plugin(self.cube)
        (cell_method,) = result.cell_methods
        self.assertEqual(cell_method.method, "max")
        self.assertEqual(cell_method.coord_names, ("time",))
        self.assertEqual(cell_method.comments, ("of precipitation_amount",))


class Test__init__(IrisTest):

    """Test error-raising behaviours unique to the init method and the private
    function _decode_comparison_operator_string."""

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
        msg = (
            "Invalid combination of keywords. Cannot specify "
            "fuzzy_factor and fuzzy_bounds together"
        )
        with self.assertRaisesRegex(ValueError, msg):
            Threshold(0.6, fuzzy_factor=fuzzy_factor, fuzzy_bounds=fuzzy_bounds)

    def test_invalid_bounds_toofew(self):
        """Test when fuzzy_bounds contains one value (invalid)."""
        threshold = 0.6
        fuzzy_bounds = (0.4,)
        # Regexp matches .* with any string.
        msg = "Invalid bounds for one threshold: .*. " "Expected 2 floats."
        with self.assertRaisesRegex(ValueError, msg):
            Threshold(threshold, fuzzy_bounds=fuzzy_bounds)

    def test_invalid_bounds_toomany(self):
        """Test when fuzzy_bounds contains three values (invalid)."""
        threshold = 0.6
        fuzzy_bounds = (0.4, 0.8, 1.2)
        # Regexp matches .* with any string.
        msg = "Invalid bounds for one threshold: .*. " "Expected 2 floats."
        with self.assertRaisesRegex(ValueError, msg):
            Threshold(threshold, fuzzy_bounds=fuzzy_bounds)

    def test_invalid_upper_bound(self):
        """Test when fuzzy_bounds do not bound threshold (invalid)."""
        threshold = 0.6
        fuzzy_bounds = (0.4, 0.5)
        # Note that back-slashes are necessary to make regexp literal.
        msg = "Threshold must be within bounds: " r"\!\( {} <= {} <= {} \)".format(
            fuzzy_bounds[0], threshold, fuzzy_bounds[1]
        )
        with self.assertRaisesRegex(ValueError, msg):
            Threshold(threshold, fuzzy_bounds=fuzzy_bounds)

    def test_invalid_lower_bound(self):
        """Test when fuzzy_bounds do not bound threshold (invalid)."""
        threshold = 0.6
        fuzzy_bounds = (0.7, 0.8)
        # Note that back-slashes are necessary to make regexp literal.
        msg = "Threshold must be within bounds: " r"\!\( {} <= {} <= {} \)".format(
            fuzzy_bounds[0], threshold, fuzzy_bounds[1]
        )
        with self.assertRaisesRegex(ValueError, msg):
            Threshold(threshold, fuzzy_bounds=fuzzy_bounds)

    def test_invalid_comparison_operator(self):
        """Test plugin throws a ValueError when comparison_operator is bad"""
        comparison_operator = "invalid"
        threshold = 0.6
        msg = (
            'String "{}" does not match any known comparison_operator '
            "method".format(comparison_operator)
        )
        with self.assertRaisesRegex(ValueError, msg):
            Threshold(threshold, comparison_operator=comparison_operator)


if __name__ == "__main__":
    unittest.main()
