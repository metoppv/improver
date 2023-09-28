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
"""Unit tests for the spatial_weights.SpatiallyVaryingWeightsFromMask
   plugin."""


import unittest
from datetime import datetime

import numpy as np
import pytest
from iris.coords import AuxCoord
from iris.cube import CubeList
from iris.tests import IrisTest
from iris.util import squeeze

from improver.blending.spatial_weights import SpatiallyVaryingWeightsFromMask
from improver.metadata.probabilistic import find_threshold_coordinate
from improver.synthetic_data.set_up_test_cubes import set_up_probability_cube


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(SpatiallyVaryingWeightsFromMask("model_id"))
        msg = "<SpatiallyVaryingWeightsFromMask: fuzzy_length: 10>"
        self.assertEqual(result, msg)


class Test__create_template_slice(IrisTest):
    """Test create_template_slice method"""

    def setUp(self):
        """
        Set up a basic input cube. Input cube has 2 thresholds on and 3
        forecast_reference_times
        """
        thresholds = [10, 20]
        data = np.ones((2, 2, 3), dtype=np.float32)
        cycle1 = set_up_probability_cube(
            data,
            thresholds,
            spatial_grid="equalarea",
            time=datetime(2017, 11, 10, 4, 0),
            frt=datetime(2017, 11, 10, 0, 0),
        )
        cycle2 = set_up_probability_cube(
            data,
            thresholds,
            spatial_grid="equalarea",
            time=datetime(2017, 11, 10, 4, 0),
            frt=datetime(2017, 11, 10, 1, 0),
        )
        cycle3 = set_up_probability_cube(
            data,
            thresholds,
            spatial_grid="equalarea",
            time=datetime(2017, 11, 10, 4, 0),
            frt=datetime(2017, 11, 10, 2, 0),
        )
        self.cube_to_collapse = CubeList([cycle1, cycle2, cycle3]).merge_cube()
        self.cube_to_collapse = squeeze(self.cube_to_collapse)
        self.cube_to_collapse.rename("weights")
        # This input array has 3 forecast reference times and 2 thresholds.
        # The two thresholds have the same weights.
        self.cube_to_collapse.data = np.array(
            [
                [[[1, 0, 1], [1, 1, 1]], [[1, 0, 1], [1, 1, 1]]],
                [[[0, 0, 1], [0, 1, 1]], [[0, 0, 1], [0, 1, 1]]],
                [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]],
            ],
            dtype=np.float32,
        )
        self.cube_to_collapse.data = np.ma.masked_equal(self.cube_to_collapse.data, 0)
        self.plugin = SpatiallyVaryingWeightsFromMask("forecast_reference_time")

    def test_multi_dim_blend_coord_fail(self):
        """Test error is raised when we have a multi-dimensional blend_coord"""
        # Add a surface altitude coordinate which covers x and y dimensions.
        altitudes = np.array([[10, 20, 30], [20, 30, 10]])
        altitudes_coord = AuxCoord(
            altitudes, standard_name="surface_altitude", units="m"
        )
        self.cube_to_collapse.add_aux_coord(altitudes_coord, data_dims=(2, 3))
        message = "Blend coordinate must only be across one dimension"
        plugin = SpatiallyVaryingWeightsFromMask("surface_altitude")
        with self.assertRaisesRegex(ValueError, message):
            plugin._create_template_slice(self.cube_to_collapse)

    def test_varying_mask_fail(self):
        """Test error is raised when mask varies along collapsing dim"""
        # Check fails when blending along threshold coordinate, as mask
        # varies along this coordinate.
        threshold_coord = find_threshold_coordinate(self.cube_to_collapse)
        message = "The mask on the input cube can only vary along the blend_coord"
        plugin = SpatiallyVaryingWeightsFromMask(threshold_coord.name())
        with self.assertRaisesRegex(ValueError, message):
            plugin._create_template_slice(self.cube_to_collapse)

    def test_scalar_blend_coord_fail(self):
        """Test error is raised when blend_coord is scalar"""
        message = "Blend coordinate .* has no associated dimension"
        with self.assertRaisesRegex(ValueError, message):
            self.plugin._create_template_slice(self.cube_to_collapse[0])

    def test_basic(self):
        """Test a correct template slice is returned for simple case"""
        expected = self.cube_to_collapse.copy()[:, 0, :, :]
        result = self.plugin._create_template_slice(self.cube_to_collapse)
        self.assertEqual(expected.metadata, result.metadata)
        self.assertArrayAlmostEqual(expected.data, result.data)

    def test_basic_no_change(self):
        """Test a correct template slice is returned for a case where
        no slicing is needed"""
        input_cube = self.cube_to_collapse.copy()[:, 0, :, :]
        expected = input_cube.copy()
        result = self.plugin._create_template_slice(input_cube)
        self.assertEqual(expected.metadata, result.metadata)
        self.assertArrayAlmostEqual(expected.data, result.data)

    def test_aux_blending_coord(self):
        """Test a correct template slice is returned when blending_coord is
        an AuxCoord"""
        expected = self.cube_to_collapse.copy()[:, 0, :, :]
        plugin = SpatiallyVaryingWeightsFromMask("forecast_period")
        result = plugin._create_template_slice(self.cube_to_collapse)
        self.assertEqual(expected.metadata, result.metadata)
        self.assertArrayAlmostEqual(expected.data, result.data)


class Test_process(IrisTest):
    """Test process method"""

    def setUp(self):
        """
        Set up a basic cube and linear weights cube for the process
        method. Input cube has 2 thresholds and 3 forecast_reference_times
        """
        thresholds = [10, 20]
        data = np.ones((2, 2, 3), dtype=np.float32)
        cycle1 = set_up_probability_cube(
            data,
            thresholds,
            spatial_grid="equalarea",
            time=datetime(2017, 11, 10, 4, 0),
            frt=datetime(2017, 11, 10, 0, 0),
        )
        cycle2 = set_up_probability_cube(
            data,
            thresholds,
            spatial_grid="equalarea",
            time=datetime(2017, 11, 10, 4, 0),
            frt=datetime(2017, 11, 10, 1, 0),
        )
        cycle3 = set_up_probability_cube(
            data,
            thresholds,
            spatial_grid="equalarea",
            time=datetime(2017, 11, 10, 4, 0),
            frt=datetime(2017, 11, 10, 2, 0),
        )
        self.cube_to_collapse = CubeList([cycle1, cycle2, cycle3]).merge_cube()
        self.cube_to_collapse = squeeze(self.cube_to_collapse)
        self.cube_to_collapse.rename("weights")
        # This input array has 3 forecast reference times and 2 thresholds.
        # The two thresholds have the same weights.
        self.cube_to_collapse.data = np.array(
            [
                [[[1, 0, 1], [1, 1, 1]], [[1, 0, 1], [1, 1, 1]]],
                [[[0, 0, 1], [0, 1, 1]], [[0, 0, 1], [0, 1, 1]]],
                [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]],
            ],
            dtype=np.float32,
        )
        self.cube_to_collapse.data = np.ma.masked_equal(self.cube_to_collapse.data, 0)
        # Create a one_dimensional weights cube by slicing the larger
        # weights cube.
        # The resulting cube only has a forecast_reference_time coordinate.
        self.one_dimensional_weights_cube = self.cube_to_collapse[:, 0, 0, 0]
        self.one_dimensional_weights_cube.remove_coord("projection_x_coordinate")
        self.one_dimensional_weights_cube.remove_coord("projection_y_coordinate")
        self.one_dimensional_weights_cube.remove_coord(
            find_threshold_coordinate(self.one_dimensional_weights_cube)
        )
        self.one_dimensional_weights_cube.data = np.array(
            [0.2, 0.5, 0.3], dtype=np.float32
        )
        self.plugin = SpatiallyVaryingWeightsFromMask(
            "forecast_reference_time", fuzzy_length=2
        )
        self.plugin_no_fuzzy = SpatiallyVaryingWeightsFromMask(
            "forecast_reference_time", fuzzy_length=1
        )

    def test_none_masked(self):
        """Test when we have no masked data in the input cube."""
        self.cube_to_collapse.data = np.ones(self.cube_to_collapse.data.shape)
        self.cube_to_collapse.data = np.ma.masked_equal(self.cube_to_collapse.data, 0)
        expected_data = np.array(
            [
                [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]],
                [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
                [[0.3, 0.3, 0.3], [0.3, 0.3, 0.3]],
            ],
            dtype=np.float32,
        )
        message = "Expected masked input"
        with pytest.warns(UserWarning, match=message):
            result = self.plugin.process(
                self.cube_to_collapse,
                self.one_dimensional_weights_cube,
            )
        self.assertArrayEqual(result.data, expected_data)
        self.assertEqual(result.dtype, np.float32)

    def test_all_masked(self):
        """Test when we have all masked data in the input cube."""
        self.cube_to_collapse.data = np.ones(self.cube_to_collapse.data.shape)
        self.cube_to_collapse.data = np.ma.masked_equal(self.cube_to_collapse.data, 1)
        result = self.plugin.process(
            self.cube_to_collapse,
            self.one_dimensional_weights_cube,
        )
        expected_data = np.zeros((3, 2, 3))
        self.assertArrayAlmostEqual(expected_data, result.data)
        self.assertTrue(result.metadata, self.cube_to_collapse.data)

    def test_no_fuzziness_no_one_dimensional_weights(self):
        """Test a simple case where we have no fuzziness in the spatial
        weights and no adjustment from the one_dimensional weights."""
        self.one_dimensional_weights_cube.data = np.ones((3))
        expected_result = np.array(
            [
                [[0.5, 0.0, 0.333333], [0.5, 0.333333, 0.333333]],
                [[0.0, 0.0, 0.333333], [0.0, 0.333333, 0.333333]],
                [[0.5, 1.0, 0.333333], [0.5, 0.333333, 0.333333]],
            ],
            dtype=np.float32,
        )
        result = self.plugin_no_fuzzy.process(
            self.cube_to_collapse,
            self.one_dimensional_weights_cube,
        )
        self.assertArrayAlmostEqual(result.data, expected_result)
        self.assertEqual(result.metadata, self.cube_to_collapse.metadata)

    def test_no_fuzziness_no_one_dimensional_weights_transpose(self):
        """Test a simple case where we have no fuzziness in the spatial
        weights and no adjustment from the one_dimensional weights and
        transpose the input cube."""
        self.one_dimensional_weights_cube.data = np.ones((3))
        expected_result = np.array(
            [
                [[0.5, 0.0, 0.333333], [0.5, 0.333333, 0.333333]],
                [[0.0, 0.0, 0.333333], [0.0, 0.333333, 0.333333]],
                [[0.5, 1.0, 0.333333], [0.5, 0.333333, 0.333333]],
            ],
            dtype=np.float32,
        )
        self.cube_to_collapse.transpose([2, 0, 1, 3])
        result = self.plugin_no_fuzzy.process(
            self.cube_to_collapse,
            self.one_dimensional_weights_cube,
        )
        self.assertArrayAlmostEqual(result.data, expected_result)
        self.assertEqual(result.metadata, self.cube_to_collapse.metadata)

    def test_no_fuzziness_with_one_dimensional_weights(self):
        """Test a simple case where we have no fuzziness in the spatial
        weights and an adjustment from the one_dimensional weights."""
        expected_result = np.array(
            [
                [[0.4, 0.0, 0.2], [0.4, 0.2, 0.2]],
                [[0.0, 0.0, 0.5], [0.0, 0.5, 0.5]],
                [[0.6, 1.0, 0.3], [0.6, 0.3, 0.3]],
            ],
            dtype=np.float32,
        )
        result = self.plugin_no_fuzzy.process(
            self.cube_to_collapse,
            self.one_dimensional_weights_cube,
        )
        self.assertArrayAlmostEqual(result.data, expected_result)
        self.assertEqual(result.metadata, self.cube_to_collapse.metadata)

    def test_fuzziness_no_one_dimensional_weights(self):
        """Test a simple case where we have some fuzziness in the spatial
        weights and no adjustment from the one_dimensional weights."""
        self.one_dimensional_weights_cube.data = np.ones((3))
        expected_result = np.array(
            [
                [[0.25, 0.0, 0.166667], [0.353553, 0.166667, 0.235702]],
                [[0.00, 0.0, 0.166667], [0.000000, 0.166667, 0.235702]],
                [[0.75, 1.0, 0.666667], [0.646447, 0.666667, 0.528595]],
            ],
            dtype=np.float32,
        )
        result = self.plugin.process(
            self.cube_to_collapse,
            self.one_dimensional_weights_cube,
        )
        self.assertArrayAlmostEqual(result.data, expected_result)
        self.assertEqual(result.metadata, self.cube_to_collapse.metadata)

    def test_fuzziness_with_one_dimensional_weights(self):
        """Test a simple case where we have some fuzziness in the spatial
        weights and with adjustment from the one_dimensional weights."""
        expected_result = np.array(
            [
                [[0.2, 0.0, 0.10], [0.282843, 0.10, 0.141421]],
                [[0.0, 0.0, 0.25], [0.000000, 0.25, 0.353553]],
                [[0.8, 1.0, 0.65], [0.717157, 0.65, 0.505025]],
            ],
            dtype=np.float32,
        )
        result = self.plugin.process(
            self.cube_to_collapse,
            self.one_dimensional_weights_cube,
        )
        self.assertArrayAlmostEqual(result.data, expected_result)
        self.assertEqual(result.metadata, self.cube_to_collapse.metadata)

    def test_fuzziness_with_unequal_weightings(self):
        """Simulate the case of two models and a nowcast at short lead times: two
        unmasked slices with low weights, and one masked slice with high weights"""
        self.cube_to_collapse.data[0].mask = np.full_like(
            self.cube_to_collapse.data[0], False
        )
        self.one_dimensional_weights_cube.data = np.array(
            [0.025, 1.0, 0.075], dtype=np.float32
        )
        expected_data = np.array(
            [
                [[0.25, 0.25, 0.136364], [0.25, 0.136364, 0.0892939]],
                [[0.0, 0.0, 0.45454544], [0.0, 0.454545, 0.642824]],
                [[0.75, 0.75, 0.409091], [0.75, 0.409091, 0.267882]],
            ],
            dtype=np.float32,
        )
        result = self.plugin.process(
            self.cube_to_collapse,
            self.one_dimensional_weights_cube,
        )
        self.assertArrayAlmostEqual(result.data, expected_data)


if __name__ == "__main__":
    unittest.main()
