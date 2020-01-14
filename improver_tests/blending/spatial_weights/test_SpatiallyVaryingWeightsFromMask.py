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
"""Unit tests for the spatial_weights.SpatiallyVaryingWeightsFromMask
   plugin."""


import unittest
from datetime import datetime

import numpy as np
from iris.coords import AuxCoord
from iris.cube import CubeList
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest
from iris.util import squeeze

from improver.blending.spatial_weights import SpatiallyVaryingWeightsFromMask
from improver.metadata.probabilistic import find_threshold_coordinate
from improver.utilities.warnings_handler import ManageWarnings

from ...set_up_test_cubes import set_up_probability_cube


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(SpatiallyVaryingWeightsFromMask())
        msg = ('<SpatiallyVaryingWeightsFromMask: fuzzy_length: 10>')
        self.assertEqual(result, msg)


class Test_create_initial_weights_from_mask(IrisTest):

    """Test the create_initial_weights_from_mask method """

    def setUp(self):
        """Set up an example cube to test with"""
        thresholds = [10]
        data = np.ones((1, 2, 3), dtype=np.float32)
        cycle1 = set_up_probability_cube(data, thresholds,
                                         time=datetime(2017, 11, 10, 4, 0),
                                         frt=datetime(2017, 11, 10, 0, 0))
        cycle2 = set_up_probability_cube(data, thresholds,
                                         time=datetime(2017, 11, 10, 4, 0),
                                         frt=datetime(2017, 11, 10, 1, 0))
        cycle3 = set_up_probability_cube(data, thresholds,
                                         time=datetime(2017, 11, 10, 4, 0),
                                         frt=datetime(2017, 11, 10, 2, 0))
        self.cube = CubeList([cycle1, cycle2, cycle3]).merge_cube()
        self.cube = squeeze(self.cube)
        self.plugin = SpatiallyVaryingWeightsFromMask()

    @ManageWarnings(record=True)
    def test_no_mask(self, warning_list=None):
        """Test what happens when no mask is on the input cube"""
        expected_data = np.ones((3, 2, 3), dtype=np.float32)
        message = ("Input cube to SpatiallyVaryingWeightsFromMask "
                   "must be masked")
        result = self.plugin.create_initial_weights_from_mask(self.cube)
        self.assertTrue(any(message in str(item)
                            for item in warning_list))
        self.assertArrayEqual(result.data, expected_data)
        self.assertEqual(result.dtype, np.float32)

    def test_basic(self):
        """Test the weights coming out of a simple masked cube."""
        input_data = np.array([[[10, 5, 10],
                                [10, 5, 10]],
                               [[5, 5, 10],
                                [5, 5, 10]],
                               [[10, 10, 10],
                                [10, 10, 10]]],
                              dtype=np.float32)
        mask = np.array([[[False, True, False],
                          [False, True, False]],
                         [[True, True, False],
                          [True, True, False]],
                         [[False, False, False],
                          [False, False, False]]])
        input_data = np.ma.MaskedArray(input_data, mask=mask)
        self.cube.data = input_data
        expected = np.array([[[1, 0, 1],
                              [1, 0, 1]],
                             [[0, 0, 1],
                              [0, 0, 1]],
                             [[1, 1, 1],
                              [1, 1, 1]]],
                            dtype=np.float32)
        result = self.plugin.create_initial_weights_from_mask(self.cube)
        self.assertArrayEqual(result.data, expected)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.name(), "weights")

    @ManageWarnings(record=True)
    def test_none_masked(self, warning_list=None):
        """Test what happens if we try to create a masked array for input, but
        where all the values should be unmasked."""
        input_data = np.array([[[10, 5, 10],
                                [10, 5, 10]],
                               [[5, 5, 10],
                                [5, 5, 10]],
                               [[10, 10, 10],
                                [10, 10, 10]]],
                              dtype=np.float32)
        expected_data = np.ones((3, 2, 3), dtype=np.float32)
        # This actually produces an array which numpy classes as NOT a masked
        # array.
        input_data = np.ma.masked_equal(input_data, 0)
        self.cube.data = input_data
        message = ("Input cube to SpatiallyVaryingWeightsFromMask "
                   "must be masked")
        result = self.plugin.create_initial_weights_from_mask(self.cube)
        self.assertTrue(any(message in str(item)
                            for item in warning_list))
        self.assertArrayEqual(result.data, expected_data)
        self.assertEqual(result.dtype, np.float32)

    def test_all_masked(self):
        """Test the weights coming out of a simple masked cube."""
        input_data = np.array([[[10, 10, 10],
                                [10, 10, 10]],
                               [[10, 10, 10],
                                [10, 10, 10]],
                               [[10, 10, 10],
                                [10, 10, 10]]],
                              dtype=np.float32)
        input_data = np.ma.masked_equal(input_data, 10)
        self.cube.data = input_data
        expected = np.array([[[0, 0, 0],
                              [0, 0, 0]],
                             [[0, 0, 0],
                              [0, 0, 0]],
                             [[0, 0, 0],
                              [0, 0, 0]]],
                            dtype=np.float32)
        result = self.plugin.create_initial_weights_from_mask(self.cube)
        self.assertArrayEqual(result.data, expected)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.name(), "weights")


class Test_smooth_initial_weights(IrisTest):
    """Test the smooth_initial_weights method"""

    def setUp(self):
        """
        Set up a basic 2D cube with a large enough grid to see the
        effect of the fuzzy weights.
        """
        thresholds = [10]
        data = np.ones((1, 7, 7), dtype=np.float32)
        self.cube = set_up_probability_cube(
            data, thresholds, spatial_grid="equalarea",
            time=datetime(2017, 11, 10, 4, 0),
            frt=datetime(2017, 11, 10, 0, 0),)
        self.cube = squeeze(self.cube)

    def test_no_fuzziness(self):
        """Test fuzziness over only 1 grid square, i.e. no fuzziness"""
        self.cube.data[3, 3] = 0.0
        plugin = SpatiallyVaryingWeightsFromMask(fuzzy_length=1)
        result = plugin.smooth_initial_weights(self.cube)
        self.assertArrayEqual(self.cube.data, result.data)

    def test_initial_weights_all_1(self):
        """Test the input cube all containing weights of one."""
        plugin = SpatiallyVaryingWeightsFromMask(fuzzy_length=4)
        result = plugin.smooth_initial_weights(self.cube)
        self.assertArrayEqual(self.cube.data, result.data)

    def test_basic_fuzziness(self):
        """Test fuzzy weights over 3 grid squares"""
        self.cube.data[3, 3] = 0.0
        plugin = SpatiallyVaryingWeightsFromMask(fuzzy_length=3)
        expected = np.array(
            [[1., 1., 1., 1., 1., 1., 1.],
             [1., 0.942809, 0.745356, 0.666667, 0.745356, 0.942809, 1.],
             [1., 0.745356, 0.471405, 0.333333, 0.471405, 0.745356, 1.],
             [1., 0.666667, 0.333333, 0., 0.333333, 0.666667, 1.],
             [1., 0.745356, 0.471405, 0.333333, 0.471405, 0.745356, 1.],
             [1., 0.942809, 0.745356, 0.666667, 0.745356, 0.942809, 1.],
             [1., 1., 1., 1., 1., 1., 1.]],
            dtype=np.float32)
        result = plugin.smooth_initial_weights(self.cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_non_integer_fuzziness(self):
        """Test fuzzy weights over 2.5 grid squares"""
        self.cube.data[3, 3] = 0.0
        plugin = SpatiallyVaryingWeightsFromMask(fuzzy_length=2.5)
        expected = np.array(
            [[1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 0.89442719, 0.8, 0.89442719, 1., 1.],
             [1., 0.89442719, 0.56568542, 0.4, 0.56568542, 0.89442719, 1.],
             [1., 0.8, 0.4, 0., 0.4, 0.8, 1.],
             [1., 0.89442719, 0.56568542, 0.4, 0.56568542, 0.89442719, 1.],
             [1., 1., 0.89442719, 0.8, 0.89442719, 1., 1.],
             [1., 1., 1., 1., 1., 1., 1.]],
            dtype=np.float32)
        result = plugin.smooth_initial_weights(self.cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_fuzziness_block_zero_weight_points(self):
        """Test fuzzy weights with a block of zero weight points"""
        # Set 4 grid points to zero in a square.
        self.cube.data[2:4, 3:5] = 0.0
        plugin = SpatiallyVaryingWeightsFromMask(fuzzy_length=2)
        expected = np.array(
            [[1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 0.70710678, 0.5, 0.5, 0.70710678, 1.],
             [1., 1., 0.5, 0., 0., 0.5, 1.],
             [1., 1., 0.5, 0., 0., 0.5, 1.],
             [1., 1., 0.70710678, 0.5, 0.5, 0.70710678, 1.],
             [1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1.]],
            dtype=np.float32)
        result = plugin.smooth_initial_weights(self.cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_fuzziness_more_zero_weight_points(self):
        """Test fuzzy weights with multiple zero weight points"""
        # Set 4 grid points to zero in a square.
        self.cube.data[2, 2] = 0.0
        self.cube.data[4, 4] = 0.0
        plugin = SpatiallyVaryingWeightsFromMask(fuzzy_length=2)
        expected = np.array(
            [[1., 1., 1., 1., 1., 1., 1.],
             [1., 0.70710678, 0.5, 0.70710678, 1., 1., 1.],
             [1.,  0.5, 0., 0.5, 1., 1., 1.],
             [1., 0.70710678, 0.5, 0.70710678, 0.5, 0.70710678, 1.],
             [1., 1., 1., 0.5, 0.,  0.5, 1.],
             [1., 1., 1., 0.70710678, 0.5, 0.70710678, 1.],
             [1., 1., 1., 1., 1., 1., 1.]],
            dtype=np.float32)
        result = plugin.smooth_initial_weights(self.cube)
        self.assertArrayAlmostEqual(result.data, expected)

    def test_basic_fuzziness_3D_input_cube(self):
        """Test fuzzy weights over 3 grid squares with 3D input cube."""
        thresholds = [10, 20, 30]
        data = np.ones((3, 7, 7), dtype=np.float32)
        cube = set_up_probability_cube(
            data, thresholds, spatial_grid="equalarea",
            time=datetime(2017, 11, 10, 4, 0),
            frt=datetime(2017, 11, 10, 0, 0),)
        cube.data[0, 3, 3] = 0.0
        cube.data[1, 0, 0] = 0.0
        cube.data[2, 6, 6] = 0.0
        plugin = SpatiallyVaryingWeightsFromMask(fuzzy_length=3)
        expected = np.array(
            [[[1., 1., 1., 1., 1., 1., 1.],
              [1., 0.942809, 0.745356, 0.666667, 0.745356, 0.942809, 1.],
              [1., 0.745356, 0.471405, 0.333333, 0.471405, 0.745356, 1.],
              [1., 0.666667, 0.333333, 0., 0.333333, 0.666667, 1.],
              [1., 0.745356, 0.471405, 0.333333, 0.471405, 0.745356, 1.],
              [1., 0.942809, 0.745356, 0.666667, 0.745356, 0.942809, 1.],
              [1., 1., 1., 1., 1., 1., 1.]],
             [[0., 0.33333334, 0.6666667, 1., 1., 1., 1.],
              [0.33333334, 0.47140452, 0.74535596, 1., 1., 1., 1.],
              [0.6666667,  0.74535596, 0.94280905, 1., 1., 1., 1.],
              [1., 1., 1., 1., 1., 1., 1.],
              [1., 1., 1., 1., 1., 1., 1.],
              [1., 1., 1., 1., 1., 1., 1.],
              [1., 1., 1., 1., 1., 1., 1.]],
             [[1., 1., 1., 1., 1., 1., 1.],
              [1., 1., 1., 1., 1., 1., 1.],
              [1., 1., 1., 1., 1., 1., 1.],
              [1., 1., 1., 1., 1., 1., 1.],
              [1., 1., 1., 1., 0.94280905, 0.74535596, 0.6666667],
              [1., 1., 1., 1., 0.74535596, 0.47140452, 0.33333334],
              [1., 1., 1., 1., 0.6666667, 0.33333334, 0.]]],
            dtype=np.float32)
        result = plugin.smooth_initial_weights(cube)
        self.assertArrayAlmostEqual(result.data, expected)


class Test_multiply_weights(IrisTest):
    """Test multiply_weights method"""

    def setUp(self):
        """
        Set up a basic weights cube with 2 thresholds to multiple with
        a cube with one_dimensional weights.
        """
        thresholds = [10, 20]
        data = np.ones((2, 2, 3), dtype=np.float32)
        cycle1 = set_up_probability_cube(
            data, thresholds, spatial_grid="equalarea",
            time=datetime(2017, 11, 10, 4, 0),
            frt=datetime(2017, 11, 10, 0, 0),)
        cycle2 = set_up_probability_cube(
            data, thresholds, spatial_grid="equalarea",
            time=datetime(2017, 11, 10, 4, 0),
            frt=datetime(2017, 11, 10, 1, 0),)
        cycle3 = set_up_probability_cube(
            data, thresholds, spatial_grid="equalarea",
            time=datetime(2017, 11, 10, 4, 0),
            frt=datetime(2017, 11, 10, 2, 0),)
        self.spatial_weights_cube = CubeList(
            [cycle1, cycle2, cycle3]).merge_cube()
        self.spatial_weights_cube = squeeze(self.spatial_weights_cube)
        self.spatial_weights_cube.rename("weights")
        # This input array has 3 forecast reference times and 2 thresholds.
        # The two thresholds have the same weights.
        self.spatial_weights_cube.data = np.array([[[[1, 0, 1],
                                                     [1, 0, 1]],
                                                    [[1, 0, 1],
                                                     [1, 0, 1]]],
                                                   [[[0, 0, 1],
                                                     [0, 0, 1]],
                                                    [[0, 0, 1],
                                                     [0, 0, 1]]],
                                                   [[[1, 1, 1],
                                                     [1, 1, 1]],
                                                    [[1, 1, 1],
                                                     [1, 1, 1]]]],
                                                  dtype=np.float32)
        # Create a one_dimensional weights cube by slicing the
        # larger weights cube.
        # The resulting cube only has a forecast_reference_time coordinate.
        self.one_dimensional_weights_cube = (
            self.spatial_weights_cube[:, 0, 0, 0])
        self.one_dimensional_weights_cube.remove_coord(
            "projection_x_coordinate")
        self.one_dimensional_weights_cube.remove_coord(
            "projection_y_coordinate")
        self.one_dimensional_weights_cube.remove_coord(
            find_threshold_coordinate(self.one_dimensional_weights_cube))
        self.one_dimensional_weights_cube.data = np.array(
            [0.2, 0.5, 0.3], dtype=np.float32)
        self.plugin = SpatiallyVaryingWeightsFromMask()

    def test_basic(self):
        """Test a basic cube multiplication with a 4D input cube"""
        expected_result = np.array([[[[0.2, 0, 0.2],
                                      [0.2, 0, 0.2]],
                                     [[0.2, 0, 0.2],
                                      [0.2, 0, 0.2]]],
                                    [[[0, 0, 0.5],
                                      [0, 0, 0.5]],
                                     [[0, 0, 0.5],
                                      [0, 0, 0.5]]],
                                    [[[0.3, 0.3, 0.3],
                                      [0.3, 0.3, 0.3]],
                                     [[0.3, 0.3, 0.3],
                                      [0.3, 0.3, 0.3]]]],
                                   dtype=np.float32)
        result = self.plugin.multiply_weights(
            self.spatial_weights_cube, self.one_dimensional_weights_cube,
            "forecast_reference_time")
        self.assertArrayAlmostEqual(result.data, expected_result)
        self.assertEqual(result.metadata, self.spatial_weights_cube.metadata)

    def test_3D_cube(self):
        """Test when we only have a 3D cube as input."""
        expected_result = np.array([[[0.2, 0, 0.2],
                                     [0.2, 0, 0.2]],
                                    [[0, 0, 0.5],
                                     [0, 0, 0.5]],
                                    [[0.3, 0.3, 0.3],
                                     [0.3, 0.3, 0.3]]],
                                   dtype=np.float32)
        result = self.plugin.multiply_weights(
            self.spatial_weights_cube[:, 0, :, :],
            self.one_dimensional_weights_cube,
            "forecast_reference_time")
        self.assertArrayAlmostEqual(result.data, expected_result)
        self.assertEqual(result.metadata,
                         self.spatial_weights_cube[:, 0, :, :].metadata)

    def test_2D_cube(self):
        """Test when we only have a 2D cube as input.
        This probably won't happen in reality, but check it does something
        sensible anyway."""
        expected_result = np.array([[0.2, 0, 0.2],
                                    [0.2, 0, 0.2]],
                                   dtype=np.float32)
        result = self.plugin.multiply_weights(
            self.spatial_weights_cube[0, 0, :, :],
            self.one_dimensional_weights_cube[0],
            "forecast_reference_time")
        self.assertArrayAlmostEqual(result.data, expected_result)
        self.assertEqual(result.metadata,
                         self.spatial_weights_cube[0, 0, :, :].metadata)

    def test_mismatching_cubes(self):
        """Test the input cubes don't match along the blend_coord dim"""
        message = ("The blend_coord forecast_reference_time does not "
                   "match on weights_from_mask and "
                   "one_dimensional_weights_cube")
        with self.assertRaisesRegex(ValueError, message):
            self.plugin.multiply_weights(
                self.spatial_weights_cube,
                self.one_dimensional_weights_cube[0],
                "forecast_reference_time")

    def test_wrong_coord(self):
        """Test when we try to multiply over a coordinate not in the weights
        cubes."""
        message = (
            "Expected to find exactly 1 model coordinate, but found none.")
        with self.assertRaisesRegex(CoordinateNotFoundError, message):
            self.plugin.multiply_weights(
                self.spatial_weights_cube, self.one_dimensional_weights_cube,
                "model")


class Test_normalised_masked_weights(IrisTest):
    """Test normalised_masked_weights method"""

    def setUp(self):
        """Set up a cube with 2 thresholds to test normalisation. We are
        testing normalising along the leading dimension in this cube."""
        thresholds = [10, 20]
        data = np.ones((2, 2, 3), dtype=np.float32)
        cycle1 = set_up_probability_cube(
            data, thresholds, spatial_grid="equalarea",
            time=datetime(2017, 11, 10, 4, 0),
            frt=datetime(2017, 11, 10, 0, 0),)
        cycle2 = set_up_probability_cube(
            data, thresholds, spatial_grid="equalarea",
            time=datetime(2017, 11, 10, 4, 0),
            frt=datetime(2017, 11, 10, 1, 0),)
        cycle3 = set_up_probability_cube(
            data, thresholds, spatial_grid="equalarea",
            time=datetime(2017, 11, 10, 4, 0),
            frt=datetime(2017, 11, 10, 2, 0),)
        self.spatial_weights_cube = CubeList(
            [cycle1, cycle2, cycle3]).merge_cube()
        self.spatial_weights_cube = squeeze(self.spatial_weights_cube)
        self.spatial_weights_cube.rename("weights")
        # This input array has 3 forecast reference times and 2 thresholds.
        # The two thresholds have the same weights.
        self.spatial_weights_cube.data = np.array([[[[0.2, 0, 0.2],
                                                     [0.2, 0, 0.2]],
                                                    [[0.2, 0, 0.2],
                                                     [0.2, 0, 0.2]]],
                                                   [[[0, 0, 0.5],
                                                     [0, 0, 0.5]],
                                                    [[0, 0, 0.5],
                                                     [0, 0, 0.5]]],
                                                   [[[0.3, 0.3, 0.3],
                                                     [0.3, 0.3, 0.3]],
                                                    [[0.3, 0.3, 0.3],
                                                     [0.3, 0.3, 0.3]]]],
                                                  dtype=np.float32)
        self.plugin = SpatiallyVaryingWeightsFromMask()

    @ManageWarnings(ignored_messages=[
        "Collapsing a non-contiguous coordinate."])
    def test_basic(self):
        """Test a basic example normalising along forecast_reference_time"""
        expected_result = np.array([[[[0.4, 0, 0.2],
                                      [0.4, 0, 0.2]],
                                     [[0.4, 0, 0.2],
                                      [0.4, 0, 0.2]]],
                                    [[[0, 0, 0.5],
                                      [0, 0, 0.5]],
                                     [[0, 0, 0.5],
                                      [0, 0, 0.5]]],
                                    [[[0.6, 1.0, 0.3],
                                      [0.6, 1.0, 0.3]],
                                     [[0.6, 1.0, 0.3],
                                      [0.6, 1.0, 0.3]]]],
                                   dtype=np.float32)
        result = self.plugin.normalised_masked_weights(
            self.spatial_weights_cube, "forecast_reference_time")
        self.assertArrayAlmostEqual(result.data, expected_result)
        self.assertEqual(result.metadata, self.spatial_weights_cube.metadata)

    @ManageWarnings(ignored_messages=[
        "Collapsing a non-contiguous coordinate."])
    def test_less_input_dims(self):
        """Test a smaller input cube"""
        expected_result = np.array([[[0.4, 0, 0.2],
                                     [0.4, 0, 0.2]],
                                    [[0, 0, 0.5],
                                     [0, 0, 0.5]],
                                    [[0.6, 1.0, 0.3],
                                     [0.6, 1.0, 0.3]]],
                                   dtype=np.float32)
        result = self.plugin.normalised_masked_weights(
            self.spatial_weights_cube[:, 0, :, :], "forecast_reference_time")
        self.assertArrayAlmostEqual(result.data, expected_result)
        self.assertEqual(result[:, 0, :, :].metadata,
                         self.spatial_weights_cube.metadata)

    @ManageWarnings(ignored_messages=[
        "Collapsing a non-contiguous coordinate."])
    def test_transpose_cube(self):
        """Test the function still works when we transpose the input cube.
           Same as test_basic except for transpose."""
        expected_result = np.array([[[[0.4, 0, 0.2],
                                      [0.4, 0, 0.2]],
                                     [[0.4, 0, 0.2],
                                      [0.4, 0, 0.2]]],
                                    [[[0, 0, 0.5],
                                      [0, 0, 0.5]],
                                     [[0, 0, 0.5],
                                      [0, 0, 0.5]]],
                                    [[[0.6, 1.0, 0.3],
                                      [0.6, 1.0, 0.3]],
                                     [[0.6, 1.0, 0.3],
                                      [0.6, 1.0, 0.3]]]],
                                   dtype=np.float32)
        # The function always puts the blend_coord as a leading dimension.
        # The process method will ensure the order of the output dimensions
        # matches those in the input.
        expected_result = np.transpose(expected_result, axes=[0, 3, 2, 1])
        self.spatial_weights_cube.transpose(new_order=[3, 2, 0, 1])
        result = self.plugin.normalised_masked_weights(
            self.spatial_weights_cube, "forecast_reference_time")
        self.assertArrayAlmostEqual(result.data, expected_result)
        self.assertEqual(result.metadata, self.spatial_weights_cube.metadata)

    @ManageWarnings(ignored_messages=[
        "Collapsing a non-contiguous coordinate."])
    def test_already_normalised(self):
        """Test nothing happens if the input data is already normalised."""
        self.spatial_weights_cube.data = np.array([[[[0.4, 0, 0.2],
                                                     [0.4, 0, 0.2]],
                                                    [[0.4, 0, 0.2],
                                                     [0.4, 0, 0.2]]],
                                                   [[[0, 0, 0.5],
                                                     [0, 0, 0.5]],
                                                    [[0, 0, 0.5],
                                                     [0, 0, 0.5]]],
                                                   [[[0.6, 1.0, 0.3],
                                                     [0.6, 1.0, 0.3]],
                                                    [[0.6, 1.0, 0.3],
                                                     [0.6, 1.0, 0.3]]]],
                                                  dtype=np.float32)
        result = self.plugin.normalised_masked_weights(
            self.spatial_weights_cube, "forecast_reference_time")
        self.assertArrayAlmostEqual(
            result.data, self.spatial_weights_cube.data)
        self.assertEqual(result.metadata, self.spatial_weights_cube.metadata)

    @ManageWarnings(ignored_messages=[
        "Collapsing a non-contiguous coordinate."])
    def test_weights_sum_to_zero(self):
        """Test all x-y slices have zero weight in the same index.
           This case corresponds to the case when all the input fields are
           all masked in the same place."""
        self.spatial_weights_cube.data[:, :, :, 0] = 0
        expected_result = np.array([[[[0, 0, 0.2],
                                      [0, 0, 0.2]],
                                     [[0, 0, 0.2],
                                      [0, 0, 0.2]]],
                                    [[[0, 0, 0.5],
                                      [0, 0, 0.5]],
                                     [[0, 0, 0.5],
                                      [0, 0, 0.5]]],
                                    [[[0, 1.0, 0.3],
                                      [0, 1.0, 0.3]],
                                     [[0, 1.0, 0.3],
                                      [0, 1.0, 0.3]]]],
                                   dtype=np.float32)
        result = self.plugin.normalised_masked_weights(
            self.spatial_weights_cube, "forecast_reference_time")
        self.assertArrayAlmostEqual(result.data, expected_result)
        self.assertEqual(result.metadata, self.spatial_weights_cube.metadata)


class Test_create_template_slice(IrisTest):
    """Test create_template_slice method"""

    def setUp(self):
        """
        Set up a basic input cube. Input cube has 2 thresholds on and 3
        forecast_reference_times
        """
        thresholds = [10, 20]
        data = np.ones((2, 2, 3), dtype=np.float32)
        cycle1 = set_up_probability_cube(
            data, thresholds, spatial_grid="equalarea",
            time=datetime(2017, 11, 10, 4, 0),
            frt=datetime(2017, 11, 10, 0, 0),)
        cycle2 = set_up_probability_cube(
            data, thresholds, spatial_grid="equalarea",
            time=datetime(2017, 11, 10, 4, 0),
            frt=datetime(2017, 11, 10, 1, 0),)
        cycle3 = set_up_probability_cube(
            data, thresholds, spatial_grid="equalarea",
            time=datetime(2017, 11, 10, 4, 0),
            frt=datetime(2017, 11, 10, 2, 0),)
        self.cube_to_collapse = CubeList(
            [cycle1, cycle2, cycle3]).merge_cube()
        self.cube_to_collapse = squeeze(self.cube_to_collapse)
        self.cube_to_collapse.rename("weights")
        # This input array has 3 forecast reference times and 2 thresholds.
        # The two thresholds have the same weights.
        self.cube_to_collapse.data = np.array([[[[1, 0, 1],
                                                 [1, 1, 1]],
                                                [[1, 0, 1],
                                                 [1, 1, 1]]],
                                               [[[0, 0, 1],
                                                 [0, 1, 1]],
                                                [[0, 0, 1],
                                                 [0, 1, 1]]],
                                               [[[1, 1, 1],
                                                 [1, 1, 1]],
                                                [[1, 1, 1],
                                                 [1, 1, 1]]]],
                                              dtype=np.float32)
        self.cube_to_collapse.data = np.ma.masked_equal(
            self.cube_to_collapse.data, 0)
        self.plugin = SpatiallyVaryingWeightsFromMask()

    def test_multi_dim_blend_coord_fail(self):
        """Test error is raised when we have a multi-dimensional blend_coord"""
        # Add a surface altitude coordinate which covers x and y dimensions.
        altitudes = np.array([[10, 20, 30],
                              [20, 30, 10]])
        altitudes_coord = AuxCoord(
            altitudes, standard_name="surface_altitude", units="m")
        self.cube_to_collapse.add_aux_coord(altitudes_coord, data_dims=(2, 3))
        message = ("Blend coordinate must only be across one dimension.")
        with self.assertRaisesRegex(ValueError, message):
            self.plugin.create_template_slice(
                self.cube_to_collapse, "surface_altitude")

    def test_varying_mask_fail(self):
        """Test error is raised when mask varies along collapsing dim"""
        # Check fails when blending along threshold coordinate, as mask
        # varies along this coordinate.
        threshold_coord = find_threshold_coordinate(self.cube_to_collapse)
        message = (
            "The mask on the input cube can only vary along the blend_coord")
        with self.assertRaisesRegex(ValueError, message):
            self.plugin.create_template_slice(
                self.cube_to_collapse, threshold_coord.name())

    def test_scalar_blend_coord_fail(self):
        """Test error is raised when blend_coord is scalar"""
        message = (
            "Blend coordinate must only be across one dimension.")
        with self.assertRaisesRegex(ValueError, message):
            self.plugin.create_template_slice(
                self.cube_to_collapse[0], "forecast_reference_time")

    def test_basic(self):
        """Test a correct template slice is returned for simple case"""
        expected = self.cube_to_collapse.copy()[:, 0, :, :]
        result = self.plugin.create_template_slice(
            self.cube_to_collapse, "forecast_reference_time")
        self.assertEqual(expected.metadata, result.metadata)
        self.assertArrayAlmostEqual(expected.data, result.data)

    def test_basic_no_change(self):
        """Test a correct template slice is returned for a case where
           no slicing is needed"""
        input_cube = self.cube_to_collapse.copy()[:, 0, :, :]
        expected = input_cube.copy()
        result = self.plugin.create_template_slice(
            self.cube_to_collapse, "forecast_reference_time")
        self.assertEqual(expected.metadata, result.metadata)
        self.assertArrayAlmostEqual(expected.data, result.data)

    def test_aux_blending_coord(self):
        """Test a correct template slice is returned when blending_coord is
           an AuxCoord"""
        expected = self.cube_to_collapse.copy()[:, 0, :, :]
        result = self.plugin.create_template_slice(
            self.cube_to_collapse, "forecast_period")
        self.assertEqual(expected.metadata, result.metadata)
        self.assertArrayAlmostEqual(expected.data, result.data)


class Test_process(IrisTest):
    """Test process method"""

    def setUp(self):
        """
        Set up a basic cube and linear weights cube for the process
        method. Input cube has 2 thresholds on and 3
        forecast_reference_times
        """
        thresholds = [10, 20]
        data = np.ones((2, 2, 3), dtype=np.float32)
        cycle1 = set_up_probability_cube(
            data, thresholds, spatial_grid="equalarea",
            time=datetime(2017, 11, 10, 4, 0),
            frt=datetime(2017, 11, 10, 0, 0),)
        cycle2 = set_up_probability_cube(
            data, thresholds, spatial_grid="equalarea",
            time=datetime(2017, 11, 10, 4, 0),
            frt=datetime(2017, 11, 10, 1, 0),)
        cycle3 = set_up_probability_cube(
            data, thresholds, spatial_grid="equalarea",
            time=datetime(2017, 11, 10, 4, 0),
            frt=datetime(2017, 11, 10, 2, 0),)
        self.cube_to_collapse = CubeList(
            [cycle1, cycle2, cycle3]).merge_cube()
        self.cube_to_collapse = squeeze(self.cube_to_collapse)
        self.cube_to_collapse.rename("weights")
        # This input array has 3 forecast reference times and 2 thresholds.
        # The two thresholds have the same weights.
        self.cube_to_collapse.data = np.array([[[[1, 0, 1],
                                                 [1, 1, 1]],
                                                [[1, 0, 1],
                                                 [1, 1, 1]]],
                                               [[[0, 0, 1],
                                                 [0, 1, 1]],
                                                [[0, 0, 1],
                                                 [0, 1, 1]]],
                                               [[[1, 1, 1],
                                                 [1, 1, 1]],
                                                [[1, 1, 1],
                                                 [1, 1, 1]]]],
                                              dtype=np.float32)
        self.cube_to_collapse.data = np.ma.masked_equal(
            self.cube_to_collapse.data, 0)
        # Create a one_dimensional weights cube by slicing the larger
        # weights cube.
        # The resulting cube only has a forecast_reference_time coordinate.
        self.one_dimensional_weights_cube = self.cube_to_collapse[:, 0, 0, 0]
        self.one_dimensional_weights_cube.remove_coord(
            "projection_x_coordinate")
        self.one_dimensional_weights_cube.remove_coord(
            "projection_y_coordinate")
        self.one_dimensional_weights_cube.remove_coord(
            find_threshold_coordinate(self.one_dimensional_weights_cube))
        self.one_dimensional_weights_cube.data = np.array(
            [0.2, 0.5, 0.3], dtype=np.float32)
        self.plugin = SpatiallyVaryingWeightsFromMask(fuzzy_length=4)

    @ManageWarnings(record=True)
    def test_none_masked(self, warning_list=None):
        """Test when we have no masked data in the input cube."""
        self.cube_to_collapse.data = np.ones(self.cube_to_collapse.data.shape)
        self.cube_to_collapse.data = np.ma.masked_equal(
            self.cube_to_collapse.data, 0)
        expected_data = np.array([[[0.2, 0.2, 0.2],
                                   [0.2, 0.2, 0.2]],
                                  [[0.5, 0.5, 0.5],
                                   [0.5, 0.5, 0.5]],
                                  [[0.3, 0.3, 0.3],
                                   [0.3, 0.3, 0.3]]],
                                 dtype=np.float32)
        message = ("Input cube to SpatiallyVaryingWeightsFromMask "
                   "must be masked")
        result = self.plugin.process(
            self.cube_to_collapse, self.one_dimensional_weights_cube,
            "forecast_reference_time")
        self.assertTrue(any(message in str(item)
                            for item in warning_list))
        self.assertArrayEqual(result.data, expected_data)
        self.assertEqual(result.dtype, np.float32)

    @ManageWarnings(ignored_messages=[
        "Collapsing a non-contiguous coordinate."])
    def test_all_masked(self):
        """Test when we have all masked data in the input cube."""
        self.cube_to_collapse.data = np.ones(self.cube_to_collapse.data.shape)
        self.cube_to_collapse.data = np.ma.masked_equal(
            self.cube_to_collapse.data, 1)
        result = self.plugin.process(
                self.cube_to_collapse, self.one_dimensional_weights_cube,
                "forecast_reference_time")
        expected_data = np.zeros((3, 2, 3))
        self.assertArrayAlmostEqual(expected_data, result.data)
        self.assertTrue(result.metadata, self.cube_to_collapse.data)

    @ManageWarnings(ignored_messages=[
        "Collapsing a non-contiguous coordinate."])
    def test_no_fuzziness_no_one_dimensional_weights(self):
        """Test a simple case where we have no fuzziness in the spatial
        weights and no adjustment from the one_dimensional weights."""
        plugin = SpatiallyVaryingWeightsFromMask(fuzzy_length=1)
        self.one_dimensional_weights_cube.data = np.ones((3))
        expected_result = np.array([[[0.5, 0., 0.33333333],
                                     [0.5, 0.33333333, 0.33333333]],
                                    [[0., 0., 0.33333333],
                                     [0., 0.33333333, 0.33333333]],
                                    [[0.5, 1., 0.33333333],
                                     [0.5, 0.33333333, 0.33333333]]],
                                   dtype=np.float32)
        result = plugin.process(
            self.cube_to_collapse, self.one_dimensional_weights_cube,
            "forecast_reference_time")
        self.assertArrayAlmostEqual(result.data, expected_result)
        self.assertEqual(result.metadata, self.cube_to_collapse.metadata)

    @ManageWarnings(ignored_messages=[
        "Collapsing a non-contiguous coordinate."])
    def test_no_fuzziness_no_one_dimensional_weights_transpose(self):
        """Test a simple case where we have no fuzziness in the spatial
        weights and no adjustment from the one_dimensional weights and
        transpose the input cube."""
        plugin = SpatiallyVaryingWeightsFromMask(fuzzy_length=1)
        self.one_dimensional_weights_cube.data = np.ones((3))
        expected_result = np.array([[[0.5, 0., 0.33333333],
                                     [0.5, 0.33333333, 0.33333333]],
                                    [[0., 0., 0.33333333],
                                     [0., 0.33333333, 0.33333333]],
                                    [[0.5, 1., 0.33333333],
                                     [0.5, 0.33333333, 0.33333333]]],
                                   dtype=np.float32)
        self.cube_to_collapse.transpose([2, 0, 1, 3])
        result = plugin.process(
            self.cube_to_collapse, self.one_dimensional_weights_cube,
            "forecast_reference_time")
        self.assertArrayAlmostEqual(result.data, expected_result)
        self.assertEqual(result.metadata, self.cube_to_collapse.metadata)

    @ManageWarnings(ignored_messages=[
        "Collapsing a non-contiguous coordinate."])
    def test_no_fuzziness_with_one_dimensional_weights(self):
        """Test a simple case where we have no fuzziness in the spatial
        weights and an adjustment from the one_dimensional weights."""
        plugin = SpatiallyVaryingWeightsFromMask(fuzzy_length=1)
        expected_result = np.array([[[0.4, 0., 0.2],
                                     [0.4, 0.2, 0.2]],
                                    [[0., 0., 0.5],
                                     [0., 0.5, 0.5]],
                                    [[0.6, 1., 0.3],
                                     [0.6, 0.3, 0.3]]],
                                   dtype=np.float32)
        result = plugin.process(
            self.cube_to_collapse, self.one_dimensional_weights_cube,
            "forecast_reference_time")
        self.assertArrayAlmostEqual(result.data, expected_result)
        self.assertEqual(result.metadata, self.cube_to_collapse.metadata)

    @ManageWarnings(ignored_messages=[
        "Collapsing a non-contiguous coordinate."])
    def test_fuzziness_no_one_dimensional_weights(self):
        """Test a simple case where we have some fuzziness in the spatial
        weights and no adjustment from the one_dimensional weights."""
        plugin = SpatiallyVaryingWeightsFromMask(fuzzy_length=2)
        self.one_dimensional_weights_cube.data = np.ones((3))
        expected_result = np.array([[[0.33333334, 0., 0.25],
                                     [0.41421354, 0.25, 0.2928932]],
                                    [[0., 0., 0.25],
                                     [0., 0.25, 0.2928932]],
                                    [[0.6666667, 1., 0.5],
                                     [0.5857864, 0.5, 0.41421354]]],
                                   dtype=np.float32)
        result = plugin.process(
            self.cube_to_collapse, self.one_dimensional_weights_cube,
            "forecast_reference_time")
        self.assertArrayAlmostEqual(result.data, expected_result)
        self.assertEqual(result.metadata, self.cube_to_collapse.metadata)

    @ManageWarnings(ignored_messages=[
        "Collapsing a non-contiguous coordinate."])
    def test_fuzziness_with_one_dimensional_weights(self):
        """Test a simple case where we have some fuzziness in the spatial
        sweights and with adjustment from the one_dimensional weights."""
        plugin = SpatiallyVaryingWeightsFromMask(fuzzy_length=2)
        expected_result = np.array([[[0.25, 0., 0.15384616],
                                     [0.32037723, 0.15384616, 0.17789416]],
                                    [[0., 0., 0.3846154],
                                     [0., 0.3846154, 0.44473538]],
                                    [[0.75, 1., 0.4615385],
                                     [0.6796227, 0.4615385, 0.3773705]]],
                                   dtype=np.float32)
        result = plugin.process(
            self.cube_to_collapse, self.one_dimensional_weights_cube,
            "forecast_reference_time")
        self.assertArrayAlmostEqual(result.data, expected_result)
        self.assertEqual(result.metadata, self.cube_to_collapse.metadata)


if __name__ == '__main__':
    unittest.main()
