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
"""Unit tests for the GenerateTopographicZoneWeights plugin."""

import unittest

import iris
import numpy as np
from cf_units import Unit
from iris.exceptions import InvalidCubeError
from iris.tests import IrisTest

from improver.generate_ancillaries.generate_topographic_zone_weights import (
    GenerateTopographicZoneWeights)
from improver.utilities.warnings_handler import ManageWarnings

from ..set_up_test_cubes import set_up_variable_cube


def set_up_orography_cube(data):
    """
    Set up a static orography cube using the centralised cube
    setup utility but removing all time coordinates

    Args:
        data (numpy.ndarray):
            Orography data to populate the cube
    """
    orography = set_up_variable_cube(
        data.astype(np.float32), name="altitude", units="m")
    for coord in ["time", "forecast_reference_time",
                  "forecast_period"]:
        orography.remove_coord(coord)
    return orography


class Test_add_weight_to_upper_adjacent_band(IrisTest):
    """Test for adding weights to the upper adjacent band."""

    def setUp(self):
        """Set up plugin."""
        self.plugin = GenerateTopographicZoneWeights()

    def test_equal_to_max_band_number(self):
        """Test that the results are as expected when the band number is equal
        to the max band number."""
        expected_weights = np.array([[[0.0, 0.0],
                                      [1.0, 1.0]]])

        topographic_zone_weights = np.zeros((1, 2, 2))
        orography_band = np.array([[25., 50.],
                                   [75., 100.]])
        midpoint = 50.
        band_number = 0
        max_band_number = 0
        topographic_zone_weights = (
            self.plugin.add_weight_to_upper_adjacent_band(
                topographic_zone_weights, orography_band, midpoint,
                band_number, max_band_number))
        self.assertIsInstance(topographic_zone_weights, np.ndarray)
        self.assertArrayAlmostEqual(topographic_zone_weights, expected_weights)

    def test_not_equal_to_max_band_number(self):
        """Test that the results are as expected when the band number is not
        equal to the max band number."""
        expected_weights = np.array([[[1.0, 1.0],
                                      [0.75, 0.5]],
                                     [[0.0, 0.0],
                                      [0.25, 0.5]]])
        topographic_zone_weights = np.array([[[1.0, 1.0],
                                              [0.75, 0.5]],
                                             [[0.0, 0.0],
                                              [0.0, 0.0]]])
        orography_band = np.array([[25., 50.],
                                   [75., 100.]])
        midpoint = 50.
        band_number = 0
        max_band_number = 1
        topographic_zone_weights = (
            self.plugin.add_weight_to_upper_adjacent_band(
                topographic_zone_weights, orography_band, midpoint,
                band_number, max_band_number))
        self.assertIsInstance(topographic_zone_weights, np.ndarray)
        self.assertArrayAlmostEqual(topographic_zone_weights, expected_weights)

    def test_none_above_midpoint(self):
        """Test that the results are as expected when none of the points
        are above the midpoint."""
        expected_weights = np.array([[[0.75, 1.0],
                                      [0.65, 0.7]],
                                     [[0., 0.],
                                      [0., 0.]]])
        topographic_zone_weights = np.array([[[0.75, 1.0],
                                              [0.65, 0.7]],
                                             [[0.0, 0.0],
                                              [0.0, 0.0]]])
        orography_band = np.array([[25., 50.],
                                   [15., 30.]])
        midpoint = 50.
        band_number = 0
        max_band_number = 1
        topographic_zone_weights = (
            self.plugin.add_weight_to_upper_adjacent_band(
                topographic_zone_weights, orography_band, midpoint,
                band_number, max_band_number))
        self.assertIsInstance(topographic_zone_weights, np.ndarray)
        self.assertArrayAlmostEqual(topographic_zone_weights, expected_weights)

    def test_all_above_midpoint(self):
        """Test that the results are as expected when all of the points
        are above the midpoint."""
        expected_weights = np.array([[[0.75, 0.7],
                                      [0.6, 0.55]],
                                     [[0.25, 0.3],
                                      [0.4, 0.45]]])
        topographic_zone_weights = np.array([[[0.75, 0.7],
                                              [0.6, 0.55]],
                                             [[0.0, 0.0],
                                              [0.0, 0.0]]])
        orography_band = np.array([[75., 80.],
                                   [90., 95.]])
        midpoint = 50.
        band_number = 0
        max_band_number = 1
        topographic_zone_weights = (
            self.plugin.add_weight_to_upper_adjacent_band(
                topographic_zone_weights, orography_band, midpoint,
                band_number, max_band_number))
        self.assertIsInstance(topographic_zone_weights, np.ndarray)
        self.assertArrayAlmostEqual(topographic_zone_weights, expected_weights)


class Test_add_weight_to_lower_adjacent_band(IrisTest):
    """Test for adding weights to the lower adjacent band."""

    def setUp(self):
        """Set up plugin."""
        self.plugin = GenerateTopographicZoneWeights()

    def test_equal_to_zeroth_band_number(self):
        """Test that the results are as expected when the band number is equal
        to the zeroth band."""
        expected_weights = np.array([[[1.0, 0.0],
                                      [0.0, 0.0]]])

        topographic_zone_weights = np.zeros((1, 2, 2))
        orography_band = np.array([[25., 50.],
                                   [75., 100.]])
        midpoint = 50.
        band_number = 0
        topographic_zone_weights = (
            self.plugin.add_weight_to_lower_adjacent_band(
                topographic_zone_weights, orography_band, midpoint,
                band_number))
        self.assertIsInstance(topographic_zone_weights, np.ndarray)
        self.assertArrayAlmostEqual(topographic_zone_weights, expected_weights)

    def test_not_equal_to_zeroth_band_number(self):
        """Test that the results are as expected when the band number is not
        equal to the zeroth band number."""
        expected_weights = np.array([[[0.25, 0.0],
                                      [0.0, 0.0]],
                                     [[0.75, 1.0],
                                      [1.0, 1.0]]])
        topographic_zone_weights = np.array([[[0.0, 0.0],
                                              [0.0, 0.0]],
                                             [[0.75, 1.0],
                                              [1.0, 1.0]]])
        orography_band = np.array([[25., 50.],
                                   [75., 100.]])
        midpoint = 50.
        band_number = 1
        topographic_zone_weights = (
            self.plugin.add_weight_to_lower_adjacent_band(
                topographic_zone_weights, orography_band, midpoint,
                band_number))
        self.assertIsInstance(topographic_zone_weights, np.ndarray)
        self.assertArrayAlmostEqual(topographic_zone_weights, expected_weights)

    def test_none_below_midpoint(self):
        """Test that the results are as expected when none of the points
        are below the midpoint."""
        expected_weights = np.array([[[0.0, 0.0],
                                      [0.0, 0.0]],
                                     [[1.0, 1.0],
                                      [1.0, 1.0]]])
        topographic_zone_weights = np.array([[[0.0, 0.0],
                                              [0.0, 0.0]],
                                             [[1.0, 1.0],
                                              [1.0, 1.0]]])
        orography_band = np.array([[75., 50.],
                                   [85., 70.]])
        midpoint = 50.
        band_number = 1
        topographic_zone_weights = (
            self.plugin.add_weight_to_lower_adjacent_band(
                topographic_zone_weights, orography_band, midpoint,
                band_number))
        self.assertIsInstance(topographic_zone_weights, np.ndarray)
        self.assertArrayAlmostEqual(topographic_zone_weights, expected_weights)

    def test_all_below_midpoint(self):
        """Test that the results are as expected when all of the points
        are below the midpoint."""
        expected_weights = np.array([[[0.25, 0.3],
                                      [0.4, 0.45]],
                                     [[0.75, 0.7],
                                      [0.6, 0.55]]])
        topographic_zone_weights = np.array([[[0.0, 0.0],
                                              [0.0, 0.0]],
                                             [[0.75, 0.7],
                                              [0.6, 0.55]]])
        orography_band = np.array([[25., 20.],
                                   [10., 5.]])
        midpoint = 50.
        band_number = 1
        topographic_zone_weights = (
            self.plugin.add_weight_to_lower_adjacent_band(
                topographic_zone_weights, orography_band, midpoint,
                band_number))
        self.assertIsInstance(topographic_zone_weights, np.ndarray)
        self.assertArrayAlmostEqual(topographic_zone_weights, expected_weights)


class Test_calculate_weights(IrisTest):
    """Test the calculation of weights."""

    def setUp(self):
        """Set up plugin."""
        self.plugin = GenerateTopographicZoneWeights()

    def test_one_point(self):
        """Test when the input array has one point."""
        expected = np.array([0.75])
        points = np.array([125])
        band = [100, 200]
        result = self.plugin.calculate_weights(points, band)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected)

    def test_multiple_points_matching_points(self):
        """Test when the input array has multiple points, which match the
        midpoint and band limits."""
        expected = np.array([0.5, 1.0, 0.5])
        points = np.array([100, 150, 200])
        band = [100, 200]
        result = self.plugin.calculate_weights(points, band)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected)

    def test_multiple_points_with_points_not_matching(self):
        """Test when the input array has multiple points, which do not match
        the midpoint and band limits."""
        expected = np.array([0.6, 0.9, 0.6])
        points = np.array([110, 140, 190])
        band = [100, 200]
        result = self.plugin.calculate_weights(points, band)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected)

    def test_point_beyond_bands(self):
        """Test when the input array has points beyond the band limits.
        The default behaviour is for values beyond the band limits to be 0.5,
        as 0.5 is the value at the band limits."""
        expected = np.array([0.5, 1.0, 0.5])
        points = np.array([90, 150, 210])
        band = [100, 200]
        result = self.plugin.calculate_weights(points, band)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected)


class Test_process(IrisTest):
    """Test the process method."""

    def setUp(self):
        """Set up data for testing."""
        self.plugin = GenerateTopographicZoneWeights()
        orography_data = np.array([[10., 25.],
                                   [75., 100.]])
        self.orography = set_up_orography_cube(orography_data)

        landmask_data = np.array([[0, 1],
                                  [1, 1]], dtype=np.float32)
        landmask = self.orography.copy(data=landmask_data)
        landmask.rename("land_binary_mask")
        landmask.units = Unit("1")
        self.landmask = landmask
        self.thresholds_dict = {'bounds': [[0, 50], [50, 200]], 'units': 'm'}

    def test_basic(self):
        """Test that the output is a cube with the expected format."""
        result = self.plugin.process(
            self.orography, self.thresholds_dict, self.landmask)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), "topographic_zone_weights")
        self.assertEqual(result.units, Unit("1"))
        self.assertTrue(result.coord("topographic_zone"))
        self.assertEqual(result.coord("topographic_zone").units, Unit("m"))

    def test_invalid_orography(self):
        """Test that the appropriate exception is raised if the orography has
        more than two dimensions."""
        orography_data = np.array([[[0., 25.],
                                    [75., 100.]]])
        orography = set_up_orography_cube(orography_data)
        msg = "The input orography cube should be two-dimensional"
        with self.assertRaisesRegex(InvalidCubeError, msg):
            self.plugin.process(orography, self.thresholds_dict, self.landmask)

    def test_data(self):
        """Test that the result data and mask is as expected."""
        expected_weights_data = np.array([[[1e20, 1.0],
                                           [0.33, 0.17]],
                                          [[1e20, 0.0],
                                           [0.67, 0.83]]],
                                         dtype=np.float32)
        expected_weights_mask = np.array([[[True, False],
                                           [False, False]],
                                          [[True, False],
                                           [False, False]]])
        result = self.plugin.process(
            self.orography, self.thresholds_dict, self.landmask)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.data.data, expected_weights_data, decimal=2)
        self.assertArrayAlmostEqual(result.data.mask, expected_weights_mask)

    def test_data_no_mask(self):
        """Test that the result data is as expected, when none of the points
        are masked."""
        expected_weights_data = np.array([[[1.0, 1.0],
                                           [0.33, 0.17]],
                                          [[0.0, 0.0],
                                           [0.67, 0.83]]],
                                         dtype=np.float32)
        landmask_data = np.array([[1, 1],
                                  [1, 1]])
        landmask = self.landmask.copy(landmask_data)
        result = self.plugin.process(
            self.orography, self.thresholds_dict, landmask)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.data, expected_weights_data, decimal=2)

    def test_data_no_mask_input(self):
        """Test that the result data is as expected, when no landsea
           mask is input."""
        expected_weights_data = np.array([[[1.0, 1.0],
                                           [0.33, 0.17]],
                                          [[0.0, 0.0],
                                           [0.67, 0.83]]],
                                         dtype=np.float32)
        result = self.plugin.process(
            self.orography, self.thresholds_dict)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.data, expected_weights_data, decimal=2)

    def test_data_no_mask_input_metatdata(self):
        """Test that the result metadata is as expected, when no landsea
           mask is input."""
        result = self.plugin.process(
            self.orography, self.thresholds_dict)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(
            result.attributes["topographic_zones_include_seapoints"], "True")

    def test_data_no_mask_three_bands(self):
        """Test that the result data is as expected, when none of the points
        are masked and there are three bands defined."""
        orography_data = np.array([[10., 40., 45.],
                                   [70., 80., 95.],
                                   [115., 135., 145.]])
        orography = set_up_orography_cube(orography_data)

        landmask_data = np.array([[1, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]])
        landmask = orography.copy(data=landmask_data)
        landmask.rename("land_binary_mask")
        landmask.units = Unit("1")

        thresholds_dict = {'bounds': [[0, 50], [50, 100], [100, 150]],
                           'units': 'm'}
        expected_weights_data = np.array([[[1.0, 0.7, 0.6],
                                           [0.1, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]],
                                          [[0.0, 0.3, 0.4],
                                           [0.9, 0.9, 0.6],
                                           [0.2, 0.0, 0.0]],
                                          [[0.0, 0.0, 0.0],
                                           [0.0, 0.1, 0.4],
                                           [0.8, 1.0, 1.0]]])
        result = self.plugin.process(orography, thresholds_dict, landmask)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.data, expected_weights_data, decimal=2)

    def test_different_band_units(self):
        """Test for if the thresholds are specified in a different unit to
        the orography. The thresholds are converted to match the units of the
        orography."""
        expected_weights_data = np.array([[[1e20, 1.0],
                                           [0.333, 0.167]],
                                          [[1e20, 0.0],
                                           [0.67, 0.83]]], dtype=np.float32)
        expected_weights_mask = np.array([[[True, False],
                                           [False, False]],
                                          [[True, False],
                                           [False, False]]])
        thresholds_dict = {'bounds': [[0, 0.05], [0.05, 0.2]],
                           'units': 'km'}
        result = self.plugin.process(
            self.orography, thresholds_dict, self.landmask)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.data.data, expected_weights_data, decimal=2)
        self.assertArrayAlmostEqual(result.data.mask, expected_weights_mask)

    def test_one_band_with_orography_in_band(self):
        """Test that if only one band is specified, the results are as
        expected."""
        expected_weights_data = np.array([[1e20, 1.0],
                                          [1.0, 1.0]], dtype=np.float32)
        expected_weights_mask = np.array([[True, False],
                                          [False, False]])
        orography_data = np.array([[10., 20.],
                                   [30., 40.]], dtype=np.float32)
        orography = self.orography.copy(data=orography_data)
        thresholds_dict = {'bounds': [[0, 50]], 'units': 'm'}
        result = self.plugin.process(
            orography, thresholds_dict, self.landmask)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(
            result.data.data, expected_weights_data, decimal=2)
        self.assertArrayAlmostEqual(result.data.mask, expected_weights_mask)

    @ManageWarnings(record=True)
    def test_warning_if_orography_above_bands(self, warning_list=None):
        """Test that a warning is raised if the orography is greater than the
        maximum band."""
        orography_data = np.array([[60., 70.],
                                   [80., 90.]])
        orography = self.orography.copy(data=orography_data)
        thresholds_dict = {'bounds': [[0, 50]], 'units': 'm'}
        msg = "The maximum orography is greater than the uppermost band"
        self.plugin.process(orography, thresholds_dict, self.landmask)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(msg in str(item)
                            for item in warning_list))

    @ManageWarnings(record=True)
    def test_warning_if_orography_below_bands(self, warning_list=None):
        """Test that a warning is raised if the orography is lower than the
        minimum band."""
        orography_data = np.array([[60., 70.],
                                   [80., 90.]])
        orography = self.orography.copy(data=orography_data)
        thresholds_dict = {'bounds': [[100, 150]], 'units': 'm'}
        msg = "The minimum orography is lower than the lowest band"
        self.plugin.process(orography, thresholds_dict, self.landmask)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(msg in str(item)
                            for item in warning_list))


if __name__ == "__main__":
    unittest.main()
