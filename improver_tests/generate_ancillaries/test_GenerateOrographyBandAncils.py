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
"""Unit tests for the generate_ancillary.GenerateOrogBandAncils plugin."""

import unittest

import numpy as np
from cf_units import Unit
from iris.cube import Cube
from iris.tests import IrisTest

from improver.generate_ancillaries.generate_ancillary import \
    GenerateOrographyBandAncils as GenOrogMasks


def set_up_landmask_cube(landmask_data=None):
    """Set up a basic landmask cube."""
    if landmask_data is None:
        landmask_data = np.array([[1, 0, 0],
                                  [1, 0, 0],
                                  [1, 1, 1]])
    return Cube(landmask_data, long_name='test land', units="1")


def set_up_orography_cube(orog_data=None):
    """Set up a basic orography cube."""
    if orog_data is None:
        orog_data = np.array([[10., 0., 0.],
                              [20., 100., 15.],
                              [-10., 100., 40.]])
    return Cube(orog_data, long_name='test orog', units="m")


class Test_sea_mask(IrisTest):
    """Test the masking out of sea points with the sea_mask method."""

    def setUp(self):
        """Set up for tests."""
        self.landmask = set_up_landmask_cube()
        self.orography = set_up_orography_cube()

    def test_basic(self):
        """Test that the expected data is returned when the landmask specifies
        a mix of land and sea points."""
        expected_data = np.array([[10., 1e20, 1e20],
                                  [20., 1e20, 1e20],
                                  [-10, 100., 40]])
        expected_mask = np.array([[False, True, True],
                                  [False, True, True],
                                  [False, False, False]])
        result = GenOrogMasks().sea_mask(
            self.landmask.data, self.orography.data)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertArrayAlmostEqual(result.mask, expected_mask)

    def test_basic_sea_fill_value(self):
        """Test that the expected data is returned when the landmask specifies
        a mix of land and sea points and a fill value is given."""
        expected_data = np.array([[10., 0, 0],
                                  [20., 0, 0],
                                  [-10, 100., 40]])
        result = GenOrogMasks().sea_mask(
            self.landmask.data, self.orography.data, sea_fill_value=0)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected_data)
        self.assertEqual(np.ma.is_masked(result), False)

    def test_all_land_points(self):
        """Test that the expected data is returned when the landmask specifies
        only land points."""
        expected = np.array([[10., 0., 0.],
                             [20., 100., 15.],
                             [-10., 100., 40.]])
        landmask_data = np.array([[1, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]])
        landmask = set_up_landmask_cube(landmask_data=landmask_data)
        result = GenOrogMasks().sea_mask(landmask.data, self.orography.data)
        self.assertIsInstance(result, np.ndarray)
        self.assertArrayAlmostEqual(result, expected)


class Test_gen_orography_masks(IrisTest):
    """
    Test the gen_orography_masks method orography band mask
    ancillary generation plugin.
    """
    def setUp(self):
        """setting up test input and output data sets"""
        self.landmask = set_up_landmask_cube()
        self.orography = set_up_orography_cube()
        self.valley_threshold = [-10, 10]
        self.exp_valleymask = np.array([[[1, 0, 0],
                                         [0, 0, 0],
                                         [0, 0, 0]]])
        self.land_threshold = [0, 50]
        self.exp_landmask = np.array([[[1., 0, 0],
                                       [1., 0, 0],
                                       [0., 0., 1.]]])

        self.nonzero_land_threshold = [30, 100]
        self.exp_nonzero_landmask = np.array([[[0, 0, 0],
                                               [0, 0, 0],
                                               [0, 1, 1]]])

        self.high_land_threshold = [500, 600]
        self.exp_high_landmask = np.array([[[0., 0, 0],
                                            [0., 0, 0],
                                            [0., 0., 0.]]])

    def test_valleyband_data(self):
        """test correct mask is produced for land bands < 0m"""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask,
            self.valley_threshold)
        self.assertArrayAlmostEqual(result.data, self.exp_valleymask)

    def test_valleyband_cube(self):
        """test correct cube data is produced for land bands < 0m"""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask,
            self.valley_threshold)
        self.assertEqual(
            result.attributes['topographic_zones_include_seapoints'], "False")
        self.assertEqual(result.coord('topographic_zone').points,
                         np.mean(self.valley_threshold))
        self.assertEqual(result.coord('topographic_zone').bounds[0][0],
                         self.valley_threshold[0])
        self.assertEqual(result.coord('topographic_zone').bounds[0][1],
                         self.valley_threshold[1])

    def test_landband_data(self):
        """test correct mask is produced for land bands > 0m"""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask,
            self.land_threshold)
        self.assertArrayAlmostEqual(result.data, self.exp_landmask)

    def test_landband_cube(self):
        """test correct cube data is produced for land bands > 0m"""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask,
            self.land_threshold)
        self.assertEqual(
            result.attributes['topographic_zones_include_seapoints'], "False")
        self.assertEqual(result.coord('topographic_zone').points,
                         np.mean(self.land_threshold))
        self.assertEqual(result.coord('topographic_zone').bounds[0][0],
                         self.land_threshold[0])
        self.assertEqual(result.coord('topographic_zone').bounds[0][1],
                         self.land_threshold[1])

    def test_nonzero_landband_data(self):
        """test that correct data is produced when neither landband
        bound is zero."""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask,
            self.nonzero_land_threshold)
        self.assertArrayAlmostEqual(result.data,
                                    self.exp_nonzero_landmask)

    def test_nonzero_landband_cube(self):
        """test that a correct cube is produced when neither landband
        bound is zero."""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask,
            self.nonzero_land_threshold)
        self.assertEqual(result.coord('topographic_zone').points,
                         np.mean(self.nonzero_land_threshold))

    def test_high_landband_cube(self):
        """test that a correct cube is produced when the land band is
        higher than any land in the test cube."""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask,
            self.high_land_threshold)
        self.assertEqual(result.coord('topographic_zone').points,
                         np.mean(self.high_land_threshold))

    def test_high_landband_data(self):
        """test that a correct mask is produced when the land band is
        higher than any land in the test cube."""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask,
            self.high_land_threshold)
        self.assertArrayAlmostEqual(result.data, self.exp_high_landmask)

    def test_all_land_points(self):
        """Test that a correct mask is produced when the landsea mask only has
           land points in it."""
        land_mask_cube = self.landmask.copy()
        land_mask_cube.data = np.ones((3, 3))
        result = GenOrogMasks().gen_orography_masks(
            self.orography, land_mask_cube,
            self.valley_threshold)
        expected_data = np.array([[[1.0, 1.0, 1.0],
                                   [0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0]]])
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_any_surface_type_mask(self):
        """Test that the correct mask is produced when no landsea mask is
           provided. This is equivalent to the all_land_points test above."""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, None,
            self.valley_threshold)
        expected_data = np.array([[[1.0, 1.0, 1.0],
                                   [0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0]]])
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertEqual(
            result.attributes['topographic_zones_include_seapoints'], "True")

    def test_unit_conversion_for_landband_data(self):
        """test correct mask is produced for land bands > 0m"""
        land_threshold = [0, 0.05]
        threshold_units = "km"
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask,
            land_threshold, units=threshold_units)
        self.assertArrayAlmostEqual(result.data, self.exp_landmask)
        self.assertEqual(result.coord("topographic_zone").units, Unit("m"))


class Test_process(IrisTest):
    """
    Test the process method orography zone mask ancillary generation plugin.
    """

    def setUp(self):
        """setting up test input and output data sets"""
        self.landmask = set_up_landmask_cube()
        self.orography = set_up_orography_cube()
        self.threshold_dict = {'bounds': [[-10, 0], [0, 50]], 'units': 'm'}

    def test_thresholdset(self):
        """test the plugin produces correct number of cubes"""
        result = GenOrogMasks().process(
            self.orography, self.threshold_dict, landmask=self.landmask)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
