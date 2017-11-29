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
"""Unit tests for the generate_ancillary.GenerateOrogBandAncils plugin."""


import unittest
from iris.cube import Cube
from iris.tests import IrisTest
import numpy as np

from improver.generate_ancillaries.generate_ancillary import (
    GenerateOrographyBandAncils as GenOrogMasks)


class Test_gen_orography_masks(IrisTest):
    """
    Test the gen_orography_masks method orography band mask
    ancillary generation plugin.
    """
    def setUp(self):
        """setting up test input and output data sets"""
        landmask_data = np.array([[1, 0, 0],
                                  [1, 0, 0],
                                  [1, 1, 1]])
        self.landmask = Cube(landmask_data, long_name='test land')
        orog_data = np.array([[10., 0., 0.],
                              [20., 100., 15.],
                              [-10., 100., 40.]])
        self.orography = Cube(orog_data, long_name='test orog')
        self.valley_key = 'land'
        self.valley_threshold = [-10, 10]
        self.exp_valleymask = np.array([[[1, 999999, 999999],
                                         [0, 999999, 999999],
                                         [0, 0, 0]]])
        self.land_key = 'land'
        self.land_threshold = [0, 50]
        self.exp_landmask = np.array([[[1., 999999, 999999],
                                       [1., 999999, 999999],
                                       [0., 0., 1.]]])

        self.nonzero_land_threshold = [30, 100]
        self.exp_nonzero_landmask = np.array([[[0, 999999, 999999],
                                               [0, 999999, 999999],
                                               [0, 1, 1]]])

        self.high_land_threshold = [500, 600]
        self.exp_high_landmask = np.array([[[0., 999999, 999999],
                                            [0., 999999, 999999],
                                            [0., 0., 0.]]])

    def test_nonsensekey(self):
        """test the correct exception is raised for unknown keys"""
        exception_dict = {'nonsense': [[0, 10]]}
        with self.assertRaisesRegexp(KeyError, 'Unknown threshold_dict key'):
            GenOrogMasks().gen_orography_masks(
                self.orography, self.landmask, "nonsense",
                exception_dict["nonsense"])

    def test_valleyband_data(self):
        """test correct mask is produced for land bands < 0m"""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask, self.valley_key,
            self.valley_threshold)
        self.assertArrayAlmostEqual(result.data.data, self.exp_valleymask)

    def test_valleyband_cube(self):
        """test correct cube data is produced for land bands < 0m"""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask, self.valley_key,
            self.valley_threshold)
        self.assertEqual(result.attributes['Topographical Type'], 'Land')
        self.assertEqual(result.coord('topographic_zone').points,
                         np.mean(self.valley_threshold))
        self.assertEqual(result.coord('topographic_zone').bounds[0][0],
                         self.valley_threshold[0])
        self.assertEqual(result.coord('topographic_zone').bounds[0][1],
                         self.valley_threshold[1])

    def test_landband_data(self):
        """test correct mask is produced for land bands > 0m"""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask, self.land_key,
            self.land_threshold)
        self.assertArrayAlmostEqual(result.data.data, self.exp_landmask)

    def test_landband_cube(self):
        """test correct cube data is produced for land bands > 0m"""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask, self.land_key,
            self.land_threshold)
        self.assertEqual(result.attributes['Topographical Type'], 'Land')
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
            self.orography, self.landmask, self.land_key,
            self.nonzero_land_threshold)
        self.assertArrayAlmostEqual(result.data.data,
                                    self.exp_nonzero_landmask)

    def test_nonzero_landband_cube(self):
        """test that a correct cube is produced when neither landband
        bound is zero."""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask, self.land_key,
            self.nonzero_land_threshold)
        self.assertEqual(result.coord('topographic_zone').points,
                         np.mean(self.nonzero_land_threshold))

    def test_high_landband_cube(self):
        """test that a correct cube is produced when the land band is
        higher than any land in the test cube."""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask, self.land_key,
            self.high_land_threshold)
        self.assertEqual(result.coord('topographic_zone').points,
                         np.mean(self.high_land_threshold))

    def test_high_landband_data(self):
        """test that a correct mask is produced when the land band is
        higher than any land in the test cube."""
        result = GenOrogMasks().gen_orography_masks(
            self.orography, self.landmask, self.land_key,
            self.high_land_threshold)
        self.assertArrayAlmostEqual(result.data.data, self.exp_high_landmask)

    def test_nothreshold(self):
        """test the correct exception is raised for key without threshold"""
        key = 'land'
        threshold = []
        with self.assertRaises(ValueError):
            GenOrogMasks().gen_orography_masks(
                self.orography, self.landmask, key, threshold)

    def test_all_land_points(self):
        """Test that a correct mask is produced when the landsea mask only has
           land points in it."""
        land_mask_cube = self.landmask.copy()
        land_mask_cube.data = np.ones((3, 3))
        result = GenOrogMasks().gen_orography_masks(
            self.orography, land_mask_cube, self.land_key,
            self.valley_threshold)
        expected_data = np.array([[[1.0, 1.0, 1.0],
                                   [0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0]]])
        self.assertArrayAlmostEqual(result.data, expected_data)


class Test_process(IrisTest):
    """
    Test the process method orography zone mask ancillary generation plugin.
    """

    def setUp(self):
        """setting up test input and output data sets"""
        landmask_data = np.array([[1, 0, 0],
                                  [1, 0, 0],
                                  [1, 1, 1]])
        self.landmask = Cube(landmask_data, long_name='test land')
        orog_data = np.array([[10., 0., 0.],
                              [20., 100., 15.],
                              [-10., 100., 40.]])
        self.orography = Cube(orog_data, long_name='test orog')
        self.threshold_dict = {'land': {'bounds': [[-10, 0], [0, 50]],
                                        'units': 'm'}}

    def test_thresholdset(self):
        """test the plugin produces correct number of cubes"""
        result = GenOrogMasks().process(
            self.orography, self.landmask, self.threshold_dict)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
