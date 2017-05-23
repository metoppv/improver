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
from iris.exceptions import CoordinateNotFoundError
import numpy as np

from improver.generate_ancillary import GenerateOrographyBandAncils \
    as GenOrogMasks


class TestGenAncil(IrisTest):
    """Test the orography band mask ancillary generation plugin."""

    def setUp(self):
        """setting up test input and output data sets"""
        landmask_data = np.array([[0.25, 0., 0.],
                                  [0.75, 0.25, 0.],
                                  [1., 1., 0.75]])
        self.landmask = Cube(landmask_data, long_name='test land')
        orog_data = np.array([[10., 0., 0.],
                              [20., 100., 15.],
                              [-10., 100., 40.]])
        self.orography = Cube(orog_data, long_name='test orog')
        self.threshold_dict = {'land': [[-10, 0], [0, 50]],
                               'max land threshold': [[80]]}
        self.valley_dict = {'land': [[-10, 10]]}
        self.exp_valleymask = np.array([[1., 999999., 999999.],
                                        [0., 0., 999999.],
                                        [1., 0., 0.]])
        self.land_dict = {'land': [[0, 50]]}
        self.exp_landmask = np.array([[1., 999999., 999999.],
                                      [1., 0., 999999.],
                                      [0., 0., 1.]])
        self.max_dict = {'max land threshold': [[80]]}
        self.exp_maxmask = np.array([[0., 999999., 999999.],
                                     [0., 1., 999999.],
                                     [0., 1., 0.]])

    def test_thresholdset(self):
        """test the plugin produces correct number of cubes"""
        result = GenOrogMasks().process(self.orography,
                                        self.landmask,
                                        self.threshold_dict)
        self.assertEqual(len(result), 3)

    def test_valleyband_data(self):
        """test correct mask is produced for land bands < 0m"""
        result = GenOrogMasks().process(self.orography,
                                        self.landmask,
                                        self.valley_dict)[0]
        self.assertArrayAlmostEqual(result.data.data, self.exp_valleymask)

    def test_valleyband_cube(self):
        """test correct cube data is produced for land bands < 0m"""
        result = GenOrogMasks().process(self.orography,
                                        self.landmask,
                                        self.valley_dict)[0]
        self.assertEqual(result.attributes['Topographical Type'], 'Land')
        self.assertEqual(result.coord('topographic_bound_lower').points,
                         self.valley_dict['land'][0][0])
        self.assertEqual(result.coord('topographic_bound_upper').points,
                         self.valley_dict['land'][0][1])

    def test_landband_data(self):
        """test correct mask is produced for land bands > 0m"""
        result = GenOrogMasks().process(self.orography,
                                        self.landmask,
                                        self.land_dict)[0]
        self.assertArrayAlmostEqual(result.data.data, self.exp_landmask)

    def test_landband_cube(self):
        """test correct cube data is produced for land bands > 0m"""
        result = GenOrogMasks().process(self.orography,
                                        self.landmask,
                                        self.land_dict)[0]
        self.assertEqual(result.attributes['Topographical Type'], 'Land')
        self.assertEqual(result.coord('topographic_bound_lower').points,
                         self.land_dict['land'][0][0])
        self.assertEqual(result.coord('topographic_bound_upper').points,
                         self.land_dict['land'][0][1])

    def test_maxband_data(self):
        """test correct mask is produced for land bands > max"""
        result = GenOrogMasks().process(self.orography,
                                        self.landmask,
                                        self.max_dict)[0]
        self.assertArrayAlmostEqual(result.data.data, self.exp_maxmask)

    def test_maxband_cube(self):
        """test correct cube data is produced for land bands > max"""
        result = GenOrogMasks().process(self.orography,
                                        self.landmask,
                                        self.max_dict)[0]
        self.assertEqual(result.attributes['Topographical Type'],
                         'Max Land Threshold')
        self.assertEqual(result.coord('topographic_bound_lower').points,
                         self.max_dict['max land threshold'][0][0])
        msg = 'Expected to find exactly 1  coordinate, but found none.'
        with self.assertRaisesRegexp(CoordinateNotFoundError, msg):
            result.coord('topographic_bound_upper')

    def test_nonsensekey(self):
        """test the correct exception is raised for unknown keys"""
        exception_dict = {'nonsense': [[0, 10]]}
        with self.assertRaisesRegexp(KeyError, 'Unknown threshold_dict key'):
            result = GenOrogMasks().process(self.orography,
                                            self.landmask,
                                            exception_dict)[0]

    def test_nothreshold(self):
        """test the correct exception is raised for key without threshold"""
        exception_dict = {'land': []}
        with self.assertRaises(ValueError):
            result = GenOrogMasks().process(self.orography,
                                            self.landmask,
                                            exception_dict)[0]
