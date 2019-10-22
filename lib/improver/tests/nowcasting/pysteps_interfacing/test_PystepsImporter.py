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
"""Unit tests for pysteps plotting interface"""

import unittest
import numpy as np

from iris.tests import IrisTest

from improver.nowcasting.pysteps_interfacing import PystepsImporter
from improver.tests.set_up_test_cubes import set_up_variable_cube


class Test__init__(IrisTest):
    """Test the __init__ function"""

    def test_basic(self):
        """Test initialisation behaves as expected"""
        plugin = PystepsImporter()
        self.assertEqual(plugin.metadata['unit'], 'mm/h')
        self.assertEqual(plugin.metadata['accutime'], 0)
        self.assertIsNone(plugin.metadata['transform'])


class Test_process_cube(IrisTest):
    """Class to test the process_cube method"""

    def setUp(self):
        """Set up test cube to import"""
        precip_data_ms = 1e-7*np.ones((5, 5), dtype=np.float32)
        self.cube = set_up_variable_cube(
            precip_data_ms, name='rainfall_rate', units='m s-1',
            spatial_grid='equalarea', attributes={'institution': 'Met Office'})
        self.precip_data_mmh = 0.36*np.ones((5, 5), dtype=np.float32)

    def test_basic(self):
        """Test outputs are of correct types"""
        (data, metadata) = PystepsImporter().process_cube(self.cube)
        self.assertIsInstance(data, np.ndarray)
        self.assertIsInstance(metadata, dict)

    def test_rainfall_values(self):
        """Test array contains correct values in mm/h"""
        (data, _) = PystepsImporter().process_cube(self.cube)
        self.assertArrayAlmostEqual(data, self.precip_data_mmh)

    def test_with_mask(self):
        """Test plugin treats masked data correctly"""
        mask = np.full((5, 5), False)
        mask[:1, :] = True
        mask[:, :1] = True
        expected_data = np.where(mask, np.nan, self.precip_data_mmh)
        self.cube.data = np.ma.MaskedArray(self.cube.data, mask=mask)
        (data, _) = PystepsImporter().process_cube(self.cube)
        self.assertIsInstance(data, np.ndarray)
        self.assertTrue(np.allclose(data, expected_data, equal_nan=True))

    def test_dict_values(self):
        """Test dictionary contains expected metadata"""
        projection = ('+a=6378137.000 +b=6356752.314 +proj=laea +lon_0=-2.500'
                      ' +lat_0=54.900 +x_0=0.000 +y_0=0.000 +ellps=WGS84')
        expected_metadata = {
            'unit': 'mm/h',
            'accutime': 0,
            'transform': None,
            'institution': 'Met Office',
            'projection': projection,
            'xpixelsize': 200000.0,
            'x1': -400000.0,
            'x2': 400000.0,
            'ypixelsize': 200000.0,
            'y1': -100000.0,
            'y2': 700000.0,
            'yorigin': 'lower'}

        (_, metadata) = PystepsImporter().process_cube(self.cube)
        self.assertDictEqual(metadata, expected_metadata)

    def test_no_institution(self):
        """Test plugin deals correctly with missing attribute"""
        self.cube.attributes.pop('institution')
        (_, metadata) = PystepsImporter().process_cube(self.cube)
        self.assertEqual(metadata['institution'], 'unknown')

    def test_error_non_rate_cube(self):
        """Test plugin rejects cube of non-rate data"""
        invalid_cube = set_up_variable_cube(
            275*np.ones((5, 5), dtype=np.float32))
        msg = 'air_temperature is not a precipitation rate cube'
        with self.assertRaisesRegex(ValueError, msg):
            PystepsImporter().process_cube(invalid_cube)


if __name__ == '__main__':
    unittest.main()
