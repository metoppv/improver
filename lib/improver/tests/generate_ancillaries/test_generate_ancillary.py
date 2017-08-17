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
"""Unit tests for the the stand alone functions in generate_ancillary.py"""

import unittest
from glob import glob
import os
from iris.cube import Cube
from iris.tests import IrisTest
from iris.coords import DimCoord
from iris.coord_systems import GeogCS
from iris.fileformats.pp import EARTH_RADIUS
from iris import save
import numpy as np

from improver.generate_ancillaries.generate_ancillary import (
    _make_mask_cube, find_standard_ancil)


def _make_test_cube(long_name, stash=None):
    """Make a cube to run tests upon"""
    cs = GeogCS(EARTH_RADIUS)
    data = np.array([[1., 1., 1.],
                     [0., 0., 0.],
                     [1., 0., 1.]])
    cube = Cube(data, long_name=long_name)
    x_coord = DimCoord(np.linspace(-45.0, 45.0, 3), 'latitude',
                       units='degrees', coord_system=cs)
    y_coord = DimCoord(np.linspace(120, 180, 3), 'longitude',
                       units='degrees', coord_system=cs)
    cube.add_dim_coord(x_coord, 0)
    cube.add_dim_coord(y_coord, 1)
    if stash is not None:
        cube.attributes['STASH'] = stash
    return cube


class Test__make_mask_cube(IrisTest):
    """Test private function to make cube from generated mask"""

    def setUp(self):
        """setting up test data"""
        premask = np.array([[0., 3., 2.],
                            [0.5, 0., 1.5],
                            [0.2, 0., 0]])
        self.mask = np.ma.masked_where(premask > 1., premask)
        self.key = 'test key'
        self.x_coord = DimCoord([1, 2, 3], long_name='longitude')
        self.y_coord = DimCoord([1, 2, 3], long_name='latitude')
        self.coords = [self.x_coord, self.y_coord]
        self.upper = 100.
        self.lower = 0.

    def test_wrong_number_of_bounds(self):
        """test creating cube with neither upper nor lower threshold set"""
        emsg = "should have only an upper and lower limit"
        with self.assertRaisesRegexp(TypeError, emsg):
            result = _make_mask_cube(self.mask, self.key, self.coords, [0])
        with self.assertRaisesRegexp(TypeError, emsg):
            result = _make_mask_cube(self.mask,
                                     self.key,
                                     self.coords,
                                     [0, 2, 4])

    def test_upperbound_fails(self):
        """test creating cube with upper threshold only set"""
        emsg = "should have both an upper and lower limit"
        with self.assertRaisesRegexp(TypeError, emsg):
            _make_mask_cube(self.mask, self.key, self.coords,
                            topographic_bounds=[None, self.upper])

    def test_bothbounds(self):
        """test creating cube with both thresholds set"""
        result = _make_mask_cube(self.mask, self.key, self.coords,
                                 topographic_bounds=[self.lower, self.upper])
        self.assertEqual(result.coord('topographic_zone').bounds[0][1],
                         self.upper)
        self.assertEqual(result.coord('topographic_zone').bounds[0][0],
                         self.lower)
        self.assertEqual(result.coord('topographic_zone').points,
                         (self.lower+self.upper)/2)


class Test_find_standard_ancil(IrisTest):
    """Test function to find input for ancillary generation."""

    def setUp(self):
        """setting up test dir and test files"""
        self.test_dir = './anciltest/'
        self.stage = self.test_dir + 'stage.nc'
        os.mkdir(self.test_dir)
        save(_make_test_cube('stage test'), self.stage)

    def tearDown(self):
        """"remove test directories and files"""
        if os.path.exists(self.test_dir):
            files = glob(self.test_dir + '*')
            for f in files:
                os.remove(f)
            os.rmdir(self.test_dir)

    def test_findstage(self):
        """test case where stage file is present and read"""
        result = find_standard_ancil(self.stage)
        self.assertIsInstance(result, Cube)

    def test_findstage_fail(self):
        """test the correct exception is raised when stage ancillaries
           are not found"""
        os.remove(self.stage)
        with self.assertRaisesRegexp(IOError, 'Cannot find input ancillary'):
            find_standard_ancil(self.stage)

    def test_custom_msg_fail(self):
        """test the correct exception is raised when the optional msg
           argument is passed in and the file cannot be found"""
        os.remove(self.stage)
        msg = "That file doesn't exist"
        with self.assertRaisesRegexp(IOError, msg):
            find_standard_ancil(self.stage, msg)

if __name__ == "__main__":
    unittest.main()
