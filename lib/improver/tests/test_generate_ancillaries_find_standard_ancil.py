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
"""Unit tests for the generate_ancillary.find_standard_ancil function."""


import unittest
from glob import glob
import os
from iris.cube import Cube
from iris.cube import CubeList
from iris.tests import IrisTest
from iris.coords import AuxCoord, DimCoord
from iris.coord_systems import GeogCS
from iris.fileformats.pp import EARTH_RADIUS
from iris import save
import numpy as np

from improver.generate_ancillaries.generate_ancillary import (
    find_standard_ancil)


def _make_test_cube(long_name, stash=None):
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


class Test_find_standard_ancil(IrisTest):
    """Test function to find input for ancillary generation."""

    def setUp(self):
        """setting up test dir and test files"""
        self.test_dir = './anciltest/'
        self.stage = self.test_dir + 'stage.nc'
        os.mkdir(self.test_dir)
        save(_make_test_cube('stage test'), self.stage)
        self.grid = 'glm'

    def tearDown(self):
        if os.path.exists(self.test_dir):
            files = glob(self.test_dir + '*')
            for f in files:
                os.remove(f)
            os.rmdir(self.test_dir)

    def test_findstage(self):
        """test case where stage file is present and read"""
        result = find_standard_ancil(self.grid, self.stage)
        self.assertEqual(result.name(), 'stage test')

    def test_findstage_fail(self):
        """test the correct exception is raised when stage ancillaries
           are not found"""
        os.remove(self.stage)
        with self.assertRaisesRegexp(IOError, 'Cannot find input ancillary'):
            find_standard_ancil(self.grid, self.stage)

if __name__ == "__main__":
    unittest.main()
