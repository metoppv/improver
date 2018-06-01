# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
"""Unit tests for spotdata.write_output"""

import os
import unittest
import numpy as np
from tempfile import mkdtemp
from subprocess import call as Call
import iris
from iris.tests import IrisTest
from iris.cube import Cube
from iris.coords import DimCoord

from improver.spotdata.write_output import WriteOutput as Plugin
from improver.utilities.load import load_cube


class Test_write_output(IrisTest):
    """Test the writing of SpotData output."""

    def setUp(self):
        """Create a cube containing a regular lat-lon grid and other necessary
        ingredients for unit tests."""

        data = np.zeros((20, 20))
        latitudes = np.linspace(-90, 90, 20)
        longitudes = np.linspace(-180, 180, 20)
        latitude = DimCoord(latitudes, standard_name='latitude',
                            units='degrees')
        longitude = DimCoord(longitudes, standard_name='longitude',
                             units='degrees')

        cube = Cube(data,
                    long_name="test_data",
                    dim_coords_and_dims=[(latitude, 0), (longitude, 1)],
                    units="1")
        self.cube = cube
        self.data_directory = mkdtemp()

    def tearDown(self):
        """Remove temporary directories created for testing."""
        Call(['rm', '-f', os.path.join(self.data_directory, 'test_data.nc')])
        Call(['rmdir', self.data_directory])

    def test_write_netcdf(self):
        """Test writing of iris.cube.Cube to netcdf file."""

        method = 'as_netcdf'
        Plugin(method, self.data_directory).process(self.cube)
        result = load_cube(os.path.join(self.data_directory, 'test_data.nc'))
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), 'test_data')
        self.assertEqual(result.data.shape, (20, 20))

    def test_invalid_method(self):
        """Test attempt to write with invalid method."""

        method = 'unknown_file_type'
        msg = 'Unknown method ".*" passed to WriteOutput.'
        with self.assertRaisesRegex(AttributeError, msg):
            Plugin(method, self.data_directory).process(self.cube)


if __name__ == '__main__':
    unittest.main()
