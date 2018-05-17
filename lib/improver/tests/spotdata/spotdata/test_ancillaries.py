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
"""Unit tests for spotdata.ancillaries."""

import os
import unittest
import numpy as np
from iris.tests import IrisTest
from subprocess import call as Call
from tempfile import mkdtemp
import iris
from iris.coords import DimCoord
from iris.cube import Cube

from improver.spotdata.ancillaries import get_ancillary_data as Plugin
from improver.utilities.save import save_netcdf


class Test_get_ancillary_data(IrisTest):
    """Test the reading of ancillary data files and creation of an ancillaries
    dictionary."""

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
                    long_name="template",
                    dim_coords_and_dims=[(latitude, 0), (longitude, 1)],
                    units="1")

        orography = cube.copy()
        orography.data[0:10, 0:10] = 2.
        orography.rename('surface_altitude')
        land = cube.copy()
        land.rename('land_binary_mask')
        land.data = land.data + 1

        self.orography = orography
        self.land = land
        self.directory = mkdtemp()
        self.orography_path = os.path.join(self.directory, 'highres_orog.nc')
        self.land_path = os.path.join(self.directory, 'land_mask.nc')
        save_netcdf(orography, self.orography_path)
        save_netcdf(land, self.land_path)

        self.diagnostics = {
            "wind_speed": {
                "diagnostic_name": "wind_speed",
                "extrema": False,
                "filepath": "horizontal_wind_speed_and_direction_at_10m",
                "interpolation_method": "use_nearest",
                "neighbour_finding": {
                    "land_constraint": True,
                    "method": "fast_nearest_neighbour",
                    "vertical_bias": None
                    }
                }
            }

    def tearDown(self):
        """Remove temporary directories created for testing."""
        Call(['rm', '-f', self.orography_path])
        Call(['rm', '-f', self.land_path])
        Call(['rmdir', self.directory])

    def test_return_type(self):
        """Test return type is a dictionary containing cubes."""

        diagnostics = {}
        result = Plugin(diagnostics, self.directory)
        self.assertIsInstance(result, dict)
        for item in list(result.values()):
            self.assertIsInstance(item, Cube)

    def test_read_orography(self):
        """Test reading an orography netcdf file."""

        diagnostics = {}
        result = Plugin(diagnostics, self.directory)
        self.assertIn('orography', list(result.keys()))
        self.assertIsInstance(result['orography'], Cube)
        self.assertArrayEqual(result['orography'].data, self.orography.data)

    def test_read_land_mask(self):
        """Test reading a landmask netcdf file if a diagnostic makes use of a
        land constraint condition."""

        result = Plugin(self.diagnostics, self.directory)
        self.assertIn('land_mask', list(result.keys()))
        self.assertIsInstance(result['land_mask'], Cube)
        self.assertArrayEqual(result['land_mask'].data, self.land.data)

    def test_missing_orography(self):
        """Test attempt to read orography with missing orography netcdf
        file."""

        Call(['rm', self.orography_path])
        diagnostics = {}
        msg = 'Orography file not found.'
        with self.assertRaisesRegex(IOError, msg):
            Plugin(diagnostics, self.directory)

    def test_missing_land_mask(self):
        """Test attempt to read land_mask with missing land_mask netcdf
        file."""

        Call(['rm', self.land_path])
        msg = 'Land mask file not found.'
        with self.assertRaisesRegex(IOError, msg):
            Plugin(self.diagnostics, self.directory)


if __name__ == '__main__':
    unittest.main()
