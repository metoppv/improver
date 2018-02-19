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
"""Unit tests for saving functionality."""

import os
import unittest
import numpy as np
from subprocess import call
from tempfile import mkdtemp

import iris
from iris.coords import DimCoord
from iris.tests import IrisTest
from iris.fileformats.cf import CFReader

from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf


def set_up_test_cube():
    """Create a cube with metadata and values suitable for air temperature."""
    data = (np.linspace(-45.0, 45.0, 9).reshape(1, 3, 3) + 273.15)
    realization = DimCoord(1, "realization")
    y_coord = DimCoord(np.linspace(-45.0, 45.0, 3),
                       'latitude', units='degrees')
    x_coord = DimCoord(np.linspace(120, 180, 3),
                       'longitude', units='degrees')
    attributes = {'Conventions': 'CF-1.5', 'source_realizations': 12}
    cube = iris.cube.Cube(data, 'air_temperature', units='K',
                          attributes=attributes,
                          dim_coords_and_dims=[(realization, 0), (y_coord, 1),
                                               (x_coord, 2)])
    return cube


class Test_save_netcdf(IrisTest):

    """ Test function to save iris cubes as netcdf. """

    def setUp(self):
        """ Set up cube to write, read and check """
        self.global_keys_ref = ['title', 'um_version', 'grid_id', 'source',
                                'Conventions', 'institution', 'history']
        self.directory = mkdtemp()
        self.filepath = os.path.join(self.directory, "temp.nc")
        self.cube = set_up_test_cube()

    def tearDown(self):
        """ Remove temporary directories created for testing. """
        call(['rm', '-f', self.filepath])
        call(['rmdir', self.directory])

    def test_basic(self):
        """ Test saves file in required location """
        self.assertFalse(os.path.exists(self.filepath))
        save_netcdf(self.cube, self.filepath)
        self.assertTrue(os.path.exists(self.filepath))

    def test_cube_list(self):
        """ Test functionality for saving iris.cube.CubeList """
        cube_list = ([self.cube, self.cube])
        save_netcdf(cube_list, self.filepath)
        read_cubes = iris.load(self.filepath)
        self.assertTrue(isinstance(read_cubes, iris.cube.CubeList))
        self.assertEqual(len(read_cubes), 2)

    def test_cube_data(self):
        """ Test valid cube can be read from saved file """
        save_netcdf(self.cube, self.filepath)
        cube = load_cube(self.filepath)
        self.assertTrue(isinstance(cube, iris.cube.Cube))
        self.assertTrue(np.array_equal(cube.data, self.cube.data))

    def test_cube_dimensions(self):
        """ Test cube dimension coordinates are preserved """
        save_netcdf(self.cube, self.filepath)
        cube = load_cube(self.filepath)
        coord_names = []
        [coord_names.append(coord.name())
            for coord in cube.coords(dim_coords=True)]
        reference_names = []
        [reference_names.append(coord.name())
            for coord in self.cube.coords(dim_coords=True)]
        self.assertItemsEqual(coord_names, reference_names)

    def test_cf_global_attributes(self):
        """
        Test that the saved NetCDF file only contains the expected global
        attributes.

        NOTE Loading the file as an iris.cube.Cube does not distinguish global
        from local attributes, and therefore cannot test for the correct
        behaviour here.
        """
        save_netcdf(self.cube, self.filepath)
        cube_keys = CFReader(self.filepath).cf_group.global_attributes.keys()
        self.assertTrue(all(key in self.global_keys_ref
                            for key in cube_keys))


if __name__ == '__main__':
    unittest.main()
