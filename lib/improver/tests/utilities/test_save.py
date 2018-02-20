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
import copy
import unittest
import numpy as np
from subprocess import call
from tempfile import mkdtemp

import iris
from iris.tests import IrisTest
from iris.fileformats.cf import CFReader
from iris.fileformats.cf import CFDataVariable

from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf
from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import set_up_cube


def set_up_test_cube():
    """ Set up a temperature cube with additional global attributes. """
    data = (np.linspace(-45.0, 45.0, 9).reshape(1, 1, 3, 3) + 273.15)
    cube = set_up_cube(data, 'air_temperature', 'K', realizations=([0]))
    cube.attributes['Conventions'] = 'CF-1.5'
    cube.attributes['source_realizations'] = np.arange(12)
    return cube


class Test_save_netcdf(IrisTest):

    """ Test function to save iris cubes as NetCDF files. """

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

    def test_basic_cube(self):
        """ Test saves file in required location """
        self.assertFalse(os.path.exists(self.filepath))
        save_netcdf(self.cube, self.filepath)
        self.assertTrue(os.path.exists(self.filepath))

    def test_basic_cube_list(self):
        """ Test functionality for saving iris.cube.CubeList """
        cube_list = ([self.cube, self.cube])
        save_netcdf(cube_list, self.filepath)
        read_cubes = iris.load(self.filepath)
        self.assertIsInstance(read_cubes, iris.cube.CubeList)
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
        coord_names = [coord.name() for coord in cube.coords(dim_coords=True)]
        reference_names = [coord.name()
                           for coord in self.cube.coords(dim_coords=True)]
        self.assertItemsEqual(coord_names, reference_names)

    def test_cf_global_attributes(self):
        """ Test that a NetCDF file saved from one cube only contains the
        expected global attributes.

        NOTE Loading the file as an iris.cube.Cube does not distinguish global
        from local attributes, and therefore cannot test for the correct
        behaviour here.
        """
        save_netcdf(self.cube, self.filepath)
        global_keys = CFReader(self.filepath).cf_group.global_attributes.keys()
        self.assertTrue(all(key in self.global_keys_ref
                            for key in global_keys))


    def test_cf_data_attributes(self):
        """ Test that forbidden global metadata are saved as data variable
        attributes

        TODO get this working.  Need to be able to inspect the data variable!
        """
        save_netcdf(self.cube, self.filepath)

        cf_group = copy.deepcopy(CFReader(self.filepath).cf_group)
        print cf_group.values()

        #group = CFReader(self.filepath).cf_group

        #print group.items()

        #data_variable = CFDataVariable(group.data_variables.keys()[0],
        #                               group.data_variables)

        #print data_variable


    def test_cf_shared_attributes_list(self):
        """ Test that a NetCDF file saved from a list of cubes that share
        non-global attributes does not promote these attributes to global.
        """
        cube_list = ([self.cube, self.cube])
        save_netcdf(cube_list, self.filepath)
        global_keys = CFReader(self.filepath).cf_group.global_attributes.keys()
        self.assertEqual(len(global_keys), 1)
        self.assertTrue(all(key in self.global_keys_ref
                            for key in global_keys))


if __name__ == '__main__':
    unittest.main()
