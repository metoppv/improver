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
"""Unit tests for saving functionality."""

import os
import unittest
import numpy as np
from subprocess import call
from tempfile import mkdtemp

import iris
from iris.tests import IrisTest
from netCDF4 import Dataset

from improver.utilities.load import load_cube
from improver.utilities.save import save_netcdf
from improver.utilities.save import append_metadata_cube
from improver.tests.ensemble_calibration.ensemble_calibration. \
    helper_functions import set_up_cube


def set_up_test_cube():
    """ Set up a temperature cube with additional global attributes. """
    data = (np.linspace(-45.0, 45.0, 9).reshape(1, 1, 3, 3) + 273.15)
    cube = set_up_cube(data, 'air_temperature', 'K', realizations=([0]))
    cube.attributes['source_realizations'] = np.arange(12)
    # Desired attributes that will be global in netCDF file
    cube.attributes['title'] = 'Operational MOGREPS-UK Forecast Model'
    cube.attributes['um_version'] = '10.4'
    cube.attributes['grid_id'] = 'enukx_standard_v1'
    cube.attributes['source'] = 'Met Office Unified Model'
    cube.attributes['Conventions'] = 'CF-1.5'
    cube.attributes['institution'] = 'Met Office'
    cube.attributes['history'] = ''

    return cube


class Test_save_netcdf(IrisTest):
    """ Test function to save iris cubes as NetCDF files. """

    def setUp(self):
        """ Set up cube to write, read and check """
        self.global_keys_ref = ['title', 'um_version', 'grid_id', 'source',
                                'Conventions', 'institution', 'history',
                                'bald__isPrefixedBy']
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
        """
        Test functionality for saving iris.cube.CubeList

        Both cubes are saved into one single file which breaks the convention
        of one cube per file. Therefore can't use IMPROVER specific load
        utilities since they don't have the ability to handle multiple
        cubes in one file.

        """
        cube_list = ([self.cube, self.cube])
        save_netcdf(cube_list, self.filepath)
        read_cubes = iris.load(self.filepath)
        self.assertIsInstance(read_cubes, iris.cube.CubeList)
        # Length of read_cubes now increased to 3 as Iris 2 saves metadata
        # as separate cube rather than as attributes on other other cubes in
        # the file (Iris 1.13)
        self.assertEqual(len(read_cubes), 3)

    def test_cube_data(self):
        """ Test valid cube can be read from saved file """
        save_netcdf(self.cube, self.filepath)
        cube = load_cube(self.filepath)
        self.assertTrue(isinstance(cube, iris.cube.Cube))
        self.assertArrayEqual(cube.data, self.cube.data)

    def test_cube_dimensions(self):
        """ Test cube dimension coordinates are preserved """
        save_netcdf(self.cube, self.filepath)
        cube = load_cube(self.filepath)
        coord_names = [coord.name() for coord in cube.coords(dim_coords=True)]
        reference_names = [coord.name()
                           for coord in self.cube.coords(dim_coords=True)]
        self.assertCountEqual(coord_names, reference_names)

    def test_cf_global_attributes(self):
        """ Test that a NetCDF file saved from one cube only contains the
        expected global attributes.

        NOTE Loading the file as an iris.cube.Cube does not distinguish global
        from local attributes, and therefore cannot test for the correct
        behaviour here.
        """
        save_netcdf(self.cube, self.filepath)
        global_keys = Dataset(self.filepath, mode='r').ncattrs()
        self.assertTrue(all(key in self.global_keys_ref
                            for key in global_keys))

    def test_cf_data_attributes(self):
        """ Test that forbidden global metadata are saved as data variable
        attributes
        """
        save_netcdf(self.cube, self.filepath)
        # cast explicitly to dictionary, as pylint does not recognise
        # OrderedDict as subscriptable
        cf_data_dict = dict(Dataset(self.filepath, mode='r').variables)
        self.assertTrue('source_realizations' in
                        cf_data_dict['air_temperature'].ncattrs())
        self.assertArrayEqual(
            cf_data_dict['air_temperature'].getncattr('source_realizations'),
            np.arange(12))

    def test_cf_shared_attributes_list(self):
        """ Test that a NetCDF file saved from a list of cubes that share
        non-global attributes does not promote these attributes to global.
        """
        cube_list = ([self.cube, self.cube])
        save_netcdf(cube_list, self.filepath)
        global_keys = Dataset(self.filepath, mode='r').ncattrs()
        self.assertEqual(len(global_keys), 8)
        self.assertTrue(all(key in self.global_keys_ref
                            for key in global_keys))


class Test_append_metadata_cube(IrisTest):
    """Test that appropriate metadata cube and attributes have been appended
            to the cubes in the cube list"""

    def setUp(self):
        """ Set up cube to write, read and check """
        self.global_keys_ref = ['title', 'um_version', 'grid_id', 'source',
                                'Conventions', 'institution', 'history',
                                'bald__isPrefixedBy']
        self.directory = mkdtemp()
        self.filepath = os.path.join(self.directory, "temp.nc")
        self.cube = set_up_test_cube()

    def tearDown(self):
        """ Remove temporary directories created for testing. """
        call(['rm', '-f', self.filepath])
        call(['rmdir', self.directory])

    def test_bald_attribute_added(self):
        """Test that the bald__isPrefixedBy attribute is added to each cube
        and points to prefix_list"""
        cube_list = ([self.cube, self.cube])
        metadata_cubelist = append_metadata_cube(
            cube_list, self.global_keys_ref)
        for cube in metadata_cubelist:
            self.assertTrue(
                cube.attributes['bald__isPrefixedBy']
                is 'prefix_list')

    def test_prefix_cube_attributes(self):
        """Test that metadata prefix cube contains the correct attributes"""
        prefix_dict = {
            'spp__': 'http://reference.metoffice.gov.uk/statistical-process'
                     '/properties/',
            'bald__isPrefixedBy': 'prefix_list',
            'bald__': 'http://binary-array-ld.net/latest/',
            'spv__': 'http://reference.metoffice.gov.uk/statistical-process'
                     '/values/',
            'rdf__': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'spd__': 'http://reference.metoffice.gov.uk/statistical-process'
                     '/def/'}
        metadata_cubelist = append_metadata_cube([], self.global_keys_ref)
        self.assertDictEqual(metadata_cubelist[0].attributes, prefix_dict)

    def test_global_attributes_present(self):
        """Test that desired global attributes are added to the prefix cube
        so that Iris2 keeps these attributes global in any resultant
        netCDF file saved using these cubes"""

        cube_list = ([self.cube])
        metadata_cubelist = append_metadata_cube(
            cube_list, self.global_keys_ref)

        keys_in_prefix_cube = metadata_cubelist[1].attributes

        # Get the global keys from both prefix and data cubes
        prefix_global_keys = [
            k for k in keys_in_prefix_cube.keys()
            if k in self.global_keys_ref]
        data_cube_global_keys = [
            k for k in self.cube.attributes.keys()
            if k in self.global_keys_ref]

        # Check the keys are the same for prefix and data cube
        self.assertListEqual(
            sorted(prefix_global_keys), sorted(data_cube_global_keys))

        # Check the key values are the same for prefix and data cube.
        for key in prefix_global_keys:
            self.assertEqual(metadata_cubelist[-1].attributes[key],
                             self.cube.attributes[key])


if __name__ == '__main__':
    unittest.main()
