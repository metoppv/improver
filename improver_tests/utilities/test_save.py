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
"""Unit tests for saving functionality."""

import os
import unittest
from tempfile import mkdtemp

import iris
import numpy as np
from iris.coords import CellMethod
from iris.tests import IrisTest
from netCDF4 import Dataset

from improver.utilities.load import load_cube
from improver.utilities.save import (
    _append_metadata_cube, _order_cell_methods, save_netcdf)

from ..set_up_test_cubes import set_up_variable_cube


def set_up_test_cube():
    """ Set up a temperature cube with additional global attributes. """
    data = np.linspace(
        -45.0, 45.0, 9, dtype=np.float32).reshape((1, 3, 3)) + 273.15

    attributes = {
        'um_version': '10.4',
        'source': 'Met Office Unified Model',
        'Conventions': 'CF-1.5',
        'institution': 'Met Office',
        'history': ''}

    cube = set_up_variable_cube(
        data, attributes=attributes, standard_grid_metadata='uk_ens')

    return cube


class Test_save_netcdf(IrisTest):
    """ Test function to save iris cubes as NetCDF files. """

    def setUp(self):
        """ Set up cube to write, read and check """
        self.global_keys_ref = ['title', 'um_version', 'grid_id', 'source',
                                'mosg__grid_type', 'mosg__model_configuration',
                                'mosg__grid_domain', 'mosg__grid_version',
                                'Conventions', 'institution', 'history',
                                'bald__isPrefixedBy']
        self.directory = mkdtemp()
        self.filepath = os.path.join(self.directory, "temp.nc")
        self.cube = set_up_test_cube()
        self.cell_methods = (
            CellMethod(method='maximum', coords='time', intervals='1 hour'),
            CellMethod(method='mean', coords='realization'))
        self.cube.cell_methods = self.cell_methods

    def tearDown(self):
        """ Remove temporary directories created for testing. """
        try:
            os.remove(self.filepath)
        except FileNotFoundError:
            pass
        os.rmdir(self.directory)

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

    def test_cell_method_reordering_in_saved_file(self):
        """ Test cell methods are in the correct order when written out and
        read back in."""
        self.cube.cell_methods = (self.cell_methods[1], self.cell_methods[0])
        save_netcdf(self.cube, self.filepath)
        cube = load_cube(self.filepath)
        self.assertEqual(cube.cell_methods, self.cell_methods)

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
        self.cube.attributes['test_attribute'] = np.arange(12)
        save_netcdf(self.cube, self.filepath)
        # cast explicitly to dictionary, as pylint does not recognise
        # OrderedDict as subscriptable
        cf_data_dict = dict(Dataset(self.filepath, mode='r').variables)
        self.assertTrue('test_attribute' in
                        cf_data_dict['air_temperature'].ncattrs())
        self.assertArrayEqual(
            cf_data_dict['air_temperature'].getncattr('test_attribute'),
            np.arange(12))

    def test_cf_shared_attributes_list(self):
        """ Test that a NetCDF file saved from a list of cubes that share
        non-global attributes does not promote these attributes to global.
        """
        cube_list = ([self.cube, self.cube])
        save_netcdf(cube_list, self.filepath)
        global_keys_in_file = Dataset(self.filepath, mode='r').ncattrs()
        self.assertEqual(len(global_keys_in_file), 10)
        self.assertTrue(all(key in self.global_keys_ref
                            for key in global_keys_in_file))

    def test_error_unknown_units(self):
        """Test key error when trying to save a cube with no units"""
        no_units_cube = iris.cube.Cube(np.array([1], dtype=np.float32))
        msg = 'has unknown units'
        with self.assertRaisesRegex(ValueError, msg):
            save_netcdf(no_units_cube, self.filepath)


class Test__order_cell_methods(IrisTest):
    """ Test function that sorts cube cell_methods before saving. """

    def setUp(self):
        """ Set up cube with cell_methods."""
        self.cube = set_up_test_cube()
        self.cell_methods = (
            CellMethod(method='maximum', coords='time', intervals='1 hour'),
            CellMethod(method='mean', coords='realization'))
        self.cube.cell_methods = self.cell_methods

    def test_no_reordering_cube(self):
        """ Test the order is preserved is no reordering required."""
        _order_cell_methods(self.cube)
        self.assertEqual(self.cube.cell_methods, self.cell_methods)

    def test_reordering_cube(self):
        """ Test the order is changed when reordering is required."""
        self.cube.cell_methods = (self.cell_methods[1], self.cell_methods[0])
        # Test that following the manual reorder above the cube cell methods
        # and the tuple don't match.
        self.assertNotEqual(self.cube.cell_methods, self.cell_methods)

        _order_cell_methods(self.cube)
        # Test that they do match once sorting has occured.
        self.assertEqual(self.cube.cell_methods, self.cell_methods)


class Test__append_metadata_cube(IrisTest):
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
        try:
            os.remove(self.filepath)
        except FileNotFoundError:
            pass
        os.rmdir(self.directory)

    def test_bald_attribute_added(self):
        """Test that the bald__isPrefixedBy attribute is added to each cube
        and points to prefix_list"""
        cube_list = ([self.cube, self.cube])
        metadata_cubelist = _append_metadata_cube(
            cube_list, self.global_keys_ref)
        for cube in metadata_cubelist:
            self.assertTrue(
                cube.attributes['bald__isPrefixedBy'] == 'prefix_list')

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
        metadata_cubelist = _append_metadata_cube([], self.global_keys_ref)
        self.assertDictEqual(metadata_cubelist[0].attributes, prefix_dict)

    def test_global_attributes_present(self):
        """Test that desired global attributes are added to the prefix cube
        so that Iris2 keeps these attributes global in any resultant
        netCDF file saved using these cubes"""

        cube_list = ([self.cube])
        metadata_cubelist = _append_metadata_cube(
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
