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
"""Unit tests for the cube_metadata utilities."""
import unittest

import numpy as np

import iris
from copy import copy
from iris.tests import IrisTest
from iris.cube import Cube
from iris.coords import DimCoord
from cf_units import Unit

from improver.utilities.cube_metadata import (
    add_coord, update_coord, update_attribute,
    amend_metadata, resolve_metadata_diff, delete_attributes)
from improver.utilities.warnings_handler import ManageWarnings


def create_cube_with_threshold(data=None,
                               long_name=None,
                               threshold_values=None,
                               units=None):
    """Create a cube with threshold coord."""
    if threshold_values is None:
        threshold_values = [1.0]
    if data is None:
        data = np.zeros((len(threshold_values), 2, 2, 2))
        data[:, 0, :, :] = 0.5
        data[:, 1, :, :] = 0.6
    if long_name is None:
        long_name = "probability_of_rainfall_rate"
    if units is None:
        units = "m s^-1"

    cube = Cube(data, long_name=long_name, units='1')
    cube.add_dim_coord(DimCoord(np.linspace(-45.0, 45.0, 2), 'latitude',
                                units='degrees'), 2)
    cube.add_dim_coord(DimCoord(np.linspace(120, 180, 2), 'longitude',
                                units='degrees'), 3)
    time_origin = "hours since 1970-01-01 00:00:00"
    calendar = "gregorian"
    tunit = Unit(time_origin, calendar)
    cube.add_dim_coord(DimCoord([402192.5, 402193.5],
                                "time", units=tunit), 1)
    cube.add_dim_coord(DimCoord(threshold_values,
                                long_name='threshold',
                                units=units), 0)
    cube.attributes['relative_to_threshold'] = 'above'
    return cube


class Test_add_coord(IrisTest):

    """Test the add_coord method."""

    def test_basic(self):
        """Test that add_coord returns a Cube and adds coord correctly. """
        coord_name = 'threshold'
        cube = create_cube_with_threshold()
        cube.remove_coord(coord_name)
        cube = iris.util.squeeze(cube)
        changes = {'points': [2.0], 'bounds': [0.1, 2.0], 'units': 'mm'}
        result = add_coord(cube, coord_name, changes)
        self.assertIsInstance(result, Cube)
        self.assertArrayEqual(result.coord(coord_name).points,
                              np.array([2.0]))
        self.assertArrayEqual(result.coord(coord_name).bounds,
                              np.array([[0.1, 2.0]]))
        self.assertEqual(str(result.coord(coord_name).units),
                         'mm')

    def test_fails_no_points(self):
        """Test that add_coord fails if points not included in metadata """
        coord_name = 'threshold'
        cube = create_cube_with_threshold()
        cube.remove_coord(coord_name)
        cube = iris.util.squeeze(cube)
        changes = {'bounds': [0.1, 2.0], 'units': 'mm'}
        msg = 'Trying to add new coord but no points defined'
        with self.assertRaisesRegex(ValueError, msg):
            add_coord(cube, coord_name, changes)

    def test_fails_points_greater_than_1(self):
        """Test that add_coord fails if points greater than 1 """
        coord_name = 'threshold'
        cube = create_cube_with_threshold()
        cube.remove_coord(coord_name)
        cube = iris.util.squeeze(cube)
        changes = {'points': [0.1, 2.0]}
        msg = 'Can not add a coordinate of length > 1'
        with self.assertRaisesRegex(ValueError, msg):
            add_coord(cube, coord_name, changes)

    @ManageWarnings(record=True)
    def test_warning_messages(self, warning_list=None):
        """Test that warning messages is raised correctly. """
        coord_name = 'threshold'
        cube = create_cube_with_threshold()
        cube.remove_coord(coord_name)
        cube = iris.util.squeeze(cube)
        changes = {'points': [2.0], 'bounds': [0.1, 2.0], 'units': 'mm'}
        warning_msg = "Adding new coordinate"
        add_coord(cube, coord_name, changes, warnings_on=True)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))


class Test_update_coord(IrisTest):

    """Test the update_coord method."""

    def test_basic(self):
        """Test update_coord returns a Cube and updates coord correctly. """
        cube = create_cube_with_threshold()
        changes = {'points': [2.0], 'bounds': [0.1, 2.0], 'units': 'mm'}
        result = update_coord(cube, 'threshold', changes)
        self.assertIsInstance(result, Cube)
        self.assertArrayEqual(result.coord('threshold').points,
                              np.array([2.0]))
        self.assertArrayEqual(result.coord('threshold').bounds,
                              np.array([[0.1, 2.0]]))
        self.assertEqual(str(result.coord('threshold').units),
                         'mm')

    def test_coords_deleted(self):
        """Test update_coord deletes coordinate. """
        cube = create_cube_with_threshold()
        changes = 'delete'
        result = update_coord(cube, 'threshold', changes)
        found_key = 'threshold' in [coord.name() for coord in result.coords()]
        self.assertArrayEqual(found_key,
                              False)

    def test_coords_deleted_fails(self):
        """Test update_coord fails to delete coord of len > 1. """
        cube = create_cube_with_threshold()
        changes = 'delete'
        msg = "Can only remove a coordinate of length 1"
        with self.assertRaisesRegex(ValueError, msg):
            update_coord(cube, 'time', changes)

    @ManageWarnings(record=True)
    def test_warning_messages_with_delete(self, warning_list=None):
        """Test warning message is raised correctly when deleting coord. """
        coord_name = 'threshold'
        cube = create_cube_with_threshold()
        changes = 'delete'
        warning_msg = "Deleted coordinate"
        update_coord(cube, coord_name, changes, warnings_on=True)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))

    def test_coords_update_fail_points(self):
        """Test that update_coord fails if points do not match. """
        cube = create_cube_with_threshold()
        changes = {'points': [2.0, 3.0]}
        msg = "Mismatch in points in existing coord and updated metadata"
        with self.assertRaisesRegex(ValueError, msg):
            update_coord(cube, 'threshold', changes)

    def test_coords_update_fail_bounds(self):
        """Test update_coord fails if shape of new bounds do not match. """
        cube = create_cube_with_threshold(threshold_values=[2.0, 3.0])
        changes = {'bounds': [0.1, 2.0]}
        msg = "The shape of the bounds array should be"
        with self.assertRaisesRegex(ValueError, msg):
            update_coord(cube, 'threshold', changes)

    def test_coords_update_bounds_succeed(self):
        """Test that update_coord succeeds if bounds do match """
        cube = create_cube_with_threshold(threshold_values=[2.0, 3.0])
        cube.coord('threshold').guess_bounds()
        changes = {'bounds': [[0.1, 2.0], [2.0, 3.0]]}
        result = update_coord(cube, 'threshold', changes)
        self.assertArrayEqual(result.coord('threshold').bounds,
                              np.array([[0.1, 2.0], [2.0, 3.0]]))

    def test_coords_update_fails_bounds_differ(self):
        """Test that update_coord fails if bounds differ."""
        cube = create_cube_with_threshold(threshold_values=[2.0, 3.0])
        cube.coord('threshold').guess_bounds()
        changes = {'bounds': [[0.1, 2.0], [2.0, 3.0], [3.0, 4.0]]}
        msg = "Mismatch in bounds in existing coord and updated metadata"
        with self.assertRaisesRegex(ValueError, msg):
            update_coord(cube, 'threshold', changes)

    @ManageWarnings(record=True)
    def test_warning_messages_with_update(self, warning_list=None):
        """Test warning message is raised correctly when updating coord. """
        coord_name = 'threshold'
        cube = create_cube_with_threshold()
        changes = {'points': [2.0], 'bounds': [0.1, 2.0], 'units': 'mm'}
        warning_msg = "Updated coordinate"
        update_coord(cube, coord_name, changes, warnings_on=True)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))


class Test_update_attribute(IrisTest):

    """Test the update_attribute method."""

    def test_basic(self):
        """Test that update_attribute returns a Cube and updates OK. """
        cube = create_cube_with_threshold()
        attribute_name = 'relative_to_threshold'
        changes = 'between'
        result = update_attribute(cube, attribute_name, changes)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.attributes['relative_to_threshold'],
                         'between')

    @ManageWarnings(record=True)
    def test_attributes_updated_warnings(self, warning_list=None):
        """Test update_attribute updates attributes and gives warning. """
        cube = create_cube_with_threshold()
        attribute_name = 'relative_to_threshold'
        changes = 'between'
        warning_msg = "Adding or updating attribute"
        result = update_attribute(cube, attribute_name, changes,
                                  warnings_on=True)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertEqual(result.attributes['relative_to_threshold'],
                         'between')

    def test_attributes_added(self):
        """Test update_attribute adds attributeOK. """
        cube = create_cube_with_threshold()
        attribute_name = 'new_attribute'
        changes = 'new_value'
        result = update_attribute(cube, attribute_name, changes)
        self.assertEqual(result.attributes['new_attribute'],
                         'new_value')

    def test_attributes_deleted(self):
        """Test update_attribute deletes attribute OK. """
        cube = create_cube_with_threshold()
        attribute_name = 'relative_to_threshold'
        changes = 'delete'
        result = update_attribute(cube, attribute_name, changes)
        self.assertFalse('relative_to_threshold' in result.attributes)

    @ManageWarnings(record=True)
    def test_attributes_deleted_warnings(self, warning_list=None):
        """Test update_attribute deletes and gives warning. """
        cube = create_cube_with_threshold()
        attribute_name = 'relative_to_threshold'
        changes = 'delete'
        warning_msg = "Deleted attribute"
        result = update_attribute(cube, attribute_name, changes,
                                  warnings_on=True)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertFalse('relative_to_threshold' in result.attributes)


class Test_amend_metadata(IrisTest):

    """Test the amend_metadata method."""

    def test_basic(self):
        """Test that the function returns a Cube. """
        cube = create_cube_with_threshold()
        result = amend_metadata(cube, 'new_cube_name', np.dtype,
                                None, None)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), 'new_cube_name')

    def test_attributes_updated_and_added(self):
        """Test amend_metadata  updates and adds attributes OK. """
        cube = create_cube_with_threshold()
        attributes = {'relative_to_threshold': 'between',
                      'new_attribute': 'new_value'}
        result = amend_metadata(cube, 'new_cube_name', np.dtype,
                                None, attributes)
        self.assertEqual(result.attributes['relative_to_threshold'],
                         'between')
        self.assertEqual(result.attributes['new_attribute'],
                         'new_value')

    def test_attributes_deleted(self):
        """Test amend_metadata  updates attributes OK. """
        cube = create_cube_with_threshold()
        attributes = {'relative_to_threshold': 'delete'}
        result = amend_metadata(cube, 'new_cube_name', np.dtype,
                                None, attributes)
        self.assertFalse('relative_to_threshold' in result.attributes)

    def test_coords_updated(self):
        """Test amend_metadata returns a Cube and updates coord correctly. """
        cube = create_cube_with_threshold()
        updated_coords = {'threshold': {'points': [2.0]},
                          'time': {'points': [402193.5, 402194.5]}}
        result = amend_metadata(cube, 'new_cube_name', np.dtype,
                                updated_coords, None)
        self.assertArrayEqual(result.coord('threshold').points,
                              np.array([2.0]))
        self.assertArrayEqual(result.coord('time').points,
                              np.array([402193.5, 402194.5]))

    def test_coords_deleted_and_adds(self):
        """Test amend metadata deletes and adds coordinate. """
        cube = create_cube_with_threshold()
        coords = {'threshold': 'delete',
                  'new_coord': {'points': [2.0]}}
        result = amend_metadata(cube, 'new_cube_name', np.dtype,
                                coords, None)
        found_key = 'threshold' in [coord.name() for coord in result.coords()]
        self.assertFalse(found_key)
        self.assertArrayEqual(result.coord('new_coord').points,
                              np.array([2.0]))

    @ManageWarnings(record=True)
    def test_warnings_on_works(self, warning_list=None):
        """Test amend_metadata raises warnings """
        cube = create_cube_with_threshold()
        updated_attributes = {'new_attribute': 'new_value'}
        updated_coords = {'threshold': {'points': [2.0]}}
        warning_msg_attr = "Adding or updating attribute"
        warning_msg_coord = "Updated coordinate"
        result = amend_metadata(cube, 'new_cube_name', np.dtype,
                                updated_coords, updated_attributes,
                                warnings_on=True)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg_attr in str(item)
                            for item in warning_list))
        self.assertTrue(any(warning_msg_coord in str(item)
                            for item in warning_list))
        self.assertEqual(result.attributes['new_attribute'],
                         'new_value')


class Test_resolve_metadata_diff(IrisTest):

    """Test the resolve_metadata_diff method."""

    def setUp(self):
        """Create a cube with threshold coord is not first coord."""
        threshold_values = [1.0]
        data = np.zeros((2, len(threshold_values), 2, 2, 2))
        data[:, :, 0, :, :] = 0.5
        data[:, :, 1, :, :] = 0.6
        long_name = "probability_of_rainfall_rate"
        units = "m s^-1"

        self.cube = Cube(data, long_name=long_name, units='1')
        self.cube.add_dim_coord(DimCoord(np.linspace(-45.0, 45.0, 2),
                                         'latitude',
                                         units='degrees'), 3)
        self.cube.add_dim_coord(DimCoord(np.linspace(120, 180, 2),
                                         'longitude',
                                         units='degrees'), 4)
        time_origin = "hours since 1970-01-01 00:00:00"
        calendar = "gregorian"
        tunit = Unit(time_origin, calendar)
        self.cube.add_dim_coord(DimCoord([402192.5, 402193.5],
                                         "time", units=tunit), 2)
        self.cube.add_dim_coord(DimCoord(threshold_values,
                                         long_name='threshold',
                                         units=units), 1)
        self.cube.add_dim_coord(DimCoord([0, 1],
                                         long_name='realization',
                                         units=units), 0)
        self.cube.attributes['relative_to_threshold'] = 'above'

    def test_basic(self):
        """Test that the function returns a tuple of Cubes. """
        cube1 = create_cube_with_threshold()
        cube2 = cube1.copy()
        result = resolve_metadata_diff(cube1, cube2)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], Cube)
        self.assertIsInstance(result[1], Cube)

    def test_mismatching_coords_wrong_shape(self):
        """Test raises an error if shape do not match. """
        cube1 = create_cube_with_threshold()
        cube2 = create_cube_with_threshold(threshold_values=[1.0, 2.0])
        msg = "Can not combine cubes, mismatching shapes"
        with self.assertRaisesRegex(ValueError, msg):
            resolve_metadata_diff(cube1, cube2)

    def test_mismatching_coords_1d_coord_pos0_on_cube1(self):
        """Test missing coord on cube2. Coord is leading coord in cube1."""
        cube1 = create_cube_with_threshold()
        cube2 = cube1.copy()
        cube2.remove_coord('threshold')
        cube2 = iris.util.squeeze(cube2)
        result = resolve_metadata_diff(cube1, cube2)
        self.assertIsInstance(result, tuple)
        self.assertArrayEqual(result[0].shape, np.array([1, 2, 2, 2]))
        self.assertArrayEqual(result[1].shape, np.array([1, 2, 2, 2]))

    def test_mismatching_coords_1d_coord_pos0_on_cube2(self):
        """Test missing coord on cube1. Coord is leading coord in cube2."""
        cube1 = create_cube_with_threshold()
        cube2 = cube1.copy()
        cube1.remove_coord('threshold')
        cube1 = iris.util.squeeze(cube1)
        result = resolve_metadata_diff(cube1, cube2)
        self.assertIsInstance(result, tuple)
        self.assertArrayEqual(result[0].shape, np.array([2, 2, 2]))
        self.assertArrayEqual(result[1].shape, np.array([2, 2, 2]))

    def test_mismatching_coords_1d_coord_pos1_cube1(self):
        """Test missing 1d coord on cube2.
           Coord is not leading coord in cube1."""
        cube1 = self.cube
        cube2 = cube1.copy()
        cube2.remove_coord('threshold')
        cube2 = iris.util.squeeze(cube2)
        result = resolve_metadata_diff(cube1, cube2)
        self.assertIsInstance(result, tuple)
        self.assertArrayEqual(result[0].shape, np.array([2, 1, 2, 2, 2]))
        self.assertArrayEqual(result[1].shape, np.array([2, 1, 2, 2, 2]))

    def test_mismatching_coords_1d_coord_pos1_cube2(self):
        """Test missing 1d coord on cube1.
           Coord is not leading coord in cube2."""
        cube1 = self.cube
        cube2 = cube1.copy()
        cube1.remove_coord('threshold')
        cube1 = iris.util.squeeze(cube1)
        result = resolve_metadata_diff(cube1, cube2)
        self.assertIsInstance(result, tuple)
        self.assertArrayEqual(result[0].shape, np.array([2, 2, 2, 2]))
        self.assertArrayEqual(result[1].shape, np.array([2, 2, 2, 2]))

    def test_mismatching_coords_same_shape(self):
        """Test works with mismatching coords but coords same shape."""
        cube1 = create_cube_with_threshold()
        cube2 = create_cube_with_threshold(threshold_values=[2.0])
        result = resolve_metadata_diff(cube1, cube2)
        self.assertIsInstance(result, tuple)
        self.assertArrayEqual(result[0].coord('threshold').points,
                              np.array([1.0]))
        self.assertArrayEqual(result[1].coord('threshold').points,
                              np.array([2.0]))

    @ManageWarnings(record=True)
    def test_warnings_on_work(self, warning_list=None):
        """Test warning messages are given if warnings_on is set."""
        cube1 = create_cube_with_threshold()
        cube2 = cube1.copy()
        cube2.remove_coord('threshold')
        cube2 = iris.util.squeeze(cube2)
        warning_msg = 'Adding new coordinate'
        result = resolve_metadata_diff(cube1, cube2,
                                       warnings_on=True)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertIsInstance(result, tuple)
        self.assertArrayEqual(result[0].shape, np.array([1, 2, 2, 2]))
        self.assertArrayEqual(result[1].shape, np.array([1, 2, 2, 2]))


class Test_delete_attributes(IrisTest):

    """Test the delete_attributes method."""

    def setUp(self):
        """Create a cube with attributes to be deleted."""
        data = np.zeros((2, 2))
        long_name = "probability_of_rainfall_rate"
        units = "m s^-1"
        attributes = {'title': 'This is a cube',
                      'tithe': '10 percent',
                      'mosg_model': 'gl_det',
                      'mosg_grid_version': 1.0,
                      'mosg_grid_name': 'global'}

        self.cube = Cube(data, long_name=long_name, units=units)
        self.cube.attributes = attributes

    def test_basic(self):
        """Test that an empty call leaves the cube unchanged."""
        cube = self.cube.copy()
        delete_attributes(cube, [])

        self.assertDictEqual(self.cube.attributes, cube.attributes)

    def test_accepts_string(self):
        """Test that a single string passed as an argument works."""
        attributes_to_delete = 'title'
        attributes = copy(self.cube.attributes)
        attributes.pop(attributes_to_delete)
        delete_attributes(self.cube, attributes_to_delete)

        self.assertDictEqual(attributes, self.cube.attributes)

    def test_accepts_list_of_complete_matches(self):
        """Test that a list of complete attribute names removes the expected
        attributes."""
        attributes_to_delete = ['title', 'tithe', 'mosg_model']
        attributes = copy(self.cube.attributes)
        for item in attributes_to_delete:
            attributes.pop(item)
        delete_attributes(self.cube, attributes_to_delete)

        self.assertDictEqual(attributes, self.cube.attributes)

    def test_accepts_list_of_partial_matches(self):
        """Test that a list of partial patterns removes the expected
        attributes."""
        attributes_to_delete = ['tit', 'mosg_grid']
        expected = {'mosg_model': 'gl_det'}
        delete_attributes(self.cube, attributes_to_delete)

        self.assertDictEqual(expected, self.cube.attributes)


if __name__ == '__main__':
    unittest.main()
