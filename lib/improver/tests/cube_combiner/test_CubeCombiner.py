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
"""Unit tests for the cube_combiner.CubeCombiner plugin."""
import unittest

import warnings
import numpy as np

import iris
from iris.tests import IrisTest
from iris.cube import Cube
from iris.coords import DimCoord
from cf_units import Unit

from improver.cube_combiner import CubeCombiner


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


class Test__init__(IrisTest):

    """Test the __init__ method."""

    def test_basic(self):
        """Test that the __init__ sets things up correctly"""
        plugin = CubeCombiner('+')
        self.assertEqual(plugin.operation, '+')

    def test_raise_error_wrong_operation(self):
        """Test __init__ raises a ValueError for invalid operation"""
        msg = 'Unknown operation '
        with self.assertRaisesRegexp(ValueError, msg):
            CubeCombiner('%')


class Test__repr__(IrisTest):

    """Test the repr method."""

    def test_basic(self):
        """Test that the __repr__ returns the expected string."""
        result = str(CubeCombiner('+'))
        msg = '<CubeCombiner: operation=+, warnings_on = False>'
        self.assertEqual(result, msg)


class Test_resolve_metadata_diff(IrisTest):

    """Test the remove_metadata_diff method."""

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
        plugin = CubeCombiner('-')
        cube1 = create_cube_with_threshold()
        cube2 = cube1.copy()
        result = plugin.resolve_metadata_diff(cube1, cube2)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], Cube)
        self.assertIsInstance(result[1], Cube)

    def test_mismatching_coords_wrong_shape(self):
        """Test raises an error if shape do not match. """
        plugin = CubeCombiner('-')
        cube1 = create_cube_with_threshold()
        cube2 = create_cube_with_threshold(threshold_values=[1.0, 2.0])
        msg = "Can not combine cubes, mismatching shapes"
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.resolve_metadata_diff(cube1, cube2)

    def test_mismatching_coords_missing_1d_coord(self):
        """Test that the function returns a tuple of Cubes. """
        plugin = CubeCombiner('-')
        cube1 = create_cube_with_threshold()
        cube2 = cube1.copy()
        cube2.remove_coord('threshold')
        cube2 = iris.util.squeeze(cube2)
        result = plugin.resolve_metadata_diff(cube1, cube2)
        self.assertIsInstance(result, tuple)
        self.assertArrayEqual(result[0].shape, np.array([1, 2, 2, 2]))
        self.assertArrayEqual(result[1].shape, np.array([1, 2, 2, 2]))

    def test_mismatching_coords_missing_1d_coord_v2(self):
        """Test that the function returns a tuple of Cubes. """
        plugin = CubeCombiner('-')
        cube1 = create_cube_with_threshold()
        cube2 = cube1.copy()
        cube1.remove_coord('threshold')
        cube1 = iris.util.squeeze(cube1)
        result = plugin.resolve_metadata_diff(cube1, cube2)
        self.assertIsInstance(result, tuple)
        self.assertArrayEqual(result[0].shape, np.array([2, 2, 2]))
        self.assertArrayEqual(result[1].shape, np.array([2, 2, 2]))

    def test_mismatching_coords_missing_1d_coord_v3(self):
        """Test that the function returns a tuple of Cubes. """
        plugin = CubeCombiner('-')
        cube1 = self.cube
        cube2 = cube1.copy()
        cube2.remove_coord('threshold')
        cube2 = iris.util.squeeze(cube2)
        result = plugin.resolve_metadata_diff(cube1, cube2)
        self.assertIsInstance(result, tuple)
        self.assertArrayEqual(result[0].shape, np.array([2, 1, 2, 2, 2]))
        self.assertArrayEqual(result[1].shape, np.array([2, 1, 2, 2, 2]))

    def test_mismatching_coords_missing_1d_coord_v4(self):
        """Test that the function returns a tuple of Cubes. """
        plugin = CubeCombiner('-')
        cube1 = self.cube
        cube2 = cube1.copy()
        cube1.remove_coord('threshold')
        cube1 = iris.util.squeeze(cube1)
        result = plugin.resolve_metadata_diff(cube1, cube2)
        self.assertIsInstance(result, tuple)
        self.assertArrayEqual(result[0].shape, np.array([2, 2, 2, 2]))
        self.assertArrayEqual(result[1].shape, np.array([2, 2, 2, 2]))

    def test_mismatching_coords_same_shape(self):
        """Test that the function returns a tuple of Cubes. """
        plugin = CubeCombiner('-')
        cube1 = create_cube_with_threshold()
        cube2 = create_cube_with_threshold(threshold_values=[2.0])
        result = plugin.resolve_metadata_diff(cube1, cube2)
        self.assertIsInstance(result, tuple)
        self.assertArrayEqual(result[0].coord('threshold').points,
                              np.array([1.0]))
        self.assertArrayEqual(result[1].coord('threshold').points,
                              np.array([2.0]))


class Test_add_coord(IrisTest):

    """Test the update_coord method."""

    def test_basic(self):
        """Test that add_coord returns a Cube and adds coord correctly. """
        plugin = CubeCombiner('-')
        coord_name = 'threshold'
        cube = create_cube_with_threshold()
        cube.remove_coord(coord_name)
        cube = iris.util.squeeze(cube)
        changes = {'points': [2.0], 'bounds': [0.1, 2.0], 'units': 'mm'}
        result = plugin.add_coord(cube, coord_name, changes)
        self.assertIsInstance(result, Cube)
        self.assertArrayEqual(result.coord(coord_name).points,
                              np.array([2.0]))
        self.assertArrayEqual(result.coord(coord_name).bounds,
                              np.array([[0.1, 2.0]]))
        self.assertEqual(str(result.coord(coord_name).units),
                         'mm')

    def test_fails_no_points(self):
        """Test that add_coord fails if points not included in metadata """
        plugin = CubeCombiner('-')
        coord_name = 'threshold'
        cube = create_cube_with_threshold()
        cube.remove_coord(coord_name)
        cube = iris.util.squeeze(cube)
        changes = {'bounds': [0.1, 2.0], 'units': 'mm'}
        msg = 'Trying to add new coord but no points defined'
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.add_coord(cube, coord_name, changes)

    def test_fails_points_greater_than_1(self):
        """Test that add_coord fails if points greater than 1 """
        plugin = CubeCombiner('-')
        coord_name = 'threshold'
        cube = create_cube_with_threshold()
        cube.remove_coord(coord_name)
        cube = iris.util.squeeze(cube)
        changes = {'points': [0.1, 2.0]}
        msg = 'Can not add a coordinate of length > 1'
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.add_coord(cube, coord_name, changes)

    def test_warning_messages(self):
        """Test that warning messages is raised correctly. """
        plugin = CubeCombiner('-', warnings_on=True)
        coord_name = 'threshold'
        cube = create_cube_with_threshold()
        cube.remove_coord(coord_name)
        cube = iris.util.squeeze(cube)
        changes = {'points': [2.0], 'bounds': [0.1, 2.0], 'units': 'mm'}
        warning_msg = "Adding new coordinate"
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            plugin.add_coord(cube, coord_name, changes)
            self.assertTrue(any(item.category == UserWarning
                                for item in warning_list))
            self.assertTrue(any(warning_msg in str(item)
                                for item in warning_list))


class Test_update_coord(IrisTest):

    """Test the update_coord method."""

    def test_basic(self):
        """Test update_coord returns a Cube and updates coord correctly. """
        plugin = CubeCombiner('-')
        cube = create_cube_with_threshold()
        changes = {'points': [2.0], 'bounds': [0.1, 2.0], 'units': 'mm'}
        result = plugin.update_coord(cube, 'threshold', changes)
        self.assertIsInstance(result, Cube)
        self.assertArrayEqual(result.coord('threshold').points,
                              np.array([2.0]))
        self.assertArrayEqual(result.coord('threshold').bounds,
                              np.array([[0.1, 2.0]]))
        self.assertEqual(str(result.coord('threshold').units),
                         'mm')

    def test_coords_deleted(self):
        """Test update_coord deletes coordinate. """
        plugin = CubeCombiner('-')
        cube = create_cube_with_threshold()
        changes = 'delete'
        result = plugin.update_coord(cube, 'threshold', changes)
        found_key = 'threshold' in [coord.name() for coord in result.coords()]
        self.assertArrayEqual(found_key,
                              False)

    def test_coords_deleted_fails(self):
        """Test update_coord fails to delete coord of len > 1. """
        plugin = CubeCombiner('-')
        cube = create_cube_with_threshold()
        changes = 'delete'
        msg = "Can only remove a coordinate of length 1"
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.update_coord(cube, 'time', changes)

    def test_warning_messages_with_delete(self):
        """Test warning message is raised correctly when deleting coord. """
        plugin = CubeCombiner('-', warnings_on=True)
        coord_name = 'threshold'
        cube = create_cube_with_threshold()
        changes = 'delete'
        warning_msg = "Deleted coordinate"
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            plugin.update_coord(cube, coord_name, changes)
            self.assertTrue(any(item.category == UserWarning
                                for item in warning_list))
            self.assertTrue(any(warning_msg in str(item)
                                for item in warning_list))

    def test_coords_update_fail_points(self):
        """Test that update_coord fails if points do not match. """
        plugin = CubeCombiner('-')
        cube = create_cube_with_threshold()
        changes = {'points': [2.0, 3.0]}
        msg = "Mismatch in points in existing coord and updated metadata"
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.update_coord(cube, 'threshold', changes)

    def test_coords_update_fail_bounds(self):
        """Test update_coord fails if shape of new bounds do not match. """
        plugin = CubeCombiner('-')
        cube = create_cube_with_threshold(threshold_values=[2.0, 3.0])
        changes = {'bounds': [0.1, 2.0]}
        msg = "The shape of the bounds array should be"
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.update_coord(cube, 'threshold', changes)

    def test_coords_update_bounds_succeed(self):
        """Test that update_coord succeeds if bounds do match """
        plugin = CubeCombiner('-')
        cube = create_cube_with_threshold(threshold_values=[2.0, 3.0])
        cube.coord('threshold').guess_bounds()
        changes = {'bounds': [[0.1, 2.0], [2.0, 3.0]]}
        result = plugin.update_coord(cube, 'threshold', changes)
        self.assertArrayEqual(result.coord('threshold').bounds,
                              np.array([[0.1, 2.0], [2.0, 3.0]]))

    def test_coords_update_fails_bounds_differ(self):
        """Test that update_coord succeeds if bounds do match """
        plugin = CubeCombiner('-')
        cube = create_cube_with_threshold(threshold_values=[2.0, 3.0])
        cube.coord('threshold').guess_bounds()
        changes = {'bounds': [[0.1, 2.0], [2.0, 3.0], [3.0, 4.0]]}
        msg = "Mismatch in bounds in existing coord and updated metadata"
        with self.assertRaisesRegexp(ValueError, msg):
            plugin.update_coord(cube, 'threshold', changes)

    def test_warning_messages_with_update(self):
        """Test warning message is raised correctly when updating coord. """
        plugin = CubeCombiner('-', warnings_on=True)
        coord_name = 'threshold'
        cube = create_cube_with_threshold()
        changes = {'points': [2.0], 'bounds': [0.1, 2.0], 'units': 'mm'}
        warning_msg = "Updated coordinate"
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            plugin.update_coord(cube, coord_name, changes)
            self.assertTrue(any(item.category == UserWarning
                                for item in warning_list))
            self.assertTrue(any(warning_msg in str(item)
                                for item in warning_list))


class Test_update_attribute(IrisTest):

    """Test the update_attribute method."""

    def test_basic(self):
        """Test that update_attribute returns a Cube and updates OK. """
        plugin = CubeCombiner('-')
        cube = create_cube_with_threshold()
        attribute_name = 'relative_to_threshold'
        changes = 'between'
        result = plugin.update_attribute(cube, attribute_name, changes)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.attributes['relative_to_threshold'],
                         'between')

    def test_attributes_updated_warnings(self):
        """Test update_attribute updates attributes and gives warning. """
        plugin = CubeCombiner('-', warnings_on=True)
        cube = create_cube_with_threshold()
        attribute_name = 'relative_to_threshold'
        changes = 'between'
        warning_msg = "Adding or updating attribute"
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            result = plugin.update_attribute(cube, attribute_name, changes)
            self.assertTrue(any(item.category == UserWarning
                                for item in warning_list))
            self.assertTrue(any(warning_msg in str(item)
                                for item in warning_list))
            self.assertEqual(result.attributes['relative_to_threshold'],
                             'between')

    def test_attributes_added(self):
        """Test update_attribute adds attributeOK. """
        plugin = CubeCombiner('-')
        cube = create_cube_with_threshold()
        attribute_name = 'new_attribute'
        changes = 'new_value'
        result = plugin.update_attribute(cube, attribute_name, changes)
        self.assertEqual(result.attributes['new_attribute'],
                         'new_value')

    def test_attributes_deleted(self):
        """Test update_attribute deletes attribute OK. """
        plugin = CubeCombiner('-')
        cube = create_cube_with_threshold()
        attribute_name = 'relative_to_threshold'
        changes = 'delete'
        result = plugin.update_attribute(cube, attribute_name, changes)
        self.assertFalse('relative_to_threshold' in result.attributes)

    def test_attributes_deleted_warnings(self):
        """Test update_attribute deletes and gives warning. """
        plugin = CubeCombiner('-', warnings_on=True)
        cube = create_cube_with_threshold()
        attribute_name = 'relative_to_threshold'
        changes = 'delete'
        warning_msg = "Deleted attribute"
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            result = plugin.update_attribute(cube, attribute_name, changes)
            self.assertTrue(any(item.category == UserWarning
                                for item in warning_list))
            self.assertTrue(any(warning_msg in str(item)
                                for item in warning_list))
            self.assertFalse('relative_to_threshold' in result.attributes)


class Test_amend_metadata(IrisTest):

    """Test the amend_metadata method."""

    def test_basic(self):
        """Test that the function returns a Cube. """
        plugin = CubeCombiner('-')
        cube = create_cube_with_threshold()
        result = plugin.amend_metadata(cube, 'new_cube_name', np.dtype,
                                       None, None)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), 'new_cube_name')

    def test_attributes_updated_and_added(self):
        """Test amend_metadata  updates and adds attributes OK. """
        plugin = CubeCombiner('-')
        cube = create_cube_with_threshold()
        attributes = {'relative_to_threshold': 'between',
                      'new_attribute': 'new_value'}
        result = plugin.amend_metadata(cube, 'new_cube_name', np.dtype,
                                       None, attributes)
        self.assertEqual(result.attributes['relative_to_threshold'],
                         'between')
        self.assertEqual(result.attributes['new_attribute'],
                         'new_value')

    def test_attributes_deleted(self):
        """Test amend_metadata  updates attributes OK. """
        plugin = CubeCombiner('-')
        cube = create_cube_with_threshold()
        attributes = {'relative_to_threshold': 'delete'}
        result = plugin.amend_metadata(cube, 'new_cube_name', np.dtype,
                                       None, attributes)
        self.assertFalse('relative_to_threshold' in result.attributes)

    def test_coords_updated(self):
        """Test amend_metadata returns a Cube and updates coord correctly. """
        plugin = CubeCombiner('-')
        cube = create_cube_with_threshold()
        updated_coords = {'threshold': {'points': [2.0]},
                          'time': {'points': [402193.5, 402194.5]}}
        result = plugin.amend_metadata(cube, 'new_cube_name', np.dtype,
                                       updated_coords, None)
        self.assertArrayEqual(result.coord('threshold').points,
                              np.array([2.0]))
        self.assertArrayEqual(result.coord('time').points,
                              np.array([402193.5, 402194.5]))

    def test_coords_deleted_and_adds(self):
        """Test amend metadata deletes and adds coordinate. """
        plugin = CubeCombiner('-')
        cube = create_cube_with_threshold()
        coords = {'threshold': 'delete',
                  'new_coord': {'points': [2.0]}}
        result = plugin.amend_metadata(cube, 'new_cube_name', np.dtype,
                                       coords, None)
        found_key = 'threshold' in [coord.name() for coord in result.coords()]
        self.assertFalse(found_key)
        self.assertArrayEqual(result.coord('new_coord').points,
                              np.array([2.0]))


class Test_combine(IrisTest):

    """Test the combine method."""

    def setUp(self):
        """ Set up cubes for testing. """
        self.cube1 = create_cube_with_threshold()
        data = np.zeros((1, 2, 2, 2))
        data[0, 0, :, :] = 0.1
        data[0, 1, :, :] = 0.4
        self.cube2 = create_cube_with_threshold(data=data)
        data2 = np.zeros((1, 2, 2, 2))
        data2[0, 0, :, :] = 0.1
        data2[0, 1, :, :] = 0.8
        self.cube3 = create_cube_with_threshold(data=data2)

    def test_basic(self):
        """Test that the function returns a Cube. """
        operation = '*'
        plugin = CubeCombiner(operation)
        cube1 = self.cube1
        cube2 = cube1.copy()
        result = plugin.combine(cube1, cube2, operation)
        self.assertIsInstance(result, Cube)
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[0, 0, :, :] = 0.25
        expected_data[0, 1, :, :] = 0.36
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_add(self):
        """Test combine adds the cubes correctly. """
        operation = '+'
        plugin = CubeCombiner(operation)
        result = plugin.combine(self.cube1, self.cube2,
                                operation)
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[0, 0, :, :] = 0.6
        expected_data[0, 1, :, :] = 1.0
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_minus(self):
        """Test combine minus the cubes correctly. """
        operation = '-'
        plugin = CubeCombiner(operation)
        result = plugin.combine(self.cube1, self.cube2,
                                operation)
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[0, 0, :, :] = 0.4
        expected_data[0, 1, :, :] = 0.2
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_max(self):
        """Test combine finds the max of the cubes correctly."""
        operation = 'max'
        plugin = CubeCombiner(operation)
        result = plugin.combine(self.cube1, self.cube3,
                                operation)
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[0, 0, :, :] = 0.5
        expected_data[0, 1, :, :] = 0.8
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_min(self):
        """Test combine finds the min of the cubes correctly."""
        operation = 'min'
        plugin = CubeCombiner(operation)
        result = plugin.combine(self.cube1, self.cube3,
                                operation)
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[0, 0, :, :] = 0.1
        expected_data[0, 1, :, :] = 0.6
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_mean(self):
        """Test that the function adds the cubes correctly for mean."""
        operation = '+'
        plugin = CubeCombiner(operation)
        result = plugin.combine(self.cube1, self.cube3,
                                operation)
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[0, 0, :, :] = 0.6
        expected_data[0, 1, :, :] = 1.4
        self.assertArrayAlmostEqual(result.data, expected_data)


class Test_process(IrisTest):

    """Test the plugin combines the cubelist into a cube."""

    def setUp(self):
        """ Set up cubes for testing. """
        self.cube1 = create_cube_with_threshold()
        data = np.zeros((1, 2, 2, 2))
        data[0, 0, :, :] = 0.1
        data[0, 1, :, :] = 0.4
        self.cube2 = create_cube_with_threshold(data=data)
        data2 = np.zeros((1, 2, 2, 2))
        data2[0, 0, :, :] = 0.9
        data2[0, 1, :, :] = 0.2
        self.cube3 = create_cube_with_threshold(data=data2)

    def test_basic(self):
        """Test that the plugin returns a Cube. """
        plugin = CubeCombiner('+')
        cubelist = iris.cube.CubeList([self.cube1, self.cube1])
        result = plugin.process(cubelist, 'new_cube_name')
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), 'new_cube_name')
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[:, 0, :, :] = 1.0
        expected_data[:, 1, :, :] = 1.2
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_mean(self):
        """Test that the plugin calculates the mean correctly. """
        plugin = CubeCombiner('mean')
        cubelist = iris.cube.CubeList([self.cube1, self.cube2])
        result = plugin.process(cubelist, 'new_cube_name')
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[:, 0, :, :] = 0.3
        expected_data[:, 1, :, :] = 0.5
        self.assertEqual(result.name(), 'new_cube_name')
        self.assertArrayAlmostEqual(result.data, expected_data)

    def test_mean_multi_cube(self):
        """Test that the plugin calculates the mean for three cubes. """
        plugin = CubeCombiner('mean')
        cubelist = iris.cube.CubeList([self.cube1,
                                       self.cube2,
                                       self.cube3])
        result = plugin.process(cubelist, 'new_cube_name')
        expected_data = np.zeros((1, 2, 2, 2))
        expected_data[:, 0, :, :] = 0.5
        expected_data[:, 1, :, :] = 0.4
        self.assertEqual(result.name(), 'new_cube_name')
        self.assertArrayAlmostEqual(result.data, expected_data)


if __name__ == '__main__':
    unittest.main()
