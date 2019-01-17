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
"""Unit tests for the cube_metadata utilities."""

import unittest
import numpy as np
from copy import copy, deepcopy
from datetime import datetime as dt

import iris
from iris.cube import Cube
from iris.tests import IrisTest

from improver.utilities.cube_metadata import (
    add_coord,
    add_history_attribute,
    amend_metadata,
    delete_attributes,
    resolve_metadata_diff,
    update_attribute,
    update_cell_methods,
    update_coord,
    update_stage_v110_metadata)
from improver.utilities.warnings_handler import ManageWarnings

from improver.tests.set_up_test_cubes import (
    set_up_variable_cube, set_up_probability_cube, add_coordinate)


def create_cube_with_threshold(data=None, threshold_values=None):
    """
    Create a cube with threshold coord.  Data and threshold values MUST be
    provided as float32 (not float64), or cube setup will fail.
    """
    if threshold_values is None:
        threshold_values = np.array([1.0], dtype=np.float32)

    if data is None:
        data = np.zeros((len(threshold_values), 2, 2, 2), dtype=np.float32)
        data[:, 0, :, :] = 0.5
        data[:, 1, :, :] = 0.6

    long_name = "probability_of_rainfall_rate"
    units = "m s^-1"

    cube = set_up_probability_cube(
        data[:, 0, :, :], threshold_values, variable_name=long_name,
        threshold_units=units, time=dt(2015, 11, 19, 1, 30),
        frt=dt(2015, 11, 18, 22, 0))

    time_points = [dt(2015, 11, 19, 0, 30), dt(2015, 11, 19, 1, 30)]
    cube = add_coordinate(
        cube, time_points, "time", order=[1, 0, 2, 3], is_datetime=True)

    cube.data = data

    return cube


class Test_update_stage_v110_metadata(IrisTest):
    """Test the update_stage_v110_metadata function"""

    def setUp(self):
        """Set up variables for use in testing."""
        data = 275.*np.ones((3, 3), dtype=np.float32)
        self.cube = set_up_variable_cube(data)

    def test_basic(self):
        """Test that cube is unchanged and function returns False"""
        result = self.cube.copy()
        output = update_stage_v110_metadata(result)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(result.data, self.cube.data)
        self.assertEqual(result.attributes, self.cube.attributes)
        self.assertFalse(output)

    def test_update_ukv(self):
        """Test that cube attributes from ukv 1.1.0 are updated"""
        self.cube.attributes['grid_id'] = 'ukvx_standard_v1'
        output = update_stage_v110_metadata(self.cube)
        self.assertTrue('mosg__grid_type' in self.cube.attributes.keys())
        self.assertTrue('mosg__model_configuration' in
                        self.cube.attributes.keys())
        self.assertTrue('mosg__grid_domain' in self.cube.attributes.keys())
        self.assertTrue('mosg__grid_version' in self.cube.attributes.keys())
        self.assertFalse('grid_id' in self.cube.attributes.keys())
        self.assertEqual('standard', self.cube.attributes['mosg__grid_type'])
        self.assertEqual('uk_det',
                         self.cube.attributes['mosg__model_configuration'])
        self.assertEqual('uk_extended',
                         self.cube.attributes['mosg__grid_domain'])
        self.assertEqual('1.1.0', self.cube.attributes['mosg__grid_version'])
        self.assertTrue(output)


class Test_add_coord(IrisTest):
    """Test the add_coord method."""

    def setUp(self):
        """Set up information for testing."""
        self.cube = create_cube_with_threshold()
        self.changes = {
            'points': [2.0],
            'bounds': [0.1, 2.0],
            'units': 'mm'
            }

    def test_basic(self):
        """Test that add_coord returns a Cube and adds coord correctly. """
        coord_name = 'threshold'
        self.cube.remove_coord(coord_name)
        cube = iris.util.squeeze(self.cube)
        result = add_coord(cube, coord_name, self.changes)
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
        self.cube.remove_coord(coord_name)
        cube = iris.util.squeeze(self.cube)
        changes = {'bounds': [0.1, 2.0], 'units': 'mm'}
        msg = 'Trying to add new coord but no points defined'
        with self.assertRaisesRegex(ValueError, msg):
            add_coord(cube, coord_name, changes)

    def test_fails_points_greater_than_1(self):
        """Test that add_coord fails if points greater than 1 """
        coord_name = 'threshold'
        self.cube.remove_coord(coord_name)
        cube = iris.util.squeeze(self.cube)
        changes = {'points': [0.1, 2.0]}
        msg = 'Can not add a coordinate of length > 1'
        with self.assertRaisesRegex(ValueError, msg):
            add_coord(cube, coord_name, changes)

    @ManageWarnings(record=True)
    def test_warning_messages(self, warning_list=None):
        """Test that warning messages is raised correctly. """
        coord_name = 'threshold'
        self.cube.remove_coord(coord_name)
        cube = iris.util.squeeze(self.cube)
        warning_msg = "Adding new coordinate"
        add_coord(cube, coord_name, self.changes, warnings_on=True)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))


class Test_update_coord(IrisTest):
    """Test the update_coord method."""

    def setUp(self):
        """Set up test cube and thresholds"""
        self.cube = create_cube_with_threshold()
        self.thresholds = np.array([2.0, 3.0], dtype=np.float32)

    def test_basic(self):
        """Test update_coord returns a Cube and updates coord correctly. """
        changes = {'points': [2.0], 'bounds': [0.1, 2.0]}
        result = update_coord(self.cube, 'threshold', changes)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.coord('threshold').points,
                                    np.array([2.0], dtype=np.float32))
        self.assertArrayAlmostEqual(result.coord('threshold').bounds,
                                    np.array([[0.1, 2.0]], dtype=np.float32))

    def test_convert_units(self):
        """Test update_coord returns a Cube and converts units correctly. """
        cube = create_cube_with_threshold()
        changes = {'units': 'km s-1'}
        result = update_coord(cube, 'threshold', changes)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.coord('threshold').points,
                         np.array([0.001], dtype=np.float32))
        self.assertEqual(str(result.coord('threshold').units), 'km s-1')

    def test_coords_deleted(self):
        """Test update_coord deletes coordinate. """
        changes = 'delete'
        result = update_coord(self.cube, 'threshold', changes)
        found_key = 'threshold' in [coord.name() for coord in result.coords()]
        self.assertArrayEqual(found_key,
                              False)

    def test_coords_deleted_fails(self):
        """Test update_coord fails to delete coord of len > 1. """
        changes = 'delete'
        msg = "Can only remove a coordinate of length 1"
        with self.assertRaisesRegex(ValueError, msg):
            update_coord(self.cube, 'time', changes)

    @ManageWarnings(record=True)
    def test_warning_messages_with_delete(self, warning_list=None):
        """Test warning message is raised correctly when deleting coord. """
        coord_name = 'threshold'
        changes = 'delete'
        warning_msg = "Deleted coordinate"
        update_coord(self.cube, coord_name, changes, warnings_on=True)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))

    def test_coords_update_fail_points(self):
        """Test that update_coord fails if points do not match. """
        changes = {'points': [2.0, 3.0]}
        msg = "Mismatch in points in existing coord and updated metadata"
        with self.assertRaisesRegex(ValueError, msg):
            update_coord(self.cube, 'threshold', changes)

    def test_coords_update_fail_bounds(self):
        """Test update_coord fails if shape of new bounds do not match. """
        cube = create_cube_with_threshold(threshold_values=self.thresholds)
        changes = {'bounds': [0.1, 2.0]}
        msg = "The shape of the bounds array should be"
        with self.assertRaisesRegex(ValueError, msg):
            update_coord(cube, 'threshold', changes)

    def test_coords_update_bounds_succeed(self):
        """Test that update_coord succeeds if bounds do match """
        cube = create_cube_with_threshold(threshold_values=self.thresholds)
        cube.coord('threshold').guess_bounds()
        changes = {'bounds': [[0.1, 2.0], [2.0, 3.0]]}
        result = update_coord(cube, 'threshold', changes)
        self.assertArrayEqual(result.coord('threshold').bounds,
                              np.array([[0.1, 2.0], [2.0, 3.0]],
                                       dtype=np.float32))

    def test_coords_update_fails_bounds_differ(self):
        """Test that update_coord fails if bounds differ."""
        cube = create_cube_with_threshold(threshold_values=self.thresholds)
        cube.coord('threshold').guess_bounds()
        changes = {'bounds': [[0.1, 2.0], [2.0, 3.0], [3.0, 4.0]]}
        msg = "Mismatch in bounds in existing coord and updated metadata"
        with self.assertRaisesRegex(ValueError, msg):
            update_coord(cube, 'threshold', changes)

    @ManageWarnings(record=True)
    def test_warning_messages_with_update(self, warning_list=None):
        """Test warning message is raised correctly when updating coord. """
        coord_name = 'threshold'
        changes = {'points': [2.0], 'bounds': [0.1, 2.0]}
        warning_msg = "Updated coordinate"
        update_coord(self.cube, coord_name, changes, warnings_on=True)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))

    def test_incompatible_changes_requested(self):
        """Test that update_coord raises an exception if 'points' and 'units'
        are requested to be changed."""
        cube = create_cube_with_threshold()
        changes = {'points': [2.0, 3.0], 'units': 'mm/hr'}
        msg = "When updating a coordinate"
        with self.assertRaisesRegex(ValueError, msg):
            update_coord(cube, 'threshold', changes)

    def test_alternative_incompatible_changes_requested(self):
        """Test that update_coord raises an exception if 'bounds' and 'units'
        are requested to be changed."""
        cube = create_cube_with_threshold()
        changes = {'bounds': [0.1, 2.0], 'units': 'mm/hr'}
        msg = "When updating a coordinate"
        with self.assertRaisesRegex(ValueError, msg):
            update_coord(cube, 'threshold', changes)


class Test_update_attribute(IrisTest):
    """Test the update_attribute method."""

    def setUp(self):
        """Set up test cube"""
        self.cube = create_cube_with_threshold()

    def test_basic(self):
        """Test that update_attribute returns a Cube and updates OK. """
        attribute_name = 'relative_to_threshold'
        changes = 'between'
        result = update_attribute(self.cube, attribute_name, changes)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.attributes['relative_to_threshold'],
                         'between')

    @ManageWarnings(record=True)
    def test_attributes_updated_warnings(self, warning_list=None):
        """Test update_attribute updates attributes and gives warning. """
        attribute_name = 'relative_to_threshold'
        changes = 'between'
        warning_msg = "Adding or updating attribute"
        result = update_attribute(self.cube, attribute_name, changes,
                                  warnings_on=True)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertEqual(result.attributes['relative_to_threshold'],
                         'between')

    def test_attributes_added(self):
        """Test update_attribute adds attribute OK. """
        attribute_name = 'new_attribute'
        changes = 'new_value'
        result = update_attribute(self.cube, attribute_name, changes)
        self.assertEqual(result.attributes['new_attribute'],
                         'new_value')

    def test_history_attribute_added(self):
        """Test update_attribute adds attribute OK. """
        attribute_name = 'history'
        changes = ['add', "Nowcast"]
        result = update_attribute(self.cube, attribute_name, changes)
        self.assertTrue("history" in result.attributes.keys())

    def test_failure_to_add_history_attribute(self):
        """Test update_attribute doesn't adds non-history attribute. """
        attribute_name = 'new_attribute'
        changes = 'add'
        msg = "Only the history attribute can be added"
        with self.assertRaisesRegex(ValueError, msg):
            update_attribute(self.cube, attribute_name, changes)

    def test_attributes_deleted(self):
        """Test update_attribute deletes attribute OK. """
        attribute_name = 'relative_to_threshold'
        changes = 'delete'
        result = update_attribute(self.cube, attribute_name, changes)
        self.assertFalse('relative_to_threshold' in result.attributes)

    @ManageWarnings(record=True)
    def test_attributes_deleted_warnings(self, warning_list=None):
        """Test update_attribute deletes and gives warning. """
        attribute_name = 'relative_to_threshold'
        changes = 'delete'
        warning_msg = "Deleted attribute"
        result = update_attribute(self.cube, attribute_name, changes,
                                  warnings_on=True)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertFalse('relative_to_threshold' in result.attributes)

    def test_attributes_deleted_when_not_present(self):
        """Test update_attribute copes when an attribute is requested to be
        deleted, but this attribute is not available on the input cube."""
        attribute_name = 'invalid_name'
        changes = 'delete'
        result = update_attribute(self.cube, attribute_name, changes)
        self.assertFalse('invalid_name' in result.attributes)


class Test_update_cell_methods(IrisTest):
    """Test that the cell methods are updated."""

    def setUp(self):
        """Set up test cube"""
        self.cube = create_cube_with_threshold()

    def test_add_cell_method(self):
        """Test adding a cell method, where all cell method elements are
        present i.e. method, coords, intervals and comments."""
        cell_methods = {'action': 'add',
                        'method': 'point',
                        'coords': 'time',
                        'intervals': (),
                        'comments': ()}
        cm = deepcopy(cell_methods)
        cm.pop('action')
        expected_cell_method = iris.coords.CellMethod(**cm)
        update_cell_methods(self.cube, cell_methods)
        self.assertEqual((expected_cell_method,), self.cube.cell_methods)

    def test_add_cell_method_partial_information(self):
        """Test adding a cell method, where there is only partial
        information available i.e. just method and coords."""
        cell_methods = {'action': 'add',
                        'method': 'point',
                        'coords': 'time'}
        cm = deepcopy(cell_methods)
        cm.pop('action')
        expected_cell_method = iris.coords.CellMethod(**cm)
        update_cell_methods(self.cube, cell_methods)
        self.assertEqual((expected_cell_method,), self.cube.cell_methods)

    def test_add_cell_method_empty_method(self):
        """Test add a cell method, where the method element is specified
        as an emtpy string."""
        cell_methods = {'action': 'add',
                        'method': '',
                        'coords': 'time',
                        'intervals': (),
                        'comments': ()}
        msg = "No method has been specified within the cell method"
        with self.assertRaisesRegex(ValueError, msg):
            update_cell_methods(self.cube, cell_methods)

    def test_add_cell_method_no_coords(self):
        """Test add a cell method, where no coords element is specified."""
        cell_methods = {'action': 'add',
                        'method': 'point',
                        'coords': (),
                        'intervals': (),
                        'comments': ()}
        cm = deepcopy(cell_methods)
        cm.pop('action')
        expected_cell_method = iris.coords.CellMethod(**cm)
        update_cell_methods(self.cube, cell_methods)
        self.assertEqual((expected_cell_method,), self.cube.cell_methods)

    def test_add_cell_method_already_on_cube(self):
        """Test that there is no change to the cell method, if the specified
        cell method is already on the cube."""
        cell_methods = {'action': 'add',
                        'method': 'point',
                        'coords': 'time'}
        cm = deepcopy(cell_methods)
        cm.pop('action')
        self.cube.cell_methods = (iris.coords.CellMethod(**cm),)
        expected_cell_method = iris.coords.CellMethod(**cm)
        update_cell_methods(self.cube, cell_methods)
        self.assertEqual((expected_cell_method,), self.cube.cell_methods)

    def test_add_additional_cell_method_to_cube(self):
        """Test that there is no change to the cell method, if the specified
        cell method is already on the cube."""
        existing_cell_methods = {'action': 'add',
                                 'method': 'point',
                                 'coords': 'time'}
        additional_cell_methods = {'action': 'add',
                                   'method': 'mean',
                                   'coords': 'realization'}
        cm = deepcopy(existing_cell_methods)
        cm.pop('action')
        self.cube.cell_methods = (iris.coords.CellMethod(**cm),)
        cm = deepcopy(additional_cell_methods)
        cm.pop('action')
        expected_cell_method = iris.coords.CellMethod(**cm)
        update_cell_methods(self.cube, additional_cell_methods)
        self.assertTrue(expected_cell_method in self.cube.cell_methods)

    def test_remove_cell_method(self):
        """Test removing a cell method, when the specified cell method is
        already on the input cube."""
        cell_methods = {'action': 'delete',
                        'method': 'point',
                        'coords': 'time',
                        'intervals': (),
                        'comments': ()}
        cm = deepcopy(cell_methods)
        cm.pop('action')
        self.cube.cell_methods = (iris.coords.CellMethod(**cm),)
        update_cell_methods(self.cube, cell_methods)
        self.assertEqual(self.cube.cell_methods, ())

    def test_add_cell_method_no_action(self):
        """Test adding a cell method, where no action is specified."""
        cell_methods = {'method': 'point',
                        'coords': 'time',
                        'intervals': (),
                        'comments': ()}
        msg = "No action has been specified within the cell method definition."
        with self.assertRaisesRegex(ValueError, msg):
            update_cell_methods(self.cube, cell_methods)


class Test_amend_metadata(IrisTest):
    """Test the amend_metadata method."""

    def setUp(self):
        """Set up test cube"""
        self.cube = create_cube_with_threshold()

    def test_basic(self):
        """Test that the function returns a Cube. """
        result = amend_metadata(
            self.cube, name='new_cube_name', data_type=np.dtype)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), 'new_cube_name')

    def test_attributes_updated_and_added(self):
        """Test amend_metadata updates and adds attributes OK. """
        attributes = {'relative_to_threshold': 'between',
                      'new_attribute': 'new_value'}
        result = amend_metadata(
            self.cube, name='new_cube_name', data_type=np.dtype,
            attributes=attributes)
        self.assertEqual(result.attributes['relative_to_threshold'],
                         'between')
        self.assertEqual(result.attributes['new_attribute'],
                         'new_value')

    def test_attributes_deleted(self):
        """Test amend_metadata updates attributes OK. """
        attributes = {'relative_to_threshold': 'delete'}
        result = amend_metadata(
            self.cube, name='new_cube_name', data_type=np.dtype,
            attributes=attributes)
        self.assertFalse('relative_to_threshold' in result.attributes)

    def test_coords_updated(self):
        """Test amend_metadata returns a Cube and updates coord correctly. """
        updated_coords = {'threshold': {'points': [2.0]},
                          'time': {'points': [402193.5, 402194.5]}}
        result = amend_metadata(
            self.cube, name='new_cube_name', data_type=np.dtype,
            coordinates=updated_coords)
        self.assertArrayEqual(result.coord('threshold').points,
                              np.array([2.0]))
        self.assertArrayEqual(result.coord('time').points,
                              np.array([402193.5, 402194.5]))

    def test_coords_deleted_and_adds(self):
        """Test amend metadata deletes and adds coordinate. """
        coords = {'threshold': 'delete',
                  'new_coord': {'points': [2.0]}}
        result = amend_metadata(
            self.cube, name='new_cube_name', data_type=np.dtype,
            coordinates=coords)
        found_key = 'threshold' in [coord.name() for coord in result.coords()]
        self.assertFalse(found_key)
        self.assertArrayEqual(result.coord('new_coord').points,
                              np.array([2.0]))

    def test_cell_method_updated_and_added(self):
        """Test amend_metadata updates and adds a cell method. """
        cell_methods = {"1": {"action": "add",
                              "method": "point",
                              "coords": "time"}}
        cm = deepcopy(cell_methods)
        cm["1"].pop("action")
        expected_cell_method = iris.coords.CellMethod(**cm["1"])
        result = amend_metadata(
            self.cube, name='new_cube_name', data_type=np.dtype,
            cell_methods=cell_methods)
        self.assertTrue(expected_cell_method in result.cell_methods)

    def test_cell_method_deleted(self):
        """Test amend_metadata updates attributes OK. """
        cell_methods = {"1": {"action": "delete",
                              "method": "point",
                              "coords": "time"}}
        cm = deepcopy(cell_methods)
        cm["1"].pop("action")
        cell_method = iris.coords.CellMethod(**cm["1"])
        self.cube.cell_methods = (cell_method,)
        result = amend_metadata(
            self.cube, name='new_cube_name', data_type=np.dtype,
            cell_methods=cell_methods)
        self.assertEqual(result.cell_methods, ())

    def test_convert_units(self):
        """Test amend_metadata updates attributes OK. """
        changes = "Celsius"
        cube = set_up_variable_cube(
            np.ones((3, 3), dtype=np.float32), units='K')
        result = amend_metadata(cube, units=changes)
        self.assertEqual(result.units, "Celsius")

    @ManageWarnings(record=True)
    def test_warnings_on_works(self, warning_list=None):
        """Test amend_metadata raises warnings """
        updated_attributes = {'new_attribute': 'new_value'}
        updated_coords = {'threshold': {'points': [2.0]}}
        warning_msg_attr = "Adding or updating attribute"
        warning_msg_coord = "Updated coordinate"
        result = amend_metadata(
            self.cube, name='new_cube_name', data_type=np.dtype,
            coordinates=updated_coords, attributes=updated_attributes,
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
        self.cube1 = create_cube_with_threshold()
        self.cube2 = add_coordinate(
            self.cube1, np.arange(2).astype(np.float32),
            "realization", dtype=np.float32)

    def test_basic(self):
        """Test that the function returns a tuple of Cubes. """
        cube2 = self.cube1.copy()
        result = resolve_metadata_diff(self.cube1, cube2)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], Cube)
        self.assertIsInstance(result[1], Cube)

    def test_mismatching_coords_wrong_shape(self):
        """Test raises an error if shape do not match. """
        cube2 = create_cube_with_threshold(
            threshold_values=np.array([1.0, 2.0], dtype=np.float32))
        msg = "Can not combine cubes, mismatching shapes"
        with self.assertRaisesRegex(ValueError, msg):
            resolve_metadata_diff(self.cube1, cube2)

    def test_mismatching_coords_1d_coord_pos0_on_cube1(self):
        """Test missing coord on cube2. Coord is leading coord in cube1."""
        cube2 = self.cube1.copy()
        cube2.remove_coord('threshold')
        cube2 = iris.util.squeeze(cube2)
        result = resolve_metadata_diff(self.cube1, cube2)
        self.assertIsInstance(result, tuple)
        self.assertArrayEqual(result[0].shape, np.array([1, 2, 2, 2]))
        self.assertArrayEqual(result[1].shape, np.array([1, 2, 2, 2]))

    def test_mismatching_coords_1d_coord_pos0_on_cube2(self):
        """Test missing coord on cube1. Coord is leading coord in cube2."""
        cube2 = self.cube1.copy()
        self.cube1.remove_coord('threshold')
        self.cube1 = iris.util.squeeze(self.cube1)
        result = resolve_metadata_diff(self.cube1, cube2)
        self.assertIsInstance(result, tuple)
        self.assertArrayEqual(result[0].shape, np.array([2, 2, 2]))
        self.assertArrayEqual(result[1].shape, np.array([2, 2, 2]))

    def test_mismatching_coords_1d_coord_pos1_cube1(self):
        """Test missing 1d coord on cube2.
           Coord is not leading coord in cube1."""
        cube2 = self.cube2.copy()
        cube2.remove_coord('threshold')
        cube2 = iris.util.squeeze(cube2)
        result = resolve_metadata_diff(self.cube2, cube2)
        self.assertIsInstance(result, tuple)
        self.assertArrayEqual(result[0].shape, np.array([2, 1, 2, 2, 2]))
        self.assertArrayEqual(result[1].shape, np.array([2, 1, 2, 2, 2]))

    def test_mismatching_coords_1d_coord_pos1_cube2(self):
        """Test missing 1d coord on cube1.
           Coord is not leading coord in cube2."""
        cube2 = self.cube2.copy()
        self.cube2.remove_coord('threshold')
        self.cube2 = iris.util.squeeze(self.cube2)
        result = resolve_metadata_diff(self.cube2, cube2)
        self.assertIsInstance(result, tuple)
        self.assertArrayEqual(result[0].shape, np.array([2, 2, 2, 2]))
        self.assertArrayEqual(result[1].shape, np.array([2, 2, 2, 2]))

    def test_mismatching_coords_same_shape(self):
        """Test works with mismatching coords but coords same shape."""
        cube2 = create_cube_with_threshold(
            threshold_values=np.array([2.0], dtype=np.float32))
        result = resolve_metadata_diff(self.cube1, cube2)
        self.assertIsInstance(result, tuple)
        self.assertArrayEqual(result[0].coord('threshold').points,
                              np.array([1.0]))
        self.assertArrayEqual(result[1].coord('threshold').points,
                              np.array([2.0]))

    @ManageWarnings(record=True)
    def test_warnings_on_work(self, warning_list=None):
        """Test warning messages are given if warnings_on is set."""
        cube2 = self.cube1.copy()
        cube2.remove_coord('threshold')
        cube2 = iris.util.squeeze(cube2)
        warning_msg = 'Adding new coordinate'
        result = resolve_metadata_diff(self.cube1, cube2,
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


class Test_add_history_attribute(IrisTest):
    """Test the add_history_attribute function."""

    def test_add_history(self):
        """Test that a history attribute has been added."""
        cube = set_up_variable_cube(np.ones((3, 3), dtype=np.float32))
        add_history_attribute(cube, "Nowcast")
        self.assertTrue("history" in cube.attributes)
        self.assertTrue("Nowcast" in cube.attributes["history"])

    def test_history_already_exists(self):
        """Test that the history attribute is overwritten, if it
        already exists."""
        cube = set_up_variable_cube(np.ones((3, 3), dtype=np.float32))
        old_history = "2018-09-13T11:28:29: StaGE"
        cube.attributes["history"] = old_history
        add_history_attribute(cube, "Nowcast")
        self.assertTrue("history" in cube.attributes)
        self.assertTrue("Nowcast" in cube.attributes["history"])
        self.assertFalse(old_history in cube.attributes["history"])

    def test_history_append(self):
        """Test that the history attribute can be updated."""
        cube = set_up_variable_cube(np.ones((3, 3), dtype=np.float32))
        old_history = "2018-09-13T11:28:29: StaGE"
        cube.attributes["history"] = old_history
        add_history_attribute(cube, "Nowcast", append=True)
        self.assertTrue("history" in cube.attributes)
        self.assertTrue("Nowcast" in cube.attributes["history"])
        self.assertTrue(old_history in cube.attributes["history"])

    def test_history_append_no_existing(self):
        """Test the "append" option doesn't crash when no history exists."""
        cube = set_up_variable_cube(np.ones((3, 3), dtype=np.float32))
        add_history_attribute(cube, "Nowcast", append=True)
        self.assertTrue("history" in cube.attributes)
        self.assertTrue("Nowcast" in cube.attributes["history"])


if __name__ == '__main__':
    unittest.main()
