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
from copy import copy, deepcopy
from datetime import datetime as dt

import iris
import numpy as np
from iris.cube import Cube
from iris.tests import IrisTest

from improver.tests.set_up_test_cubes import (
    set_up_variable_cube, set_up_probability_cube, add_coordinate)
from improver.utilities.cube_checker import find_threshold_coordinate
from improver.utilities.cube_metadata import (
    add_coord,
    add_history_attribute,
    amend_metadata,
    delete_attributes,
    resolve_metadata_diff,
    update_attribute,
    update_cell_methods,
    update_coord,
    update_stage_v110_metadata,
    in_vicinity_name_format,
    extract_diagnostic_name,
    generate_hash,
    create_coordinate_hash)
from improver.utilities.warnings_handler import ManageWarnings


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

    cube = set_up_probability_cube(
        data[:, 0, :, :], threshold_values, variable_name="rainfall_rate",
        threshold_units="m s-1", time=dt(2015, 11, 19, 1, 30),
        frt=dt(2015, 11, 18, 22, 0))

    time_points = [dt(2015, 11, 19, 0, 30), dt(2015, 11, 19, 1, 30)]
    cube = add_coordinate(
        cube, time_points, "time", order=[1, 0, 2, 3], is_datetime=True)

    cube.attributes["attribute_to_update"] = "first_value"

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
        self.changes = {
            'points': [2.0],
            'bounds': [0.1, 2.0],
            'units': 'mm',
            'var_name': 'threshold'
            }
        cube = create_cube_with_threshold()
        self.coord_name = find_threshold_coordinate(cube).name()
        cube.remove_coord(self.coord_name)
        self.cube = iris.util.squeeze(cube)

    def test_basic(self):
        """Test that add_coord returns a Cube and adds coord correctly"""
        result = add_coord(self.cube, self.coord_name, self.changes)
        result_coord = result.coord(self.coord_name)
        self.assertIsInstance(result, Cube)
        self.assertArrayEqual(result_coord.points, np.array([2.0]))
        self.assertArrayEqual(result_coord.bounds, np.array([[0.1, 2.0]]))
        self.assertEqual(str(result_coord.units), 'mm')
        self.assertEqual(result_coord.var_name, "threshold")

    def test_standard_name(self):
        """Test default is for coordinate to be added as standard name"""
        result = add_coord(self.cube, self.coord_name, self.changes)
        self.assertEqual(
            result.coord(self.coord_name).standard_name, self.coord_name)

    def test_long_name(self):
        """Test a coordinate can be added with a name that is not standard"""
        result = add_coord(self.cube, "threshold", self.changes)
        self.assertEqual(result.coord("threshold").long_name, "threshold")

    def test_non_name_value_error(self):
        """Test value errors thrown by iris.Coord (eg invalid units) are
        still raised"""
        self.changes['units'] = 'narwhal'
        msg = 'Failed to parse unit "narwhal"'
        with self.assertRaisesRegex(ValueError, msg):
            add_coord(self.cube, self.coord_name, self.changes)

    def test_fails_no_points(self):
        """Test that add_coord fails if points not included in metadata """
        changes = {'bounds': [0.1, 2.0], 'units': 'mm'}
        msg = 'Trying to add new coord but no points defined'
        with self.assertRaisesRegex(ValueError, msg):
            add_coord(self.cube, self.coord_name, changes)

    def test_fails_points_greater_than_1(self):
        """Test that add_coord fails if points greater than 1 """
        changes = {'points': [0.1, 2.0]}
        msg = 'Can not add a coordinate of length > 1'
        with self.assertRaisesRegex(ValueError, msg):
            add_coord(self.cube, self.coord_name, changes)

    @ManageWarnings(record=True)
    def test_warning_messages(self, warning_list=None):
        """Test that warning messages is raised correctly. """
        warning_msg = "Adding new coordinate"
        add_coord(self.cube, self.coord_name, self.changes, warnings_on=True)
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
        self.coord_name = find_threshold_coordinate(self.cube).name()

    def test_basic(self):
        """Test update_coord returns a Cube and updates coord correctly. """
        changes = {'points': [2.0], 'bounds': [0.1, 2.0]}
        result = update_coord(self.cube, self.coord_name, changes)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.coord(self.coord_name).points,
                                    np.array([2.0], dtype=np.float32))
        self.assertArrayAlmostEqual(result.coord(self.coord_name).bounds,
                                    np.array([[0.1, 2.0]], dtype=np.float32))

    def test_convert_units(self):
        """Test update_coord returns a Cube and converts units correctly. """
        cube = create_cube_with_threshold()
        changes = {'units': 'km s-1'}
        result = update_coord(cube, self.coord_name, changes)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.coord(self.coord_name).points,
                         np.array([0.001], dtype=np.float32))
        self.assertEqual(str(result.coord(self.coord_name).units), 'km s-1')

    def test_coords_deleted(self):
        """Test update_coord deletes coordinate. """
        changes = 'delete'
        result = update_coord(self.cube, self.coord_name, changes)
        found_key = self.coord_name in [
            coord.name() for coord in result.coords()]
        self.assertArrayEqual(found_key, False)

    def test_coords_deleted_fails(self):
        """Test update_coord fails to delete coord of len > 1. """
        changes = 'delete'
        msg = "Can only remove a coordinate of length 1"
        with self.assertRaisesRegex(ValueError, msg):
            update_coord(self.cube, 'time', changes)

    @ManageWarnings(record=True)
    def test_warning_messages_with_delete(self, warning_list=None):
        """Test warning message is raised correctly when deleting coord. """
        changes = 'delete'
        warning_msg = "Deleted coordinate"
        update_coord(self.cube, self.coord_name, changes, warnings_on=True)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))

    def test_coords_update_fail_points(self):
        """Test that update_coord fails if points do not match. """
        changes = {'points': [2.0, 3.0]}
        msg = "Mismatch in points in existing coord and updated metadata"
        with self.assertRaisesRegex(ValueError, msg):
            update_coord(self.cube, self.coord_name, changes)

    def test_coords_update_fail_bounds(self):
        """Test update_coord fails if shape of new bounds do not match. """
        cube = create_cube_with_threshold(threshold_values=self.thresholds)
        changes = {'bounds': [0.1, 2.0]}
        msg = "The shape of the bounds array should be"
        with self.assertRaisesRegex(ValueError, msg):
            update_coord(cube, self.coord_name, changes)

    def test_coords_update_bounds_succeed(self):
        """Test that update_coord succeeds if bounds do match """
        cube = create_cube_with_threshold(threshold_values=self.thresholds)
        cube.coord(self.coord_name).guess_bounds()
        changes = {'bounds': [[0.1, 2.0], [2.0, 3.0]]}
        result = update_coord(cube, self.coord_name, changes)
        self.assertArrayEqual(result.coord(self.coord_name).bounds,
                              np.array([[0.1, 2.0], [2.0, 3.0]],
                                       dtype=np.float32))

    def test_coords_update_fails_bounds_differ(self):
        """Test that update_coord fails if bounds differ."""
        cube = create_cube_with_threshold(threshold_values=self.thresholds)
        cube.coord(self.coord_name).guess_bounds()
        changes = {'bounds': [[0.1, 2.0], [2.0, 3.0], [3.0, 4.0]]}
        msg = "Mismatch in bounds in existing coord and updated metadata"
        with self.assertRaisesRegex(ValueError, msg):
            update_coord(cube, self.coord_name, changes)

    def test_update_attributes(self):
        """Test update attributes associated with a coordinate."""
        cube = create_cube_with_threshold()
        changes = {'attributes': {'spp__relative_to_threshold': "below"}}
        result = update_coord(cube, self.coord_name, changes)
        self.assertIsInstance(result, Cube)
        self.assertEqual(
            result.coord(self.coord_name).attributes, changes["attributes"])

    @ManageWarnings(record=True)
    def test_warning_messages_with_update(self, warning_list=None):
        """Test warning message is raised correctly when updating coord. """
        changes = {'points': [2.0], 'bounds': [0.1, 2.0]}
        warning_msg = "Updated coordinate"
        update_coord(self.cube, self.coord_name, changes, warnings_on=True)
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
            update_coord(cube, self.coord_name, changes)

    def test_alternative_incompatible_changes_requested(self):
        """Test that update_coord raises an exception if 'bounds' and 'units'
        are requested to be changed."""
        cube = create_cube_with_threshold()
        changes = {'bounds': [0.1, 2.0], 'units': 'mm/hr'}
        msg = "When updating a coordinate"
        with self.assertRaisesRegex(ValueError, msg):
            update_coord(cube, self.coord_name, changes)


class Test_update_attribute(IrisTest):
    """Test the update_attribute method."""

    def setUp(self):
        """Set up test cube"""
        self.cube = create_cube_with_threshold()

    def test_basic(self):
        """Test that update_attribute returns a Cube and updates OK. """
        attribute_name = 'attribute_to_update'
        changes = 'second_value'
        result = update_attribute(self.cube, attribute_name, changes)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.attributes['attribute_to_update'],
                         'second_value')

    @ManageWarnings(record=True)
    def test_attributes_updated_warnings(self, warning_list=None):
        """Test update_attribute updates attributes and gives warning. """
        attribute_name = 'attribute_to_update'
        changes = 'second_value'
        warning_msg = "Adding or updating attribute"
        result = update_attribute(self.cube, attribute_name, changes,
                                  warnings_on=True)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertEqual(result.attributes['attribute_to_update'],
                         'second_value')

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
        attribute_name = 'attribute_to_update'
        changes = 'delete'
        result = update_attribute(self.cube, attribute_name, changes)
        self.assertFalse('attribute_to_update' in result.attributes)

    @ManageWarnings(record=True)
    def test_attributes_deleted_warnings(self, warning_list=None):
        """Test update_attribute deletes and gives warning. """
        attribute_name = 'attribute_to_update'
        changes = 'delete'
        warning_msg = "Deleted attribute"
        result = update_attribute(self.cube, attribute_name, changes,
                                  warnings_on=True)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertFalse('attribute_to_update' in result.attributes)

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
        self.threshold_coord = find_threshold_coordinate(self.cube).name()

    def test_basic(self):
        """Test that the function returns a Cube. """
        result = amend_metadata(
            self.cube, name='new_cube_name', data_type=np.dtype)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), 'new_cube_name')

    def test_attributes_updated_and_added(self):
        """Test amend_metadata updates and adds attributes OK. """
        attributes = {'attribute_to_update': 'second_value',
                      'new_attribute': 'new_value'}
        result = amend_metadata(
            self.cube, name='new_cube_name', data_type=np.dtype,
            attributes=attributes)
        self.assertEqual(result.attributes['attribute_to_update'],
                         'second_value')
        self.assertEqual(result.attributes['new_attribute'],
                         'new_value')

    def test_attributes_deleted(self):
        """Test amend_metadata updates attributes OK. """
        attributes = {'attribute_to_update': 'delete'}
        result = amend_metadata(
            self.cube, name='new_cube_name', data_type=np.dtype,
            attributes=attributes)
        self.assertFalse('attribute_to_update' in result.attributes)

    def test_coords_updated(self):
        """Test amend_metadata returns a Cube and updates coord correctly. """
        updated_coords = {self.threshold_coord: {'points': [2.0]},
                          'time': {'points': [402193.5, 402194.5]}}
        result = amend_metadata(
            self.cube, name='new_cube_name', data_type=np.dtype,
            coordinates=updated_coords)
        self.assertArrayEqual(result.coord(self.threshold_coord).points,
                              np.array([2.0]))
        self.assertArrayEqual(result.coord('time').points,
                              np.array([402193.5, 402194.5]))

    def test_coords_deleted_and_adds(self):
        """Test amend metadata deletes and adds coordinate. """
        coords = {self.threshold_coord: 'delete',
                  'new_coord': {'points': [2.0]}}
        result = amend_metadata(
            self.cube, name='new_cube_name', data_type=np.dtype,
            coordinates=coords)
        found_key = self.threshold_coord in [
            coord.name() for coord in result.coords()]
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
        updated_coords = {self.threshold_coord: {'points': [2.0]}}
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
        self.coord_name = find_threshold_coordinate(self.cube1).name()

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
        cube2.remove_coord(self.coord_name)
        cube2 = iris.util.squeeze(cube2)
        result = resolve_metadata_diff(self.cube1, cube2)
        self.assertIsInstance(result, tuple)
        self.assertArrayEqual(result[0].shape, np.array([1, 2, 2, 2]))
        self.assertArrayEqual(result[1].shape, np.array([1, 2, 2, 2]))

    def test_mismatching_coords_1d_coord_pos0_on_cube2(self):
        """Test missing coord on cube1. Coord is leading coord in cube2."""
        cube2 = self.cube1.copy()
        self.cube1.remove_coord(self.coord_name)
        self.cube1 = iris.util.squeeze(self.cube1)
        result = resolve_metadata_diff(self.cube1, cube2)
        self.assertIsInstance(result, tuple)
        self.assertArrayEqual(result[0].shape, np.array([2, 2, 2]))
        self.assertArrayEqual(result[1].shape, np.array([2, 2, 2]))

    def test_mismatching_coords_1d_coord_pos1_cube1(self):
        """Test missing 1d coord on cube2.
           Coord is not leading coord in cube1."""
        cube2 = self.cube2.copy()
        cube2.remove_coord(self.coord_name)
        cube2 = iris.util.squeeze(cube2)
        result = resolve_metadata_diff(self.cube2, cube2)
        self.assertIsInstance(result, tuple)
        self.assertArrayEqual(result[0].shape, np.array([2, 1, 2, 2, 2]))
        self.assertArrayEqual(result[1].shape, np.array([2, 1, 2, 2, 2]))

    def test_mismatching_coords_1d_coord_pos1_cube2(self):
        """Test missing 1d coord on cube1.
           Coord is not leading coord in cube2."""
        cube2 = self.cube2.copy()
        self.cube2.remove_coord(self.coord_name)
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
        self.assertArrayEqual(result[0].coord(self.coord_name).points,
                              np.array([1.0]))
        self.assertArrayEqual(result[1].coord(self.coord_name).points,
                              np.array([2.0]))

    @ManageWarnings(record=True)
    def test_warnings_on_work(self, warning_list=None):
        """Test warning messages are given if warnings_on is set."""
        cube2 = self.cube1.copy()
        cube2.remove_coord(self.coord_name)
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
        long_name = "probability_of_rainfall_rate_above_threshold"
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


class Test_in_vicinity_name_format(IrisTest):
    """Test that the 'in_vicinity' above/below threshold probability
    cube naming function produces the correctly formatted names."""

    def setUp(self):
        """Set up test cube"""
        self.cube = create_cube_with_threshold()
        self.cube.long_name = 'probability_of_X_rate_above_threshold'

    def test_in_vicinity_name_format(self):
        """Test that 'in_vicinity' is added correctly to the name for both
        above and below threshold cases"""
        correct_name_above = (
            'probability_of_X_rate_in_vicinity_above_threshold')
        new_name_above = in_vicinity_name_format(self.cube.name())
        self.cube.rename('probability_of_X_below_threshold')
        correct_name_below = (
            'probability_of_X_in_vicinity_below_threshold')
        new_name_below = in_vicinity_name_format(self.cube.name())
        self.assertEqual(new_name_above, correct_name_above)
        self.assertEqual(new_name_below, correct_name_below)

    def test_between_thresholds(self):
        """Test for "between_thresholds" suffix"""
        self.cube.rename('probability_of_visibility_between_thresholds')
        correct_name = (
            'probability_of_visibility_in_vicinity_between_thresholds')
        new_name = in_vicinity_name_format(self.cube.name())
        self.assertEqual(new_name, correct_name)

    def test_no_above_below_threshold(self):
        """Test the case of name without above/below_threshold is handled
        correctly"""
        self.cube.rename('probability_of_X')
        correct_name_no_threshold = (
            'probability_of_X_in_vicinity')
        new_name_no_threshold = in_vicinity_name_format(self.cube.name())
        self.assertEqual(new_name_no_threshold, correct_name_no_threshold)

    def test_in_vicinity_already_exists(self):
        """Test the case of 'in_vicinity' already existing in the cube name"""
        self.cube.rename('probability_of_X_in_vicinity')
        result = in_vicinity_name_format(self.cube.name())
        self.assertEqual(result, 'probability_of_X_in_vicinity')


class Test_extract_diagnostic_name(IrisTest):
    """Test utility to extract diagnostic name from probability cube name"""

    def test_basic(self):
        """Test correct name is returned from a standard (above threshold)
        probability field"""
        result = extract_diagnostic_name(
            'probability_of_air_temperature_above_threshold')
        self.assertEqual(result, 'air_temperature')

    def test_below_threshold(self):
        """Test correct name is returned from a probability below threshold"""
        result = extract_diagnostic_name(
            'probability_of_air_temperature_below_threshold')
        self.assertEqual(result, 'air_temperature')

    def test_between_thresholds(self):
        """Test correct name is returned from a probability between thresholds
        """
        result = extract_diagnostic_name(
            'probability_of_visibility_in_air_between_thresholds')
        self.assertEqual(result, 'visibility_in_air')

    def test_in_vicinity(self):
        """Test correct name is returned from an "in vicinity" probability.
        Name "cloud_height" is used in this test to illustrate why suffix
        cannot be removed with "rstrip"."""
        diagnostic = 'cloud_height'
        result = extract_diagnostic_name(
            'probability_of_{}_in_vicinity_above_threshold'.format(diagnostic))
        self.assertEqual(result, diagnostic)

    def test_error_not_probability(self):
        """Test exception if input is not a probability cube name"""
        with self.assertRaises(ValueError):
            extract_diagnostic_name('lwe_precipitation_rate')


class Test_generate_hash(IrisTest):
    """Test utility to generate md5 hash codes from a multitude of inputs."""

    def test_string_input(self):
        """Test the expected hash is returned when input is a string type."""

        hash_input = 'this is a test string'
        result = generate_hash(hash_input)
        expected = "8e502f6a5b4a2e0f226649210895cebc"
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_numeric_input(self):
        """Test the expected hash is returned when input is a numeric type."""

        hash_input = 1000
        result = generate_hash(hash_input)
        expected = "d017763f19ef64f920c43fc57413d171"
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_dictionary_input(self):
        """Test the expected hash is returned when input is a dictionary."""

        hash_input = {'one': 1, 'two': 2}
        result = generate_hash(hash_input)
        expected = "4735f4a74dd17d27b383de504a87e324"
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_dictionary_order_variant(self):
        """Test the expected hash is different if the dictionary order is
        different."""

        hash_input1 = {'one': 1, 'two': 2}
        hash_input2 = {'two': 2, 'one': 1}
        result1 = generate_hash(hash_input1)
        result2 = generate_hash(hash_input2)
        self.assertNotEqual(result1, result2)

    def test_cube_input(self):
        """Test the expected hash is returned when input is a cube."""

        hash_input = set_up_variable_cube(np.ones((3, 3)).astype(np.float32))
        result = generate_hash(hash_input)
        expected = "ad664992debed0bdf8f20804e4164691"
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_coordinate_input(self):
        """Test the expected hash is returned when input is a cube
        coordinate."""

        cube = set_up_variable_cube(np.ones((3, 3)).astype(np.float32))
        hash_input = cube.coord('latitude')
        result = generate_hash(hash_input)
        expected = "8c8846a4be49f7aab487353d9ecf623c"
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_numpy_array_type_variant(self):
        """Test the expected hash is different if the numpy array type is
        different."""

        hash_input32 = np.array([np.sqrt(2.)], dtype=np.float32)
        hash_input64 = np.array([np.sqrt(2.)], dtype=np.float64)
        result32 = generate_hash(hash_input32)
        result64 = generate_hash(hash_input64)
        self.assertNotEqual(result32, result64)

    def test_equivalent_input_gives_equivalent_hash(self):
        """Test that creating a hash twice using the same input results in the
        same hash being generated."""

        cube = set_up_variable_cube(np.ones((3, 3)).astype(np.float32))
        hash_input = cube.coord('latitude')
        result1 = generate_hash(hash_input)
        result2 = generate_hash(hash_input)
        self.assertEqual(result1, result2)


class Test_create_coordinate_hash(IrisTest):
    """Test wrapper to hash generation to return a hash based on the x and y
    coordinates of a given cube."""

    def test_basic(self):
        """Test the expected hash is returned for a given cube."""

        hash_input = set_up_variable_cube(np.zeros((3, 3)).astype(np.float32))
        result = create_coordinate_hash(hash_input)
        expected = "fd40f6d5a8e0a347f181d87bcfd445fa"
        self.assertIsInstance(result, str)
        self.assertEqual(result, expected)

    def test_variation(self):
        """Test that two cubes with slightly different coordinates return
        different hashes."""

        hash_input1 = set_up_variable_cube(np.zeros((3, 3)).astype(np.float32))
        hash_input2 = hash_input1.copy()
        latitude = hash_input2.coord('latitude')
        latitude_values = latitude.points * 1.001
        latitude = latitude.copy(points=latitude_values)
        hash_input2.remove_coord("latitude")
        hash_input2.add_dim_coord(latitude, 0)

        result1 = create_coordinate_hash(hash_input1)
        result2 = create_coordinate_hash(hash_input2)
        self.assertNotEqual(result1, result2)


if __name__ == '__main__':
    unittest.main()
