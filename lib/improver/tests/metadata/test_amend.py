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
"""Tests for the improver.metadata.amend module"""

import unittest
from copy import copy, deepcopy
from datetime import datetime as dt

import iris
import numpy as np
from iris.cube import Cube
from iris.tests import IrisTest

from improver.metadata.amend import (
    add_coord,
    set_history_attribute,
    amend_attributes,
    amend_metadata,
    _update_attribute,
    _update_cell_methods,
    _update_coord,
    update_stage_v110_metadata)
from improver.metadata.probabilistic import find_threshold_coordinate
from improver.tests.set_up_test_cubes import (
    set_up_variable_cube, set_up_probability_cube, add_coordinate)
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


class Test_amend_attributes(IrisTest):
    """Test the amend_attributes method."""

    def setUp(self):
        """Set up a cube and dict"""
        self.cube = set_up_variable_cube(
            280*np.ones((3, 3), dtype=np.float32),
            attributes={"mosg__grid_version": "1.3.0",
                        "mosg__model_configuration": "uk_det"})
        self.metadata_dict = {
            "mosg__grid_version": "remove",
            "source": "IMPROVER unit tests",
            "mosg__model_configuration": "other_model"}

    def test_basic(self):
        """Test function adds, removes and modifies attributes as expected"""
        expected_attributes = {
            "source": "IMPROVER unit tests",
            "mosg__model_configuration": "other_model"}
        amend_attributes(self.cube, self.metadata_dict)
        self.assertDictEqual(self.cube.attributes, expected_attributes)


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
        """Test that add_coord returns a Cube and adds coord correctly and does
        not modify the input cube"""
        original_cube = self.cube.copy()
        result = add_coord(self.cube, self.coord_name, self.changes)
        result_coord = result.coord(self.coord_name)
        self.assertIsInstance(result, Cube)
        self.assertArrayEqual(result_coord.points, np.array([2.0]))
        self.assertArrayEqual(result_coord.bounds, np.array([[0.1, 2.0]]))
        self.assertEqual(str(result_coord.units), 'mm')
        self.assertEqual(result_coord.var_name, "threshold")
        self.assertEqual(self.cube, original_cube)

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


class Test__update_coord(IrisTest):
    """Test the _update_coord method."""

    def setUp(self):
        """Set up test cube and thresholds"""
        self.cube = create_cube_with_threshold()
        self.thresholds = np.array([2.0, 3.0], dtype=np.float32)
        self.coord_name = find_threshold_coordinate(self.cube).name()

    def test_basic(self):
        """Test _update_coord returns a Cube and updates coord correctly and
        does not modify the input cube. """
        original_cube = self.cube.copy()
        changes = {'points': [2.0], 'bounds': [0.1, 2.0]}
        result = _update_coord(self.cube, self.coord_name, changes)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(result.coord(self.coord_name).points,
                                    np.array([2.0], dtype=np.float32))
        self.assertArrayAlmostEqual(result.coord(self.coord_name).bounds,
                                    np.array([[0.1, 2.0]], dtype=np.float32))
        self.assertEqual(self.cube, original_cube)

    def test_convert_units(self):
        """Test _update_coord returns a Cube and converts units correctly. """
        cube = create_cube_with_threshold()
        changes = {'units': 'km s-1'}
        result = _update_coord(cube, self.coord_name, changes)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.coord(self.coord_name).points,
                         np.array([0.001], dtype=np.float32))
        self.assertEqual(str(result.coord(self.coord_name).units), 'km s-1')

    def test_coords_deleted(self):
        """Test _update_coord deletes coordinate. """
        changes = 'delete'
        result = _update_coord(self.cube, self.coord_name, changes)
        found_key = self.coord_name in [
            coord.name() for coord in result.coords()]
        self.assertArrayEqual(found_key, False)

    def test_coords_deleted_fails(self):
        """Test _update_coord fails to delete coord of len > 1. """
        changes = 'delete'
        msg = "Can only remove a coordinate of length 1"
        with self.assertRaisesRegex(ValueError, msg):
            _update_coord(self.cube, 'time', changes)

    @ManageWarnings(record=True)
    def test_warning_messages_with_delete(self, warning_list=None):
        """Test warning message is raised correctly when deleting coord. """
        changes = 'delete'
        warning_msg = "Deleted coordinate"
        _update_coord(self.cube, self.coord_name, changes, warnings_on=True)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))

    def test_coords_update_fail_points(self):
        """Test that _update_coord fails if points do not match. """
        changes = {'points': [2.0, 3.0]}
        msg = "Mismatch in points in existing coord and updated metadata"
        with self.assertRaisesRegex(ValueError, msg):
            _update_coord(self.cube, self.coord_name, changes)

    def test_coords_update_fail_bounds(self):
        """Test _update_coord fails if shape of new bounds do not match. """
        cube = create_cube_with_threshold(threshold_values=self.thresholds)
        changes = {'bounds': [0.1, 2.0]}
        msg = "The shape of the bounds array should be"
        with self.assertRaisesRegex(ValueError, msg):
            _update_coord(cube, self.coord_name, changes)

    def test_coords_update_bounds_succeed(self):
        """Test that _update_coord succeeds if bounds do match """
        cube = create_cube_with_threshold(threshold_values=self.thresholds)
        cube.coord(self.coord_name).guess_bounds()
        changes = {'bounds': [[0.1, 2.0], [2.0, 3.0]]}
        result = _update_coord(cube, self.coord_name, changes)
        self.assertArrayEqual(result.coord(self.coord_name).bounds,
                              np.array([[0.1, 2.0], [2.0, 3.0]],
                                       dtype=np.float32))

    def test_coords_update_fails_bounds_differ(self):
        """Test that _update_coord fails if bounds differ."""
        cube = create_cube_with_threshold(threshold_values=self.thresholds)
        cube.coord(self.coord_name).guess_bounds()
        changes = {'bounds': [[0.1, 2.0], [2.0, 3.0], [3.0, 4.0]]}
        msg = "Mismatch in bounds in existing coord and updated metadata"
        with self.assertRaisesRegex(ValueError, msg):
            _update_coord(cube, self.coord_name, changes)

    def test__update_attributes(self):
        """Test update attributes associated with a coordinate."""
        cube = create_cube_with_threshold()
        changes = {'attributes': {'spp__relative_to_threshold': "below"}}
        result = _update_coord(cube, self.coord_name, changes)
        self.assertIsInstance(result, Cube)
        self.assertEqual(
            result.coord(self.coord_name).attributes, changes["attributes"])

    @ManageWarnings(record=True)
    def test_warning_messages_with_update(self, warning_list=None):
        """Test warning message is raised correctly when updating coord. """
        changes = {'points': [2.0], 'bounds': [0.1, 2.0]}
        warning_msg = "Updated coordinate"
        _update_coord(self.cube, self.coord_name, changes, warnings_on=True)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))

    def test_incompatible_changes_requested(self):
        """Test that _update_coord raises an exception if 'points' and 'units'
        are requested to be changed."""
        cube = create_cube_with_threshold()
        changes = {'points': [2.0, 3.0], 'units': 'mm/hr'}
        msg = "When updating a coordinate"
        with self.assertRaisesRegex(ValueError, msg):
            _update_coord(cube, self.coord_name, changes)

    def test_alternative_incompatible_changes_requested(self):
        """Test that _update_coord raises an exception if 'bounds' and 'units'
        are requested to be changed."""
        cube = create_cube_with_threshold()
        changes = {'bounds': [0.1, 2.0], 'units': 'mm/hr'}
        msg = "When updating a coordinate"
        with self.assertRaisesRegex(ValueError, msg):
            _update_coord(cube, self.coord_name, changes)


class Test__update_attribute(IrisTest):
    """Test the _update_attribute method."""

    def setUp(self):
        """Set up test cube"""
        self.cube = create_cube_with_threshold()

    def test_basic(self):
        """Test that _update_attribute returns a Cube and updates OK and does
        not modify the input cube. """
        original_cube = self.cube.copy()
        attribute_name = 'attribute_to_update'
        changes = 'second_value'
        result = _update_attribute(self.cube, attribute_name, changes)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.attributes['attribute_to_update'],
                         'second_value')
        self.assertEqual(self.cube.metadata, original_cube.metadata)

    @ManageWarnings(record=True)
    def test_attributes_updated_warnings(self, warning_list=None):
        """Test _update_attribute updates attributes and gives warning. """
        attribute_name = 'attribute_to_update'
        changes = 'second_value'
        warning_msg = "Adding or updating attribute"
        result = _update_attribute(self.cube, attribute_name, changes,
                                   warnings_on=True)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertEqual(result.attributes['attribute_to_update'],
                         'second_value')

    def test_attributes_added(self):
        """Test _update_attribute adds attribute OK. """
        attribute_name = 'new_attribute'
        changes = 'new_value'
        result = _update_attribute(self.cube, attribute_name, changes)
        self.assertEqual(result.attributes['new_attribute'],
                         'new_value')

    def test_attributes_deleted(self):
        """Test _update_attribute deletes attribute OK. """
        attribute_name = 'attribute_to_update'
        changes = 'delete'
        result = _update_attribute(self.cube, attribute_name, changes)
        self.assertFalse('attribute_to_update' in result.attributes)

    @ManageWarnings(record=True)
    def test_attributes_deleted_warnings(self, warning_list=None):
        """Test _update_attribute deletes and gives warning. """
        attribute_name = 'attribute_to_update'
        changes = 'delete'
        warning_msg = "Deleted attribute"
        result = _update_attribute(self.cube, attribute_name, changes,
                                   warnings_on=True)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertFalse('attribute_to_update' in result.attributes)

    def test_attributes_deleted_when_not_present(self):
        """Test _update_attribute copes when an attribute is requested to be
        deleted, but this attribute is not available on the input cube."""
        attribute_name = 'invalid_name'
        changes = 'delete'
        result = _update_attribute(self.cube, attribute_name, changes)
        self.assertFalse('invalid_name' in result.attributes)


class Test__update_cell_methods(IrisTest):
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
        _update_cell_methods(self.cube, cell_methods)
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
        _update_cell_methods(self.cube, cell_methods)
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
            _update_cell_methods(self.cube, cell_methods)

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
        _update_cell_methods(self.cube, cell_methods)
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
        _update_cell_methods(self.cube, cell_methods)
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
        _update_cell_methods(self.cube, additional_cell_methods)
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
        _update_cell_methods(self.cube, cell_methods)
        self.assertEqual(self.cube.cell_methods, ())

    def test_add_cell_method_no_action(self):
        """Test adding a cell method, where no action is specified."""
        cell_methods = {'method': 'point',
                        'coords': 'time',
                        'intervals': (),
                        'comments': ()}
        msg = "No action has been specified within the cell method definition."
        with self.assertRaisesRegex(ValueError, msg):
            _update_cell_methods(self.cube, cell_methods)


class Test_amend_metadata(IrisTest):
    """Test the amend_metadata method."""

    def setUp(self):
        """Set up test cube"""
        self.cube = create_cube_with_threshold()
        self.threshold_coord = find_threshold_coordinate(self.cube).name()

    def test_basic(self):
        """Test that the function returns a Cube and the input cube is not
        modified. """
        result = amend_metadata(
            self.cube, name='new_cube_name', data_type=np.dtype)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.name(), 'new_cube_name')
        self.assertNotEqual(self.cube.name(), 'new_cube_name')

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
                          'time': {'points': [1447896600, 1447900200]}}
        result = amend_metadata(
            self.cube, name='new_cube_name', data_type=np.dtype,
            coordinates=updated_coords)
        self.assertArrayEqual(result.coord(self.threshold_coord).points,
                              np.array([2.0]))
        self.assertArrayEqual(result.coord('time').points,
                              np.array([1447896600, 1447900200]))

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


class Test_set_history_attribute(IrisTest):
    """Test the set_history_attribute function."""

    def test_add_history(self):
        """Test that a history attribute has been added."""
        cube = set_up_variable_cube(np.ones((3, 3), dtype=np.float32))
        set_history_attribute(cube, "Nowcast")
        self.assertTrue("history" in cube.attributes)
        self.assertTrue("Nowcast" in cube.attributes["history"])

    def test_history_already_exists(self):
        """Test that the history attribute is overwritten, if it
        already exists."""
        cube = set_up_variable_cube(np.ones((3, 3), dtype=np.float32))
        old_history = "2018-09-13T11:28:29: StaGE"
        cube.attributes["history"] = old_history
        set_history_attribute(cube, "Nowcast")
        self.assertTrue("history" in cube.attributes)
        self.assertTrue("Nowcast" in cube.attributes["history"])
        self.assertFalse(old_history in cube.attributes["history"])

    def test_history_append(self):
        """Test that the history attribute can be updated."""
        cube = set_up_variable_cube(np.ones((3, 3), dtype=np.float32))
        old_history = "2018-09-13T11:28:29: StaGE"
        cube.attributes["history"] = old_history
        set_history_attribute(cube, "Nowcast", append=True)
        self.assertTrue("history" in cube.attributes)
        self.assertTrue("Nowcast" in cube.attributes["history"])
        self.assertTrue(old_history in cube.attributes["history"])

    def test_history_append_no_existing(self):
        """Test the "append" option doesn't crash when no history exists."""
        cube = set_up_variable_cube(np.ones((3, 3), dtype=np.float32))
        set_history_attribute(cube, "Nowcast", append=True)
        self.assertTrue("history" in cube.attributes)
        self.assertTrue("Nowcast" in cube.attributes["history"])


if __name__ == '__main__':
    unittest.main()
