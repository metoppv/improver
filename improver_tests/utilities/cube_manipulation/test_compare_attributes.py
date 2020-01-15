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
"""
Unit tests for the function "cube_manipulation.compare_attributes".
"""

import unittest

import iris
import numpy as np
from iris.tests import IrisTest

from improver.utilities.cube_manipulation import compare_attributes
from improver.utilities.warnings_handler import ManageWarnings

from ...set_up_test_cubes import set_up_variable_cube


class Test_compare_attributes(IrisTest):
    """Test the compare_attributes utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        data = 275*np.ones((3, 3, 3), dtype=np.float32)
        self.cube = set_up_variable_cube(
            data, standard_grid_metadata='uk_ens',
            attributes={'mosg__grid_version': '1.2.0'})
        self.cube_ukv = set_up_variable_cube(
            data[0], standard_grid_metadata='uk_det',
            attributes={'mosg__grid_version': '1.1.0'})

    def test_basic(self):
        """Test that the utility returns a list and have no differences."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cubelist = iris.cube.CubeList([cube1, cube2])
        result = compare_attributes(cubelist)
        self.assertIsInstance(result, list)
        self.assertArrayEqual(result, [{}, {}])

    @ManageWarnings(record=True)
    def test_warning(self, warning_list=None):
        """Test that the utility returns warning if only one cube supplied."""
        result = (
            compare_attributes(iris.cube.CubeList([self.cube])))
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        warning_msg = "Only a single cube so no differences will be found "
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertArrayEqual(result, [])

    def test_history_attribute(self):
        """Test that the utility returns diff when history do not match"""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube1.attributes["history"] = "2017-01-18T08:59:53: StaGE Decoupler"
        cube2.attributes["history"] = "2017-01-19T08:59:53: StaGE Decoupler"
        cubelist = iris.cube.CubeList([cube1, cube2])
        result = compare_attributes(cubelist)
        self.assertArrayEqual(result,
                              [{'history':
                                '2017-01-18T08:59:53: StaGE Decoupler'},
                               {'history':
                                '2017-01-19T08:59:53: StaGE Decoupler'}])

    def test_multiple_differences(self):
        """Test that the utility returns multiple differences"""
        cubelist = iris.cube.CubeList([self.cube, self.cube_ukv])
        result = compare_attributes(cubelist)
        self.assertArrayEqual(result,
                              [{'mosg__model_configuration': 'uk_ens',
                                'mosg__grid_version': '1.2.0'},
                               {'mosg__model_configuration': 'uk_det',
                                'mosg__grid_version': '1.1.0'}])

    def test_three_cubes(self):
        """Test that the utility returns a list of differences when there are
        more than two input cubes for comparison."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube3 = self.cube.copy()
        cube1.attributes["history"] = "2017-01-18T08:59:53: StaGE Decoupler"
        cube2.attributes["mosg__model_configuration"] = "test"
        cube3.attributes["mosg__grid_version"] = "10"

        cubelist = iris.cube.CubeList([cube1, cube2, cube3])
        result = compare_attributes(cubelist)
        expected = [
            {'mosg__model_configuration': 'uk_ens',
             'mosg__grid_version': '1.2.0',
             'history': '2017-01-18T08:59:53: StaGE Decoupler'},
            {'mosg__model_configuration': 'test',
             'mosg__grid_version': '1.2.0'},
            {'mosg__model_configuration': 'uk_ens',
             'mosg__grid_version': '10'}]
        self.assertArrayEqual(result, expected)

    def test_filtered_differences(self):
        """Test that the utility returns differences only between attributes
        that match the attribute filter."""
        cubelist = iris.cube.CubeList([self.cube, self.cube_ukv])
        result = compare_attributes(cubelist, attribute_filter='mosg__grid')
        self.assertArrayEqual(result,
                              [{'mosg__grid_version': '1.2.0'},
                               {'mosg__grid_version': '1.1.0'}])

    def test_filtered_difference_three_cubes(self):
        """Test that the utility returns a list of filtered differences when
        there are more than two input cubes for comparison. In this case we
        expect the history attribute that is different to be ignored."""

        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube3 = self.cube.copy()
        cube1.attributes["history"] = "2017-01-18T08:59:53: StaGE Decoupler"
        cube2.attributes["mosg__model_configuration"] = "test"
        cube3.attributes["mosg__grid_version"] = "10"

        cubelist = iris.cube.CubeList([cube1, cube2, cube3])
        result = compare_attributes(cubelist, attribute_filter='mosg')
        expected = [
            {'mosg__model_configuration': 'uk_ens',
             'mosg__grid_version': '1.2.0'},
            {'mosg__model_configuration': 'test',
             'mosg__grid_version': '1.2.0'},
            {'mosg__model_configuration': 'uk_ens',
             'mosg__grid_version': '10'}]
        self.assertArrayEqual(result, expected)

    def test_unhashable_types_list(self):
        """Test that the utility returns differences when unhashable attributes
        are present, e.g. a list."""
        self.cube.attributes['test_attribute'] = [0, 1, 2]
        cubelist = iris.cube.CubeList([self.cube, self.cube_ukv])
        result = compare_attributes(cubelist)

        expected = [
            {'mosg__model_configuration': 'uk_ens',
             'mosg__grid_version': '1.2.0',
             'test_attribute': [0, 1, 2]},
            {'mosg__model_configuration': 'uk_det',
             'mosg__grid_version': '1.1.0'}]

        self.assertArrayEqual(result, expected)

    def test_unhashable_types_array(self):
        """Test that the utility returns differences when unhashable attributes
        are present, e.g. a numpy array."""
        self.cube.attributes['test_attribute'] = np.array([0, 1, 2])
        cubelist = iris.cube.CubeList([self.cube, self.cube_ukv])
        result = compare_attributes(cubelist)

        expected = [
            {'mosg__model_configuration': 'uk_ens',
             'mosg__grid_version': '1.2.0',
             'test_attribute': np.array([0, 1, 2])},
            {'mosg__model_configuration': 'uk_det',
             'mosg__grid_version': '1.1.0'}]

        # The numpy array prevents us comparing the whole list of dictionaries
        # in a single step, so we break it up to compare the elements.
        self.assertDictEqual(result[1], expected[1])
        self.assertEqual(result[0]['mosg__model_configuration'],
                         expected[0]['mosg__model_configuration'])
        self.assertEqual(result[0]['mosg__grid_version'],
                         expected[0]['mosg__grid_version'])
        self.assertArrayEqual(result[0]['test_attribute'],
                              expected[0]['test_attribute'])


if __name__ == '__main__':
    unittest.main()
