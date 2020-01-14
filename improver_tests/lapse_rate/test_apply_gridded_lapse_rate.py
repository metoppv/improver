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
"""Unit tests for the apply_gridded_lapse_rate method."""

import unittest

import iris
import numpy as np
from iris.tests import IrisTest

from improver.constants import DALR
from improver.lapse_rate import apply_gridded_lapse_rate

from ..set_up_test_cubes import add_coordinate, set_up_variable_cube


class Test_apply_gridded_lapse_rate(IrisTest):
    """Test the apply_gridded_lapse_rate method"""

    def setUp(self):
        """Set up some input cubes"""
        source_orog = np.array([[400., 400., 402., 402.],
                                [400., 400., 402., 402.],
                                [403., 403., 405., 405.],
                                [403., 403., 405., 405.]], dtype=np.float32)
        self.source_orog = set_up_variable_cube(
            source_orog, name='orography', units='m', spatial_grid='equalarea')

        dest_orog = np.array([[400., 401., 401., 402.],
                              [402., 402., 402., 403.],
                              [403., 404., 405., 404.],
                              [404., 405., 406., 405.]], dtype=np.float32)
        self.dest_orog = set_up_variable_cube(
            dest_orog, name='orography', units='m', spatial_grid='equalarea')

        self.lapse_rate = set_up_variable_cube(
            np.full((4, 4), DALR, dtype=np.float32), name='lapse_rate',
            units='K m-1', spatial_grid='equalarea')

        # specify temperature values ascending in 0.25 K increments
        temp_data = np.array([[276., 276.25, 276.5, 276.75],
                              [277., 277.25, 277.5, 277.75],
                              [278., 278.25, 278.5, 278.75],
                              [279., 279.25, 279.5, 279.75]], dtype=np.float32)
        self.temperature = set_up_variable_cube(
            temp_data, name='screen_temperature', spatial_grid='equalarea')

        self.expected_data = np.array([
            [276., 276.2402, 276.5098, 276.75],
            [276.9804, 277.2304, 277.5, 277.7402],
            [278., 278.2402, 278.5, 278.7598],
            [278.9902, 279.2304, 279.4902, 279.75]], dtype=np.float32)

    def test_basic(self):
        """Test output is cube with correct name, type and units"""
        result = apply_gridded_lapse_rate(self.temperature, self.lapse_rate,
                                          self.source_orog, self.dest_orog)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), 'screen_temperature')
        self.assertEqual(result.units, 'K')
        self.assertEqual(result.dtype, np.float32)

    def test_values(self):
        """Check adjusted temperature values are as expected"""
        result = apply_gridded_lapse_rate(self.temperature, self.lapse_rate,
                                          self.source_orog, self.dest_orog)

        # test that temperatures are reduced where destination orography
        # is higher than source
        source_lt_dest = np.where(self.source_orog.data < self.dest_orog.data)
        self.assertTrue(
            np.all(result.data[source_lt_dest] <
                   self.temperature.data[source_lt_dest]))

        # test that temperatures are increased where destination orography
        # is lower than source
        source_gt_dest = np.where(self.source_orog.data > self.dest_orog.data)
        self.assertTrue(
            np.all(result.data[source_gt_dest] >
                   self.temperature.data[source_gt_dest]))

        # test that temperatures are equal where destination orography
        # is equal to source
        source_eq_dest = np.where(np.isclose(self.source_orog.data,
                                             self.dest_orog.data))
        self.assertArrayAlmostEqual(result.data[source_eq_dest],
                                    self.temperature.data[source_eq_dest])

        # match specific values
        self.assertArrayAlmostEqual(result.data, self.expected_data)

    def test_unit_adjustment(self):
        """Test correct values are retrieved if input cubes have incorrect
        units"""
        self.temperature.convert_units('degC')
        self.source_orog.convert_units('km')
        result = apply_gridded_lapse_rate(self.temperature, self.lapse_rate,
                                          self.source_orog, self.dest_orog)
        self.assertEqual(result.units, 'K')
        self.assertArrayAlmostEqual(result.data, self.expected_data)

    def test_realizations(self):
        """Test processing of a cube with multiple realizations"""
        temp_3d = add_coordinate(self.temperature, [0, 1, 2], 'realization')
        lrt_3d = add_coordinate(self.lapse_rate, [0, 1, 2], 'realization')
        result = apply_gridded_lapse_rate(
            temp_3d, lrt_3d, self.source_orog, self.dest_orog)
        self.assertArrayEqual(
            result.coord('realization').points, np.array([0, 1, 2]))
        for subcube in result.slices_over('realization'):
            self.assertArrayAlmostEqual(subcube.data, self.expected_data)

    def test_unmatched_realizations(self):
        """Test error if realizations on temperature and lapse rate do not
        match"""
        temp_3d = add_coordinate(self.temperature, [0, 1, 2], 'realization')
        lrt_3d = add_coordinate(self.lapse_rate, [2, 3, 4], 'realization')
        msg = 'Lapse rate cube coordinate "realization" does not match '
        with self.assertRaisesRegex(ValueError, msg):
            _ = apply_gridded_lapse_rate(
                temp_3d, lrt_3d, self.source_orog, self.dest_orog)

    def test_missing_coord(self):
        """Test error if temperature cube has realizations but lapse rate
        does not"""
        temp_3d = add_coordinate(self.temperature, [0, 1, 2], 'realization')
        msg = 'Lapse rate cube has no coordinate "realization"'
        with self.assertRaisesRegex(ValueError, msg):
            _ = apply_gridded_lapse_rate(
                temp_3d, self.lapse_rate, self.source_orog, self.dest_orog)

    def test_spatial_mismatch(self):
        """Test error if source orography grid is not matched to temperature"""
        new_y_points = self.source_orog.coord(axis='y').points + 100.
        self.source_orog.coord(axis='y').points = new_y_points
        msg = 'Source orography spatial coordinates do not match'
        with self.assertRaisesRegex(ValueError, msg):
            _ = apply_gridded_lapse_rate(self.temperature, self.lapse_rate,
                                         self.source_orog, self.dest_orog)

    def test_spatial_mismatch_2(self):
        """Test error if destination orography grid is not matched to
        temperature"""
        new_y_points = self.dest_orog.coord(axis='y').points + 100.
        self.dest_orog.coord(axis='y').points = new_y_points
        msg = 'Destination orography spatial coordinates do not match'
        with self.assertRaisesRegex(ValueError, msg):
            _ = apply_gridded_lapse_rate(self.temperature, self.lapse_rate,
                                         self.source_orog, self.dest_orog)


if __name__ == '__main__':
    unittest.main()
