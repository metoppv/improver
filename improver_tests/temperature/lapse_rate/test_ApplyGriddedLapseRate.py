# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the ApplyGriddedLapseRate plugin."""

import unittest

import iris
import numpy as np
from iris.tests import IrisTest

from improver.constants import DALR
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_variable_cube,
)
from improver.temperature.lapse_rate import ApplyGriddedLapseRate


class Test_process(IrisTest):
    """Test the ApplyGriddedLapseRate plugin"""

    def setUp(self):
        """Set up some input cubes"""
        source_orog = np.array(
            [
                [400.0, 400.0, 402.0, 402.0],
                [400.0, 400.0, 402.0, 402.0],
                [403.0, 403.0, 405.0, 405.0],
                [403.0, 403.0, 405.0, 405.0],
            ],
            dtype=np.float32,
        )
        self.source_orog = set_up_variable_cube(
            source_orog, name="orography", units="m", spatial_grid="equalarea"
        )

        dest_orog = np.array(
            [
                [400.0, 401.0, 401.0, 402.0],
                [402.0, 402.0, 402.0, 403.0],
                [403.0, 404.0, 405.0, 404.0],
                [404.0, 405.0, 406.0, 405.0],
            ],
            dtype=np.float32,
        )
        self.dest_orog = set_up_variable_cube(
            dest_orog, name="orography", units="m", spatial_grid="equalarea"
        )

        self.lapse_rate = set_up_variable_cube(
            np.full((4, 4), DALR, dtype=np.float32),
            name="lapse_rate",
            units="K m-1",
            spatial_grid="equalarea",
        )

        # specify temperature values ascending in 0.25 K increments
        temp_data = np.array(
            [
                [276.0, 276.25, 276.5, 276.75],
                [277.0, 277.25, 277.5, 277.75],
                [278.0, 278.25, 278.5, 278.75],
                [279.0, 279.25, 279.5, 279.75],
            ],
            dtype=np.float32,
        )
        self.temperature = set_up_variable_cube(
            temp_data, name="screen_temperature", spatial_grid="equalarea"
        )

        self.expected_data = np.array(
            [
                [276.0, 276.2402, 276.5098, 276.75],
                [276.9804, 277.2304, 277.5, 277.7402],
                [278.0, 278.2402, 278.5, 278.7598],
                [278.9902, 279.2304, 279.4902, 279.75],
            ],
            dtype=np.float32,
        )

        self.plugin = ApplyGriddedLapseRate()

    def test_basic(self):
        """Test output is cube with correct name, type and units"""
        result = self.plugin(
            self.temperature, self.lapse_rate, self.source_orog, self.dest_orog
        )
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), "screen_temperature")
        self.assertEqual(result.units, "K")
        self.assertEqual(result.dtype, np.float32)

    def test_values(self):
        """Check adjusted temperature values are as expected"""
        result = self.plugin(
            self.temperature, self.lapse_rate, self.source_orog, self.dest_orog
        )

        # test that temperatures are reduced where destination orography
        # is higher than source
        source_lt_dest = np.where(self.source_orog.data < self.dest_orog.data)
        self.assertTrue(
            np.all(result.data[source_lt_dest] < self.temperature.data[source_lt_dest])
        )

        # test that temperatures are increased where destination orography
        # is lower than source
        source_gt_dest = np.where(self.source_orog.data > self.dest_orog.data)
        self.assertTrue(
            np.all(result.data[source_gt_dest] > self.temperature.data[source_gt_dest])
        )

        # test that temperatures are equal where destination orography
        # is equal to source
        source_eq_dest = np.where(
            np.isclose(self.source_orog.data, self.dest_orog.data)
        )
        self.assertArrayAlmostEqual(
            result.data[source_eq_dest], self.temperature.data[source_eq_dest]
        )

        # match specific values
        self.assertArrayAlmostEqual(result.data, self.expected_data)

    def test_unit_adjustment(self):
        """Test correct values are retrieved if input cubes have incorrect
        units"""
        self.temperature.convert_units("degC")
        self.source_orog.convert_units("km")
        result = self.plugin(
            self.temperature, self.lapse_rate, self.source_orog, self.dest_orog
        )
        self.assertEqual(result.units, "K")
        self.assertArrayAlmostEqual(result.data, self.expected_data)

    def test_realizations(self):
        """Test processing of a cube with multiple realizations"""
        temp_3d = add_coordinate(self.temperature, [0, 1, 2], "realization")
        lrt_3d = add_coordinate(self.lapse_rate, [0, 1, 2], "realization")
        result = ApplyGriddedLapseRate()(
            temp_3d, lrt_3d, self.source_orog, self.dest_orog
        )
        self.assertArrayEqual(result.coord("realization").points, np.array([0, 1, 2]))
        for subcube in result.slices_over("realization"):
            self.assertArrayAlmostEqual(subcube.data, self.expected_data)

    def test_unmatched_realizations(self):
        """Test error if realizations on temperature and lapse rate do not
        match"""
        temp_3d = add_coordinate(self.temperature, [0, 1, 2], "realization")
        lrt_3d = add_coordinate(self.lapse_rate, [2, 3, 4], "realization")
        msg = 'Lapse rate cube coordinate "realization" does not match '
        with self.assertRaisesRegex(ValueError, msg):
            ApplyGriddedLapseRate()(temp_3d, lrt_3d, self.source_orog, self.dest_orog)

    def test_missing_coord(self):
        """Test error if temperature cube has realizations but lapse rate
        does not"""
        temp_3d = add_coordinate(self.temperature, [0, 1, 2], "realization")
        msg = 'Lapse rate cube has no coordinate "realization"'
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin(temp_3d, self.lapse_rate, self.source_orog, self.dest_orog)

    def test_spatial_mismatch(self):
        """Test error if source orography grid is not matched to temperature"""
        new_y_points = self.source_orog.coord(axis="y").points + 100.0
        self.source_orog.coord(axis="y").points = new_y_points
        msg = "Source orography spatial coordinates do not match"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin(
                self.temperature, self.lapse_rate, self.source_orog, self.dest_orog
            )

    def test_spatial_mismatch_2(self):
        """Test error if destination orography grid is not matched to
        temperature"""
        new_y_points = self.dest_orog.coord(axis="y").points + 100.0
        self.dest_orog.coord(axis="y").points = new_y_points
        msg = "Destination orography spatial coordinates do not match"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin(
                self.temperature, self.lapse_rate, self.source_orog, self.dest_orog
            )


if __name__ == "__main__":
    unittest.main()
