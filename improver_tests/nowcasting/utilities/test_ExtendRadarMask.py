#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module with tests for the ExtendRadarMask plugin."""

import unittest

import iris
import numpy as np
from iris.tests import IrisTest

from improver.nowcasting.utilities import ExtendRadarMask
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube


class Test__init_(IrisTest):
    """Test the _init_ method"""

    def test_basic(self):
        """Test initialisation of class"""
        plugin = ExtendRadarMask()
        self.assertSequenceEqual(plugin.coverage_valid, [1, 2])


class Test_process(IrisTest):
    """Test the process method"""

    def setUp(self):
        """Set up some input cubes"""
        rainrate_data = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.4, 0.3, 0.0],
                [0.0, 0.2, 0.6, 0.7, 0.6],
                [0.0, 0.0, 0.4, 0.5, 0.4],
                [0.0, 0.0, 0.1, 0.2, 0.3],
            ],
            dtype=np.float32,
        )

        rainrate_mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, False],
                [True, False, False, False, False],
                [True, False, False, False, False],
                [True, False, False, False, False],
            ],
            dtype=bool,
        )

        rainrate_data = np.ma.MaskedArray(rainrate_data, mask=rainrate_mask)

        self.rainrate = set_up_variable_cube(
            rainrate_data,
            name="lwe_precipitation_rate",
            units="mm h-1",
            spatial_grid="equalarea",
        )

        coverage_data = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 2, 1, 1, 3],
                [0, 1, 1, 1, 1],
                [0, 2, 1, 1, 1],
                [0, 3, 1, 1, 1],
            ],
            dtype=np.int32,
        )

        self.coverage = set_up_variable_cube(
            coverage_data, name="radar_coverage", units="1", spatial_grid="equalarea"
        )

        self.expected_mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, False],
                [True, False, False, False, False],
                [True, True, False, False, False],
            ],
            dtype=bool,
        )

    def test_basic(self):
        """Test processing outputs a cube of precipitation rates"""
        result = ExtendRadarMask().process(self.rainrate, self.coverage)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertEqual(result.name(), self.rainrate.name())

    def test_values(self):
        """Test output cube has expected mask and underlying data is
        unchanged"""
        result = ExtendRadarMask().process(self.rainrate, self.coverage)
        self.assertArrayEqual(result.data.mask, self.expected_mask)
        self.assertArrayEqual(result.data.data, self.rainrate.data.data)

    def test_inputs_unmodified(self):
        """Test the rainrate cube is not modified in place"""
        reference = self.rainrate.copy()
        _ = ExtendRadarMask().process(self.rainrate, self.coverage)
        self.assertEqual(reference, self.rainrate)

    def test_coords_unmatched_error(self):
        """Test error is raised if coordinates do not match"""
        x_points = self.rainrate.coord(axis="x").points
        self.rainrate.coord(axis="x").points = x_points + 100.0
        msg = "Rain rate and coverage composites unmatched"
        with self.assertRaisesRegex(ValueError, msg):
            _ = ExtendRadarMask().process(self.rainrate, self.coverage)


if __name__ == "__main__":
    unittest.main()
