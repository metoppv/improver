# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the percentile.PercentileConverter plugin."""

import unittest
from datetime import datetime

import iris
import numpy as np
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest

from improver.percentile import PercentileConverter
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import get_coord_names, get_dim_coord_names


class Test_process(IrisTest):

    """Test the creation of percentiles by the plugin."""

    def setUp(self):
        """Create a cube with collapsable coordinates.

        Data is formatted to increase linearly in x/y dimensions,
        e.g.
              0 0 0 0
              1 1 1 1
              2 2 2 2
              3 3 3 3

        """
        data = [[list(range(0, 11, 1))] * 11] * 3
        data = np.array(data).astype(np.float32)
        data.resize((3, 11, 11))
        self.cube = set_up_variable_cube(data, realizations=[0, 1, 2])
        self.default_percentiles = np.array(
            [0, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 100]
        )

    def test_valid_single_coord_string(self):
        """Test that the plugin handles a valid collapse_coord passed in
        as a string."""

        collapse_coord = "longitude"

        plugin = PercentileConverter(collapse_coord)
        result = plugin.process(self.cube)

        # Check percentile values.
        self.assertArrayAlmostEqual(
            result.data[:, 0, 0], self.default_percentiles * 0.1
        )
        # Check coordinate name.
        self.assertEqual(result.coords()[0].name(), "percentile")
        # Check coordinate units.
        self.assertEqual(result.coords()[0].units, "%")
        # Check coordinate points.
        self.assertArrayEqual(
            result.coord("percentile").points,
            [0, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 100],
        )
        # Check resulting data shape.
        self.assertEqual(result.data.shape, (15, 3, 11))

    def test_valid_single_coord_string_for_time(self):
        """Test that the plugin handles time being the collapse_coord that is
        passed in as a string."""
        data = [[list(range(1, 12, 1))] * 11] * 3
        data = np.array(data).astype(np.float32)
        data.resize((3, 11, 11))
        new_cube = set_up_variable_cube(
            data,
            time=datetime(2017, 11, 11, 4, 0),
            frt=datetime(2017, 11, 11, 0, 0),
            realizations=[0, 1, 2],
        )
        cube = iris.cube.CubeList([self.cube, new_cube]).merge_cube()

        collapse_coord = "time"

        plugin = PercentileConverter(collapse_coord)
        result = plugin.process(cube)

        # Check percentile values.
        self.assertArrayAlmostEqual(
            result.data[:, 0, 0, 0], self.default_percentiles * 0.01
        )
        # Check coordinate name.
        self.assertEqual(result.coords()[0].name(), "percentile")
        # Check coordinate units.
        self.assertEqual(result.coords()[0].units, "%")
        # Check coordinate points.
        self.assertArrayEqual(
            result.coord("percentile").points,
            [0, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 100],
        )
        # Check resulting data shape.
        self.assertEqual(result.data.shape, (15, 3, 11, 11))

    def test_valid_multi_coord_string_list(self):
        """Test that the plugin handles a valid list of collapse_coords passed
        in as a list of strings."""

        collapse_coord = ["longitude", "latitude"]

        plugin = PercentileConverter(collapse_coord)
        result = plugin.process(self.cube)

        # Check percentile values.
        self.assertArrayAlmostEqual(
            result.data[:, 0],
            [
                0.0,
                0.0,
                1.0,
                2.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                8.0,
                9.0,
                10.0,
                10.0,
            ],
        )
        # Check coordinate name.
        self.assertEqual(result.coords()[0].name(), "percentile")
        # Check coordinate units.
        self.assertEqual(result.coords()[0].units, "%")
        # Check coordinate points.
        self.assertArrayEqual(
            result.coord("percentile").points,
            [0, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 100],
        )
        # Check resulting data shape.
        self.assertEqual(result.data.shape, (15, 3))

    def test_single_percentile(self):
        """Test dimensions of output at median only"""
        collapse_coord = ["realization"]
        plugin = PercentileConverter(collapse_coord, percentiles=[50])
        result = plugin.process(self.cube)
        result_coords = get_coord_names(result)
        self.assertNotIn("realization", result_coords)
        self.assertIn("percentile", result_coords)
        self.assertNotIn("percentile", get_dim_coord_names(result))

    def test_use_with_masked_data(self):
        """Test that the plugin handles masked data, this requiring the option
        fast_percentile_method=False."""

        mask = np.zeros((3, 11, 11))
        mask[:, :, 1:-1:2] = 1
        masked_data = np.ma.array(self.cube.data, mask=mask)
        cube = self.cube.copy(data=masked_data)
        collapse_coord = "longitude"

        plugin = PercentileConverter(collapse_coord, fast_percentile_method=False)
        result = plugin.process(cube)

        # Check percentile values.
        self.assertArrayAlmostEqual(
            result.data[:, 0, 0], self.default_percentiles * 0.1
        )
        # Check coordinate name.
        self.assertEqual(result.coords()[0].name(), "percentile")
        # Check coordinate units.
        self.assertEqual(result.coords()[0].units, "%")
        # Check coordinate points.
        self.assertArrayEqual(
            result.coord("percentile").points,
            [0, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 100],
        )
        # Check resulting data shape.
        self.assertEqual(result.data.shape, (15, 3, 11))

    def test_unavailable_collapse_coord(self):
        """Test that the plugin handles a collapse_coord that is not
        available in the cube."""

        collapse_coord = "not_a_coordinate"
        plugin = PercentileConverter(collapse_coord)
        msg = "Coordinate "
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            plugin.process(self.cube)

    def test_invalid_collapse_coord_type(self):
        """Test that the plugin handles invalid collapse_coord type."""

        collapse_coord = self.cube
        msg = "collapse_coord is "
        with self.assertRaisesRegex(TypeError, msg):
            PercentileConverter(collapse_coord)


if __name__ == "__main__":
    unittest.main()
