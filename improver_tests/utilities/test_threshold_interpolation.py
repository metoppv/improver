# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the
`ensemble_copula_coupling.EnsembleCopulaCouplingUtilities` class.
"""

import unittest
import pytest
import numpy as np
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest
from improver.utilities.threshold_interpolation import create_cube_with_thresholds, Threshold_interpolation
from improver.synthetic_data.set_up_test_cubes import (
    set_up_probability_cube,
    add_coordinate,
)

class Test_create_cube_with_thresholds(IrisTest):
    """Test the _create_cube_with_thresholds function."""

    def setUp(self):
        """Set up temperature cube."""
        original_visibility_probabilities = np.array(
            [
                [[1.0, 0.9, 1.0], [0.8, 0.9, 0.5], [0.5, 0.2, 0.0]],
                [[1.0, 0.5, 1.0], [0.5, 0.5, 0.3], [0.2, 0.0, 0.0]],
                [[1.0, 0.2, 0.5], [0.2, 0.0, 0.1], [0.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )

        self.cube = set_up_probability_cube(
            original_visibility_probabilities,
            thresholds=[100, 200, 300],
            variable_name="visibility_in_air",
            threshold_units="m",
            spp__relative_to_threshold="less_than",
        )
        self.cube_data = original_visibility_probabilities

    def test_basic(self):
        """Test that the plugin returns an Iris.cube.Cube with suitable units."""
        cube_data = self.cube_data
        print(self.cube)
        thresholds = [100, 150, 200, 250, 300]
        coord_name = "visibility_in_air"
        result = Threshold_interpolation(self.cube, thresholds)
        self.assertIsInstance(result, Cube)
        self.assertEqual(result.units, self.cube.units)

    def test_incompatible_thresholds(self):
        """
        Test that the plugin fails if the percentile values requested
        are not numbers.
        """
        thresholds = ["cat", "dog", "elephant"]
        cube_data = np.zeros(
            [
                len(thresholds),
                len(self.cube.coord("latitude").points),
                len(self.cube.coord("longitude").points),
            ]
        )
        msg = "could not convert string to float"
        with self.assertRaisesRegex(ValueError, msg):
            Threshold_interpolation(self.cube, thresholds)

    def test_metadata_copy(self):
        """
        Test that the metadata dictionaries within the input cube, are
        also present on the output cube.
        """
        self.cube.attributes = {"source": "ukv"}
        cube_data = self.cube_data
        thresholds = [100, 150, 200, 250, 300]
        result = Threshold_interpolation(self.cube, thresholds)
        self.assertDictEqual(self.cube.metadata._asdict(), result.metadata._asdict())

    def test_realization_coord_removed(self):
        """
        Testing that the realization coordinate has been removed if exists.
        """
        thresholds = [100, 200, 300]
        realization_cube = add_coordinate(
            self.cube, [0, 1, 2], "realization", coord_units=1, dtype=np.int32
        )
        result = Threshold_interpolation(realization_cube, thresholds)
        dim_coords = [coord.name() for coord in result.coords(dim_coords=True)]
        expected_dim_coords = [
            coord.name() for coord in realization_cube.coords(dim_coords=True)
        ]
        expected_dim_coords.remove("realization")
        self.assertSequenceEqual(dim_coords, expected_dim_coords)

    def test_cube_no_threshold_coord(self):
        """
        Testing that an error is raised if no threshold coordinate exists.
        """
        thresholds = [100, 200, 300]
        realization_cube = add_coordinate(
            self.cube, [0, 1, 2], "realization", coord_units=1, dtype=np.int32
        )
        realization_cube.remove_coord("visibility_in_air")
        msg = "No threshold coord found"
        with self.assertRaisesRegex(CoordinateNotFoundError, msg):
            Threshold_interpolation(realization_cube, thresholds)

    def test_no_thresholds_provided(self):
        """
        Testing that a warning message is raised if no thresholds are provided.
        """
        warning_msg = "No thresholds provided, using existing thresholds."

        with pytest.warns(UserWarning, match=warning_msg):
            result = Threshold_interpolation(self.cube)

    def test_thresholds_different_mask(self):
        """
        Testing that a value error message is raised if masks are different across thresholds.
        """

        mask = np.array(
            [
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
                [[0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            ],
            dtype=np.int8,
        )

        masked_data = np.ma.masked_array(self.cube_data, mask=mask)

        masked_cube = set_up_probability_cube(
            masked_data,
            thresholds=[100, 200, 300],
            variable_name="visibility_in_air",
            threshold_units="m",
            spp__relative_to_threshold="less_than",
        )
        print(masked_cube)

        thresholds = [100, 150, 200, 250, 300]

        msg = "The mask is expected to be constant across different slices of the"
        with self.assertRaisesRegex(ValueError, msg):
            Threshold_interpolation(masked_cube, thresholds)

#    def test_masked_cube(self):
#        result = Threshold_interpolation(masked_cube, thresholds)
#        self.assertIsInstance(result, Cube)


if __name__ == '__main__':
    unittest.main()