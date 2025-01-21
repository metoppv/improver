# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
Unit tests for the
`ensemble_copula_coupling.EnsembleCopulaCouplingUtilities` class.
"""

import unittest

import numpy as np
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube
from iris.exceptions import CoordinateNotFoundError
from iris.tests import IrisTest
from improver.utilities.threshold_interpolation import create_cube_with_thresholds, Threshold_interpolation
from improver.synthetic_data.set_up_test_cubes import (
    set_up_probability_cube,
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

#        self.cube = set_up_probability_cube(original_visibility_probabilities[0], thresholds=[100, 200, 300])
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

#    def test_changed_cube_units(self):
#        """Test that the plugin returns a cube with chosen units."""
#        cube_data = self.cube_data
#        percentiles = [100, 150, 200, 250, 300]
#        result = create_cube_with_thresholds(
#            thresholds, self.cube, cube_data, cube_unit="m"
#        )
#        self.assertEqual(result.units, Unit("m"))

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

    def test_coordinate_copy(self):
        """
        Testing that the realization coordinate has been removed if exists.
        """
        thresholds = [100, 200, 300]
        realizations = [0, 1, 2]
        result = Threshold_interpolation(self.cube, thresholds)
        print(result)
        self.cube.add_dim_coord(DimCoord([10.0], long_name="realization"), 3)
        print(self.cube)
        for coord in self.cube.coords():
            if coord not in result.coords():
                msg = "Coordinate: {} not found in cube {}".format(coord, result)
                raise CoordinateNotFoundError(msg)

if __name__ == '__main__':
    unittest.main()