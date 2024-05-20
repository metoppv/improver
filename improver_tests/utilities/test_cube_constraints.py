# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Utilities for creating Iris constraints."""

import unittest

import iris
from iris.tests import IrisTest

from improver.metadata.probabilistic import find_threshold_coordinate
from improver.utilities.cube_constraints import create_sorted_lambda_constraint

from .cube_extraction.test_cube_extraction import set_up_precip_probability_cube


class Test_create_sorted_lambda_constraint(IrisTest):
    """Test that a lambda constraint is created."""

    def setUp(self):
        """Set up cube with testing lambda constraint."""
        self.precip_cube = set_up_precip_probability_cube()
        self.coord_name = find_threshold_coordinate(self.precip_cube).name()
        self.precip_cube.coord(self.coord_name).convert_units("mm h-1")
        self.expected_data = self.precip_cube[:2].data

    def test_basic_ascending(self):
        """Test that a constraint is created, if the input coordinates are
        ascending."""
        values = [0.03, 0.1]
        result = create_sorted_lambda_constraint(self.coord_name, values)
        self.assertIsInstance(result, iris.Constraint)
        self.assertEqual(list(result._coord_values.keys()), [self.coord_name])
        result_cube = self.precip_cube.extract(result)
        self.assertArrayAlmostEqual(result_cube.data, self.expected_data)

    def test_basic_descending(self):
        """Test that a constraint is created, if the input coordinates are
        descending."""
        values = [0.1, 0.03]
        result = create_sorted_lambda_constraint(self.coord_name, values)
        self.assertIsInstance(result, iris.Constraint)
        self.assertEqual(list(result._coord_values.keys()), [self.coord_name])
        result_cube = self.precip_cube.extract(result)
        self.assertArrayAlmostEqual(result_cube.data, self.expected_data)

    def test_non_default_tolerance(self):
        """Test that a constraint is created, if the input coordinates are
        made more fuzzy by use of a non-standard tolerance. The default
        tolerance is 1.0E-7, here we make it large enough to extract all the
        available thresholds using two bounds; this is testing an extreme
        not a desirable use case."""
        values = [0.03, 0.6]
        result = create_sorted_lambda_constraint(self.coord_name, values, tolerance=0.9)
        self.assertIsInstance(result, iris.Constraint)
        self.assertEqual(list(result._coord_values.keys()), [self.coord_name])
        result_cube = self.precip_cube.extract(result)
        self.assertArrayAlmostEqual(result_cube.data, self.precip_cube.data)


if __name__ == "__main__":
    unittest.main()
