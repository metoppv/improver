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
"""Utilities for creating Iris constraints."""

import unittest

import iris
from iris.tests import IrisTest

from improver.metadata.probabilistic import find_threshold_coordinate
from improver.utilities.cube_constraints import create_sorted_lambda_constraint

from ..utilities.test_cube_extraction import set_up_precip_probability_cube


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
        result = create_sorted_lambda_constraint(
            self.coord_name, values, tolerance=0.9)
        self.assertIsInstance(result, iris.Constraint)
        self.assertEqual(list(result._coord_values.keys()), [self.coord_name])
        result_cube = self.precip_cube.extract(result)
        self.assertArrayAlmostEqual(result_cube.data, self.precip_cube.data)


if __name__ == '__main__':
    unittest.main()
