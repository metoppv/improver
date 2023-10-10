# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
Unit tests for the function collapsed.
"""

import unittest

import iris
import numpy as np
from iris.exceptions import CoordinateCollapseError

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import collapse_realizations


class Test_aggregate(unittest.TestCase):

    """Test the collapse_realizations utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        data = 281 * np.random.random_sample((3, 3, 3)).astype(np.float32)
        self.cube = set_up_variable_cube(data, realizations=[0, 1, 2])
        self.expected_data = self.cube.collapsed(
            ["realization"], iris.analysis.MEAN
        ).data

    def test_basic(self):
        """Test that a collapsed cube is returned with no realization coord"""
        result = collapse_realizations(self.cube)
        assert "realization" not in [
            coord.name() for coord in result.dim_coords + result.aux_coords
        ]
        assert (result.data == self.expected_data).all()

    def test_invalid_dimension(self):
        """Test that an error is raised if realization dimension
        does not exist."""
        sub_cube = self.cube.extract(iris.Constraint(realization=0))
        with self.assertRaises(CoordinateCollapseError):
            collapse_realizations(sub_cube, "mean")

    def test_different_aggregators(self):
        """Test aggregators other than mean."""
        aggregator_dict = {
            "sum": iris.analysis.SUM,
            "median": iris.analysis.MEDIAN,
            "std_dev": iris.analysis.STD_DEV,
            "min": iris.analysis.MIN,
            "max": iris.analysis.MAX,
        }
        for key, value in aggregator_dict.items():
            result = collapse_realizations(self.cube, key)
            expected_data = self.cube.collapsed(["realization"], value).data
            self.assertTrue((result.data == expected_data).all())

    def test_invalid_aggregators(self):
        """Test that an error is raised if aggregator is not
        one of the allowed types."""

        msg = "method must be one of"
        with self.assertRaises(ValueError, msg=msg):
            collapse_realizations(self.cube, method="product")

    def test_1d_std_dev(self):
        """Test that when std_dev is calculated over a dimension of size 1,
        output is all masked and underlying value is np.nan.
        """
        data = 281 * np.random.random_sample((1, 3, 3)).astype(np.float32)
        cube_1d = set_up_variable_cube(data, realizations=[0])
        result = collapse_realizations(cube_1d, "std_dev")
        self.assertTrue(np.all(np.ma.getmask(result.data)))
        self.assertTrue(np.all(np.isnan(result.data.data)))
