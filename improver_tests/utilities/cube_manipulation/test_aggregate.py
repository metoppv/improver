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
Unit tests for the function aggregate.
"""

import unittest

import iris
import numpy as np

from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import aggregate


class Test_aggregate(unittest.TestCase):

    """Test the aggregate utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        data = 281 * np.random.random_sample((3, 3, 3)).astype(np.float32)
        self.cube = set_up_variable_cube(data, realizations=[0, 1, 2])
        self.expected_data = self.cube.collapsed(
            ["realization"], iris.analysis.MEAN
        ).data

    def test_invalid_aggregator(self):
        """Test that an error is raised if aggregation method is not one
        of the allowed types."""
        msg = "aggregation must be one of"
        with self.assertRaisesRegex(ValueError, msg):
            aggregate(self.cube, "realization", "product")

    def test_default(self):
        """Test with default options."""
        result = aggregate(self.cube)
        self.assertTrue((result.data == self.expected_data).all())

    def test_broadcast(self):
        """Test that if broadcast is True, returned cube
        has same coords as original."""
        result = aggregate(self.cube, "realization", "mean", broadcast=True)
        expected_data = np.broadcast_to(self.expected_data, self.cube.data.shape)
        self.assertTrue((result.data == expected_data).all())
        self.assertTrue(result.coords() == self.cube.coords())

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
            result = aggregate(self.cube, "realization", key)
            expected_data = self.cube.collapsed(["realization"], value).data
            self.assertTrue((result.data == expected_data).all())

    def test_rename(self):
        """Test rename functionality."""
        new_name = "ensemble_mean"
        result = aggregate(self.cube, "realization", "mean", new_name=new_name)
        self.assertTrue(result.name() == new_name)


if __name__ == "__main__":
    unittest.main()
