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
"""
Unit tests for the function "cube_manipulation.equalise_cube_attributes".
"""

import unittest
import numpy as np
from datetime import datetime as dt

import iris
from iris.tests import IrisTest

from improver.utilities.cube_manipulation import equalise_cube_attributes
from improver.tests.set_up_test_cubes import set_up_variable_cube
from improver.utilities.warnings_handler import ManageWarnings


class Test_equalise_cube_attributes(IrisTest):

    """Test the equalise_cube_attributes utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        data = np.array([[[226.15, 237.40, 248.65],
                          [259.90, 271.15, 282.40],
                          [293.65, 304.90, 316.15]],
                         [[230.15, 241.40, 252.65],
                          [263.90, 275.15, 286.40],
                          [297.65, 308.90, 320.15]],
                         [[232.15, 243.40, 254.65],
                          [265.90, 277.15, 288.40],
                          [299.65, 310.90, 322.15]]], dtype=np.float32)

        cube_attrs = {"history": "2017-01-18T08:59:53: StaGE Decoupler",
                      "unknown_attribute": "1"}
        self.cube = set_up_variable_cube(
            data, time=dt(2017, 1, 10, 3), frt=dt(2017, 1, 10, 0),
            attributes=cube_attrs)

    def test_cubelist_history_removal(self):
        """Test that the utility removes history attribute,
        if they are different.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube2.attributes["history"] = "2017-01-19T08:59:53: StaGE Decoupler"

        cubelist = iris.cube.CubeList([cube1, cube2])

        equalise_cube_attributes(cubelist, silent="history")
        self.assertNotIn("history", cubelist[0].attributes.keys())
        self.assertNotIn("history", cubelist[1].attributes.keys())

    def test_cubelist_no_history_removal(self):
        """Test that the utility does not remove history attribute,
        if they are the same.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cubelist = iris.cube.CubeList([cube1, cube2])

        equalise_cube_attributes(cubelist)
        self.assertIn("history", cubelist[0].attributes.keys())
        self.assertIn("history", cubelist[1].attributes.keys())

    @ManageWarnings(record=True)
    def test_unknown_attribute(self, warning_list=None):
        """Test that the utility raises warning when removing a non-silent
        mismatched attribute."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube2.attributes.update({'unknown_attribute': '2'})

        cubelist = iris.cube.CubeList([cube1, cube2])

        equalise_cube_attributes(cubelist)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        warning_msg = "Deleting unmatched attribute "
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertNotIn("unknown_attribute",
                         cubelist[0].attributes.keys())
        self.assertNotIn("unknown_attribute",
                         cubelist[1].attributes.keys())


if __name__ == '__main__':
    unittest.main()
