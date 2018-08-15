# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
Unit tests for the function "cube_manipulation._equalise_cube_attributes".
"""

import unittest

import iris
from iris.tests import IrisTest
import numpy as np

from improver.utilities.cube_manipulation import _equalise_cube_attributes

from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import set_up_temperature_cube

from improver.utilities.warnings_handler import ManageWarnings


class Test__equalise_cube_attributes(IrisTest):

    """Test the equalise_cube_attributes utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()
        self.cube_ukv = self.cube.extract(iris.Constraint(realization=1))
        self.cube_ukv.remove_coord('realization')

    def test_cubelist_history_removal(self):
        """Test that the utility removes history attribute,
        if they are different.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube2.coord("time").points = 402195.0
        cube1.attributes["history"] = "2017-01-18T08:59:53: StaGE Decoupler"
        cube2.attributes["history"] = "2017-01-19T08:59:53: StaGE Decoupler"

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = _equalise_cube_attributes(cubelist)
        self.assertNotIn("history", result[0].attributes.keys())
        self.assertNotIn("history", result[1].attributes.keys())

    def test_cubelist_no_history_removal(self):
        """Test that the utility does not remove history attribute,
        if they are the same.
        """
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube2.coord("time").points = 402195.0
        cube1.attributes["history"] = "2017-01-18T08:59:53: StaGE Decoupler"
        cube2.attributes["history"] = "2017-01-18T08:59:53: StaGE Decoupler"

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = _equalise_cube_attributes(cubelist)
        self.assertIn("history", result[0].attributes.keys())
        self.assertIn("history", result[1].attributes.keys())

    def test_cubelist_mosg__model_configuration_identical(self):
        """
        Test that the utility does nothing to mosg__model_configuration if
        they match
        """

        cube1 = self.cube_ukv.copy()
        cube2 = self.cube.copy()
        cube1.attributes.update({'mosg__model_configuration':
                                 'uk_det'})
        cube2.attributes.update({'mosg__model_configuration':
                                 'uk_det'})

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = _equalise_cube_attributes(cubelist)
        self.assertEqual(result[0].attributes["mosg__model_configuration"],
                         result[1].attributes["mosg__model_configuration"])
        self.assertEqual(result[0].attributes["mosg__model_configuration"],
                         'uk_det')

    def test_cubelist_mosg__model_configuration(self):
        """Test that the utility adds coords for model if not matching"""

        cube1 = self.cube_ukv.copy()
        cube2 = self.cube.copy()
        cube1.attributes.update({'mosg__model_configuration':
                                 'uk_det'})
        cube2.attributes.update({'mosg__model_configuration':
                                 'uk_ens'})

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = _equalise_cube_attributes(cubelist)

        self.assertArrayAlmostEqual(result[0].coord("model_id").points,
                                    np.array([3000]))
        self.assertEqual(result[0].coord("model").points[0],
                         'uk_det')
        self.assertArrayAlmostEqual(result[1].coord("model_id").points,
                                    np.array([4000]))
        self.assertEqual(result[1].coord("model").points[0],
                         'uk_ens')
        self.assertNotIn("mosg__model_configuration",
                         result[0].attributes.keys())
        self.assertNotIn("mosg__model_configuration",
                         result[1].attributes.keys())

    @ManageWarnings(record=True)
    def test_unknown_attribute(self, warning_list=None):
        """Test that the utility returns warning and removes unknown
        mismatching attribute."""
        cube1 = self.cube_ukv.copy()
        cube2 = self.cube.copy()
        cube1.attributes.update({'unknown_attribute':
                                 '1'})
        cube2.attributes.update({'unknown_attribute':
                                 '2'})

        cubelist = iris.cube.CubeList([cube1, cube2])

        result = _equalise_cube_attributes(cubelist)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        warning_msg = "Do not know what to do with "
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertNotIn("unknown_attribute",
                         result[0].attributes.keys())
        self.assertNotIn("unknown_attribute",
                         result[1].attributes.keys())


if __name__ == '__main__':
    unittest.main()
