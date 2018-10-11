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
Unit tests for the function "cube_manipulation.compare_attributes".
"""

import unittest

import iris
from iris.tests import IrisTest

from improver.utilities.cube_manipulation import compare_attributes

from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import set_up_temperature_cube

from improver.utilities.warnings_handler import ManageWarnings


class Test_compare_attributes(IrisTest):
    """Test the compare_attributes utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()
        self.cube_ukv = self.cube.extract(iris.Constraint(realization=1))
        self.cube_ukv.remove_coord('realization')
        self.cube_ukv.attributes['mosg__grid_type'] = 'standard'
        self.cube_ukv.attributes['mosg__model_configuration'] = 'uk_det'
        self.cube_ukv.attributes['mosg__grid_domain'] = 'uk_extended'
        self.cube_ukv.attributes['mosg__grid_version'] = '1.1.0'
        self.cube.attributes['mosg__grid_type'] = 'standard'
        self.cube.attributes['mosg__model_configuration'] = 'uk_ens'
        self.cube.attributes['mosg__grid_domain'] = 'uk_extended'
        self.cube.attributes['mosg__grid_version'] = '1.2.0'

    def test_basic(self):
        """Test that the utility returns a list and have no differences."""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cubelist = iris.cube.CubeList([cube1, cube2])
        result = compare_attributes(cubelist)
        self.assertIsInstance(result, list)
        self.assertAlmostEqual(result, [{}, {}])

    @ManageWarnings(record=True)
    def test_warning(self, warning_list=None):
        """Test that the utility returns warning if only one cube supplied."""
        result = (
            compare_attributes(iris.cube.CubeList([self.cube])))
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        warning_msg = "Only a single cube so no differences will be found "
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertAlmostEqual(result, [])

    def test_history_attribute(self):
        """Test that the utility returns diff when history do not match"""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube1.attributes["history"] = "2017-01-18T08:59:53: StaGE Decoupler"
        cube2.attributes["history"] = "2017-01-19T08:59:53: StaGE Decoupler"
        cubelist = iris.cube.CubeList([cube1, cube2])
        result = compare_attributes(cubelist)
        self.assertEqual(result,
                         [{'history':
                           '2017-01-18T08:59:53: StaGE Decoupler'},
                          {'history':
                           '2017-01-19T08:59:53: StaGE Decoupler'}])

    def test_multiple_differences(self):
        """Test that the utility returns multiple differences"""
        cubelist = iris.cube.CubeList([self.cube, self.cube_ukv])
        result = compare_attributes(cubelist)
        self.assertAlmostEqual(result,
                               [{'mosg__model_configuration': 'uk_ens',
                                 'mosg__grid_version':
                                 '1.2.0'},
                                {'mosg__model_configuration': 'uk_det',
                                 'mosg__grid_version':
                                 '1.1.0'}])


if __name__ == '__main__':
    unittest.main()
