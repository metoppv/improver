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
Unit tests for the function "cube_manipulation.equalise_cubes".
"""

import unittest

import iris
from iris.tests import IrisTest
import numpy as np

from improver.utilities.cube_manipulation import equalise_cubes

from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import (
        set_up_temperature_cube,
        add_forecast_reference_time_and_forecast_period)

from improver.utilities.warnings_handler import ManageWarnings


class Test_equalise_cubes(IrisTest):

    """Test the_equalise_cubes utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()
        self.cube_ukv = self.cube.extract(iris.Constraint(realization=1))
        self.cube_ukv.remove_coord('realization')
        self.cube_ukv.attributes['mosg__grid_type'] = 'standard'
        self.cube_ukv.attributes['mosg__model_configuration'] = 'uk_det'
        self.cube_ukv.attributes['mosg__grid_domain'] = 'uk_extended'
        self.cube_ukv.attributes['mosg__grid_version'] = '1.2.0'
        add_forecast_reference_time_and_forecast_period(self.cube_ukv,
                                                        fp_point=4.0)
        add_forecast_reference_time_and_forecast_period(self.cube,
                                                        fp_point=7.0)
        self.cube.attributes['mosg__grid_type'] = 'standard'
        self.cube.attributes['mosg__model_configuration'] = 'uk_ens'
        self.cube.attributes['mosg__grid_domain'] = 'uk_extended'
        self.cube.attributes['mosg__grid_version'] = '1.2.0'
        self.cube.attributes["history"] = (
            "2017-01-18T08:59:53: StaGE Decoupler")
        self.cube_ukv.attributes["history"] = (
            "2017-01-19T08:59:53: StaGE Decoupler")

    @ManageWarnings(record=True)
    def test_basic(self, warning_list=None):
        """Test that the utility returns an iris.cube.CubeList."""
        cubes = self.cube
        if isinstance(cubes, iris.cube.Cube):
            cubes = iris.cube.CubeList([cubes])
        result = equalise_cubes(cubes)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        warning_msg = "Only a single cube "
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertIsInstance(result, iris.cube.CubeList)

    def test_equalise_attributes(self):
        """Test that the utility equalises the attributes as expected"""
        cubelist = iris.cube.CubeList([self.cube_ukv, self.cube])
        result = equalise_cubes(cubelist, "mosg__model_configuration")
        self.assertArrayAlmostEqual(result[0].coord("model_id").points,
                                    np.array([0]))
        self.assertEqual(result[0].coord("model_configuration").points[0],
                         'uk_det')
        self.assertArrayAlmostEqual(result[1].coord("model_id").points,
                                    np.array([1000]))
        self.assertEqual(result[1].coord("model_configuration").points[0],
                         'uk_ens')
        self.assertNotIn("mosg__model_configuration", result[0].attributes)
        self.assertNotIn("mosg__model_configuration", result[1].attributes)
        self.assertAlmostEqual(result[0].attributes["mosg__grid_domain"],
                               result[1].attributes["mosg__grid_domain"])
        self.assertEqual(result[0].attributes["mosg__grid_domain"],
                         'uk_extended')
        self.assertNotIn("history", result[0].attributes.keys())
        self.assertNotIn("history", result[1].attributes.keys())

    def test_strip_var_names(self):
        """Test that the utility removes var names"""
        cube1 = self.cube.copy()
        cube2 = self.cube.copy()
        cube1.coord("time").var_name = "time_0"
        cube2.coord("time").var_name = "time_1"
        cubelist = iris.cube.CubeList([cube1, cube2])
        result = equalise_cubes(cubelist)
        self.assertIsNone(result[0].coord("time").var_name)
        self.assertIsNone(result[1].coord("time").var_name)

    def test_coords_not_equalised_if_not_merging(self):
        """Test that the coords are not equalised if not merging"""
        cubelist = iris.cube.CubeList([self.cube_ukv, self.cube])
        result = equalise_cubes(cubelist, merging=False)
        self.assertEqual(len(result),
                         len(cubelist))

    def test_coords_are_equalised_if_merging(self):
        """Test that the coords are equalised if merging"""
        cubelist = iris.cube.CubeList([self.cube_ukv, self.cube])
        result = equalise_cubes(cubelist, "mosg__model_configuration")
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[3].coord('model_realization').points,
                               1002.0)


if __name__ == '__main__':
    unittest.main()
