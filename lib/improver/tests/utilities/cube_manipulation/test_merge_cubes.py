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
Unit tests for the function "cube_manipulation.merge_cubes".
"""

import unittest
import numpy as np
from datetime import datetime as dt

import iris
from iris.cube import Cube
from iris.exceptions import DuplicateDataError, MergeError
from iris.tests import IrisTest

from improver.utilities.cube_checker import find_threshold_coordinate
from improver.utilities.cube_manipulation_new import merge_cubes
from improver.utilities.warnings_handler import ManageWarnings
from improver.tests.set_up_test_cubes import (
    set_up_variable_cube, set_up_probability_cube)


class Test_merge_cubes(IrisTest):

    """Test the merge_cubes utility."""

    def setUp(self):
        """Use temperature cube to test with."""

        data = np.array([[[226.15, 237.4, 248.65],
                          [259.9, 271.15, 282.4],
                          [293.65, 304.9, 316.15]],
                         [[230.15, 241.4, 252.65],
                          [263.9, 275.15, 286.4],
                          [297.65, 308.9, 320.15]],
                         [[232.15, 243.4, 254.65],
                          [265.9, 277.15, 288.4],
                          [299.65, 310.9, 322.15]]], dtype=np.float32)

        # set up a MOGREPS-UK cube with 7 hour forecast period
        time_point = dt(2015, 11, 23, 7)
        self.cube = set_up_variable_cube(
            data.copy(), standard_grid_metadata='uk_ens', time=time_point,
            frt=dt(2015, 11, 23, 0))

        # set up a UKV cube with 4 hour forecast period
        self.cube_ukv = set_up_variable_cube(
            data[1].copy(), standard_grid_metadata='uk_det', time=time_point,
            frt=dt(2015, 11, 23, 3))

        # set up more UKV cubes with 5 and 6 hour forecast periods
        self.cube_ukv_t1 = set_up_variable_cube(
            data[1].copy(), standard_grid_metadata='uk_det', time=time_point,
            frt=dt(2015, 11, 23, 2))
        self.cube_ukv_t2 = set_up_variable_cube(
            data[1].copy(), standard_grid_metadata='uk_det', time=time_point,
            frt=dt(2015, 11, 23, 1))

        # Setup two non-Met Office model example configuration cubes.
        # Using a simple temperature data array, one cube set is setup
        # as a deterministic model, the other as an ensemble.
        self.data = (
            np.linspace(275.0, 284.0, 12).reshape(3, 4).astype(np.float32))
        self.data_3d = np.array([self.data, self.data, self.data])

        self.cube_non_mo_det = set_up_variable_cube(self.data)
        self.cube_non_mo_ens = set_up_variable_cube(
            self.data_3d, realizations=np.array([0, 3, 4]))

        self.cube_non_mo_det.attributes['non_mo_model_config'] = 'non_uk_det'
        self.cube_non_mo_ens.attributes['non_mo_model_config'] = 'non_uk_ens'

    def test_basic(self):
        """Test that the utility returns an iris.cube.Cube."""
        result = merge_cubes([self.cube_ukv, self.cube_ukv_t1])
        self.assertIsInstance(result, Cube)

    def test_identical_cubes(self):
        """Test that merging identical cubes fails."""
        cubes = iris.cube.CubeList([self.cube, self.cube])
        msg = "failed to merge into a single cube"
        with self.assertRaisesRegex(DuplicateDataError, msg):
            merge_cubes(cubes)

    def test_lagged_ukv(self):
        """Test lagged UKV merge OK (forecast periods in seconds)"""
        expected_fp_points = 3600*np.array([6, 5, 4], dtype=np.int32)
        cubes = iris.cube.CubeList([self.cube_ukv,
                                    self.cube_ukv_t1,
                                    self.cube_ukv_t2])
        result = merge_cubes(cubes)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(
            result.coord("forecast_period").points, expected_fp_points)


if __name__ == '__main__':
    unittest.main()
