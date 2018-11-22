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
Unit tests for the function "cube_manipulation.merge_cubes".
"""

import unittest

import iris
from iris.cube import Cube
from iris.exceptions import DuplicateDataError
from iris.tests import IrisTest
import numpy as np

from improver.utilities.cube_manipulation import merge_cubes

from improver.tests.ensemble_calibration.ensemble_calibration.\
    helper_functions import (
        set_up_temperature_cube,
        set_up_probability_above_threshold_temperature_cube,
        add_forecast_reference_time_and_forecast_period)

from improver.utilities.warnings_handler import ManageWarnings
from improver.tests.set_up_test_cubes import set_up_variable_cube


class Test_merge_cubes(IrisTest):

    """Test the merge_cubes utility."""

    def setUp(self):
        """Use temperature cube to test with."""
        self.cube = set_up_temperature_cube()
        self.cube_ukv = self.cube.extract(iris.Constraint(realization=1))
        self.cube_ukv.remove_coord('realization')
        self.cube_ukv.attributes['mosg__grid_type'] = 'standard'
        self.cube_ukv.attributes['mosg__model_configuration'] = 'uk_det'
        self.cube_ukv.attributes['mosg__grid_domain'] = 'uk_extended'
        self.cube_ukv.attributes['mosg__grid_version'] = '1.2.0'
        self.cube_ukv_t1 = self.cube_ukv.copy()
        self.cube_ukv_t2 = self.cube_ukv.copy()
        add_forecast_reference_time_and_forecast_period(self.cube_ukv,
                                                        fp_point=4.0)
        add_forecast_reference_time_and_forecast_period(self.cube_ukv_t1,
                                                        fp_point=5.0)
        add_forecast_reference_time_and_forecast_period(self.cube_ukv_t2,
                                                        fp_point=6.0)
        add_forecast_reference_time_and_forecast_period(self.cube,
                                                        fp_point=7.0)
        self.cube.attributes['mosg__grid_type'] = 'standard'
        self.cube.attributes['mosg__model_configuration'] = 'uk_ens'
        self.cube.attributes['mosg__grid_domain'] = 'uk_extended'
        self.cube.attributes['mosg__grid_version'] = '1.2.0'
        self.prob_ukv = set_up_probability_above_threshold_temperature_cube()
        self.prob_ukv.attributes['mosg__grid_type'] = 'standard'
        self.prob_ukv.attributes['mosg__model_configuration'] = 'uk_det'
        self.prob_ukv.attributes['mosg__grid_domain'] = 'uk_extended'
        self.prob_ukv.attributes['mosg__grid_version'] = '1.2.0'
        self.prob_enuk = set_up_probability_above_threshold_temperature_cube()
        self.prob_enuk.attributes.update({'mosg__grid_type': 'standard'})
        self.prob_enuk.attributes.update(
            {'mosg__model_configuration': 'uk_ens'})
        self.prob_enuk.attributes.update({'mosg__grid_domain': 'uk_extended'})
        self.prob_enuk.attributes.update({'mosg__grid_version': '1.2.0'})

        # Setup two non-Met Office model example configuration cubes.
        # Using a simple temperature data array, one cube set is setup
        # as a deterministic model, the other as an ensemble.
        self.data = (
            np.linspace(275.0, 284.0, 12).reshape(3, 4).astype(np.float32))
        self.data_3d = np.array([self.data, self.data, self.data])

        self.cube_non_mo_det = set_up_variable_cube(self.data_3d)

        self.cube_non_mo_ens = set_up_variable_cube(
            self.data_3d, realizations=np.array([0, 3, 4]))

        self.cube_non_mo_det.attributes['non_mo_model_config'] = 'non_uk_det'
        self.cube_non_mo_ens.attributes['non_mo_model_config'] = 'non_uk_ens'

    @ManageWarnings(record=True)
    def test_basic(self, warning_list=None):
        """Test that the utility returns an iris.cube.Cube."""
        result = merge_cubes(self.cube)
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        warning_msg = "Only a single cube "
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertIsInstance(result, Cube)

    def test_identical_cubes(self):
        """Test that merging identical cubes fails."""
        cubes = iris.cube.CubeList([self.cube, self.cube])
        msg = "failed to merge into a single cube"
        with self.assertRaisesRegex(DuplicateDataError, msg):
            merge_cubes(cubes)

    def test_lagged_ukv(self):
        """Test Lagged ukv merge OK"""
        cubes = iris.cube.CubeList([self.cube_ukv,
                                    self.cube_ukv_t1,
                                    self.cube_ukv_t2])
        result = merge_cubes(cubes)
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(
            result.coord("forecast_period").points, [6.0, 5.0, 4.0])

    def test_multi_model(self):
        """Test Multi models merge OK"""
        cubes = iris.cube.CubeList([self.cube, self.cube_ukv])
        result = merge_cubes(cubes, "mosg__model_configuration")
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(
            result.coord("model_realization").points, [0., 1.,
                                                       2., 1000.])

    def test_threshold_data(self):
        """Test threshold data merges OK"""
        cubes = iris.cube.CubeList([self.prob_ukv, self.prob_enuk])
        result = merge_cubes(cubes, "mosg__model_configuration")
        self.assertArrayAlmostEqual(
            result.coord("model_id").points, [0., 1000.])

    def test_non_mo_model_id(self):
        """Test that a model ID attribute string can be specified when
        merging multi model cubes"""
        cubes = iris.cube.CubeList(
            [self.cube_non_mo_ens, self.cube_non_mo_det])
        result = merge_cubes(cubes, 'non_mo_model_config')
        self.assertIsInstance(result, Cube)
        self.assertArrayAlmostEqual(
            result.coord(
                "model_realization").points, [0., 3., 4., 1000., 1001., 1002.])

    def test_model_id_attr_mismatch(self):
        """Test that when a model ID attribute string is specified that does
        not match the model ID attribute key name on both cubes to be merged,
        an error is thrown"""
        cubes = iris.cube.CubeList(
            [self.cube_non_mo_ens, self.cube_non_mo_det])

        # The test cubes contain the 'non_mo_model_config' attribute key.
        # We'll specify 'non_matching_model_config' as our model ID
        # attribute key argument. Merge_cubes should then raise a warning
        # as our specified model ID does not match that on the cubes and it
        # will not be able to build a model ID coordinate.
        msg = ('Cannot create model ID coordinate for grid blending '
               'as the model ID attribute specified does not match '
               'that on the cubes to be blended')

        with self.assertRaisesRegex(ValueError, msg):
            merge_cubes(cubes, 'non_matching_model_config')

    def test_model_id_attr_mismatch_one_cube(self):
        """Test that when a model ID attribute string is specified that only
        matches the model ID attribute key name on one of the cubes to be
        merged, an error is thrown"""

        # Change the model ID attribute key on one of the test cubes so that
        # it matches the model ID argument. Merge_cubes should still raise
        # an error as the model ID attribute key has to match on all cubes
        # to be blended.
        self.cube_non_mo_det.attributes.pop('non_mo_model_config')
        self.cube_non_mo_det.attributes[
            'non_matching_model_config'] = 'non_uk_det'

        cubes = iris.cube.CubeList(
            [self.cube_non_mo_ens, self.cube_non_mo_det])

        msg = ('Cannot create model ID coordinate for grid blending '
               'as the model ID attribute specified does not match '
               'that on the cubes to be blended')

        with self.assertRaisesRegex(ValueError, msg):
            merge_cubes(cubes, 'non_matching_model_config')


if __name__ == '__main__':
    unittest.main()
