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
Unit tests for MergeCubesForBlending.
"""

import unittest
import numpy as np
from datetime import datetime as dt

import iris
from iris.tests import IrisTest

from improver.utilities.cube_manipulation_new import (
    MergeCubesForWeightedBlending)
from improver.tests.set_up_test_cubes import set_up_probability_cube


class Test__init__(IrisTest):
    """Test the __init__ method"""

    def test_basic(self):
        """Test default initialisation"""
        plugin = MergeCubesForWeightedBlending()
        self.assertIsNone(plugin.model_id_attr)

    def test_model(self):
        """Test model ID attribute setting"""
        plugin = MergeCubesForWeightedBlending(
            model_id_attr="mosg__model_configuration")
        self.assertEqual(plugin.model_id_attr, "mosg__model_configuration")


class Test__create_model_coordinates(IrisTest):
    """Test the _create_model_coordinates method"""

    def setUp(self):
        """Set up some probability cubes from different models"""
        data = np.array(
            [0.9*np.ones((3, 3)), 0.5*np.ones((3, 3)), 0.1*np.ones((3, 3))],
            dtype=np.float32)
        thresholds = np.array([273., 275., 277.], dtype=np.float32)
        time_point = dt(2015, 11, 23, 7)

        # set up a MOGREPS-UK cube with 7 hour forecast period
        self.cube_enuk = set_up_probability_cube(
            data.copy(), thresholds, standard_grid_metadata='uk_ens',
            time=time_point, frt=dt(2015, 11, 23, 0))

        # set up a UKV cube with 4 hour forecast period
        self.cube_ukv = set_up_probability_cube(
            data.copy(), thresholds, standard_grid_metadata='uk_det',
            time=time_point, frt=dt(2015, 11, 23, 3))

        self.plugin = MergeCubesForWeightedBlending(
            model_id_attr="mosg__model_configuration")

    def test_basic(self):
        """Test model ID and model configuration coords are created"""
        cubelist = iris.cube.CubeList([
            self.cube_enuk.copy(), self.cube_ukv.copy()])
        self.plugin._create_model_coordinates(cubelist)
        for cube in cubelist:
            cube_coords = [coord.name() for coord in cube.coords()]
            self.assertIn("model_id", cube_coords)
            self.assertIn("model_configuration", cube_coords)
            self.assertEqual(
                cube.attributes["mosg__model_configuration"], "blend")

    def test_null(self):
        """Test no effect if model_id_attr is not set"""
        plugin = MergeCubesForWeightedBlending()
        cubelist = iris.cube.CubeList([
            self.cube_enuk.copy(), self.cube_ukv.copy()])
        plugin._create_model_coordinates(cubelist)
        for cube in cubelist:
            cube_coords = [coord.name() for coord in cube.coords()]
            self.assertNotIn("model_id", cube_coords)
            self.assertNotIn("model_configuration", cube_coords)

    def test_unmatched_model_id_attr(self):
        """Test error if model_id_attr is not present on both input cubes"""
        cubelist = iris.cube.CubeList([
            self.cube_enuk.copy(), self.cube_ukv.copy()])
        cubelist[0].attributes.pop("mosg__model_configuration")
        msg = 'Cannot create model ID coordinate for grid blending '
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin._create_model_coordinates(cubelist)


class Test_process(IrisTest):
    """Test the process method"""

    def setUp(self):
        """Set up some probability cubes from different models"""
        data = np.array(
            [0.9*np.ones((3, 3)), 0.5*np.ones((3, 3)), 0.1*np.ones((3, 3))],
            dtype=np.float32)
        thresholds = np.array([273., 275., 277.], dtype=np.float32)
        time_point = dt(2015, 11, 23, 7)
        time_bounds = [dt(2015, 11, 23, 4), time_point]

        # set up a MOGREPS-UK cube with 7 hour forecast period
        self.cube_enuk = set_up_probability_cube(
            data.copy(), thresholds, standard_grid_metadata='uk_ens',
            time=time_point, frt=dt(2015, 11, 23, 0), time_bounds=time_bounds)

        # set up a UKV cube with 4 hour forecast period
        self.cube_ukv = set_up_probability_cube(
            data.copy(), thresholds, standard_grid_metadata='uk_det',
            time=time_point, frt=dt(2015, 11, 23, 3), time_bounds=time_bounds)

        # set up some non-UK test cubes
        self.cube_non_mo_ens = self.cube_enuk.copy()
        self.cube_non_mo_ens.attributes.pop("mosg__model_configuration")
        self.cube_non_mo_ens.attributes['non_mo_model_config'] = 'non_uk_ens'
        self.cube_non_mo_det = self.cube_ukv.copy()
        self.cube_non_mo_det.attributes.pop("mosg__model_configuration")
        self.cube_non_mo_det.attributes['non_mo_model_config'] = 'non_uk_det'

        self.plugin = MergeCubesForWeightedBlending(
            model_id_attr="mosg__model_configuration")

    def test_basic(self):
        """Test single cube is returned unmodified"""
        cube = self.cube_enuk.copy()
        result = self.plugin.process(cube, "model")
        self.assertArrayAlmostEqual(result.data, self.cube_enuk.data)
        self.assertEqual(result.metadata, self.cube_enuk.metadata)

    def test_multi_model_merge(self):
        """Test models merge OK and have expected model coordinates"""
        cubes = iris.cube.CubeList([self.cube_enuk, self.cube_ukv])
        result = self.plugin.process(cubes, "model")
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(
            result.coord("model_id").points, [0, 1000])
        self.assertArrayEqual(
            result.coord("model_configuration").points, ["uk_ens", "uk_det"])

    def test_rationalise_time_coords(self):
        """Test merged cube has scalar time coordinates if weighting models
        by forecast period"""
        cubes = iris.cube.CubeList([self.cube_enuk, self.cube_ukv])
        result = self.plugin.process(
            cubes, "model", weighting_coord="forecast_period")
        # test resulting cube has single 4 hour (shorter) forecast period
        self.assertEqual(result.coord("forecast_period").points, [4*3600])
        # check time and frt points are also consistent with the UKV input cube
        self.assertEqual(
            result.coord("time").points, self.cube_ukv.coord("time").points)
        self.assertEqual(
            result.coord("forecast_reference_time").points,
            self.cube_ukv.coord("forecast_reference_time").points)

    def test_no_model_id_attr(self):
        """Test multi model blending fails if no model_id_attr is specified"""
        plugin = MergeCubesForWeightedBlending()
        cubes = iris.cube.CubeList([self.cube_enuk, self.cube_ukv])
        msg = "model_id_attr required to blend over model"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(cubes, "model")

    def test_non_mo_model_id(self):
        """Test that a model ID attribute string can be specified when
        merging multi model cubes"""
        plugin = MergeCubesForWeightedBlending(
            model_id_attr='non_mo_model_config')
        result = plugin.process(
            [self.cube_non_mo_ens, self.cube_non_mo_det], "model")
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(
            result.coord("model_id").points, [0, 1000])

    def test_model_id_attr_mismatch(self):
        """Test that when a model ID attribute string is specified that does
        not match the model ID attribute key name on both cubes to be merged,
        an error is thrown"""
        plugin = MergeCubesForWeightedBlending(
            model_id_attr='non_matching_model_config')
        msg = ('Cannot create model ID coordinate for grid blending '
               'as the model ID attribute specified is not found '
               'within the cube attributes')
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(
                [self.cube_non_mo_ens, self.cube_non_mo_det], "model")

    def test_model_id_attr_mismatch_one_cube(self):
        """Test that when a model ID attribute string is specified that only
        matches the model ID attribute key name on one of the cubes to be
        merged, an error is thrown"""
        self.cube_non_mo_det.attributes.pop('non_mo_model_config')
        self.cube_non_mo_det.attributes[
            'non_matching_model_config'] = 'non_uk_det'
        plugin = MergeCubesForWeightedBlending(
            model_id_attr='non_matching_model_config')
        msg = ('Cannot create model ID coordinate for grid blending '
               'as the model ID attribute specified is not found '
               'within the cube attributes')
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(
                [self.cube_non_mo_ens, self.cube_non_mo_det], "model")

    def test_time_bounds_mismatch(self):
        """Test failure for cycle blending when time bounds ranges are not
        matched (ie cycle blending different "accumulation periods")"""
        cube2 = self.cube_ukv.copy()
        cube2.coord("forecast_reference_time").points = (
            cube2.coord("forecast_reference_time").points + 3600)
        cube2.coord("time").bounds = [
            cube2.coord("time").bounds[0, 0] + 3600,
            cube2.coord("time").bounds[0, 1]]
        cube2.coord("forecast_period").bounds = [
            cube2.coord("forecast_period").bounds[0, 0] + 3600,
            cube2.coord("forecast_period").bounds[0, 1]]
        msg = "Cube with mismatching time bounds ranges cannot be blended"
        with self.assertRaisesRegex(ValueError, msg):
            MergeCubesForWeightedBlending().process(
                [self.cube_ukv, cube2], "forecast_reference_time")


if __name__ == '__main__':
    unittest.main()
