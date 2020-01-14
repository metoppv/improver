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
"""Unit tests for MergeCubesForWeightedBlending"""

import unittest
from datetime import datetime as dt

import iris
import numpy as np
from iris.tests import IrisTest

from improver.blending.weighted_blend import MergeCubesForWeightedBlending
from improver.utilities.warnings_handler import ManageWarnings

from ...set_up_test_cubes import set_up_probability_cube, set_up_variable_cube


class Test__init__(IrisTest):
    """Test the __init__ method"""

    def test_basic(self):
        """Test default initialisation"""
        plugin = MergeCubesForWeightedBlending("realization")
        self.assertEqual(plugin.blend_coord, "realization")
        self.assertIsNone(plugin.weighting_coord)
        self.assertIsNone(plugin.model_id_attr)

    def test_optional_args(self):
        """Test model ID and weighting coordinate setting"""
        plugin = MergeCubesForWeightedBlending(
            "model_id", weighting_coord="forecast_period",
            model_id_attr="mosg__model_configuration")
        self.assertEqual(plugin.weighting_coord, "forecast_period")
        self.assertEqual(plugin.model_id_attr, "mosg__model_configuration")

    def test_error_missing_model_id_attr(self):
        """Test exception is raised if blending over model with no identifying
        attribute"""
        msg = "model_id_attr required to blend over model_id"
        with self.assertRaisesRegex(ValueError, msg):
            MergeCubesForWeightedBlending("model_id")

    @ManageWarnings(record=True)
    def test_warning_unnecessary_model_id_attr(self, warning_list=None):
        """Test warning if model_id_attr is set for non-model blending"""
        warning_msg = "model_id_attr not required"
        plugin = MergeCubesForWeightedBlending(
            "realization", model_id_attr="mosg__model_configuration")
        self.assertTrue(any(item.category == UserWarning
                            for item in warning_list))
        self.assertTrue(any(warning_msg in str(item)
                            for item in warning_list))
        self.assertIsNone(plugin.model_id_attr)


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

        self.cubelist = iris.cube.CubeList([self.cube_enuk, self.cube_ukv])
        self.plugin = MergeCubesForWeightedBlending(
            "model", weighting_coord="forecast_period",
            model_id_attr="mosg__model_configuration")

    def test_basic(self):
        """Test model ID and model configuration coords are created and that
        the model_id_attr (in this case 'mosg__model_configuration') is
        correctly updated"""
        self.plugin._create_model_coordinates(self.cubelist)
        for cube in self.cubelist:
            cube_coords = [coord.name() for coord in cube.coords()]
            self.assertIn("model_id", cube_coords)
            self.assertIn("model_configuration", cube_coords)
            self.assertEqual(
                cube.attributes["mosg__model_configuration"], "blend")

    def test_values(self):
        """Test values of model coordinates are as expected"""
        expected_id = [0, 1000]
        expected_config = ["uk_ens", "uk_det"]
        self.plugin._create_model_coordinates(self.cubelist)
        for cube, m_id, m_conf in zip(
                self.cubelist, expected_id, expected_config):
            self.assertEqual(cube.coord("model_id").points, [m_id])
            self.assertEqual(
                cube.coord("model_configuration").points, [m_conf])

    def test_unmatched_model_id_attr(self):
        """Test error if model_id_attr is not present on both input cubes"""
        self.cubelist[0].attributes.pop("mosg__model_configuration")
        msg = 'Cannot create model ID coordinate for grid blending '
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin._create_model_coordinates(self.cubelist)

    def test_error_same_model(self):
        """Test error if input cubes are from the same model"""
        new_cubelist = iris.cube.CubeList(
            [self.cube_enuk.copy(), self.cube_enuk.copy()])
        msg = 'Cannot create model dimension'
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin._create_model_coordinates(new_cubelist)


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

        self.cubelist = iris.cube.CubeList([self.cube_enuk, self.cube_ukv])

        # set up some non-UK test cubes
        cube_non_mo_ens = self.cube_enuk.copy()
        cube_non_mo_ens.attributes.pop("mosg__model_configuration")
        cube_non_mo_ens.attributes['non_mo_model_config'] = 'non_uk_ens'
        cube_non_mo_det = self.cube_ukv.copy()
        cube_non_mo_det.attributes.pop("mosg__model_configuration")
        cube_non_mo_det.attributes['non_mo_model_config'] = 'non_uk_det'

        self.non_mo_cubelist = iris.cube.CubeList(
            [cube_non_mo_ens, cube_non_mo_det])

        # set up plugin for multi-model blending weighted by forecast period
        self.plugin = MergeCubesForWeightedBlending(
            "model", weighting_coord="forecast_period",
            model_id_attr="mosg__model_configuration")

    def test_basic(self):
        """Test single cube is returned unchanged"""
        cube = self.cube_enuk.copy()
        result = self.plugin.process(cube)
        self.assertArrayAlmostEqual(result.data, self.cube_enuk.data)
        self.assertEqual(result.metadata, self.cube_enuk.metadata)

    def test_single_item_list(self):
        """Test cube from single item list is returned unchanged"""
        cubelist = iris.cube.CubeList([self.cube_enuk.copy()])
        result = self.plugin.process(cubelist)
        self.assertArrayAlmostEqual(result.data, self.cube_enuk.data)
        self.assertEqual(result.metadata, self.cube_enuk.metadata)

    def test_multi_model_merge(self):
        """Test models merge OK and have expected model coordinates"""
        result = self.plugin.process(self.cubelist)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(
            result.coord("model_id").points, [0, 1000])
        self.assertArrayEqual(
            result.coord("model_configuration").points, ["uk_ens", "uk_det"])

    def test_time_coords(self):
        """Test merged cube has scalar time coordinates if weighting models
        by forecast period"""
        result = self.plugin.process(self.cubelist)
        # test resulting cube has single 4 hour (shorter) forecast period
        self.assertEqual(result.coord("forecast_period").points, [4*3600])
        # check time and frt points are also consistent with the UKV input cube
        self.assertEqual(
            result.coord("time").points, self.cube_ukv.coord("time").points)
        self.assertEqual(
            result.coord("forecast_reference_time").points,
            self.cube_ukv.coord("forecast_reference_time").points)

    def test_cycle_blend(self):
        """Test merge for blending over forecast_reference_time"""
        cube = self.cube_ukv.copy()
        cube.coord("forecast_reference_time").points = (
            cube.coord("forecast_reference_time").points + 3600)
        cube.coord("forecast_period").points = (
            cube.coord("forecast_reference_time").points - 3600)
        plugin = MergeCubesForWeightedBlending("forecast_reference_time")
        result = plugin.process([self.cube_ukv, cube])
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertIn(result.coord("forecast_reference_time"),
                      result.coords(dim_coords=True))
        # check no model coordinates have been added
        with self.assertRaises(iris.exceptions.CoordinateNotFoundError):
            result.coord("model_id")
        with self.assertRaises(iris.exceptions.CoordinateNotFoundError):
            result.coord("model_configuration")

    def test_blend_coord_ascending(self):
        """Test the order of the output blend coordinate is always ascending,
        independent of the input cube order"""
        frt = self.cube_ukv.coord("forecast_reference_time").points[0]
        fp = self.cube_ukv.coord("forecast_period").points[0]
        cube1 = self.cube_ukv.copy()
        cube1.coord("forecast_reference_time").points = [frt + 3600]
        cube1.coord("forecast_period").points = [fp - 3600]
        cube2 = self.cube_ukv.copy()
        cube2.coord("forecast_reference_time").points = [frt + 7200]
        cube2.coord("forecast_period").points = [fp - 7200]
        # input unordered cubes; expect ordered output
        expected_points = np.array([frt, frt+3600, frt+7200], dtype=np.int64)
        plugin = MergeCubesForWeightedBlending("forecast_reference_time")
        result = plugin.process([cube1, self.cube_ukv, cube2])
        self.assertArrayEqual(
            result.coord("forecast_reference_time").points, expected_points)

    def test_cycletime(self):
        """Test merged cube has updated forecast reference time and forecast
        period if specified using the 'cycletime' argument"""
        result = self.plugin.process(self.cubelist, cycletime="20151123T0600Z")
        # test resulting cube has forecast period consistent with cycletime
        self.assertEqual(result.coord("forecast_period").points, [3600])
        self.assertEqual(
            result.coord("forecast_reference_time").points,
            self.cube_ukv.coord("forecast_reference_time").points + 3*3600)
        # check validity time is unchanged
        self.assertEqual(
            result.coord("time").points, self.cube_ukv.coord("time").points)

    def test_non_mo_model_id(self):
        """Test that a model ID attribute string can be specified when
        merging multi model cubes"""
        plugin = MergeCubesForWeightedBlending(
            "model", model_id_attr="non_mo_model_config")
        result = plugin.process(self.non_mo_cubelist)
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(
            result.coord("model_id").points, [0, 1000])

    def test_model_id_attr_mismatch(self):
        """Test that when a model ID attribute string is specified that does
        not match the model ID attribute key name on both cubes to be merged,
        an error is thrown"""
        plugin = MergeCubesForWeightedBlending(
            "model", model_id_attr="non_matching_model_config")
        msg = "Cannot create model ID coordinate"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(self.non_mo_cubelist)

    def test_model_id_attr_mismatch_one_cube(self):
        """Test that when a model ID attribute string is specified that only
        matches the model ID attribute key name on one of the cubes to be
        merged, an error is thrown"""
        self.non_mo_cubelist[1].attributes.pop("non_mo_model_config")
        self.non_mo_cubelist[1].attributes[
            "non_matching_model_config"] = "non_uk_det"
        plugin = MergeCubesForWeightedBlending(
            "model", model_id_attr="non_matching_model_config")
        msg = "Cannot create model ID coordinate"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(self.non_mo_cubelist)

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
            MergeCubesForWeightedBlending("forecast_reference_time").process(
                [self.cube_ukv, cube2])

    def test_blend_coord_not_present(self):
        """Test exception when blend coord is not present on inputs"""
        msg = "realization coordinate is not present on all input cubes"
        with self.assertRaisesRegex(ValueError, msg):
            MergeCubesForWeightedBlending("realization").process(self.cubelist)

    def test_blend_realizations(self):
        """Test processing works for merging over coordinates that don't
        require specific setup"""
        data = np.ones((1, 3, 3), dtype=np.float32)
        cube1 = set_up_variable_cube(data, realizations=np.array([0]))
        cube1 = iris.util.squeeze(cube1)
        cube2 = set_up_variable_cube(data, realizations=np.array([1]))
        cube2 = iris.util.squeeze(cube2)
        plugin = MergeCubesForWeightedBlending("realization")
        result = plugin.process([cube1, cube2])
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayEqual(
            result.coord("realization").points, np.array([0, 1]))
        self.assertEqual(result[0].metadata, cube1.metadata)
        self.assertEqual(result[1].metadata, cube2.metadata)


if __name__ == '__main__':
    unittest.main()
