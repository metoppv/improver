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
"""Tests for the WeightAndBlend plugin"""

import unittest
import numpy as np
from cf_units import Unit
from datetime import datetime as dt

import iris
from iris.tests import IrisTest

from improver.blending.calculate_weights_and_blend import WeightAndBlend
from improver.utilities.warnings_handler import ManageWarnings
from improver.tests.set_up_test_cubes import set_up_probability_cube


MODEL_WEIGHTS = {
    "nc_det": {"forecast_period": [0, 4, 8],
               "weights": [1, 0, 0],
               "units": "hours"},
    "uk_det": {"forecast_period": [0, 4, 8],
               "weights": [0, 1, 0],
               "units": "hours"},
    "uk_ens": {"forecast_period": [0, 4, 8],
               "weights": [0, 1, 1],
               "units": "hours"}
}


class Test__init__(IrisTest):
    """Test the __init__ method"""

    def test_cycle(self):
        """Test initialisation for cycle blending"""
        plugin = WeightAndBlend(
            "forecast_reference_time", "linear", y0val=1, ynval=1)
        self.assertEqual(plugin.blend_coord, "forecast_reference_time")
        self.assertEqual(plugin.wts_calc_method, "linear")
        self.assertIsNone(plugin.weighting_coord)
        # TODO remove
        self.assertEqual(plugin.blend_coord_unit, Unit('unknown'))
        self.assertAlmostEqual(plugin.y0val, 1.0)
        self.assertAlmostEqual(plugin.ynval, 1.0)

    def test_model(self):
        """Test initialisation for model blending"""
        plugin = WeightAndBlend(
            "model_id", "dict", weighting_coord="forecast_period",
            wts_dict=MODEL_WEIGHTS)
        self.assertEqual(plugin.blend_coord, "model_id")
        self.assertEqual(plugin.wts_calc_method, "dict")
        self.assertEqual(plugin.weighting_coord, "forecast_period")
        # TODO remove
        self.assertIsNone(plugin.blend_coord_unit)
        self.assertDictEqual(plugin.wts_dict, MODEL_WEIGHTS)

    def test_unrecognised_weighting_type_error(self):
        """Test error if wts_calc_method is unrecognised"""
        msg = "Weights calculation method 'kludge' unrecognised"
        with self.assertRaisesRegex(ValueError, msg):
            WeightAndBlend("forecast_period", "kludge")


class Test_process(IrisTest):
    """Test the process method"""

    def setUp(self):
        """Set up test cubes (each with a single point and 3 thresholds)"""
        thresholds = np.array([0.5, 1, 2], dtype=np.float32)
        units = "mm h-1"
        name = "lwe_precipitation_rate"
        datatime = dt(2018, 9, 10, 7)
        cycletime = dt(2018, 9, 10, 3)

        # a UKV cube with some rain and a 4 hr forecast period
        rain_data = np.array([[[0.9]], [[0.5]], [[0]]], dtype=np.float32)
        self.ukv_cube = set_up_probability_cube(
            rain_data, thresholds, variable_name=name,
            threshold_units=units, time=datatime, frt=cycletime,
            standard_grid_metadata="uk_det")

        # a UKV cube from a more recent cycle with more rain
        more_rain_data = np.array([[[1]], [[0.6]], [[0.2]]], dtype=np.float32)
        self.ukv_cube_latest = set_up_probability_cube(
            more_rain_data, thresholds, variable_name=name,
            threshold_units=units, time=datatime, frt=dt(2018, 9, 10, 4),
            standard_grid_metadata="uk_det")

        # a nowcast cube with more rain and a 2 hr forecast period
        self.nowcast_cube = set_up_probability_cube(
            more_rain_data, thresholds, variable_name=name,
            threshold_units=units, time=datatime, frt=dt(2018, 9, 10, 5),
            attributes={"mosg__model_configuration": "nc_det"})

        # a MOGREPS-UK cube with less rain and a 4 hr forecast period
        less_rain_data = np.array([[[0.7]], [[0.3]], [[0]]], dtype=np.float32)
        self.enukx_cube = set_up_probability_cube(
            less_rain_data, thresholds, variable_name=name,
            threshold_units=units, time=datatime, frt=cycletime,
            standard_grid_metadata="uk_ens")

        self.plugin_cycle = WeightAndBlend(
            "forecast_reference_time", "linear", y0val=1, ynval=1)
        self.plugin_model = WeightAndBlend(
            "model_id", "dict", weighting_coord="forecast_period",
            wts_dict=MODEL_WEIGHTS)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate"])
    def test_basic(self):
        """Test output is a cube"""
        result = self.plugin_cycle.process(
            [self.ukv_cube, self.ukv_cube_latest], "weighted_mean")
        self.assertIsInstance(result, iris.cube.Cube)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate"])
    def test_cycle_blend_linear(self):
        """Test plugin produces correct cycle blended output with equal
        linear weightings"""
        expected_data = np.array(
            [[[0.95]], [[0.55]], [[0.1]]], dtype=np.float32)
        result = self.plugin_cycle.process(
            [self.ukv_cube, self.ukv_cube_latest], "weighted_mean")
        self.assertArrayAlmostEqual(result.data, expected_data)
        # make sure output cube has the forecast reference time and period
        # from the most recent contributing cycle
        for coord in ["time", "forecast_reference_time", "forecast_period"]:
            self.assertEqual(
                result.coord(coord), self.ukv_cube_latest.coord(coord))

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate"])
    def test_model_blend(self):
        """Test plugin produces correct output for UKV-ENUKX model blend
        with 50-50 weightings defined by dictionary"""
        expected_data = np.array([[[0.8]], [[0.4]], [[0]]], dtype=np.float32)
        result = self.plugin_model.process(
            [self.ukv_cube, self.enukx_cube], "weighted_mean",
            model_id_attr="mosg__model_configuration")
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertEqual(
            result.attributes["mosg__model_configuration"], "blend")
        result_coords = [coord.name() for coord in result.coords()]
        self.assertNotIn("model_id", result_coords)
        self.assertNotIn("model_configuration", result_coords)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate",
                          "Deleting unmatched attribute"])
    def test_blend_three_models(self):
        """Test plugin produces correct output for 3-model blend when all
        models have (equal) non-zero weights"""
        expected_data = np.array(
            [[[0.8666667]], [[0.4666667]], [[0.0666667]]], dtype=np.float32)
        result = self.plugin_model.process(
            [self.ukv_cube, self.enukx_cube, self.nowcast_cube],
            "weighted_mean", model_id_attr="mosg__model_configuration")
        self.assertArrayAlmostEqual(result.data, expected_data)
        # make sure output cube has the forecast reference time and period
        # from the most recent contributing model
        for coord in ["time", "forecast_period", "forecast_reference_time"]:
            self.assertEqual(
                result.coord(coord), self.nowcast_cube.coord(coord))

    def test_one_cube(self):
        """Test the plugin returns a single input cube with identical data and
        suitably updated metadata"""
        result = self.plugin_model.process(
            [self.enukx_cube], "weighted_mean",
            model_id_attr="mosg__model_configuration")
        self.assertArrayAlmostEqual(result.data, self.enukx_cube.data)
        self.assertEqual(
            result.attributes['mosg__model_configuration'], 'blend')
        self.assertEqual(
            result.attributes['title'], 'IMPROVER Model Forecast')

    def test_error_blend_coord_absent(self):
        """Test error is raised if blend coord is not present on input cubes"""
        plugin = WeightAndBlend("kittens", "linear", y0val=1, ynval=1)
        msg = "kittens coordinate is not present on all input cubes"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process(
                [self.ukv_cube, self.ukv_cube_latest], "weighted_mean")


class Test_process_spatial_weights(IrisTest):
    """Test the process method with spatial weights options"""

    def setUp(self):
        """Set up a masked nowcast and unmasked UKV cube"""
        thresholds = np.array([0.5, 1, 2], dtype=np.float32)
        units = "mm h-1"
        name = "lwe_precipitation_rate"
        datatime = dt(2018, 9, 10, 7)
        cycletime = dt(2018, 9, 10, 5)

        # 5x5 matrix results in grid spacing of 200 km
        base_data = np.ones((5, 5), dtype=np.float32)

        # set up a UKV cube with some rain
        rain_data = np.array([0.9*base_data, 0.5*base_data, 0*base_data])
        ukv_cube = set_up_probability_cube(
            rain_data, thresholds, variable_name=name,
            threshold_units=units, time=datatime, frt=cycletime,
            spatial_grid="equalarea", standard_grid_metadata="uk_det")

        # set up a masked nowcast cube with more rain
        more_rain_data = np.array([base_data, 0.6*base_data, 0.2*base_data])
        radar_mask = np.broadcast_to(
            np.array([False, False, False, True, True]), (3, 5, 5))
        more_rain_data = np.ma.MaskedArray(more_rain_data, mask=radar_mask)
        nowcast_cube = set_up_probability_cube(
            more_rain_data, thresholds, variable_name=name,
            threshold_units=units, time=datatime, frt=cycletime,
            spatial_grid="equalarea",
            attributes={"mosg__model_configuration": "nc_det"})

        self.cubelist = [ukv_cube, nowcast_cube]

        self.plugin = WeightAndBlend(
            "model_id", "dict", weighting_coord="forecast_period",
            wts_dict=MODEL_WEIGHTS)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate",
                          "Deleting unmatched attribute"])
    def test_default(self):
        """Test plugin returns a cube with expected values where default fuzzy
        length is less than grid length (no smoothing)"""
        # data is 50:50 where radar is valid, 100% UKV where radar is masked
        expected_data = np.array(
            [np.broadcast_to([0.95, 0.95, 0.95, 0.9, 0.9], (5, 5)),
             np.broadcast_to([0.55, 0.55, 0.55, 0.5, 0.5], (5, 5)),
             np.broadcast_to([0.1, 0.1, 0.1, 0.0, 0.0], (5, 5))],
            dtype=np.float32)
        result = self.plugin.process(
            self.cubelist, "weighted_mean", spatial_weights=True,
            model_id_attr="mosg__model_configuration")
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_data)

    @ManageWarnings(
        ignored_messages=["Collapsing a non-contiguous coordinate",
                          "Deleting unmatched attribute"])
    def test_fuzzy_length(self):
        """Test values where fuzzy length is equal to 2 grid lengths"""
        # proportion of radar data is reduced at edge of valid region; still
        # 100% UKV where radar data is masked
        expected_data = np.array(
            [np.broadcast_to([0.95, 0.95, 0.9333333, 0.9, 0.9], (5, 5)),
             np.broadcast_to([0.55, 0.55, 0.5333333, 0.5, 0.5], (5, 5)),
             np.broadcast_to([0.1, 0.1, 0.0666666, 0.0, 0.0], (5, 5))],
            dtype=np.float32)
        result = self.plugin.process(
            self.cubelist, "weighted_mean", spatial_weights=True,
            model_id_attr="mosg__model_configuration", fuzzy_length=400000)
        self.assertArrayAlmostEqual(result.data, expected_data)


if __name__ == '__main__':
    unittest.main()
