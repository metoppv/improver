# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
from datetime import datetime as dt

import iris
import numpy as np
from iris.tests import IrisTest

from improver.blending.calculate_weights_and_blend import WeightAndBlend
from improver.blending.weighted_blend import MergeCubesForWeightedBlending
from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS
from improver.synthetic_data.set_up_test_cubes import (
    set_up_probability_cube,
    set_up_variable_cube,
)
from improver.utilities.warnings_handler import ManageWarnings

MODEL_WEIGHTS = {
    "nc_det": {"forecast_period": [0, 4, 8], "weights": [1, 0, 0], "units": "hours"},
    "uk_det": {"forecast_period": [0, 4, 8], "weights": [0, 1, 0], "units": "hours"},
    "uk_ens": {"forecast_period": [0, 4, 8], "weights": [0, 1, 1], "units": "hours"},
}

MODEL_WEIGHTS_WITH_ZERO = {
    "nc_det": {"forecast_period": [0, 4, 8], "weights": [1, 0, 0], "units": "hours"},
    "uk_det": {"forecast_period": [0, 4, 8], "weights": [0, 0, 0], "units": "hours"},
    "uk_ens": {"forecast_period": [0, 4, 8], "weights": [0, 1, 1], "units": "hours"},
}


def set_up_masked_cubes():
    """
    Set up cubes with masked data for spatial weights tests

    Returns:
        iris.cube.CubeList:
            List containing a UKV cube with some rain, and a masked nowcast
            cube with more rain.
    """
    thresholds = np.array([0.5, 1, 2], dtype=np.float32)
    units = "mm h-1"
    name = "lwe_precipitation_rate"
    datatime = dt(2018, 9, 10, 7)
    cycletime = dt(2018, 9, 10, 5)
    cycletime_string = "20180910T0500Z"

    # 5x5 matrix results in grid spacing of 200 km
    base_data = np.ones((5, 5), dtype=np.float32)

    # Calculate grid spacing
    grid_spacing = np.around(1000000.0 / 5)

    # set up a UKV cube with some rain
    rain_data = np.array([0.9 * base_data, 0.5 * base_data, 0 * base_data])
    ukv_cube = set_up_probability_cube(
        rain_data,
        thresholds,
        variable_name=name,
        threshold_units=units,
        time=datatime,
        frt=cycletime,
        spatial_grid="equalarea",
        standard_grid_metadata="uk_det",
        grid_spacing=grid_spacing,
    )

    # set up a masked nowcast cube with more rain
    more_rain_data = np.array([base_data, 0.6 * base_data, 0.2 * base_data])
    radar_mask = np.broadcast_to(np.array([False, False, False, True, True]), (3, 5, 5))
    more_rain_data = np.ma.MaskedArray(more_rain_data, mask=radar_mask)
    nowcast_cube = set_up_probability_cube(
        more_rain_data,
        thresholds,
        variable_name=name,
        threshold_units=units,
        time=datatime,
        frt=cycletime,
        spatial_grid="equalarea",
        attributes={"mosg__model_configuration": "nc_det"},
        grid_spacing=grid_spacing,
    )

    return iris.cube.CubeList([ukv_cube, nowcast_cube]), cycletime_string


class Test__init__(IrisTest):
    """Test the __init__ method"""

    def test_cycle(self):
        """Test initialisation for cycle blending"""
        plugin = WeightAndBlend("forecast_reference_time", "linear", y0val=1, ynval=1)
        self.assertEqual(plugin.blend_coord, "forecast_reference_time")
        self.assertEqual(plugin.wts_calc_method, "linear")
        self.assertIsNone(plugin.weighting_coord)
        self.assertAlmostEqual(plugin.y0val, 1.0)
        self.assertAlmostEqual(plugin.ynval, 1.0)

    def test_model(self):
        """Test initialisation for model blending"""
        plugin = WeightAndBlend(
            "model_id",
            "dict",
            weighting_coord="forecast_period",
            wts_dict=MODEL_WEIGHTS,
        )
        self.assertEqual(plugin.blend_coord, "model_id")
        self.assertEqual(plugin.wts_calc_method, "dict")
        self.assertEqual(plugin.weighting_coord, "forecast_period")
        self.assertDictEqual(plugin.wts_dict, MODEL_WEIGHTS)

    def test_unrecognised_weighting_type_error(self):
        """Test error if wts_calc_method is unrecognised"""
        msg = "Weights calculation method 'kludge' unrecognised"
        with self.assertRaisesRegex(ValueError, msg):
            WeightAndBlend("forecast_period", "kludge")


class Test__calculate_blending_weights(IrisTest):
    """Test the _calculate_blending_weights method"""

    def test_default_linear(self):
        """Test linear weighting over realizations"""
        cube = set_up_variable_cube(278.0 * np.ones((4, 3, 3), dtype=np.float32))
        plugin = WeightAndBlend("realization", "linear", y0val=1, ynval=1)
        weights = plugin._calculate_blending_weights(cube)
        self.assertIsInstance(weights, iris.cube.Cube)
        weights_dims = [coord.name() for coord in weights.coords(dim_coords=True)]
        self.assertSequenceEqual(weights_dims, ["realization"])
        self.assertArrayAlmostEqual(weights.data, 0.25 * np.ones((4,)))

    def test_default_nonlinear(self):
        """Test non-linear weighting over forecast reference time, where the
        earlier cycle has a higher weighting"""
        data = np.ones((3, 3, 3), dtype=np.float32)
        thresholds = np.array([276, 277, 278], dtype=np.float32)
        ukv_cube_earlier = set_up_probability_cube(
            data, thresholds, time=dt(2018, 9, 10, 7), frt=dt(2018, 9, 10, 3)
        )
        ukv_cube_later = set_up_probability_cube(
            data, thresholds, time=dt(2018, 9, 10, 7), frt=dt(2018, 9, 10, 4)
        )
        cube = iris.cube.CubeList([ukv_cube_later, ukv_cube_earlier]).merge_cube()

        plugin = WeightAndBlend("forecast_reference_time", "nonlinear", cval=0.85)
        weights = plugin._calculate_blending_weights(cube)
        self.assertArrayAlmostEqual(weights.data, np.array([0.5405405, 0.45945945]))

    def test_default_nonlinear_inverse(self):
        """Test non-linear weighting over forecast reference time in reverse
        order, so that the more recent cycle has a higher weighting"""
        data = np.ones((3, 3, 3), dtype=np.float32)
        thresholds = np.array([276, 277, 278], dtype=np.float32)
        ukv_cube_earlier = set_up_probability_cube(
            data, thresholds, time=dt(2018, 9, 10, 7), frt=dt(2018, 9, 10, 3)
        )
        ukv_cube_later = set_up_probability_cube(
            data, thresholds, time=dt(2018, 9, 10, 7), frt=dt(2018, 9, 10, 4)
        )
        cube = iris.cube.CubeList([ukv_cube_later, ukv_cube_earlier]).merge_cube()

        plugin = WeightAndBlend(
            "forecast_reference_time", "nonlinear", cval=0.85, inverse_ordering=True
        )
        weights = plugin._calculate_blending_weights(cube)
        self.assertArrayAlmostEqual(weights.data, np.array([0.45945945, 0.5405405]))

    def test_dict(self):
        """Test dictionary option for model blending with non-equal weights"""
        data = np.ones((3, 3, 3), dtype=np.float32)
        thresholds = np.array([276, 277, 278], dtype=np.float32)
        ukv_cube = set_up_probability_cube(
            data,
            thresholds,
            time=dt(2018, 9, 10, 7),
            frt=dt(2018, 9, 10, 1),
            standard_grid_metadata="uk_det",
        )
        enukx_cube = set_up_probability_cube(
            data,
            thresholds,
            time=dt(2018, 9, 10, 7),
            frt=dt(2018, 9, 10, 1),
            standard_grid_metadata="uk_ens",
        )
        merger = MergeCubesForWeightedBlending(
            "model_id",
            weighting_coord="forecast_period",
            model_id_attr="mosg__model_configuration",
        )
        cube = merger.process([ukv_cube, enukx_cube])

        plugin = WeightAndBlend(
            "model_id",
            "dict",
            weighting_coord="forecast_period",
            wts_dict=MODEL_WEIGHTS,
        )

        # at 6 hours lead time we should have 1/3 UKV and 2/3 MOGREPS-UK,
        # according to the dictionary weights specified above
        weights = plugin._calculate_blending_weights(cube)
        self.assertArrayEqual(
            weights.coord("model_configuration").points, ["uk_det", "uk_ens"]
        )
        self.assertArrayAlmostEqual(weights.data, np.array([0.3333333, 0.6666667]))


class Test__update_spatial_weights(IrisTest):
    """Test the _update_spatial_weights method"""

    @ManageWarnings(ignored_messages=["Deleting unmatched attribute"])
    def setUp(self):
        """Set up cube and plugin"""
        cubelist, self.cycletime = set_up_masked_cubes()
        merger = MergeCubesForWeightedBlending(
            "model_id",
            weighting_coord="forecast_period",
            model_id_attr="mosg__model_configuration",
        )
        self.cube = merger.process(cubelist)
        self.plugin = WeightAndBlend(
            "model_id",
            "dict",
            weighting_coord="forecast_period",
            wts_dict=MODEL_WEIGHTS,
        )
        self.initial_weights = self.plugin._calculate_blending_weights(self.cube)

    @ManageWarnings(ignored_messages=["Collapsing a non-contiguous coordinate"])
    def test_basic(self):
        """Test function returns a cube of the expected shape"""
        expected_dims = [
            "model_id",
            "projection_y_coordinate",
            "projection_x_coordinate",
        ]
        expected_shape = (2, 5, 5)
        result = self.plugin._update_spatial_weights(
            self.cube, self.initial_weights, 20000
        )
        result_dims = [coord.name() for coord in result.coords(dim_coords=True)]
        self.assertSequenceEqual(result_dims, expected_dims)
        self.assertSequenceEqual(result.shape, expected_shape)

    @ManageWarnings(ignored_messages=["Collapsing a non-contiguous coordinate"])
    def test_values(self):
        """Test weights are fuzzified as expected"""
        expected_data = np.array(
            [
                np.broadcast_to([0.5, 0.5, 0.75, 1.0, 1.0], (5, 5)),
                np.broadcast_to([0.5, 0.5, 0.25, 0.0, 0.0], (5, 5)),
            ],
            dtype=np.float32,
        )
        result = self.plugin._update_spatial_weights(
            self.cube, self.initial_weights, 400000
        )
        self.assertArrayEqual(
            result.coord("model_configuration").points, ["uk_det", "nc_det"]
        )
        self.assertArrayAlmostEqual(result.data, expected_data)


class Test_process(IrisTest):
    """Test the process method"""

    def setUp(self):
        """Set up test cubes (each with a single point and 3 thresholds)"""
        thresholds = np.array([0.5, 1, 2], dtype=np.float32)
        units = "mm h-1"
        name = "lwe_precipitation_rate"
        datatime = dt(2018, 9, 10, 7)

        # a UKV cube with some rain and a 4 hr forecast period
        rain_data = np.array([[[0.9]], [[0.5]], [[0]]], dtype=np.float32)
        self.ukv_cube = set_up_probability_cube(
            rain_data,
            thresholds,
            variable_name=name,
            threshold_units=units,
            time=datatime,
            frt=dt(2018, 9, 10, 3),
            standard_grid_metadata="uk_det",
        )

        # a UKV cube from a more recent cycle with more rain
        more_rain_data = np.array([[[1]], [[0.6]], [[0.2]]], dtype=np.float32)
        self.ukv_cube_latest = set_up_probability_cube(
            more_rain_data,
            thresholds,
            variable_name=name,
            threshold_units=units,
            time=datatime,
            frt=dt(2018, 9, 10, 4),
            standard_grid_metadata="uk_det",
        )

        # a nowcast cube with more rain and a 2 hr forecast period
        self.nowcast_cube = set_up_probability_cube(
            more_rain_data,
            thresholds,
            variable_name=name,
            threshold_units=units,
            time=datatime,
            frt=dt(2018, 9, 10, 5),
            attributes={"mosg__model_configuration": "nc_det"},
        )

        # a MOGREPS-UK cube with less rain and a 4 hr forecast period
        less_rain_data = np.array([[[0.7]], [[0.3]], [[0]]], dtype=np.float32)
        self.enukx_cube = set_up_probability_cube(
            less_rain_data,
            thresholds,
            variable_name=name,
            threshold_units=units,
            time=datatime,
            frt=dt(2018, 9, 10, 3),
            standard_grid_metadata="uk_ens",
        )

        # cycletime from the most recent forecast (simulates current cycle)
        self.cycletime = "20180910T0500Z"
        self.expected_frt = self.nowcast_cube.coord("forecast_reference_time").copy()
        self.expected_fp = self.nowcast_cube.coord("forecast_period").copy()

        self.plugin_cycle = WeightAndBlend(
            "forecast_reference_time", "linear", y0val=1, ynval=1
        )
        self.plugin_model = WeightAndBlend(
            "model_id",
            "dict",
            weighting_coord="forecast_period",
            wts_dict=MODEL_WEIGHTS,
        )

    @ManageWarnings(ignored_messages=["Collapsing a non-contiguous coordinate"])
    def test_basic(self):
        """Test output is a cube"""
        result = self.plugin_cycle.process(
            [self.ukv_cube, self.ukv_cube_latest], cycletime=self.cycletime,
        )
        self.assertIsInstance(result, iris.cube.Cube)

    @ManageWarnings(record=True)
    def test_masked_blending_warning(self, warning_list=None):
        """Test a warning is raised if blending masked data with non-spatial
        weights."""
        ukv_cube = self.ukv_cube.copy(
            data=np.ma.masked_where(self.ukv_cube.data < 0.5, self.ukv_cube.data)
        )
        self.plugin_cycle.process(
            [ukv_cube, self.ukv_cube_latest], cycletime=self.cycletime,
        )
        message = "Blending masked data without spatial weights"
        self.assertTrue(any(message in str(item) for item in warning_list))

    @ManageWarnings(ignored_messages=["Collapsing a non-contiguous coordinate"])
    def test_cycle_blend_linear(self):
        """Test plugin produces correct cycle blended output with equal
        linear weightings"""
        expected_data = np.array([[[0.95]], [[0.55]], [[0.1]]], dtype=np.float32)
        result = self.plugin_cycle.process(
            [self.ukv_cube, self.ukv_cube_latest], cycletime=self.cycletime,
        )
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertArrayEqual(
            result.coord("time").points, self.ukv_cube_latest.coord("time").points
        )
        self.assertArrayEqual(
            result.coord("forecast_reference_time").points, self.expected_frt.points
        )
        self.assertArrayEqual(
            result.coord("forecast_period").points, self.expected_fp.points
        )
        for coord in ["forecast_reference_time", "forecast_period"]:
            self.assertIn("deprecation_message", result.coord(coord).attributes)

    @ManageWarnings(ignored_messages=["Collapsing a non-contiguous coordinate"])
    def test_model_blend(self):
        """Test plugin produces correct output for UKV-ENUKX model blend
        with 50-50 weightings defined by dictionary"""
        expected_data = np.array([[[0.8]], [[0.4]], [[0]]], dtype=np.float32)
        result = self.plugin_model.process(
            [self.ukv_cube, self.enukx_cube],
            model_id_attr="mosg__model_configuration",
            cycletime=self.cycletime,
        )
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertEqual(
            result.attributes["mosg__model_configuration"], "uk_det uk_ens"
        )
        result_coords = [coord.name() for coord in result.coords()]
        self.assertNotIn("model_id", result_coords)
        self.assertNotIn("model_configuration", result_coords)
        for coord in ["forecast_reference_time", "forecast_period"]:
            self.assertIn("deprecation_message", result.coord(coord).attributes)
            self.assertIn(
                "will be removed", result.coord(coord).attributes["deprecation_message"]
            )

    @ManageWarnings(
        ignored_messages=[
            "Collapsing a non-contiguous coordinate",
            "Deleting unmatched attribute",
        ]
    )
    def test_attributes_dict(self):
        """Test output attributes can be updated through argument"""
        attribute_changes = {
            "mosg__model_configuration": "remove",
            "source": "IMPROVER",
            "title": "IMPROVER Post-Processed Multi-Model Blend",
        }
        expected_attributes = {
            "source": "IMPROVER",
            "title": "IMPROVER Post-Processed Multi-Model Blend",
            "institution": MANDATORY_ATTRIBUTE_DEFAULTS["institution"],
        }
        result = self.plugin_model.process(
            [self.ukv_cube, self.nowcast_cube],
            model_id_attr="mosg__model_configuration",
            attributes_dict=attribute_changes,
            cycletime=self.cycletime,
        )
        self.assertDictEqual(result.attributes, expected_attributes)

    @ManageWarnings(
        ignored_messages=[
            "Collapsing a non-contiguous coordinate",
            "Deleting unmatched attribute",
        ]
    )
    def test_blend_three_models(self):
        """Test plugin produces correct output for 3-model blend when all
        models have (equal) non-zero weights. Each model in WEIGHTS_DICT has
        a weight of 0.5 at 4 hours lead time, and the total weights are
        re-normalised during the process, so the final blend contains 1/3
        contribution from each of the three models."""
        expected_data = np.array(
            [[[0.8666667]], [[0.4666667]], [[0.0666667]]], dtype=np.float32
        )
        result = self.plugin_model.process(
            [self.ukv_cube, self.enukx_cube, self.nowcast_cube],
            model_id_attr="mosg__model_configuration",
            cycletime=self.cycletime,
        )
        self.assertArrayAlmostEqual(result.data, expected_data)
        # make sure output cube has the forecast reference time and period
        # from the most recent contributing model
        for coord in ["time", "forecast_period", "forecast_reference_time"]:
            self.assertArrayEqual(
                result.coord(coord).points, self.nowcast_cube.coord(coord).points
            )

    def test_blend_with_zero_weight(self):
        """Test plugin produces correct values and attributes when some models read
        into the plugin have zero weighting"""
        plugin = WeightAndBlend(
            "model_id",
            "dict",
            weighting_coord="forecast_period",
            wts_dict=MODEL_WEIGHTS_WITH_ZERO,
        )
        expected_data = np.array([[[0.85]], [[0.45]], [[0.1]]], dtype=np.float32)
        result = plugin.process(
            [self.ukv_cube, self.enukx_cube, self.nowcast_cube],
            model_id_attr="mosg__model_configuration",
            cycletime=self.cycletime,
        )
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertEqual(
            result.attributes["mosg__model_configuration"], "nc_det uk_ens"
        )

    def test_blend_with_zero_weight_one_model_valid(self):
        """Test plugin can cope with only one remaining model in the list to blend"""
        plugin = WeightAndBlend(
            "model_id",
            "dict",
            weighting_coord="forecast_period",
            wts_dict=MODEL_WEIGHTS_WITH_ZERO,
        )
        expected_data = self.nowcast_cube.data.copy()
        result = plugin.process(
            [self.ukv_cube, self.nowcast_cube],
            model_id_attr="mosg__model_configuration",
            cycletime=self.cycletime,
        )
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertEqual(result.attributes["mosg__model_configuration"], "nc_det")

    def test_blend_with_zero_weight_one_model_input(self):
        """Test plugin returns data unchanged from a single model, even if that
        model had zero weight"""
        plugin = WeightAndBlend(
            "model_id",
            "dict",
            weighting_coord="forecast_period",
            wts_dict=MODEL_WEIGHTS_WITH_ZERO,
        )
        expected_data = self.ukv_cube.data.copy()
        result = plugin.process(
            self.ukv_cube,
            model_id_attr="mosg__model_configuration",
            cycletime=self.cycletime,
        )
        self.assertArrayAlmostEqual(result.data, expected_data)
        self.assertEqual(result.attributes["mosg__model_configuration"], "uk_det")

    def test_forecast_coord_deprecation(self):
        """Test model blending works if some (but not all) inputs have previously
        been cycle blended"""
        for cube in [self.ukv_cube, self.enukx_cube]:
            for coord in ["forecast_period", "forecast_reference_time"]:
                cube.coord(coord).attributes.update({"deprecation_message": "blah"})
        result = self.plugin_model.process(
            [self.ukv_cube, self.enukx_cube, self.nowcast_cube],
            model_id_attr="mosg__model_configuration",
            cycletime=self.cycletime,
        )
        for coord in ["forecast_reference_time", "forecast_period"]:
            self.assertIn("deprecation_message", result.coord(coord).attributes)
            self.assertIn(
                "will be removed", result.coord(coord).attributes["deprecation_message"]
            )

    def test_one_cube(self):
        """Test the plugin returns a single input cube with updated attributes
        and time coordinates"""
        expected_coords = {coord.name() for coord in self.enukx_cube.coords()}
        expected_coords.update({"blend_time"})
        result = self.plugin_model.process(
            [self.enukx_cube],
            model_id_attr="mosg__model_configuration",
            cycletime=self.cycletime,
            attributes_dict={"source": "IMPROVER"},
        )
        self.assertArrayAlmostEqual(result.data, self.enukx_cube.data)
        self.assertSetEqual(
            {coord.name() for coord in result.coords()}, expected_coords
        )
        self.assertEqual(result.attributes["source"], "IMPROVER")

    def test_one_cube_error_no_cycletime(self):
        """Test an error is raised if no cycletime is provided for model blending"""
        msg = "Current cycle time is required"
        with self.assertRaisesRegex(ValueError, msg):
            self.plugin_model.process(
                [self.enukx_cube], model_id_attr="mosg__model_configuration"
            )

    def test_one_cube_with_cycletime_model_blending(self):
        """Test the plugin returns a single input cube with an updated forecast
        reference time and period if given the "cycletime" option."""
        expected_frt = self.enukx_cube.coord("forecast_reference_time").points[0] + 3600
        expected_fp = self.enukx_cube.coord("forecast_period").points[0] - 3600
        result = self.plugin_model.process(
            [self.enukx_cube],
            model_id_attr="mosg__model_configuration",
            cycletime="20180910T0400Z",
        )
        self.assertEqual(
            result.coord("forecast_reference_time").points[0], expected_frt
        )
        self.assertEqual(result.coord("forecast_period").points[0], expected_fp)

    def test_one_cube_with_cycletime_cycle_blending(self):
        """Test the plugin returns a single input cube with an updated forecast
        reference time and period if given the "cycletime" option."""
        expected_frt = self.enukx_cube.coord("forecast_reference_time").points[0] + 3600
        expected_fp = self.enukx_cube.coord("forecast_period").points[0] - 3600
        result = self.plugin_cycle.process(
            [self.enukx_cube], cycletime="20180910T0400Z"
        )
        self.assertEqual(
            result.coord("forecast_reference_time").points[0], expected_frt
        )
        self.assertEqual(result.coord("forecast_period").points[0], expected_fp)

    def test_error_blend_coord_absent(self):
        """Test error is raised if blend coord is not present on input cubes"""
        plugin = WeightAndBlend("kittens", "linear", y0val=1, ynval=1)
        msg = "kittens coordinate is not present on all input cubes"
        with self.assertRaisesRegex(ValueError, msg):
            plugin.process([self.ukv_cube, self.ukv_cube_latest])


class Test_process_spatial_weights(IrisTest):
    """Test the process method with spatial weights options"""

    def setUp(self):
        """Set up a masked nowcast and unmasked UKV cube"""
        self.cubelist, self.cycletime = set_up_masked_cubes()
        self.plugin = WeightAndBlend(
            "model_id",
            "dict",
            weighting_coord="forecast_period",
            wts_dict=MODEL_WEIGHTS,
        )

    @ManageWarnings(
        ignored_messages=[
            "Collapsing a non-contiguous coordinate",
            "Deleting unmatched attribute",
        ]
    )
    def test_default(self):
        """Test plugin returns a cube with expected values where default fuzzy
        length is less than grid length (no smoothing)"""
        # data is 50:50 where radar is valid, 100% UKV where radar is masked
        expected_data = np.array(
            [
                np.broadcast_to([0.95, 0.95, 0.95, 0.9, 0.9], (5, 5)),
                np.broadcast_to([0.55, 0.55, 0.55, 0.5, 0.5], (5, 5)),
                np.broadcast_to([0.1, 0.1, 0.1, 0.0, 0.0], (5, 5)),
            ],
            dtype=np.float32,
        )
        result = self.plugin.process(
            self.cubelist,
            model_id_attr="mosg__model_configuration",
            spatial_weights=True,
            cycletime=self.cycletime,
        )
        self.assertIsInstance(result, iris.cube.Cube)
        self.assertArrayAlmostEqual(result.data, expected_data)

    @ManageWarnings(
        ignored_messages=[
            "Collapsing a non-contiguous coordinate",
            "Deleting unmatched attribute",
        ]
    )
    def test_fuzzy_length(self):
        """Test values where fuzzy length is equal to 2 grid lengths"""
        # proportion of radar data is reduced at edge of valid region; still
        # 100% UKV where radar data is masked
        expected_data = np.array(
            [
                np.broadcast_to([0.95, 0.95, 0.925, 0.9, 0.9], (5, 5)),
                np.broadcast_to([0.55, 0.55, 0.525, 0.5, 0.5], (5, 5)),
                np.broadcast_to([0.1, 0.1, 0.05, 0.0, 0.0], (5, 5)),
            ],
            dtype=np.float32,
        )
        result = self.plugin.process(
            self.cubelist,
            model_id_attr="mosg__model_configuration",
            spatial_weights=True,
            fuzzy_length=400000,
            cycletime=self.cycletime,
        )
        self.assertArrayAlmostEqual(result.data, expected_data)


if __name__ == "__main__":
    unittest.main()
