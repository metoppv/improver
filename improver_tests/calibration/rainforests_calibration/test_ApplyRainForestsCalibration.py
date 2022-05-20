# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
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
"""Unit tests for the ApplyRainForestsCalibration class."""
import sys

import numpy as np
import pytest
from iris.cube import CubeList

from improver.calibration.rainforest_calibration import ApplyRainForestsCalibration
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import compare_attributes, compare_coords

try:
    import treelite_runtime
except ModuleNotFoundError:
    TREELITE_ENABLED = False
else:
    TREELITE_ENABLED = True

lightgbm = pytest.importorskip("lightgbm")

EMPTY_COMPARSION_DICT = [{}, {}]


class MockBooster:
    def __init__(self, model_file, **kwargs):
        self.model_class = "lightgbm-Booster"
        self.model_file = model_file

    def reset_parameter(self, params):
        self.threads = params.get("num_threads")
        return self


class MockPredictor:
    def __init__(self, libpath, nthread, **kwargs):
        self.model_class = "treelite-Predictor"
        self.threads = nthread
        self.model_file = libpath


@pytest.mark.parametrize("lightgbm_keys", (True, False))
@pytest.mark.parametrize("ordered_inputs", (True, False))
@pytest.mark.parametrize("treelite_model", (TREELITE_ENABLED, False))
@pytest.mark.parametrize("treelite_file", (True, False))
def test__init__(
    lightgbm_keys,
    ordered_inputs,
    treelite_model,
    treelite_file,
    monkeypatch,
    model_config,
    error_thresholds,
):
    """Test treelite models are loaded if model_config correctly defines them. If all thresholds
    contain treelite model AND the treelite module is available, treelite Predictor is returned,
    otherwise return lightgbm Boosters. Checks outputs are ordered when inputs can be unordered.
    If neither treelite nor lightgbm configs are complete, a ValueError is expected."""
    if treelite_model:
        monkeypatch.setattr(treelite_runtime, "Predictor", MockPredictor)
    else:
        monkeypatch.setitem(sys.modules, "treelite_runtime", None)
    monkeypatch.setattr(lightgbm, "Booster", MockBooster)

    if not treelite_file:
        # Model type should default to lightgbm if there are any treelite models
        # missing across any thresholds
        model_config["0.0000"].pop("treelite_model", None)
    if not ordered_inputs:
        tmp_value = model_config.pop("0.0000", None)
        model_config["0.0000"] = tmp_value
    if not lightgbm_keys:
        for t, d in model_config.items():
            d.pop("lightgbm_model")

    if treelite_model and treelite_file:
        expected_class = "treelite-Predictor"
    elif lightgbm_keys:
        expected_class = "lightgbm-Booster"
    else:
        with pytest.raises(ValueError, match="Path to lightgbm model missing"):
            ApplyRainForestsCalibration(model_config, threads=8)
        return

    result = ApplyRainForestsCalibration(model_config, threads=8)

    for model in result.tree_models:
        assert model.model_class == expected_class
        assert model.threads == 8
    assert result.treelite_enabled is treelite_model
    assert np.all(result.error_thresholds == error_thresholds)
    for threshold, model in zip(result.error_thresholds, result.tree_models):
        assert f"{threshold:06.4f}" in model.model_file


def test__check_num_features_lightgbm(ensemble_features, dummy_lightgbm_models):
    """Test number of features expected by tree_models matches features passed in."""

    plugin = ApplyRainForestsCalibration(model_config_dict={}, threads=1)
    plugin.tree_models, _ = dummy_lightgbm_models

    plugin._check_num_features(ensemble_features)

    with pytest.raises(ValueError):
        plugin._check_num_features(ensemble_features[:-1])


@pytest.mark.skipif(not TREELITE_ENABLED, reason="Required dependency missing.")
def test__check_num_features_treelite(ensemble_features, dummy_treelite_models):
    """Test number of features expected by tree_models matches features passed in."""

    plugin = ApplyRainForestsCalibration(model_config_dict={}, threads=1)
    plugin.tree_models, _ = dummy_treelite_models

    plugin._check_num_features(ensemble_features)

    with pytest.raises(ValueError):
        plugin._check_num_features(ensemble_features[:-1])


def test__align_feature_variables_ensemble(ensemble_features, ensemble_forecast):
    """Check cube alignment when using feature and forecast variables when realization
    coordinate present in some cube variables."""

    (aligned_features, aligned_forecast,) = ApplyRainForestsCalibration(
        model_config_dict={}, threads=1
    )._align_feature_variables(ensemble_features, ensemble_forecast)

    input_cubes = CubeList([*ensemble_features, ensemble_forecast])
    output_cubes = CubeList([*aligned_features, aligned_forecast])

    # Check that the realization dimension is the outer-most dimension
    assert np.all([cube.coord_dims("realization") == (0,) for cube in output_cubes])

    # Check that all cubes have consistent shape
    assert np.all(
        [cube.data.shape == output_cubes[0].data.shape for cube in output_cubes[1:]]
    )

    # Check the other properties of the cubes are unchanged.
    for input_cube, output_cube in zip(input_cubes, output_cubes):
        assert (
            compare_coords([output_cube, input_cube], ignored_coords="realization")
            == EMPTY_COMPARSION_DICT
        )
        assert compare_attributes([output_cube, input_cube]) == EMPTY_COMPARSION_DICT


def test__align_feature_variables_deterministic(
    deterministic_features, deterministic_forecast
):
    """Check cube alignment when using feature and forecast variables when no realization
    coordinate present in any of the cube variables."""

    (aligned_features, aligned_forecast,) = ApplyRainForestsCalibration(
        model_config_dict={}, threads=1
    )._align_feature_variables(deterministic_features, deterministic_forecast)

    input_cubes = CubeList([*deterministic_features, deterministic_forecast])
    output_cubes = CubeList([*aligned_features, aligned_forecast])

    # Check that all cubes have realization dimension of length 1
    assert np.all([cube.coord("realization").shape == (1,) for cube in output_cubes])

    # Check that the realization dimension is the outer-most dimension
    assert np.all([cube.coord_dims("realization") == (0,) for cube in output_cubes])

    # Check that all cubes have consistent shape
    assert np.all(
        [cube.data.shape == output_cubes[0].data.shape for cube in output_cubes[1:]]
    )

    # Check the other properties of the cubes are unchanged.
    for input_cube, output_cube in zip(input_cubes, output_cubes):
        assert (
            compare_coords([output_cube, input_cube], ignored_coords="realization")
            == EMPTY_COMPARSION_DICT
        )
        assert compare_attributes([output_cube, input_cube]) == EMPTY_COMPARSION_DICT


def test__align_feature_variables_misaligned_dim_coords(ensemble_features):
    """Check ValueError raised when feature/forecast cubes have differing dimension
    coordinates."""
    # Test case where non-realization dimension differ.
    misaligned_forecast_cube = set_up_variable_cube(
        np.maximum(0, np.random.normal(0.002, 0.001, (5, 10, 15))).astype(np.float32),
        name="lwe_thickness_of_precipitation_amount",
        units="m",
        realizations=np.arange(5),
    )

    with pytest.raises(ValueError):
        ApplyRainForestsCalibration(
            model_config_dict={}, threads=1
        )._align_feature_variables(ensemble_features, misaligned_forecast_cube)

    # Test case where realization dimension differ.
    misaligned_forecast_cube = set_up_variable_cube(
        np.maximum(0, np.random.normal(0.002, 0.001, (10, 10, 10))).astype(np.float32),
        name="lwe_thickness_of_precipitation_amount",
        units="m",
        realizations=np.arange(10),
    )

    with pytest.raises(ValueError):
        ApplyRainForestsCalibration(
            model_config_dict={}, threads=1
        )._align_feature_variables(ensemble_features, misaligned_forecast_cube)


def test__prepare_error_probability_cube(
    ensemble_forecast, error_thresholds, error_threshold_cube
):
    """Test the preparation of error probability cube from input
    forecast cube."""
    plugin = ApplyRainForestsCalibration(model_config_dict={}, threads=1)
    plugin.error_thresholds = error_thresholds
    result = plugin._prepare_error_probability_cube(ensemble_forecast)

    assert result.long_name == error_threshold_cube.long_name
    assert result.units == error_threshold_cube.units
    assert result.coords() == error_threshold_cube.coords()
    assert result.attributes == error_threshold_cube.attributes


def test__prepare_features_dataframe(ensemble_features):
    """Test dataframe preparation given set of feature cubes."""
    # Note: Clearsky solar radiation cube does not have realization
    # dimension, so without calling _align_feature_variables it will
    # will differ in length from all other cubes.

    # With clearsky solar cube present, function should fail due to
    # differing size of underlying arrays.
    with pytest.raises(RuntimeError):
        ApplyRainForestsCalibration(
            model_config_dict={}, threads=1
        )._prepare_features_dataframe(ensemble_features)

    # Drop clearsky solar cube so that all cubes now have the same size.
    ensemble_features = ensemble_features[:-1]
    feature_names = [cube.name() for cube in ensemble_features]
    expected_size = ensemble_features.extract_cube(
        "lwe_thickness_of_precipitation_amount"
    ).data.size

    result = ApplyRainForestsCalibration(
        model_config_dict={}, threads=1
    )._prepare_features_dataframe(ensemble_features)

    assert list(result.columns) == list(sorted(feature_names))
    assert len(result) == expected_size


def test_make_decreasing():
    """Test that make_increasing returns an array that is non-decreasing
    in the first dimension."""
    input_array = np.array([[5, 5], [4, 3], [3, 4], [2, 2], [1, 1]])
    expected = np.array([[5, 5], [4, 3.5], [3, 3.5], [2, 2], [1, 1]])
    result = ApplyRainForestsCalibration(
        model_config_dict={}, threads=1
    )._make_decreasing(input_array)
    assert np.allclose(expected, result)


def test__calculate_error_probabilities_lightgbm(
    ensemble_features, ensemble_forecast, dummy_lightgbm_models
):
    """Test calculation of error probability cube when using lightgbm Boosters."""
    plugin = ApplyRainForestsCalibration(model_config_dict={}, threads=1)
    plugin.tree_models, plugin.error_thresholds = dummy_lightgbm_models

    aligned_features, aligned_forecast = plugin._align_feature_variables(
        ensemble_features, ensemble_forecast
    )

    result = plugin._calculate_error_probabilities(aligned_forecast, aligned_features)

    # Check that data has sensible probability values
    assert np.all(result.data >= 0.0)
    assert np.all(result.data <= 1.0)
    assert np.all(np.isfinite(result.data))
    # Check that data is monotonically decreasing
    assert np.all((result.data[1:, :, :, :] - result.data[:-1, :, :, :]) <= 0.0)


@pytest.mark.skipif(not TREELITE_ENABLED, reason="Required dependency missing.")
def test__calculate_error_probabilities_treelite(
    ensemble_features, ensemble_forecast, dummy_treelite_models
):
    """Test calculation of error probability cube when using treelite Predictors."""
    plugin = ApplyRainForestsCalibration(model_config_dict={}, threads=1)
    plugin.tree_models, plugin.error_thresholds = dummy_treelite_models

    aligned_features, aligned_forecast = plugin._align_feature_variables(
        ensemble_features, ensemble_forecast
    )

    result = plugin._calculate_error_probabilities(aligned_forecast, aligned_features)

    # Check that data has sensible probability values
    assert np.all(result.data >= 0.0)
    assert np.all(result.data <= 1.0)
    assert np.all(np.isfinite(result.data))
    # Check that data is monotonically decreasing
    assert np.all((result.data[1:, :, :, :] - result.data[:-1, :, :, :]) <= 0.0)


def test__extract_error_percentiles(error_threshold_cube, error_percentile_cube):

    result = ApplyRainForestsCalibration(
        model_config_dict={}, threads=1
    )._extract_error_percentiles(error_threshold_cube, 4)

    assert result.long_name == error_percentile_cube.long_name
    assert result.units == error_percentile_cube.units
    assert result.coords() == error_percentile_cube.coords()
    assert result.attributes == error_percentile_cube.attributes


def test__apply_error_to_forecast(ensemble_forecast, error_percentile_cube):

    result = ApplyRainForestsCalibration(
        model_config_dict={}, threads=1
    )._apply_error_to_forecast(ensemble_forecast, error_percentile_cube)

    assert result.standard_name == ensemble_forecast.standard_name
    assert result.long_name == ensemble_forecast.long_name
    assert result.var_name == ensemble_forecast.var_name
    assert result.units == ensemble_forecast.units

    assert result.coords(dim_coords=False) == ensemble_forecast.coords(dim_coords=False)
    assert result.coords(dim_coords=True) == error_percentile_cube.coords(
        dim_coords=True
    )

    assert result.attributes == ensemble_forecast.attributes

    assert np.all(result.data >= 0.0)


def test__stack_subensembles(error_percentile_cube):

    result = ApplyRainForestsCalibration(
        model_config_dict={}, threads=1
    )._stack_subensembles(error_percentile_cube)

    assert (
        compare_coords(
            [result, error_percentile_cube],
            ignored_coords=["realization", "percentile"],
        )
        == EMPTY_COMPARSION_DICT
    )
    assert result.coords("percentile") == []
    assert np.all(
        result.coord("realization").points
        == np.arange(
            error_percentile_cube.coord("realization").points.size
            * error_percentile_cube.coord("percentile").points.size
        )
    )
    assert result.coords(dim_coords=False) == error_percentile_cube.coords(
        dim_coords=False
    )
    assert result.attributes == error_percentile_cube.attributes


def test__combine_subensembles(error_percentile_cube):

    result = ApplyRainForestsCalibration(
        model_config_dict={}, threads=1
    )._combine_subensembles(error_percentile_cube, output_realizations_count=None)

    assert np.all(
        result.coord("realization").points
        == np.arange(
            error_percentile_cube.coord("realization").points.size
            * error_percentile_cube.coord("percentile").points.size
        )
    )

    result = ApplyRainForestsCalibration(
        model_config_dict={}, threads=1
    )._combine_subensembles(error_percentile_cube, output_realizations_count=10)

    assert np.all(result.coord("realization").points == np.arange(10))


def test_process_with_lightgbm(
    ensemble_forecast, ensemble_features, dummy_lightgbm_models
):

    plugin = ApplyRainForestsCalibration(model_config_dict={}, threads=1)
    plugin.tree_models, plugin.error_thresholds = dummy_lightgbm_models

    for output_realization_count in (None, 10):
        result = plugin.process(
            ensemble_forecast,
            ensemble_features,
            error_percentiles_count=4,
            output_realizations_count=output_realization_count,
        )

    assert result.standard_name == ensemble_forecast.standard_name
    assert result.long_name == ensemble_forecast.long_name
    assert result.var_name == ensemble_forecast.var_name
    assert result.units == ensemble_forecast.units

    assert (
        compare_coords([result, ensemble_forecast], ignored_coords=["realization"])
        == EMPTY_COMPARSION_DICT
    )
    if output_realization_count is None:
        assert (
            result.coord("realization").points.size
            == 4 * ensemble_forecast.coord("realization").points.size
        )
    else:
        assert result.coord("realization").points.size == output_realization_count
    assert result.coords(dim_coords=False) == ensemble_forecast.coords(dim_coords=False)
    assert result.attributes == ensemble_forecast.attributes


@pytest.mark.skipif(not TREELITE_ENABLED, reason="Required dependency missing.")
def test_process_with_treelite(
    ensemble_forecast, ensemble_features, dummy_treelite_models
):

    plugin = ApplyRainForestsCalibration(model_config_dict={}, threads=1)
    plugin.tree_models, plugin.error_thresholds = dummy_treelite_models

    for output_realization_count in (None, 10):
        result = plugin.process(
            ensemble_forecast,
            ensemble_features,
            error_percentiles_count=4,
            output_realizations_count=output_realization_count,
        )

    assert result.standard_name == ensemble_forecast.standard_name
    assert result.long_name == ensemble_forecast.long_name
    assert result.var_name == ensemble_forecast.var_name
    assert result.units == ensemble_forecast.units

    assert (
        compare_coords([result, ensemble_forecast], ignored_coords=["realization"])
        == EMPTY_COMPARSION_DICT
    )
    if output_realization_count is None:
        assert (
            result.coord("realization").points.size
            == 4 * ensemble_forecast.coord("realization").points.size
        )
    else:
        assert result.coord("realization").points.size == output_realization_count
    assert result.coords(dim_coords=False) == ensemble_forecast.coords(dim_coords=False)
    assert result.attributes == ensemble_forecast.attributes
