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
"""Unit tests for the ApplyRainForestsCalibrationLightGBM class."""
import numpy as np
import pytest
from iris import Constraint

from improver.calibration.rainforest_calibration import (
    ApplyRainForestsCalibrationLightGBM,
)
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

lightgbm = pytest.importorskip("lightgbm")


class MockBooster:
    def __init__(self, model_file, **kwargs):
        self.model_class = "lightgbm-Booster"
        self.model_file = model_file

    def reset_parameter(self, params):
        self.threads = params.get("num_threads")
        return self


def test__new__(model_config, monkeypatch):
    monkeypatch.setattr(lightgbm, "Booster", MockBooster)
    # Check that we get the expected subclass
    result = ApplyRainForestsCalibrationLightGBM(model_config)
    assert type(result).__name__ == "ApplyRainForestsCalibrationLightGBM"
    model_config
    # Test exception raised when file path is missing.
    model_config["0.0000"].pop("lightgbm_model", None)
    with pytest.raises(ValueError):
        ApplyRainForestsCalibrationLightGBM(model_config)


@pytest.mark.parametrize("ordered_inputs", (True, False))
@pytest.mark.parametrize("default_threads", (True, False))
def test__init__(
    model_config, ordered_inputs, default_threads, error_thresholds, monkeypatch
):
    monkeypatch.setattr(lightgbm, "Booster", MockBooster)

    if not ordered_inputs:
        tmp_value = model_config.pop("0.0000", None)
        model_config["0.0000"] = tmp_value

    if default_threads:
        expected_threads = 1
        result = ApplyRainForestsCalibrationLightGBM(model_config)
    else:
        expected_threads = 8
        result = ApplyRainForestsCalibrationLightGBM(
            model_config, threads=expected_threads
        )
    # Check thresholds and model types match
    assert np.all(result.error_thresholds == error_thresholds)
    for model in result.tree_models:
        assert model.model_class == "lightgbm-Booster"
        assert model.threads == expected_threads
    # Ensure threshold and files match
    for threshold, model in zip(result.error_thresholds, result.tree_models):
        # LightGBM library only introduced support for pathlib Paths in 3.3+
        assert isinstance(model.model_file, str)
        assert f"{threshold:06.4f}" in str(model.model_file)


def test__check_num_features(ensemble_features, plugin_and_dummy_models):
    """Test number of features expected by tree_models matches features passed in."""
    plugin_cls, dummy_models = plugin_and_dummy_models
    plugin = plugin_cls(model_config_dict={})
    plugin.tree_models, _ = dummy_models
    plugin._check_num_features(ensemble_features)
    with pytest.raises(ValueError):
        plugin._check_num_features(ensemble_features[:-1])


def test__align_feature_variables_ensemble(ensemble_features, ensemble_forecast):
    """Check cube alignment when using feature and forecast variables when realization
    coordinate present in some cube variables."""
    expected_features = ensemble_features.copy()
    # Drop realization coordinate from one of the ensemble features
    dervied_field_cube = ensemble_features.pop(-1).extract(Constraint(realization=0))
    dervied_field_cube.remove_coord("realization")
    ensemble_features.append(dervied_field_cube)

    (aligned_features, aligned_forecast,) = ApplyRainForestsCalibrationLightGBM(
        model_config_dict={}
    )._align_feature_variables(ensemble_features, ensemble_forecast)

    assert aligned_features == expected_features
    assert aligned_forecast == ensemble_forecast


def test__align_feature_variables_deterministic(
    deterministic_features, deterministic_forecast
):
    """Check cube alignment when using feature and forecast variables when no realization
    coordinate present in any of the cube variables."""
    expected_features = deterministic_features.copy()
    expected_forecast = deterministic_forecast.copy()
    # Drop realization from all features.
    deterministic_features = deterministic_features.extract(Constraint(realization=0))
    [feature.remove_coord("realization") for feature in deterministic_features]
    # Drop realization from forecast.
    deterministic_forecast = deterministic_forecast.extract(Constraint(realization=0))
    deterministic_forecast.remove_coord("realization")

    (aligned_features, aligned_forecast,) = ApplyRainForestsCalibrationLightGBM(
        model_config_dict={}
    )._align_feature_variables(deterministic_features, deterministic_forecast)

    assert aligned_features == expected_features
    assert aligned_forecast == expected_forecast


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
        ApplyRainForestsCalibrationLightGBM(
            model_config_dict={}
        )._align_feature_variables(ensemble_features, misaligned_forecast_cube)
    # Test case where realization dimension differ.
    misaligned_forecast_cube = set_up_variable_cube(
        np.maximum(0, np.random.normal(0.002, 0.001, (10, 10, 10))).astype(np.float32),
        name="lwe_thickness_of_precipitation_amount",
        units="m",
        realizations=np.arange(10),
    )
    with pytest.raises(ValueError):
        ApplyRainForestsCalibrationLightGBM(
            model_config_dict={}
        )._align_feature_variables(ensemble_features, misaligned_forecast_cube)


def test__prepare_error_probability_cube(
    ensemble_forecast, error_thresholds, error_threshold_cube
):
    """Test the preparation of error probability cube from input
    forecast cube."""
    plugin = ApplyRainForestsCalibrationLightGBM(model_config_dict={})
    plugin.error_thresholds = error_thresholds
    result = plugin._prepare_error_probability_cube(ensemble_forecast)

    assert result.long_name == error_threshold_cube.long_name
    assert result.units == error_threshold_cube.units
    assert result.coords() == error_threshold_cube.coords()
    assert result.attributes == error_threshold_cube.attributes


def test__prepare_features_array(ensemble_features):
    """Test array preparation given set of feature cubes."""
    expected_size = ensemble_features.extract_cube(
        "lwe_thickness_of_precipitation_amount"
    ).data.size
    result = ApplyRainForestsCalibrationLightGBM(
        model_config_dict={}
    )._prepare_features_array(ensemble_features)

    assert result.shape[0] == expected_size

    # Drop realization coordinate from one of the ensemble features, to produce
    # cubes of differing length.
    cube_lacking_realization = ensemble_features.pop(-1).extract(
        Constraint(realization=0)
    )
    cube_lacking_realization.remove_coord("realization")
    ensemble_features.append(cube_lacking_realization)
    with pytest.raises(ValueError):
        ApplyRainForestsCalibrationLightGBM(
            model_config_dict={}
        )._prepare_features_array(ensemble_features)


def test__evaluate_probabilities(
    ensemble_features, ensemble_forecast, error_threshold_cube, plugin_and_dummy_models
):
    """Test that _evaluate_probabilities populates error_threshold_cube.data with
    probability data."""
    plugin_cls, dummy_models = plugin_and_dummy_models
    plugin = plugin_cls(model_config_dict={})
    plugin.tree_models, plugin.error_thresholds = dummy_models
    input_dataset = plugin._prepare_features_array(ensemble_features)
    forecast_data = ensemble_forecast.data.ravel()
    data_before = error_threshold_cube.data.copy()
    plugin._evaluate_probabilities(
        forecast_data,
        input_dataset,
        ensemble_forecast.name(),
        ensemble_forecast.units,
        error_threshold_cube.data,
    )
    diff = error_threshold_cube.data - data_before
    # check each error threshold has been populated
    assert np.all(np.any(diff != 0, axis=0))
    # check data is between 0 and 1
    assert np.all(error_threshold_cube.data >= 0)
    assert np.all(error_threshold_cube.data <= 1)
    # check data is 1 where forecast + error < 0
    for i, t in enumerate(plugin.error_thresholds):
        invalid_error = ensemble_forecast.data + t < 0
        np.testing.assert_almost_equal(error_threshold_cube.data[i, invalid_error], 1)


def test_make_decreasing():
    """Test that make_increasing returns an array that is non-decreasing
    in the first dimension."""
    # Test on standard use case.
    input_array = np.array([[5, 5], [2, 3], [3, 4], [4, 2], [1, 1]]) / 5.0
    expected = np.array([[5, 5], [3, 3.5], [3, 3.5], [3, 2], [1, 1]]) / 5.0
    result = ApplyRainForestsCalibrationLightGBM(model_config_dict={})._make_decreasing(
        input_array
    )
    np.testing.assert_almost_equal(expected, result)
    # Test on case where all data is already monotonically decreasing.
    input_array = np.array([[5, 5], [4, 4], [3, 3], [2, 2], [1, 1]]) / 5.0
    expected = np.array([[5, 5], [4, 4], [3, 3], [2, 2], [1, 1]]) / 5.0
    result = ApplyRainForestsCalibrationLightGBM(model_config_dict={})._make_decreasing(
        input_array
    )
    np.testing.assert_almost_equal(expected, result)
    # Test on case where some data is monotonically increasing.
    input_array = np.array([[1, 5], [2, 3], [3, 4], [4, 2], [5, 1]]) / 5.0
    expected = np.array([[3, 5], [3, 3.5], [3, 3.5], [3, 2], [3, 1]]) / 5.0
    result = ApplyRainForestsCalibrationLightGBM(model_config_dict={})._make_decreasing(
        input_array
    )
    np.testing.assert_almost_equal(expected, result)
    # Test on case where data increasing along second dimension; this
    # should be preserved in final output, with leading dimension monotonically
    # decreasing.
    input_array = np.array([[4, 5], [2, 3], [3, 4], [1, 2], [1, 1]]) / 5.0
    expected = np.array([[4, 5], [2.5, 3.5], [2.5, 3.5], [1, 2], [1, 1]]) / 5.0
    result = ApplyRainForestsCalibrationLightGBM(model_config_dict={})._make_decreasing(
        input_array
    )
    np.testing.assert_almost_equal(expected, result)
    # Test on case where data has shape (n, 1).
    input_array = np.array([[5], [3], [4], [2], [1]]) / 5.0
    expected = np.array([[5], [3.5], [3.5], [2], [1]]) / 5.0
    result = ApplyRainForestsCalibrationLightGBM(model_config_dict={})._make_decreasing(
        input_array
    )
    np.testing.assert_almost_equal(expected, result)
    # Test on case where data has shape (1, n).
    input_array = np.array([[5, 3, 4, 2, 1]]) / 5.0
    expected = np.array([[5, 3, 4, 2, 1]]) / 5.0
    result = ApplyRainForestsCalibrationLightGBM(model_config_dict={})._make_decreasing(
        input_array
    )
    np.testing.assert_almost_equal(expected, result)
    # Test on case where data has shape (n).
    input_array = np.array([5, 3, 4, 2, 1]) / 5.0
    expected = np.array([5, 3.5, 3.5, 2, 1]) / 5.0
    result = ApplyRainForestsCalibrationLightGBM(model_config_dict={})._make_decreasing(
        input_array
    )
    np.testing.assert_almost_equal(expected, result)


def test__calculate_error_probabilities(
    ensemble_features, ensemble_forecast, plugin_and_dummy_models
):
    """Test calculation of error probability cube."""
    plugin_cls, dummy_models = plugin_and_dummy_models
    plugin = plugin_cls(model_config_dict={})
    plugin.tree_models, plugin.error_thresholds = dummy_models
    result = plugin._calculate_error_probabilities(ensemble_forecast, ensemble_features)

    # Check that data has sensible probability values.
    # Note: here we are NOT checking the returned value against an expected value
    # as this will be sensitive to changes in associated GBDT libraries, given that
    # the tree models are created dynamically within fixtures. Here we implicitly trust
    # the output from the tree models are correct based on the specified inputs, and so
    # only test to ensure that the dataset overall conforms to the bounds for probability
    # data.
    assert np.all(result.data >= 0.0)
    assert np.all(result.data <= 1.0)
    assert np.all(np.isfinite(result.data))
    # Check that data is monotonically decreasing
    assert np.all(np.diff(result.data, axis=0) <= 0.0)


def test__get_ensemble_distributions(
    error_threshold_cube, ensemble_forecast, plugin_and_dummy_models
):
    """Test _get_ensemble_distributions."""
    plugin_cls, _ = plugin_and_dummy_models
    plugin = plugin_cls(model_config_dict={})
    output_thresholds = [0.0, 0.0005, 0.001]
    result = plugin._get_ensemble_distributions(
        error_threshold_cube, ensemble_forecast, output_thresholds
    )
    # Check that probabilities are between 0 and 1
    assert np.all(result.data >= 0.0)
    assert np.all(result.data <= 1.0)
    # Check that data is monotonically decreasing
    assert np.all(np.diff(result.data, axis=0) <= 0.0)
    # Check that threshold coordinate of output has same values as output_thresholds
    forecast_variable = ensemble_forecast.name()
    assert result.coord(forecast_variable).var_name == "threshold"
    np.testing.assert_almost_equal(
        result.coord(forecast_variable).points, output_thresholds
    )
    # Check that other dimensions are same as ensemble_forecast
    assert result.coords(dim_coords=True)[1:] == ensemble_forecast.coords(
        dim_coords=True
    )
    assert result.coords(dim_coords=False) == ensemble_forecast.coords(dim_coords=False)
    assert result.attributes == ensemble_forecast.attributes


def test_process_ensemble(
    ensemble_forecast, ensemble_features, plugin_and_dummy_models
):
    """Test process routine with ensemble data."""
    plugin_cls, dummy_models = plugin_and_dummy_models
    plugin = plugin_cls(model_config_dict={})
    plugin.tree_models, plugin.error_thresholds = dummy_models

    output_thresholds = [0.0, 0.0005, 0.001]
    result = plugin.process(ensemble_forecast, ensemble_features, output_thresholds,)

    forecast_variable = ensemble_forecast.name()
    assert result.long_name == f"probability_of_{forecast_variable}_above_threshold"

    # Check that first dimension is threshold and its values are same as output_thresholds
    threshold_coord = result.coord(forecast_variable)
    np.testing.assert_almost_equal(threshold_coord.points, output_thresholds)
    threshold_dim = result.coord_dims(forecast_variable)
    assert threshold_dim == (0,)

    # Check that dimensions are same as the input forecast, apart from first dimension
    assert (
        result.coords(dim_coords=True)[1:]
        == ensemble_forecast.coords(dim_coords=True)[1:]
    )
    assert result.coords(dim_coords=False) == ensemble_forecast.coords(dim_coords=False)
    assert result.attributes == ensemble_forecast.attributes


def test_process_ensemble_different_threshold_unit(
    ensemble_forecast, ensemble_features, plugin_and_dummy_models
):
    """Test process routine with ensemble data where unit of threshold is different from
    unit of forecast cube."""
    plugin_cls, dummy_models = plugin_and_dummy_models
    plugin = plugin_cls(model_config_dict={})
    plugin.tree_models, plugin.error_thresholds = dummy_models

    output_thresholds_m = [0.0, 0.0005, 0.001]
    result_m = plugin.process(ensemble_forecast, ensemble_features, output_thresholds_m)
    output_thresholds_mm = [0.0, 0.5, 1.0]
    result_mm = plugin.process(
        ensemble_forecast, ensemble_features, output_thresholds_mm, "mm"
    )

    assert result_m.coords() == result_mm.coords()
    np.testing.assert_array_almost_equal(result_m.data, result_mm.data)


def test_process_deterministic(
    deterministic_forecast, deterministic_features, plugin_and_dummy_models
):
    """Test process routine with deterministic data."""
    plugin_cls, dummy_models = plugin_and_dummy_models
    plugin = plugin_cls(model_config_dict={})
    plugin.tree_models, plugin.error_thresholds = dummy_models

    output_thresholds = [0.0, 0.0005, 0.001]
    result = plugin.process(
        deterministic_forecast, deterministic_features, output_thresholds,
    )

    forecast_variable = deterministic_forecast.name()
    assert result.long_name == f"probability_of_{forecast_variable}_above_threshold"

    # Check that first dimension is threshold and its values are same as output_thresholds
    threshold_coord = result.coord(forecast_variable)
    np.testing.assert_almost_equal(threshold_coord.points, output_thresholds)
    threshold_dim = result.coord_dims(forecast_variable)
    assert threshold_dim == (0,)

    # Check that dimensions are same as the input forecast, apart from first dimension
    assert (
        result.coords(dim_coords=True)[1:]
        == deterministic_forecast.coords(dim_coords=True)[1:]
    )
    assert result.coords(dim_coords=False) == deterministic_forecast.coords(
        dim_coords=False
    )
    assert result.attributes == deterministic_forecast.attributes
