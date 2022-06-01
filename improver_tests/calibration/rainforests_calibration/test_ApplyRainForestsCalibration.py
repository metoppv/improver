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
from iris import Constraint

from improver.calibration.rainforest_calibration import ApplyRainForestsCalibration
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube

try:
    import treelite_runtime
except ModuleNotFoundError:
    TREELITE_ENABLED = False
else:
    TREELITE_ENABLED = True

lightgbm = pytest.importorskip("lightgbm")
treelite_available = pytest.mark.skipif(
    not TREELITE_ENABLED, reason="Required dependency missing."
)


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
    plugin = ApplyRainForestsCalibration(model_config_dict={})
    plugin.tree_models, _ = dummy_lightgbm_models
    plugin._check_num_features(ensemble_features)
    with pytest.raises(ValueError):
        plugin._check_num_features(ensemble_features[:-1])


@treelite_available
def test__check_num_features_treelite(ensemble_features, dummy_treelite_models):
    """Test number of features expected by tree_models matches features passed in."""
    plugin = ApplyRainForestsCalibration(model_config_dict={})
    plugin.tree_models, _ = dummy_treelite_models
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

    (aligned_features, aligned_forecast,) = ApplyRainForestsCalibration(
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

    (aligned_features, aligned_forecast,) = ApplyRainForestsCalibration(
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
        ApplyRainForestsCalibration(model_config_dict={})._align_feature_variables(
            ensemble_features, misaligned_forecast_cube
        )
    # Test case where realization dimension differ.
    misaligned_forecast_cube = set_up_variable_cube(
        np.maximum(0, np.random.normal(0.002, 0.001, (10, 10, 10))).astype(np.float32),
        name="lwe_thickness_of_precipitation_amount",
        units="m",
        realizations=np.arange(10),
    )
    with pytest.raises(ValueError):
        ApplyRainForestsCalibration(model_config_dict={})._align_feature_variables(
            ensemble_features, misaligned_forecast_cube
        )


def test__prepare_error_probability_cube(
    ensemble_forecast, error_thresholds, error_threshold_cube
):
    """Test the preparation of error probability cube from input
    forecast cube."""
    plugin = ApplyRainForestsCalibration(model_config_dict={})
    plugin.error_thresholds = error_thresholds
    result = plugin._prepare_error_probability_cube(ensemble_forecast)

    assert result.long_name == error_threshold_cube.long_name
    assert result.units == error_threshold_cube.units
    assert result.coords() == error_threshold_cube.coords()
    assert result.attributes == error_threshold_cube.attributes


def test__prepare_features_dataframe(ensemble_features):
    """Test dataframe preparation given set of feature cubes."""
    feature_names = [cube.name() for cube in ensemble_features]
    expected_size = ensemble_features.extract_cube(
        "lwe_thickness_of_precipitation_amount"
    ).data.size
    result = ApplyRainForestsCalibration(
        model_config_dict={}
    )._prepare_features_dataframe(ensemble_features)

    assert list(result.columns) == list(sorted(feature_names))
    assert len(result) == expected_size

    # Drop realization coordinate from one of the ensemble features, to produce
    # cubes of differing length.
    cube_lacking_realization = ensemble_features.pop(-1).extract(
        Constraint(realization=0)
    )
    cube_lacking_realization.remove_coord("realization")
    ensemble_features.append(cube_lacking_realization)
    with pytest.raises(ValueError):
        ApplyRainForestsCalibration(model_config_dict={})._prepare_features_dataframe(
            ensemble_features
        )


def test_make_decreasing():
    """Test that make_increasing returns an array that is non-decreasing
    in the first dimension."""
    # Test on standard use case.
    input_array = np.array([[5, 5], [2, 3], [3, 4], [4, 2], [1, 1]]) / 5.0
    expected = np.array([[5, 5], [3, 3.5], [3, 3.5], [3, 2], [1, 1]]) / 5.0
    result = ApplyRainForestsCalibration(model_config_dict={})._make_decreasing(
        input_array
    )
    np.testing.assert_almost_equal(expected, result)
    # Test on case where all data is already monotonically decreasing.
    input_array = np.array([[5, 5], [4, 4], [3, 3], [2, 2], [1, 1]]) / 5.0
    expected = np.array([[5, 5], [4, 4], [3, 3], [2, 2], [1, 1]]) / 5.0
    result = ApplyRainForestsCalibration(model_config_dict={})._make_decreasing(
        input_array
    )
    np.testing.assert_almost_equal(expected, result)
    # Test on case where some data is monotonically increasing.
    input_array = np.array([[1, 5], [2, 3], [3, 4], [4, 2], [5, 1]]) / 5.0
    expected = np.array([[3, 5], [3, 3.5], [3, 3.5], [3, 2], [3, 1]]) / 5.0
    result = ApplyRainForestsCalibration(model_config_dict={})._make_decreasing(
        input_array
    )
    np.testing.assert_almost_equal(expected, result)
    # Test on case where data increasing along second dimension; this
    # should be preserved in final output, with leading dimension monotonically
    # decreasing.
    input_array = np.array([[4, 5], [2, 3], [3, 4], [1, 2], [1, 1]]) / 5.0
    expected = np.array([[4, 5], [2.5, 3.5], [2.5, 3.5], [1, 2], [1, 1]]) / 5.0
    result = ApplyRainForestsCalibration(model_config_dict={})._make_decreasing(
        input_array
    )
    np.testing.assert_almost_equal(expected, result)
    # Test on case where data has shape (n, 1).
    input_array = np.array([[5], [3], [4], [2], [1]]) / 5.0
    expected = np.array([[5], [3.5], [3.5], [2], [1]]) / 5.0
    result = ApplyRainForestsCalibration(model_config_dict={})._make_decreasing(
        input_array
    )
    np.testing.assert_almost_equal(expected, result)
    # Test on case where data has shape (1, n).
    input_array = np.array([[5, 3, 4, 2, 1]]) / 5.0
    expected = np.array([[5, 3, 4, 2, 1]]) / 5.0
    result = ApplyRainForestsCalibration(model_config_dict={})._make_decreasing(
        input_array
    )
    np.testing.assert_almost_equal(expected, result)
    # Test on case where data has shape (n).
    input_array = np.array([5, 3, 4, 2, 1]) / 5.0
    expected = np.array([5, 3.5, 3.5, 2, 1]) / 5.0
    result = ApplyRainForestsCalibration(model_config_dict={})._make_decreasing(
        input_array
    )
    np.testing.assert_almost_equal(expected, result)


def test__calculate_error_probabilities_lightgbm(
    ensemble_features, ensemble_forecast, dummy_lightgbm_models
):
    """Test calculation of error probability cube when using lightgbm Boosters."""
    plugin = ApplyRainForestsCalibration(model_config_dict={})
    plugin.tree_models, plugin.error_thresholds = dummy_lightgbm_models
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


@treelite_available
def test__calculate_error_probabilities_treelite(
    ensemble_features, ensemble_forecast, dummy_treelite_models
):
    """Test calculation of error probability cube when using treelite Predictors."""
    plugin = ApplyRainForestsCalibration(model_config_dict={})
    plugin.tree_models, plugin.error_thresholds = dummy_treelite_models
    result = plugin._calculate_error_probabilities(ensemble_forecast, ensemble_features)

    # Check that data has sensible probability values
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


def test__extract_error_percentiles(error_threshold_cube, error_percentile_cube):
    """Test the extraction of error percentiles from error-probability cube."""
    result = ApplyRainForestsCalibration(
        model_config_dict={}
    )._extract_error_percentiles(error_threshold_cube, 4)

    assert result.long_name == error_percentile_cube.long_name
    assert result.units == error_percentile_cube.units
    assert result.coords() == error_percentile_cube.coords()
    assert result.attributes == error_percentile_cube.attributes


def test__apply_error_to_forecast(ensemble_forecast, error_percentile_cube):
    """Test the application of forecast error (percentile) values to the forecast cube."""
    result = ApplyRainForestsCalibration(model_config_dict={})._apply_error_to_forecast(
        ensemble_forecast, error_percentile_cube
    )

    assert result.standard_name == ensemble_forecast.standard_name
    assert result.long_name == ensemble_forecast.long_name
    assert result.var_name == ensemble_forecast.var_name
    assert result.units == ensemble_forecast.units
    # Aux coords should be consistent with forecast
    assert result.coords(dim_coords=False) == ensemble_forecast.coords(dim_coords=False)
    # Dim coords should be consistent with forecast error.
    assert result.coords(dim_coords=True) == error_percentile_cube.coords(
        dim_coords=True
    )
    assert result.attributes == ensemble_forecast.attributes
    assert np.all(result.data >= 0.0)


def test__stack_subensembles(error_percentile_cube):
    """Test the stacking of realization-percentile dimensions into a single
    realization dimension."""
    column = np.arange(20, dtype=np.float32)

    error_percentile_cube.data = np.broadcast_to(
        column.reshape(5, 4)[:, :, np.newaxis, np.newaxis], (5, 4, 10, 10)
    )
    expected_data = np.broadcast_to(column[:, np.newaxis, np.newaxis], (20, 10, 10))
    result = ApplyRainForestsCalibration(model_config_dict={})._stack_subensembles(
        error_percentile_cube
    )
    # Result should not contain percentile coordinate
    assert result.coords("percentile") == []
    # Result should have realization coordinate composed on of length
    # realization.points.size * percentile.points.size
    assert (
        result.coord("realization").points.size
        == error_percentile_cube.coord("realization").points.size
        * error_percentile_cube.coord("percentile").points.size
    )
    # All remaining coords should be consistent
    assert (
        result.coords(dim_coords=True)[1:]
        == error_percentile_cube.coords(dim_coords=True)[2:]
    )
    assert result.coords(dim_coords=False) == error_percentile_cube.coords(
        dim_coords=False
    )
    assert result.attributes == error_percentile_cube.attributes
    # Check data ordered as expected.
    np.testing.assert_equal(result.data, expected_data)

    # Test the case where cubes are not in expected order.
    error_percentile_cube.transpose([1, 0, 2, 3])
    with pytest.raises(ValueError):
        ApplyRainForestsCalibration(model_config_dict={})._stack_subensembles(
            error_percentile_cube
        )


def test__combine_subensembles(error_percentile_cube):
    """Test extraction of realization values from full superensemble."""
    result = ApplyRainForestsCalibration(model_config_dict={})._combine_subensembles(
        error_percentile_cube, output_realizations_count=None
    )

    assert (
        result.coord("realization").points.size
        == error_percentile_cube.coord("realization").points.size
        * error_percentile_cube.coord("percentile").points.size
    )

    result = ApplyRainForestsCalibration(model_config_dict={})._combine_subensembles(
        error_percentile_cube, output_realizations_count=10
    )

    assert result.coord("realization").points.size == 10


def test_process_with_lightgbm(
    ensemble_forecast, ensemble_features, dummy_lightgbm_models
):
    """Test process routine using lightgbm booster."""
    plugin = ApplyRainForestsCalibration(model_config_dict={})
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

    # Check that all non-realization are equal
    assert result.coords(dim_coords=True)[1:], ensemble_forecast.coords(
        dim_coords=True
    )[1:]
    if output_realization_count is None:
        assert (
            result.coord("realization").points.size
            == 4 * ensemble_forecast.coord("realization").points.size
        )
    else:
        assert result.coord("realization").points.size == output_realization_count
    assert result.coords(dim_coords=False) == ensemble_forecast.coords(dim_coords=False)
    assert result.attributes == ensemble_forecast.attributes


@treelite_available
def test_process_with_treelite(
    ensemble_forecast, ensemble_features, dummy_treelite_models
):
    """Test process routine using treelite Predictor."""
    plugin = ApplyRainForestsCalibration(model_config_dict={})
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

    # Check that all non-realization are equal
    assert result.coords(dim_coords=True)[1:], ensemble_forecast.coords(
        dim_coords=True
    )[1:]
    if output_realization_count is None:
        assert (
            result.coord("realization").points.size
            == 4 * ensemble_forecast.coord("realization").points.size
        )
    else:
        assert result.coord("realization").points.size == output_realization_count
    assert result.coords(dim_coords=False) == ensemble_forecast.coords(dim_coords=False)
    assert result.attributes == ensemble_forecast.attributes
