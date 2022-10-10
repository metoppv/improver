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
"""Unit tests for the ApplyRainForestsCalibrationTreelite class."""
import sys

import numpy as np
import pytest

from improver.calibration.rainforest_calibration import (
    ApplyRainForestsCalibrationTreelite,
)

treelite_runtime = pytest.importorskip("treelite_runtime")


class MockPredictor:
    def __init__(self, libpath, nthread, **kwargs):
        self.model_class = "treelite-Predictor"
        self.threads = nthread
        self.model_file = libpath


def test__new__(model_config, monkeypatch):
    monkeypatch.setattr(treelite_runtime, "Predictor", MockPredictor)

    # Check that we get the expected subclass
    result = ApplyRainForestsCalibrationTreelite(model_config)
    assert type(result).__name__ == "ApplyRainForestsCalibrationTreelite"
    # Test exception raised when file path is missing.
    model_config["0.0000"].pop("treelite_model", None)
    with pytest.raises(ValueError):
        ApplyRainForestsCalibrationTreelite(model_config)

    monkeypatch.setitem(sys.modules, "treelite_runtime", None)
    with pytest.raises(ModuleNotFoundError):
        ApplyRainForestsCalibrationTreelite(model_config)


@pytest.mark.parametrize("ordered_inputs", (True, False))
@pytest.mark.parametrize("default_threads", (True, False))
def test__init__(
    model_config, ordered_inputs, default_threads, error_thresholds, monkeypatch
):
    monkeypatch.setattr(treelite_runtime, "Predictor", MockPredictor)

    if not ordered_inputs:
        tmp_value = model_config.pop("0.0000", None)
        model_config["0.0000"] = tmp_value

    if default_threads:
        expected_threads = 1
        result = ApplyRainForestsCalibrationTreelite(model_config)
    else:
        expected_threads = 8
        result = ApplyRainForestsCalibrationTreelite(
            model_config, threads=expected_threads
        )
    # Check thresholds and model types match
    assert np.all(result.error_thresholds == error_thresholds)
    for model in result.tree_models:
        assert model.model_class == "treelite-Predictor"
        assert model.threads == expected_threads
    # Ensure threshold and files match
    for threshold, model in zip(result.error_thresholds, result.tree_models):
        # Treelite library requires paths to be passed as strings
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


def test__calculate_error_probabilities(
    ensemble_features, ensemble_forecast, plugin_and_dummy_models
):
    """Test calculation of error probability cube when using treelite Predictors."""
    plugin_cls, dummy_models = plugin_and_dummy_models
    plugin = plugin_cls(model_config_dict={})
    plugin.tree_models, plugin.error_thresholds = dummy_models
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


def test_process_with_treelite(
    ensemble_forecast, ensemble_features, plugin_and_dummy_models
):
    """Test process routine using treelite Predictor."""
    plugin_cls, dummy_models = plugin_and_dummy_models
    plugin = plugin_cls(model_config_dict={})
    plugin.tree_models, plugin.error_thresholds = dummy_models

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
