# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the ApplyRainForestsCalibration class."""
import sys

import numpy as np
import pytest

from improver.calibration.rainforest_calibration import ApplyRainForestsCalibration

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
@pytest.mark.parametrize("treelite_model", (TREELITE_ENABLED, False))
@pytest.mark.parametrize("treelite_file", (True, False))
def test__new__(
    lightgbm_keys, treelite_model, treelite_file, monkeypatch, model_config,
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
        model_config["24"]["0.0000"].pop("treelite_model", None)
    if not lightgbm_keys:
        model_config["24"]["0.0000"].pop("lightgbm_model", None)

    if treelite_model and treelite_file:
        expected_class = "ApplyRainForestsCalibrationTreelite"
    elif lightgbm_keys:
        expected_class = "ApplyRainForestsCalibrationLightGBM"
    else:
        with pytest.raises(ValueError, match="Path to lightgbm model missing"):
            ApplyRainForestsCalibration(model_config)
        return

    result = ApplyRainForestsCalibration(model_config)
    assert type(result).__name__ == expected_class


@pytest.mark.parametrize("treelite_file", (True, False))
def test__get_feature_splits(
    treelite_file, model_config, plugin_and_dummy_models, lightgbm_model_files
):
    """Test that _get_feature_splits returns a dict in the expected format.
    The lightgbm_model_files parameter is not used explicitly, but it is
    required in order to make the files available."""
    if not treelite_file:
        # Model type should default to lightgbm if there are any treelite models
        # missing across any thresholds
        model_config["24"]["0.0000"].pop("treelite_model", None)

    plugin_cls, dummy_models = plugin_and_dummy_models
    plugin = plugin_cls(model_config_dict={})
    plugin.tree_models, plugin.lead_times, plugin.model_thresholds = dummy_models

    splits = plugin._get_feature_splits(model_config)

    lead_times = sorted([np.float32(x) for x in model_config.keys()])
    assert sorted(list(splits.keys())) == lead_times

    model_path = model_config["24"]["0.0000"].get("lightgbm_model")
    model = lightgbm.Booster(model_file=model_path)
    num_features = len(model.feature_name())
    assert all([len(x) == num_features for x in splits.values()])


def test_check_filenames(model_config):
    """Test that check_filenames raises an error if an invalid
    key_name is specified."""

    msg = "key_name must be 'lightgbm_model' or 'treelite_model'"
    with pytest.raises(ValueError, match=msg):
        ApplyRainForestsCalibration.check_filenames(
            key_name="tensorflow_models", model_config_dict=model_config
        )
