# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the ApplyRainForestsCalibration class."""

import re
import sys

import numpy as np
import pytest

from improver.calibration.rainforest_calibration import (
    ApplyRainForestsCalibration,
    ApplyRainForestsCalibrationLightGBM,
    ApplyRainForestsCalibrationTreelite,
    ModelFileNotFoundError,
)

from .utils import MockBooster, MockPredictor

try:
    import tl2cgen
    import treelite  # noqa: F401
except ModuleNotFoundError:
    TREELITE_ENABLED = False
else:
    TREELITE_ENABLED = True

lightgbm = pytest.importorskip("lightgbm")
treelite_available = pytest.mark.skipif(
    not TREELITE_ENABLED, reason="Required dependency missing."
)


class TestConstructorCorrectClass:
    """Test that the ApplyRainForestsCalibration constructor correctly
    selects which class to use based on availability of modules.
    """

    def test_correct_class_when_lightgbm_available(self, monkeypatch, model_config):
        """Test that the ApplyRainForestsCalibration constructor creates an
        object of the correct type when lightgbm is available."""
        monkeypatch.setattr(lightgbm, "Booster", MockBooster)
        monkeypatch.setitem(sys.modules, "tl2cgen", None)
        result = ApplyRainForestsCalibration(model_config)
        assert type(result) is ApplyRainForestsCalibrationLightGBM

    def test_correct_class_when_treelite_available(self, monkeypatch, model_config):
        """Test that the ApplyRainForestsCalibration constructor creates an
        object of the correct type when tl2cgen is available."""
        monkeypatch.setattr(tl2cgen, "Predictor", MockPredictor)
        result = ApplyRainForestsCalibration(model_config)
        assert type(result) is ApplyRainForestsCalibrationTreelite


class TestLoadTreeliteModels:
    """Test that treelite models are loaded correctly."""

    def test_treelite_model_and_module_available(self, monkeypatch, model_config):
        """Test when model config is loaded and tl2cgen module available.

        Tests that treelite models are loaded correctly when:

        - all thresholds contain treelite model
        - tl2cgen modules are available
        - lightgbm module is available.
        """
        monkeypatch.setattr(tl2cgen, "Predictor", MockPredictor)
        monkeypatch.setattr(lightgbm, "Booster", MockBooster)
        result = ApplyRainForestsCalibration(model_config)
        assert type(result) is ApplyRainForestsCalibrationTreelite

    def test_treelite_model_unavailable_module_available(
        self, monkeypatch, model_config
    ):
        """Test when model config is loaded and tl2cgen module unavailable.

        Tests that treelite models are loaded correctly when:

        - all thresholds contain treelite model
        - tl2cgen modules is unavailable
        - lightgbm module is available.
        """
        monkeypatch.setitem(sys.modules, "tl2cgen", None)
        monkeypatch.setattr(lightgbm, "Booster", MockBooster)
        result = ApplyRainForestsCalibration(model_config)
        assert type(result) is ApplyRainForestsCalibrationLightGBM

    def test_treelite_model_and_module_unavailable(self, monkeypatch, model_config):
        """Test when model and tl2cgen module both unavailable.

        Tests that treelite models are loaded correctly when:

        - treelite module unavailable
        - tl2cgen module unavailable
        - lightgbm module available.
        """
        monkeypatch.setitem(sys.modules, "tl2cgen", None)
        monkeypatch.setattr(lightgbm, "Booster", MockBooster)
        model_config["24"]["0.0000"].pop("treelite_model", None)
        result = ApplyRainForestsCalibration(model_config)
        assert type(result) is ApplyRainForestsCalibrationLightGBM

    def test_treelite_unavailable_lightgbm_keys_unavailable(
        self, monkeypatch, model_config
    ):
        """Test when treelite model unavailable, tl2cgen unavailable, no lightgbm config.

        Test that the correct exception is raised when:
        - treelite model is unavailable
        - tl2cgen module is unavailable
        - there are no lightgbm configs
        - lightgbm module is available.
        """
        monkeypatch.setitem(sys.modules, "tl2cgen", None)
        monkeypatch.setattr(lightgbm, "Booster", MockBooster)
        model_config["24"]["0.0000"].pop("treelite_model", None)
        model_config["24"]["0.0000"].pop("lightgbm_model", None)
        with pytest.raises(
            ModelFileNotFoundError, match="Path to lightgbm model missing"
        ):
            ApplyRainForestsCalibration(model_config)


@pytest.mark.parametrize("treelite_file", (True, False))
def test__get_feature_splits(
    treelite_file, model_config, plugin_and_dummy_models, lightgbm_model_files
):
    """Test that _get_feature_splits returns a dict in the expected format.

    Note: The lightgbm_model_files parameter is not used explicitly, but it is
    required in order to make the files available.
    """
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


@pytest.mark.parametrize(
    "key_name", ("tensorflow_models", 123, -1, True, False, "treelite")
)
def test_check_filenames_with_invalid_key_name(model_config, key_name):
    """Test that check_filenames() raises an error if an invalid
    key_name is specified."""

    msg = "key_name must be one of the following: ('lightgbm_model', 'treelite_model')"
    with pytest.raises(ValueError, match=re.escape(msg)):
        ApplyRainForestsCalibration.check_filenames(
            key_name=key_name, model_config_dict=model_config
        )
