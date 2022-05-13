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

try:
    import treelite_runtime

    TREELITE_ENABLED = True
except ModuleNotFoundError:
    TREELITE_ENABLED = False

from improver.calibration.rainforest_calibration import ApplyRainForestsCalibration

lightgbm = pytest.importorskip("lightgbm")


class MockBooster:
    def __init__(self, model_file, **kwargs):
        self.model_class = "lightgbm-Booster"

    def reset_parameter(self, params):
        self.threads = params.get("num_threads")
        return self


class MockPredictor:
    def __init__(self, libpath, nthread, **kwargs):
        self.model_class = "treelite-Predictor"
        self.threads = nthread


def test__init_lightgbm_models(monkeypatch, lightgbm_model_config, error_thresholds):
    """Test lightgbm models are loaded if model_config contains path to lightgbm models only."""
    monkeypatch.setattr(lightgbm, "Booster", MockBooster)

    result = ApplyRainForestsCalibration(lightgbm_model_config, threads=8)

    for model in result.tree_models:
        assert model.model_class == "lightgbm-Booster"
        assert model.threads == 8
    assert result.treelite_enabled == TREELITE_ENABLED
    assert np.all(result.error_thresholds == error_thresholds)


@pytest.mark.skipif(not TREELITE_ENABLED, reason="Required dependency missing.")
def test__init_treelite_models(monkeypatch, treelite_model_config, error_thresholds):
    """Test treelite models are loaded if model_config correctly. If all thresholds
    contain treelite model, treelite Predictor is returned, otherwise return lightgbm
    Boosters."""
    monkeypatch.setattr(treelite_runtime, "Predictor", MockPredictor)
    monkeypatch.setattr(lightgbm, "Booster", MockBooster)

    result = ApplyRainForestsCalibration(treelite_model_config, threads=8)

    for model in result.tree_models:
        assert model.model_class == "treelite-Predictor"
        assert model.threads == 8
    assert result.treelite_enabled is True
    assert np.all(result.error_thresholds == error_thresholds)

    # Model type should default to lightgbm if there are any treelite models
    # missing across any thresholds
    treelite_model_config["0.0000"].pop("treelite_model", None)
    result = ApplyRainForestsCalibration(treelite_model_config, threads=8)

    for model in result.tree_models:
        assert model.model_class == "lightgbm-Booster"
        assert model.threads == 8
    assert result.treelite_enabled is True
    assert np.all(result.error_thresholds == error_thresholds)


def test__init_treelite_missing(monkeypatch, treelite_model_config, error_thresholds):
    """Test that lightgbm Booster returned when model_config references treelite models,
    but treelite dependency is missing."""
    # Simulate environment which does not have treelite loaded.
    monkeypatch.setitem(sys.modules, "treelite_runtime", None)
    monkeypatch.setattr(lightgbm, "Booster", MockBooster)

    result = ApplyRainForestsCalibration(treelite_model_config, threads=8)

    for model in result.tree_models:
        assert model.model_class == "lightgbm-Booster"
        assert model.threads == 8
    assert result.treelite_enabled is False
    assert np.all(result.error_thresholds == error_thresholds)
