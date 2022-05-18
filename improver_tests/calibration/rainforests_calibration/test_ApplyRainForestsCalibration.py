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
        self.model_file = model_file

    def reset_parameter(self, params):
        self.threads = params.get("num_threads")
        return self


class MockPredictor:
    def __init__(self, libpath, nthread, **kwargs):
        self.model_class = "treelite-Predictor"
        self.threads = nthread
        self.model_file = libpath


@pytest.mark.skipif(not TREELITE_ENABLED, reason="Required dependency missing.")
@pytest.mark.parametrize("lightgbm_keys", (True, False))
@pytest.mark.parametrize("ordered_inputs", (True, False))
@pytest.mark.parametrize("treelite_model", (True, False))
@pytest.mark.parametrize("treelite_file", (True, False))
def test__init_treelite_models(
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
