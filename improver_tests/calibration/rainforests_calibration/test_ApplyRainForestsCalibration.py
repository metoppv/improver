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
    print(model_config)
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
