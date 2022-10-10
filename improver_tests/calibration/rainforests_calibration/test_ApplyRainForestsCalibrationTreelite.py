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
