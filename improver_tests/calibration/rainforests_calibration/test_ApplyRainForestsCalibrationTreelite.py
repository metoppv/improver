# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
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
    model_config["24"]["0.0000"].pop("treelite_model", None)
    with pytest.raises(ValueError):
        ApplyRainForestsCalibrationTreelite(model_config)

    monkeypatch.setitem(sys.modules, "treelite_runtime", None)
    with pytest.raises(ModuleNotFoundError):
        ApplyRainForestsCalibrationTreelite(model_config)


@pytest.mark.parametrize("ordered_inputs", (True, False))
@pytest.mark.parametrize("default_threads", (True, False))
def test__init__(
    model_config, ordered_inputs, default_threads, lead_times, thresholds, monkeypatch
):
    monkeypatch.setattr(treelite_runtime, "Predictor", MockPredictor)

    if not ordered_inputs:
        tmp_value = model_config["24"].pop("0.0000", None)
        model_config["24"]["0.0000"] = tmp_value

    if default_threads:
        expected_threads = 1
        result = ApplyRainForestsCalibrationTreelite(model_config)
    else:
        expected_threads = 8
        result = ApplyRainForestsCalibrationTreelite(
            model_config, threads=expected_threads
        )
    # Check thresholds and model types match
    assert np.all(result.lead_times == lead_times)
    assert np.all(result.model_thresholds == thresholds)
    for lead_time in lead_times:
        for threshold in thresholds:
            model = result.tree_models[lead_time, threshold]
            assert model.model_class == "treelite-Predictor"
            assert model.threads == expected_threads
    # Ensure threshold and files match
    for lead_time in lead_times:
        for threshold in thresholds:
            model = result.tree_models[lead_time, threshold]
            # Treelite library requires paths to be passed as strings
            assert isinstance(model.model_file, str)
            assert f"{lead_time:03d}H" in str(model.model_file)
            assert f"{threshold:06.4f}" in str(model.model_file)
    # Test error is raised if lead times have different thresholds
    val = model_config["24"].pop("0.0000")
    model_config["24"]["1.0000"] = val
    msg = "same thresholds must be used"
    with pytest.raises(ValueError, match=msg):
        ApplyRainForestsCalibrationTreelite(model_config, threads=expected_threads)
