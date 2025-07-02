# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the ApplyRainForestsCalibrationTreelite class."""

import sys

import numpy as np
import pytest

from improver.calibration.rainforest_calibration import (
    ApplyRainForestsCalibrationTreelite,
    ModelFileNotFoundError,
)

from .utils import MockPredictor

tl2cgen = pytest.importorskip("tl2cgen")


class TestNew:
    """Tests for the __new__ method of ApplyRainForestsCalibrationTreelite."""

    def test_new_with_tl2cgen_available(self, model_config, monkeypatch):
        """Test that the __new__ method creates an object of the expected
        subclass when tl2cgen is available."""
        monkeypatch.setattr(tl2cgen, "Predictor", MockPredictor)
        result = ApplyRainForestsCalibrationTreelite(model_config)
        assert type(result) is ApplyRainForestsCalibrationTreelite

    def test_new_with_file_path_missing(self, model_config, monkeypatch):
        """Test that the __new__ method raises the correct exception type
        when file path is missing in model config."""
        monkeypatch.setattr(tl2cgen, "Predictor", MockPredictor)
        model_config["24"]["0.0000"].pop("treelite_model", None)
        with pytest.raises(ModelFileNotFoundError):
            ApplyRainForestsCalibrationTreelite(model_config)

    def test_new_with_tl2cgen_unavailable(self, model_config, monkeypatch):
        """Test that the __new__ method raises the correct error when
        tl2cgen is unavailable."""
        monkeypatch.setitem(sys.modules, "tl2cgen", None)
        with pytest.raises(ModuleNotFoundError):
            ApplyRainForestsCalibrationTreelite(model_config)


@pytest.mark.parametrize("ordered_inputs", (True, False))
@pytest.mark.parametrize("expected_threads", (None, 1, 3, 8))
def test_tree_models(
    monkeypatch, model_config, ordered_inputs, expected_threads, lead_times, thresholds
):
    """Test for the correct values when using the
    ApplyRainForestsCalibrationTreelite class.

    Asserts that:
    - Thresholds and model types match
    - Thresholds and files match
    """
    # Setup
    monkeypatch.setattr(tl2cgen, "Predictor", MockPredictor)
    if not ordered_inputs:
        tmp_value = model_config["24"].pop("0.0000", None)
        model_config["24"]["0.0000"] = tmp_value

    # Act
    if expected_threads is None:
        result = ApplyRainForestsCalibrationTreelite(model_config)
    else:
        result = ApplyRainForestsCalibrationTreelite(
            model_config, threads=expected_threads
        )

    # Assert
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


@pytest.mark.parametrize("expected_threads", (1, 8))
def test_correct_error_raised(monkeypatch, model_config, expected_threads):
    """Test that an error is raised if lead times have different thresholds."""
    monkeypatch.setattr(tl2cgen, "Predictor", MockPredictor)
    val = model_config["24"].pop("0.0000")
    model_config["24"]["1.0000"] = val
    with pytest.raises(ValueError, match="same thresholds must be used"):
        ApplyRainForestsCalibrationTreelite(model_config, threads=expected_threads)
