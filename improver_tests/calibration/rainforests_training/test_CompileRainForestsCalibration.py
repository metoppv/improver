# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

from pathlib import Path

import pytest
from lightgbm import Booster

from improver.calibration.rainforest_training import (
    CompileRainForestsCalibration,
)


def test__init__(lightgbm_available, treelite_available, tmp_path):
    """Test class is created if treelight libraries are available.
    Test class is not created if treelight libraries not available."""

    if treelite_available and lightgbm_available:
        expected_class = "CompileRainForestsCalibration"
        result = CompileRainForestsCalibration()
        assert type(result).__name__ == expected_class
    else:
        with pytest.raises(ModuleNotFoundError):
            result = CompileRainForestsCalibration()


def test_process(dummy_lightgbm_models, tmp_path):
    """Test models are compiled."""

    tree_models, lead_times, thresholds = dummy_lightgbm_models

    compiler = CompileRainForestsCalibration()

    model: Booster = tree_models[lead_times[0], thresholds[0]]
    model_path = tmp_path / f"model{lead_times[0]}{thresholds[0]}.txt"
    model.save_model(model_path)

    compiled_path = tmp_path / f"compiled{lead_times[0]}{thresholds[0]}.o"
    compiler.process(model_path, compiled_path)

    assert Path.exists(compiled_path)
