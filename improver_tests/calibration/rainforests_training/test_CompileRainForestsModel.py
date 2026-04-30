# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

from pathlib import Path

import pytest

from improver.calibration.rainforest_compilation import (
    CompileRainForestsModel,
)

tl2cgen = pytest.importorskip("tl2cgen")
treelite = pytest.importorskip("treelite")


def test__init__(treelite_available, model_config):
    """Test class is created if treelight libraries are available.
    Test class is not created if treelight libraries not available."""

    if treelite_available:
        expected_class = "CompileRainForestsModel"
        result = CompileRainForestsModel(model_config)
        assert type(result).__name__ == expected_class
    else:
        with pytest.raises(ModuleNotFoundError):
            result = CompileRainForestsModel(model_config)


def test_process(model_config_with_trained_models):
    """Test models are compiled."""

    model_config = model_config_with_trained_models

    compiler = CompileRainForestsModel(model_config, parallel_comp=8)

    # Call the process method
    compiler.process()

    # Check that all models were compiled
    for lead_time, thresholds in model_config.items():
        for threshold in thresholds:
            path = model_config[lead_time][threshold]["treelite_model"]
            assert Path(path).exists()


def test_process_fails_with_missing_models(model_config_with_trained_models):
    """Test models are not compiled when any models are missing."""

    model_config = model_config_with_trained_models

    compiler = CompileRainForestsModel(model_config, parallel_comp=8)

    lead_times = list(model_config.keys())
    thresholds = list(model_config[lead_times[0]].keys())

    # Remove one of the trained models
    missing_model_path = model_config[lead_times[0]][thresholds[2]]["lightgbm_model"]
    Path(missing_model_path).unlink()

    # Call the process method
    with pytest.raises(ValueError) as e:
        compiler.process(allow_missing=False)

    # Should identify the missing model in the error
    assert missing_model_path in str(e)

    # Check that none of the models were compiled
    for lead_time, thresholds in model_config.items():
        for threshold in thresholds:
            treelite_path = Path(model_config[lead_time][threshold]["treelite_model"])
            assert not treelite_path.exists()


def test_process_with_missing_models_allowed(model_config_with_trained_models):
    """Test that available models are compiled when other models are missing."""

    model_config = model_config_with_trained_models

    compiler = CompileRainForestsModel(model_config, parallel_comp=8)

    lead_times = list(model_config.keys())
    thresholds = list(model_config[lead_times[0]].keys())

    # Remove one of the trained models
    Path(model_config[lead_times[0]][thresholds[2]]["lightgbm_model"]).unlink()

    # Call the process method
    compiler.process(allow_missing=True)

    # Check that only available models were compiled
    for lead_time, thresholds in model_config.items():
        for threshold in thresholds:
            lightgbm_path = Path(model_config[lead_time][threshold]["lightgbm_model"])
            treelite_path = Path(model_config[lead_time][threshold]["treelite_model"])
            assert treelite_path.exists() == lightgbm_path.exists()
