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

    compiler = CompileRainForestsModel(
        model_config_with_trained_models, parallel_comp=8
    )

    compiler.process()
    for lead_time, thresholds in model_config_with_trained_models.items():
        for threshold in thresholds:
            path = model_config_with_trained_models[lead_time][threshold][
                "treelite_model"
            ]
            assert Path(path).exists()
