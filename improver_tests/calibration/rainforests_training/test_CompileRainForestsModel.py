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


def test__init__(treelite_available):
    """Test class is created if treelight libraries are available.
    Test class is not created if treelight libraries not available."""

    if treelite_available:
        expected_class = "CompileRainForestsModel"
        result = CompileRainForestsModel()
        assert type(result).__name__ == expected_class
    else:
        with pytest.raises(ModuleNotFoundError):
            result = CompileRainForestsModel()


def test_process(rainforests_model_files, tmp_path):
    """Test models are compiled."""

    compiler = CompileRainForestsModel(parallel_comp=8)

    output_dir = Path(tmp_path) / "compiled"
    output_dir.mkdir(exist_ok=True)

    for model_file in rainforests_model_files:
        compiler.process(model_file, output_dir)

        assert Path.exists(output_dir / f"{model_file.stem}.so")
