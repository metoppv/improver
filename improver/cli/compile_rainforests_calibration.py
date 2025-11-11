# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to compile a Rainforests calibration model."""

from pathlib import Path

from improver import cli


@cli.clizefy
def process(lightgbm_model: cli.inputpath):
    """
    Train a set of Rainforests models.

    """

    from improver.calibration.rainforest_training import CompileRainForestsCalibration

    if not Path.is_file(lightgbm_model):
        raise ValueError("--output_dir must be an existing file")

    plugin = CompileRainForestsCalibration()

    input_path = Path(lightgbm_model)
    output_path = input_path.with_suffix(".o")
    plugin.process(input_path, output_path)
