# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""CLI to apply rainforests calibration."""

from pathlib import Path

from improver import cli


@cli.clizefy
def process(
    *,
    training_data: cli.inputpath,
    training_columns: cli.comma_separated_list,
    observation_column: str,
    thresholds: cli.comma_separated_list_of_float,
    output_dir: cli.inputpath,
):
    """
    Train a set of Rainforests models.

    """
    import pandas as pd

    from improver.calibration.rainforest_training import TrainRainForestsCalibration

    if not Path.is_dir(output_dir):
        raise ValueError("--output_dir must be a directory")

    plugin = TrainRainForestsCalibration(pd.read_parquet(training_data))

    for threshold in thresholds:
        output_path = Path(output_dir) / f"model_{threshold}.txt"
        plugin.process(
            training_columns=training_columns,
            observation_column=observation_column,
            threshold=threshold,
            output_path=output_path,
        )
