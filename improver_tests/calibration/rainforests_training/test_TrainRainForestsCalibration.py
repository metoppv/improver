# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

from pathlib import Path

import pytest

from improver.calibration.rainforest_training import (
    TrainRainForestsCalibration,
)

lightgbm = pytest.importorskip("lightgbm")


def test__init__lightgmb_available(lightgbm_available, deterministic_training_data):
    """Test class is created if lightgbm library is available.
    Test class is not created if lightgbm library not available."""

    training_data, observation_column, training_columns = (
        deterministic_training_data
    )

    if lightgbm_available:
        expected_class = "TrainRainForestsCalibration"
        result = TrainRainForestsCalibration(
            training_data, observation_column, training_columns
        )
        assert type(result).__name__ == expected_class
    else:
        with pytest.raises(ModuleNotFoundError):
            result = TrainRainForestsCalibration(
                training_data, observation_column, training_columns
            )


def test__init__(deterministic_training_data):
    """Test class is created with training data."""
    training_data, observation_column, training_columns = (
        deterministic_training_data
    )

    result = TrainRainForestsCalibration(
        training_data, observation_column, training_columns
    )
    assert result.training_columns == training_columns
    assert result.observation_column == observation_column


def test_process(thresholds, deterministic_training_data):
    """Test lightgbm models are created."""

    training_data, observation_column, training_columns = (
        deterministic_training_data
    )

    threshold = thresholds[0]

    trainer = TrainRainForestsCalibration(
        training_data, observation_column, training_columns
    )
    result = trainer.process(threshold)
    assert isinstance(result, str)


def test_process_with_path(thresholds, deterministic_training_data, tmp_path):
    """Test lightgbm models are created at specified path."""

    training_data, observation_column, training_columns = (
        deterministic_training_data
    )

    trainer = TrainRainForestsCalibration(
        training_data, observation_column, training_columns
    )

    result_path = tmp_path / "output.txt"

    threshold = thresholds[0]
    trainer.process(threshold, result_path)

    assert Path.exists(result_path)
