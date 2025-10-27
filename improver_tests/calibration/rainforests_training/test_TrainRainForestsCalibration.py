# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

from pathlib import Path

import pytest

from improver.calibration.rainforest_training import (
    TrainRainForestsCalibration,
)


def test__init__lightgmb_available(lightgbm_available):
    """Test class is created if lightgbm library is available.
    Test class is not created if lightgbm library not available."""

    if lightgbm_available:
        expected_class = "TrainRainForestsCalibration"
        result = TrainRainForestsCalibration({})
        assert type(result).__name__ == expected_class
    else:
        with pytest.raises(ModuleNotFoundError):
            result = TrainRainForestsCalibration({})


def test__init__(deterministic_training_data):
    """Test class is created with lead times and thresholds."""
    training_data, fcst_column, observation_column, training_columns = (
        deterministic_training_data
    )
    # This data contains several lead times and thresholds. Choose one of each to train on.
    lead_time = 24
    curr_training_data = training_data.loc[
        training_data["lead_time_hours"] == lead_time
    ]
    result = TrainRainForestsCalibration(curr_training_data)
    assert result.training_data is curr_training_data


def test_process(thresholds, deterministic_training_data):
    """Test lightgbm models are created."""

    training_data, fcst_column, observation_column, training_columns = (
        deterministic_training_data
    )

    # This data contains several lead times and thresholds. Choose one of each to train on.
    threshold = thresholds[0]
    lead_time = 24
    curr_training_data = training_data.loc[
        training_data["lead_time_hours"] == lead_time
    ]

    trainer = TrainRainForestsCalibration(curr_training_data)
    result = trainer.process(threshold, observation_column, training_columns)
    assert isinstance(result, str)


def test_process_with_path(thresholds, deterministic_training_data, tmp_path):
    """Test lightgbm models are created at specified path."""

    training_data, fcst_column, observation_column, training_columns = (
        deterministic_training_data
    )

    # This data contains several lead times and thresholds. Choose one of each to train on.
    threshold = thresholds[0]
    lead_time = 24
    curr_training_data = training_data.loc[
        training_data["lead_time_hours"] == lead_time
    ]

    trainer = TrainRainForestsCalibration(curr_training_data)

    result_path = tmp_path / "output.txt"

    trainer.process(threshold, observation_column, training_columns, result_path)

    assert Path.exists(result_path)
