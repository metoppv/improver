# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

from pathlib import Path

import pytest

from improver.calibration.rainforest_training import (
    TrainRainForestsModel,
)

lightgbm = pytest.importorskip("lightgbm")


def test__init__lightgmb_available(
    lightgbm_available, model_config, deterministic_training_data
):
    """Test class is created if lightgbm library is available.
    Test class is not created if lightgbm library not available."""

    training_data, observation_column, training_columns = deterministic_training_data

    if lightgbm_available:
        expected_class = "TrainRainForestsModel"
        result = TrainRainForestsModel(
            model_config, training_data, observation_column, training_columns
        )
        assert type(result).__name__ == expected_class
    else:
        with pytest.raises(ModuleNotFoundError):
            result = TrainRainForestsModel(
                model_config, training_data, observation_column, training_columns
            )


def test__init__(model_config, deterministic_training_data):
    """Test class is created with training data."""
    training_data, observation_column, training_columns = deterministic_training_data

    result = TrainRainForestsModel(
        model_config, training_data, observation_column, training_columns
    )
    assert result.training_columns == training_columns
    assert result.observation_column == observation_column


def test__init__missing_obs_column(model_config, deterministic_training_data):
    """Test class creation fails when observation column isn't present in the training data."""
    training_data, _, training_columns = deterministic_training_data

    dummy_obs_column = "dummy_obs_column"

    with pytest.raises(KeyError) as e:
        TrainRainForestsModel(
            model_config, training_data, dummy_obs_column, training_columns
        )
    assert dummy_obs_column in str(e)


def test__init__missing_train_column(model_config, deterministic_training_data):
    """Test class creation fails when one of the training columns isn't present in the training data."""
    training_data, observation_column, training_columns = deterministic_training_data

    dummy_train_column = "dummy_train_column"
    training_columns.append(dummy_train_column)

    with pytest.raises(KeyError) as e:
        TrainRainForestsModel(
            model_config, training_data, observation_column, training_columns
        )
    assert dummy_train_column in str(e)


def test__init__obs_column_is_train_column(model_config, deterministic_training_data):
    """Test class creation fails when the observation column is one of the training columns."""
    training_data, _, training_columns = deterministic_training_data

    dummy_obs_column = training_columns[2]

    with pytest.raises(KeyError) as e:
        TrainRainForestsModel(
            model_config, training_data, dummy_obs_column, training_columns
        )
        assert dummy_obs_column not in str(e)


def test_process(model_config, deterministic_training_data):
    """Test lightgbm models are created at specified path."""

    training_data, observation_column, training_columns = deterministic_training_data

    trainer = TrainRainForestsModel(
        model_config, training_data, observation_column, training_columns
    )

    lead_time = list(model_config.keys())[0]
    thresholds = model_config[lead_time].keys()

    trainer.process(lead_time, thresholds)

    for threshold in thresholds:
        # Should format unique filename for each threshold
        expected_path = model_config[lead_time][threshold]["lightgbm_model"]
        assert Path(expected_path).exists()


def test_process_missing_lead_time(model_config, deterministic_training_data):
    """Test lightgbm models are not created for invalid lead time."""

    training_data, observation_column, training_columns = deterministic_training_data

    trainer = TrainRainForestsModel(
        model_config, training_data, observation_column, training_columns
    )

    lead_time = list(model_config.keys())[0]
    thresholds = model_config[lead_time].keys()
    invalid_lead_time = 1

    with pytest.raises(KeyError):
        trainer.process(invalid_lead_time, thresholds)


def test_process_missing_threshold(model_config, deterministic_training_data):
    """Test lightgbm models are not created for invalid threshold."""

    training_data, observation_column, training_columns = deterministic_training_data

    trainer = TrainRainForestsModel(
        model_config, training_data, observation_column, training_columns
    )

    lead_time = list(model_config.keys())[0]
    invalid_thresholds = list(model_config[lead_time].keys()) + ["1234.5"]

    with pytest.raises(KeyError):
        trainer.process(lead_time, invalid_thresholds)
