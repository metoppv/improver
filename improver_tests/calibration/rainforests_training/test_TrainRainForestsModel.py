# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

from pathlib import Path
from unittest.mock import patch

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

    _, training_data, observation_column, training_columns = deterministic_training_data

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
    _, training_data, observation_column, training_columns = deterministic_training_data

    result = TrainRainForestsModel(
        model_config, training_data, observation_column, training_columns
    )
    assert result.training_columns == training_columns
    assert result.observation_column == observation_column


def test__init__missing_obs_column(model_config, deterministic_training_data):
    """Test class creation fails when observation column isn't present in the training data."""
    _, training_data, _, training_columns = deterministic_training_data

    dummy_obs_column = "dummy_obs_column"

    with pytest.raises(KeyError) as e:
        TrainRainForestsModel(
            model_config, training_data, dummy_obs_column, training_columns
        )
    assert dummy_obs_column in str(e)


def test__init__missing_train_column(model_config, deterministic_training_data):
    """Test class creation fails when one of the training columns isn't present in the training data."""
    _, training_data, observation_column, training_columns = deterministic_training_data

    dummy_train_column = "dummy_train_column"
    training_columns.append(dummy_train_column)

    with pytest.raises(KeyError) as e:
        TrainRainForestsModel(
            model_config, training_data, observation_column, training_columns
        )
    assert dummy_train_column in str(e)


def test__init__obs_column_is_train_column(model_config, deterministic_training_data):
    """Test class creation fails when the observation column is one of the training columns."""
    _, training_data, _, training_columns = deterministic_training_data

    dummy_obs_column = training_columns[2]

    with pytest.raises(KeyError) as e:
        TrainRainForestsModel(
            model_config, training_data, dummy_obs_column, training_columns
        )
    assert dummy_obs_column in str(e)


def test_process(model_config, deterministic_training_data):
    """Test lightgbm models are created at specified path."""

    lead_time, training_data, observation_column, training_columns = (
        deterministic_training_data
    )

    trainer = TrainRainForestsModel(
        model_config, training_data, observation_column, training_columns
    )

    thresholds = model_config[lead_time].keys()

    trainer.process(lead_time, thresholds)

    for threshold in thresholds:
        expected_path = model_config[lead_time][threshold]["lightgbm_model"]
        assert Path(expected_path).exists()


@patch("lightgbm.train")
def test_process_calls_train(mock_train, model_config, deterministic_training_data):
    """Test lightgbm models are created at specified path."""

    lead_time, training_data, observation_column, training_columns = (
        deterministic_training_data
    )

    trainer = TrainRainForestsModel(
        model_config, training_data, observation_column, training_columns
    )

    thresholds = model_config[lead_time].keys()

    trainer.process(lead_time, thresholds)
    assert mock_train.call_count == len(thresholds)


@patch("lightgbm.train")
def test_process_calls_train_with_default_params(
    mock_train, model_config, deterministic_training_data
):
    """Test default lightgbm params are used if none are specified."""

    lead_time, training_data, observation_column, training_columns = (
        deterministic_training_data
    )

    trainer = TrainRainForestsModel(
        model_config, training_data, observation_column, training_columns
    )

    # Process just one threshold
    trainer.process(lead_time, ["0.0000"])

    # Test the parameters actually used in the train call.
    params = mock_train.call_args.args[0]
    assert params == {
        "objective": "binary",
        "num_leaves": 5,
        "seed": 0,
    }


@patch("lightgbm.train")
def test_process_calls_train_with_override_params(
    mock_train, model_config, deterministic_training_data
):
    """Test default lightgbm params can be overridden."""

    lead_time, training_data, observation_column, training_columns = (
        deterministic_training_data
    )

    override_params = {
        # Override a param already in the defaults
        "objective": "regression",
        # Add a param not in the defaults
        "num_threads": 48,
    }

    # Pass the override params into constructor.
    trainer = TrainRainForestsModel(
        model_config,
        training_data,
        observation_column,
        training_columns,
        override_params,
    )

    # Process just one threshold
    trainer.process(lead_time, ["0.0000"])

    # Test the parameters actually used in the train call.
    params = mock_train.call_args.args[0]

    # Default params not present in constructor args should be retained.
    assert params["num_leaves"] == 5
    # Params that are only in the constructor arg should be included.
    assert params["num_threads"] == 48
    # Where there is a clash, param in constructor argument is used.
    assert params["objective"] == "regression"


@patch("lightgbm.train")
def test_process_calls_train_with_all_columns(
    mock_train, model_config, deterministic_training_data
):
    """Test lightgbm training is provided with only the specified training columns."""

    lead_time, training_data, observation_column, training_columns = (
        deterministic_training_data
    )

    # Pass the override params into constructor.
    trainer = TrainRainForestsModel(
        model_config,
        training_data,
        observation_column,
        training_columns,
    )

    # Process just one threshold
    trainer.process(lead_time, ["0.0000"])

    # Check the data actually used in the train call (a lightgbm.Dataset) has only
    # the training columns.
    dataset = mock_train.call_args.args[1]
    assert set(dataset.data.columns) == set(training_columns)


def test_process_missing_lead_time(model_config, deterministic_training_data):
    """Test lightgbm models are not created for invalid lead time."""

    lead_time, training_data, observation_column, training_columns = (
        deterministic_training_data
    )

    trainer = TrainRainForestsModel(
        model_config, training_data, observation_column, training_columns
    )

    thresholds = model_config[lead_time].keys()
    invalid_lead_time = 1

    with pytest.raises(KeyError):
        trainer.process(invalid_lead_time, thresholds)

    # Check that none of the models were trained
    for lead_time, thresholds in model_config.items():
        for threshold in thresholds:
            lightgbm_path = Path(model_config[lead_time][threshold]["lightgbm_model"])
            assert not lightgbm_path.exists()


def test_process_missing_threshold(model_config, deterministic_training_data):
    """Test lightgbm models are not created for invalid threshold."""

    lead_time, training_data, observation_column, training_columns = (
        deterministic_training_data
    )

    trainer = TrainRainForestsModel(
        model_config, training_data, observation_column, training_columns
    )

    invalid_thresholds = list(model_config[lead_time].keys()) + ["1234.5"]

    with pytest.raises(KeyError):
        trainer.process(lead_time, invalid_thresholds)

    # Check that none of the models were trained
    for lead_time, thresholds in model_config.items():
        for threshold in thresholds:
            lightgbm_path = Path(model_config[lead_time][threshold]["lightgbm_model"])
            assert not lightgbm_path.exists()
