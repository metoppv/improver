# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

from pathlib import Path

import pytest

from improver.calibration.rainforest_compilation import (
    CompileRainForestsModel,
)
from improver.calibration.rainforest_training import (
    TrainRainForestsModel,
)

lightgbm = pytest.importorskip("lightgbm")


def test__init__lightgmb_available(
    lightgbm_available, deterministic_training_data, tmp_path
):
    """Test class is created if lightgbm library is available.
    Test class is not created if lightgbm library not available."""

    training_data, observation_column, training_columns = deterministic_training_data

    if lightgbm_available:
        expected_class = "TrainRainForestsModel"
        result = TrainRainForestsModel(
            training_data, observation_column, training_columns, tmp_path
        )
        assert type(result).__name__ == expected_class
    else:
        with pytest.raises(ModuleNotFoundError):
            result = TrainRainForestsModel(
                training_data, observation_column, training_columns, tmp_path
            )


def test__init__(deterministic_training_data, tmp_path):
    """Test class is created with training data."""
    training_data, observation_column, training_columns = deterministic_training_data

    result = TrainRainForestsModel(
        training_data, observation_column, training_columns, tmp_path
    )
    assert result.training_columns == training_columns
    assert result.observation_column == observation_column


def test__init__missing_obs_column(deterministic_training_data, tmp_path):
    """Test class creation fails when observation column isn't present in the training data."""
    training_data, _, training_columns = deterministic_training_data

    dummy_obs_column = "dummy_obs_column"

    with pytest.raises(KeyError) as e:
        TrainRainForestsModel(
            training_data, dummy_obs_column, training_columns, tmp_path
        )
    assert dummy_obs_column in str(e)


def test__init__missing_train_column(deterministic_training_data, tmp_path):
    """Test class creation fails when one of the training columns isn't present in the training data."""
    training_data, observation_column, training_columns = deterministic_training_data

    dummy_train_column = "dummy_train_column"
    training_columns.append(dummy_train_column)

    with pytest.raises(KeyError) as e:
        TrainRainForestsModel(
            training_data, observation_column, training_columns, tmp_path
        )
    assert dummy_train_column in str(e)


def test__init__obs_column_is_train_column(deterministic_training_data, tmp_path):
    """Test class creation fails when the observation column is one of the training columns."""
    training_data, _, training_columns = deterministic_training_data

    dummy_obs_column = training_columns[2]

    with pytest.raises(KeyError) as e:
        TrainRainForestsModel(
            training_data, dummy_obs_column, training_columns, tmp_path
        )
        assert dummy_obs_column not in str(e)


def test_process(thresholds, deterministic_training_data, tmp_path):
    """Test lightgbm models are created at specified path."""

    training_data, observation_column, training_columns = deterministic_training_data

    trainer = TrainRainForestsModel(
        training_data, observation_column, training_columns, tmp_path
    )

    trainer.process(thresholds)

    for threshold in thresholds:
        # Should format unique filename for each threshold
        expected_path = tmp_path / f"lgb_model-threshold_{threshold:04.2f}.txt"
        assert Path.exists(expected_path)


def test_process_format_filename(thresholds, deterministic_training_data, tmp_path):
    """Test lightgbm models are created at specified path."""

    training_data, observation_column, training_columns = deterministic_training_data

    trainer = TrainRainForestsModel(
        training_data, observation_column, training_columns, tmp_path
    )

    trainer.model_file_name_formatter = lambda threshold: (
        f"_my_output_file_with_threshold_{threshold:07.5f}.txt"
    )

    trainer.process(thresholds)

    for threshold in thresholds:
        # Should format unique filename for each threshold
        expected_path = (
            tmp_path / f"_my_output_file_with_threshold_{threshold:07.5f}.txt"
        )
        assert Path.exists(expected_path)


def test_process_with_compile_missing_compiler(
    thresholds, deterministic_training_data, tmp_path
):
    """Test lightgbm models are created at specified path."""

    training_data, observation_column, training_columns = deterministic_training_data

    # Do not provide compiler to training plugin
    trainer = TrainRainForestsModel(
        training_data, observation_column, training_columns, tmp_path
    )

    # Error when trying to use compile option
    with pytest.raises(ValueError):
        trainer.process(thresholds, compile=True)


def test_process_compile(thresholds, deterministic_training_data, tmp_path):
    """Test lightgbm models are created at specified path."""

    training_data, observation_column, training_columns = deterministic_training_data

    compiler = CompileRainForestsModel(parallel_comp=8)
    trainer = TrainRainForestsModel(
        training_data, observation_column, training_columns, tmp_path, compiler=compiler
    )

    trainer.process(thresholds, compile=True)

    for threshold in thresholds:
        expected_path = tmp_path / f"lgb_model-threshold_{threshold:04.2f}.txt"
        assert Path.exists(expected_path)
        # Should also produce a compiled file.
        assert Path.exists(expected_path.with_suffix(".so"))
