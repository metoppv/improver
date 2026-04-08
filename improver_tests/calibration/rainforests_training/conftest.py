# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import sys
from pathlib import Path

import pytest

from improver.calibration import lightgbm_package_available, treelite_packages_available

from ..rainforests_calibration.conftest import (
    deterministic_features,
    deterministic_forecast,
    ensemble_features,
    ensemble_forecast,
    lead_times,
    model_config,
    prepare_dummy_training_data,
    thresholds,
)

_ = (
    deterministic_features,
    deterministic_forecast,
    ensemble_features,
    ensemble_forecast,
    lead_times,
    model_config,
    prepare_dummy_training_data,
    thresholds,
)


@pytest.fixture(params=[True, False])
def lightgbm_available(request, monkeypatch):
    """Make lightgbm module available or unavailable"""

    available = request.param and lightgbm_package_available()
    if not available:
        monkeypatch.setitem(sys.modules, "lightgbm", None)
    return available


@pytest.fixture(params=[True, False])
def treelite_available(request, monkeypatch):
    """Make treelite module available or unavailable"""

    available = request.param and treelite_packages_available()
    if not available:
        monkeypatch.setitem(sys.modules, "treelite", None)
    return available


@pytest.fixture
def deterministic_training_data(deterministic_features, deterministic_forecast):
    """Make some dummy training data for one lead time"""
    lead_time = 24
    training_data, _, observation_column, training_columns = (
        prepare_dummy_training_data(
            deterministic_features, deterministic_forecast, [lead_time]
        )
    )

    return training_data, observation_column, training_columns


@pytest.fixture
def model_config_with_trained_models(
    model_config, ensemble_features, ensemble_forecast, thresholds, lead_times
):
    pytest.importorskip("lightgbm")
    """Return the RainForests model config, first performing the lightgbm training step
    so that the models are available for compiling with the compiler plugin."""
    training_data, _, obs_column, train_columns = prepare_dummy_training_data(
        ensemble_features, ensemble_forecast, lead_times
    )

    lightgbm = pytest.importorskip("lightgbm")

    params = {"objective": "binary", "num_leaves": 5, "verbose": -1, "seed": 0}
    for lead_time, thresholds in model_config.items():
        for threshold in thresholds:
            curr_training_data = training_data.loc[
                training_data["lead_time_hours"] == int(lead_time)
            ]
            data = lightgbm.Dataset(
                curr_training_data[train_columns],
                label=(curr_training_data[obs_column] >= float(threshold)).astype(int),
            )
            booster = lightgbm.train(params, data, num_boost_round=10)
            model_path = model_config[lead_time][threshold]["lightgbm_model"]
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            booster.save_model(model_path)

    return model_config
