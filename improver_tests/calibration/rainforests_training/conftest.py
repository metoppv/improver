# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import sys
from pathlib import Path

import numpy as np
import pytest

from improver.calibration import lightgbm_package_available, treelite_packages_available

from ..rainforests_calibration.conftest import (
    generate_aligned_feature_cubes,
    generate_forecast_cubes,
    prepare_dummy_training_data,
)

ATTRIBUTES = {
    "title": "Test forecast",
    "source": "IMPROVER",
    "institution": "Australian Bureau of Meteorology",
}


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
def model_config(tmp_path):
    """Make a dummy model config object with valid file paths"""
    lead_times = np.array([24, 48], dtype=np.int32)
    thresholds = np.array([0.0000, 0.0001, 0.0010, 0.0100], dtype=np.float32)

    lightgbm_model_dir = tmp_path / "lightgbm_model_dir"
    treelite_model_dir = tmp_path / "treelite_model_dir"
    return {
        str(lead_time): {
            f"{threshold:06.4f}": {
                "lightgbm_model": f"{lightgbm_model_dir}/test_model_{lead_time:03d}H_{threshold:06.4f}.txt",  # noqa: E501
                "treelite_model": f"{treelite_model_dir}/test_model_{lead_time:03d}H_{threshold:06.4f}.so",  # noqa: E501
            }
            for threshold in thresholds
        }
        for lead_time in lead_times
    }


@pytest.fixture
def deterministic_training_data():
    """Make some dummy training data for one lead time"""

    deterministic_features = generate_aligned_feature_cubes(realizations=np.arange(1))
    deterministic_forecast = generate_forecast_cubes(realizations=np.arange(1))
    lead_time = 24
    training_data, _, observation_column, training_columns = (
        prepare_dummy_training_data(
            deterministic_features, deterministic_forecast, [lead_time]
        )
    )

    return str(lead_time), training_data, observation_column, training_columns


@pytest.fixture
def model_config_with_trained_models(model_config):
    """Return the RainForests model config, first performing the lightgbm training step
    so that the models are available for compiling with the compiler plugin."""
    lightgbm = pytest.importorskip("lightgbm")

    lead_times = [int(l) for l in model_config.keys()]

    deterministic_features = generate_aligned_feature_cubes(realizations=np.arange(1))
    deterministic_forecast = generate_forecast_cubes(realizations=np.arange(1))
    training_data, _, obs_column, train_columns = prepare_dummy_training_data(
        deterministic_features, deterministic_forecast, lead_times
    )

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
