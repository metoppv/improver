# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import sys

import pytest

from improver.calibration import lightgbm_package_available, treelite_packages_available

from ..rainforests_calibration.conftest import (
    deterministic_features,
    deterministic_forecast,
    dummy_lightgbm_models,
    ensemble_features,
    ensemble_forecast,
    lead_times,
    prepare_dummy_training_data,
    thresholds,
)

_ = (
    deterministic_features,
    deterministic_forecast,
    dummy_lightgbm_models,
    ensemble_features,
    ensemble_forecast,
    lead_times,
    prepare_dummy_training_data,
    thresholds,
)

dummy_lightgbm_models = dummy_lightgbm_models


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
def deterministic_training_data(
    deterministic_features, deterministic_forecast, lead_times
):
    training_data, fcst_column, observation_column, training_columns = (
        prepare_dummy_training_data(
            deterministic_features, deterministic_forecast, lead_times
        )
    )

    # This data contains several lead times. Filter the data to one leadtime.
    lead_time = 24
    curr_training_data = training_data.loc[
        training_data["lead_time_hours"] == lead_time
    ]

    return curr_training_data, observation_column, training_columns


@pytest.fixture
def rainforests_model_files(dummy_lightgbm_models, tmp_path):
    """Export some LightGBM Boosters to file"""

    tree_models, lead_times, thresholds = dummy_lightgbm_models

    output_dir = tmp_path / "models"
    output_dir.mkdir(exist_ok=True)

    def saved_path(lead_time, threshold):
        path = output_dir / f"model_{lead_time:0}_{threshold:06.4f}.txt"
        tree_models[lead_time, threshold].save_model(path)
        return path

    return [saved_path(l, t) for l in lead_times for t in thresholds]
