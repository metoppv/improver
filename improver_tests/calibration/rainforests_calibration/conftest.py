# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Fixtures for rainforests calibration."""
import numpy as np
import pandas as pd
import pytest
from iris import Constraint
from iris.cube import CubeList

from improver.calibration.rainforest_calibration import (
    ApplyRainForestsCalibrationLightGBM,
    ApplyRainForestsCalibrationTreelite,
)
from improver.synthetic_data.set_up_test_cubes import (
    add_coordinate,
    set_up_probability_cube,
    set_up_variable_cube,
)

ATTRIBUTES = {
    "title": "Test forecast",
    "source": "IMPROVER",
    "institution": "Australian Bureau of Meteorology",
}


@pytest.fixture
def error_thresholds():
    return np.array(
        [-0.0010, -0.0001, 0.0000, 0.0001, 0.0010, 0.0100], dtype=np.float32
    )


@pytest.fixture
def model_config(error_thresholds):
    return {
        f"{threshold:06.4f}": {
            "lightgbm_model": f"lightgbm_model_dir/test_model_{threshold:06.4f}.txt",
            "treelite_model": f"treelite_model_dir/test_model_{threshold:06.4f}.so",
        }
        for threshold in error_thresholds
    }


def generate_forecast_cubes(realizations):
    rng = np.random.default_rng(0)
    data_shape = (len(realizations), 10, 10)
    return set_up_variable_cube(
        np.maximum(0, rng.normal(0.002, 0.001, data_shape)).astype(np.float32),
        name="lwe_thickness_of_precipitation_amount",
        units="m",
        realizations=realizations,
        attributes=ATTRIBUTES,
    )


def generate_aligned_feature_cubes(realizations):
    """Generate a set of aligned feature cubes consistent with forecast cube.

    Note: The clearsky_solar_rad field will contain copy over realization dimension.
    This is to simulate the resultant broadcasting that will occur after aligning
    feature variables."""
    data_shape = (len(realizations), 10, 10)
    rng = np.random.default_rng(0)
    cape = set_up_variable_cube(
        np.maximum(0, rng.normal(15, 5, data_shape)).astype(np.float32),
        name="cape",
        units="J kg-1",
        realizations=realizations,
        attributes=ATTRIBUTES,
    )
    precipitation_accumulation_from_convection = set_up_variable_cube(
        np.maximum(0, rng.normal(0.001, 0.001, data_shape)).astype(np.float32),
        name="lwe_thickness_of_convective_precipitation_amount",
        units="m",
        realizations=realizations,
        attributes=ATTRIBUTES,
    )
    precipitation_accumulation = set_up_variable_cube(
        np.maximum(0, rng.normal(0.002, 0.001, data_shape)).astype(np.float32),
        name="lwe_thickness_of_precipitation_amount",
        units="m",
        realizations=realizations,
        attributes=ATTRIBUTES,
    )
    wind_speed = set_up_variable_cube(
        np.maximum(0, rng.normal(5, 5, data_shape)).astype(np.float32),
        name="wind_speed",
        units="m s-1",
        realizations=realizations,
        attributes=ATTRIBUTES,
    )
    clearsky_solar_rad = set_up_variable_cube(
        np.maximum(0, rng.normal(5000000, 2000000, data_shape)).astype(np.float32),
        name="integral_of_surface_downwelling_shortwave_flux_in_air_assuming_clear_sky_wrt_time",
        units="W s m-2",
        realizations=realizations,
        attributes=ATTRIBUTES,
    )
    # For clearsky_solar_rad, we want all realization values equal to simulate derived field.
    if realizations is not None:
        clearsky_solar_rad.data = np.broadcast_to(
            clearsky_solar_rad.data[0, :, :], data_shape
        )
    return CubeList(
        [
            cape,
            precipitation_accumulation_from_convection,
            precipitation_accumulation,
            wind_speed,
            clearsky_solar_rad,
        ]
    )


@pytest.fixture
def ensemble_forecast():
    """Create ensemble forecast cube."""
    return generate_forecast_cubes(realizations=np.arange(5))


@pytest.fixture
def ensemble_features():
    """Create a set of aligned ensemble feature cube."""
    return generate_aligned_feature_cubes(realizations=np.arange(5))


@pytest.fixture
def deterministic_forecast():
    """Create deterministic forecast cube."""
    return generate_forecast_cubes(realizations=np.arange(1))


@pytest.fixture
def deterministic_features():
    """Create a set of aligned deterministic feature cubes."""
    return generate_aligned_feature_cubes(realizations=np.arange(1))


def prepare_dummy_training_data(features, forecast):
    """Create a dummy training set for tree-models."""
    # Set column names for reference in training
    fcst_column = forecast.name()
    obs_column = fcst_column + "_obs"
    train_columns = [cube.name() for cube in features]
    train_columns.sort()

    training_data = pd.DataFrame()
    # Initialise feature variables
    for cube in (*features, forecast):
        # Initialise forecast variable
        if cube.coords("realization"):
            training_data[cube.name()] = cube.extract(
                Constraint(realization=0)
            ).data.flatten()

    # mock y data so that it is correlated with predicted precipitation_accumulation
    # and other variables, with some noise
    non_target_columns = list(set(training_data.columns) - set(list(fcst_column)))
    rng = np.random.default_rng(0)
    training_data[obs_column] = (
        training_data[fcst_column] + training_data[non_target_columns].sum(axis=1) / 5
    )
    training_data[obs_column] += rng.normal(
        0, training_data[obs_column].std(), len(training_data)
    )

    # rescale to have same mean and standard deviation as prediction
    training_data[obs_column] = (
        training_data[obs_column] - training_data[obs_column].mean()
    ) / training_data[obs_column].std()
    training_data[obs_column] = (
        training_data[obs_column] * training_data[fcst_column].std()
        + training_data[fcst_column].mean()
    )
    return training_data, fcst_column, obs_column, train_columns


@pytest.fixture
def dummy_lightgbm_models(ensemble_features, ensemble_forecast, error_thresholds):
    """Create sample lightgbm models for evaluating forecast error probabilities."""
    import lightgbm

    training_data, fcst_column, obs_column, train_columns = prepare_dummy_training_data(
        ensemble_features, ensemble_forecast
    )
    # train a model for each threshold
    tree_models = []
    params = {"objective": "binary", "num_leaves": 5, "verbose": -1, "seed": 0}
    training_columns = train_columns
    for threshold in error_thresholds:
        data = lightgbm.Dataset(
            training_data[training_columns],
            label=(
                training_data[obs_column] - training_data[fcst_column] > threshold
            ).astype(int),
        )
        booster = lightgbm.train(params, data, num_boost_round=10)
        tree_models.append(booster)

    return tree_models, error_thresholds


@pytest.fixture
def dummy_treelite_models(dummy_lightgbm_models, tmp_path):
    """Create sample treelite models for evaluating forecast error probabilities."""
    import treelite
    import treelite_runtime

    lightgbm_models, error_thresholds = dummy_lightgbm_models
    tree_models = []
    for model in lightgbm_models:
        treelite_model = treelite.Model.from_lightgbm(model)
        treelite_model.export_lib(
            toolchain="gcc",
            libpath=str(tmp_path / "model.so"),
            verbose=False,
            params={"parallel_comp": 8, "quantize": 1},
        )
        predictor = treelite_runtime.Predictor(
            str(tmp_path / "model.so"), verbose=True, nthread=1
        )
        tree_models.append(predictor)

    return tree_models, error_thresholds


@pytest.fixture(params=["lightgbm", "treelite"])
def plugin_and_dummy_models(request):
    if request.param == "lightgbm":
        _ = pytest.importorskip("lightgbm")
        return (
            ApplyRainForestsCalibrationLightGBM,
            request.getfixturevalue("dummy_lightgbm_models"),
        )
    elif request.param == "treelite":
        _ = pytest.importorskip("treelite")
        return (
            ApplyRainForestsCalibrationTreelite,
            request.getfixturevalue("dummy_treelite_models"),
        )
    else:
        pytest.fail("unknown plugin type")


@pytest.fixture
def error_threshold_cube(error_thresholds):
    """Create sample error-threshold cube"""
    prob = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    data = np.broadcast_to(prob[:, np.newaxis, np.newaxis], (6, 10, 10))
    probability_cube = set_up_probability_cube(
        data.astype(np.float32),
        thresholds=error_thresholds,
        variable_name="forecast_error_of_lwe_thickness_of_precipitation_amount",
        threshold_units="m",
        attributes=ATTRIBUTES,
        spp__relative_to_threshold="above",
    )
    error_threshold_cube = add_coordinate(
        probability_cube,
        coord_points=np.arange(5),
        coord_name="realization",
        coord_units="1",
        order=[1, 0, 2, 3],
    )
    return error_threshold_cube
