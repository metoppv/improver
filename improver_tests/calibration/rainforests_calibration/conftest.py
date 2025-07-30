# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Fixtures for rainforests calibration."""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from iris import Constraint
from iris.analysis import MEAN, STD_DEV
from iris.cube import CubeList

from improver.calibration.rainforest_calibration import (
    ApplyRainForestsCalibrationLightGBM,
    ApplyRainForestsCalibrationTreelite,
)
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
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
def thresholds():
    return np.array([0.0000, 0.0001, 0.0010, 0.0100], dtype=np.float32)


@pytest.fixture
def lead_times():
    return np.array([24, 48], dtype=np.int32)


@pytest.fixture
def model_config(lead_times, thresholds, tmp_path):
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


def generate_forecast_cubes(realizations):
    rng = np.random.default_rng(0)
    data_shape = (len(realizations), 10, 10)
    return set_up_variable_cube(
        np.maximum(0, rng.normal(0.002, 0.001, data_shape)).astype(np.float32),
        name="lwe_thickness_of_precipitation_amount",
        units="m",
        frt=datetime(2017, 11, 10, 0, 0),
        time=datetime(2017, 11, 11, 0, 0),
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
        frt=datetime(2017, 11, 10, 0, 0),
        time=datetime(2017, 11, 11, 0, 0),
        realizations=realizations,
        attributes=ATTRIBUTES,
    )
    precipitation_accumulation_from_convection = set_up_variable_cube(
        np.maximum(0, rng.normal(0.001, 0.001, data_shape)).astype(np.float32),
        name="lwe_thickness_of_convective_precipitation_amount",
        units="m",
        frt=datetime(2017, 11, 10, 0, 0),
        time=datetime(2017, 11, 11, 0, 0),
        realizations=realizations,
        attributes=ATTRIBUTES,
    )
    precipitation_accumulation = set_up_variable_cube(
        np.maximum(0, rng.normal(0.002, 0.001, data_shape)).astype(np.float32),
        name="lwe_thickness_of_precipitation_amount",
        units="m",
        frt=datetime(2017, 11, 10, 0, 0),
        time=datetime(2017, 11, 11, 0, 0),
        realizations=realizations,
        attributes=ATTRIBUTES,
    )
    wind_speed = set_up_variable_cube(
        np.maximum(0, rng.normal(5, 5, data_shape)).astype(np.float32),
        name="wind_speed",
        units="m s-1",
        frt=datetime(2017, 11, 10, 0, 0),
        time=datetime(2017, 11, 11, 0, 0),
        realizations=realizations,
        attributes=ATTRIBUTES,
    )
    clearsky_solar_rad = set_up_variable_cube(
        np.maximum(0, rng.normal(5000000, 2000000, data_shape)).astype(np.float32),
        name="integral_of_surface_downwelling_shortwave_flux_in_air_assuming_clear_sky_wrt_time",
        units="W s m-2",
        frt=datetime(2017, 11, 10, 0, 0),
        time=datetime(2017, 11, 11, 0, 0),
        realizations=realizations,
        attributes=ATTRIBUTES,
    )
    # For clearsky_solar_rad, we want all realization values equal to simulate derived field.
    if realizations is not None:
        clearsky_solar_rad.data = np.broadcast_to(
            clearsky_solar_rad.data[0, :, :], data_shape
        )
    cubes = [
        cape,
        precipitation_accumulation_from_convection,
        precipitation_accumulation,
        wind_speed,
        clearsky_solar_rad,
    ]
    if len(realizations) > 1:
        precipitation_accumulation_mean = create_new_diagnostic_cube(
            name="ensemble_mean_lwe_thickness_of_precipitation_amount",
            units=precipitation_accumulation.units,
            template_cube=precipitation_accumulation,
            mandatory_attributes=generate_mandatory_attributes(
                [precipitation_accumulation]
            ),
            optional_attributes=precipitation_accumulation.attributes,
            data=np.broadcast_to(
                precipitation_accumulation.collapsed("realization", MEAN).data,
                precipitation_accumulation.data.shape,
            ),
        )
        precipitation_accumulation_std = create_new_diagnostic_cube(
            name="ensemble_std_lwe_thickness_of_precipitation_amount",
            units=precipitation_accumulation.units,
            template_cube=precipitation_accumulation,
            mandatory_attributes=generate_mandatory_attributes(
                [precipitation_accumulation]
            ),
            optional_attributes=precipitation_accumulation.attributes,
            data=np.broadcast_to(
                precipitation_accumulation.collapsed("realization", STD_DEV).data,
                precipitation_accumulation.data.shape,
            ),
        )
        cubes += [precipitation_accumulation_mean, precipitation_accumulation_std]
    return CubeList(cubes)


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


def prepare_dummy_training_data(features, forecast, lead_times):
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

    # add lead time column
    training_data["lead_time_hours"] = rng.choice(lead_times, len(training_data))

    return training_data, fcst_column, obs_column, train_columns


def train_models(training_data, obs_column, train_columns, thresholds, lead_times):
    """Train a model for each threshold."""
    import lightgbm

    tree_models = {}
    params = {"objective": "binary", "num_leaves": 5, "verbose": -1, "seed": 0}
    for lead_time in lead_times:
        for threshold in thresholds:
            curr_training_data = training_data.loc[
                training_data["lead_time_hours"] == lead_time
            ]
            data = lightgbm.Dataset(
                curr_training_data[train_columns],
                label=(curr_training_data[obs_column] >= threshold).astype(int),
            )
            booster = lightgbm.train(params, data, num_boost_round=10)
            tree_models[lead_time, threshold] = booster
    return tree_models


@pytest.fixture
def dummy_lightgbm_models(ensemble_features, ensemble_forecast, thresholds, lead_times):
    """Create sample lightgbm models for evaluating forecast probabilities."""
    training_data, fcst_column, obs_column, train_columns = prepare_dummy_training_data(
        ensemble_features, ensemble_forecast, lead_times
    )
    tree_models = train_models(
        training_data, obs_column, train_columns, thresholds, lead_times
    )

    return tree_models, lead_times, thresholds


def compile_models(lightgbm_models, lead_times, thresholds, tmp_path):
    """Compile lightgbm models."""
    import tl2cgen
    import treelite

    tree_models = {}
    for lead_time in lead_times:
        for threshold in thresholds:
            model = lightgbm_models[lead_time, threshold]
            treelite_model = treelite.Model.from_lightgbm(model)
            tl2cgen.export_lib(
                treelite_model,
                toolchain="gcc",
                libpath=str(tmp_path / "model.so"),
                verbose=False,
                params={"parallel_comp": 8, "quantize": 1},
            )
            predictor = tl2cgen.Predictor(
                str(tmp_path / "model.so"), verbose=True, nthread=1
            )
            tree_models[lead_time, threshold] = predictor
    return tree_models


@pytest.fixture
def dummy_treelite_models(dummy_lightgbm_models, tmp_path):
    """Create sample treelite models for evaluating forecast probabilities."""

    lightgbm_models, lead_times, thresholds = dummy_lightgbm_models
    tree_models = compile_models(lightgbm_models, lead_times, thresholds, tmp_path)
    return tree_models, lead_times, thresholds


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
        _ = pytest.importorskip("tl2cgen")
        return (
            ApplyRainForestsCalibrationTreelite,
            request.getfixturevalue("dummy_treelite_models"),
        )
    else:
        pytest.fail("unknown plugin type")


@pytest.fixture()
def lightgbm_model_files(dummy_lightgbm_models, model_config):
    """Write the lightgbm model files to the location specified in model config."""
    tree_models, lead_times, thresholds = dummy_lightgbm_models
    for lead_time in lead_times:
        for threshold in thresholds:
            model = tree_models[lead_time, threshold]
            model_path = model_config[str(lead_time)][f"{threshold:06.4f}"][
                "lightgbm_model"
            ]
            model_dir = os.path.dirname(model_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model.save_model(model_path)


@pytest.fixture
def dummy_lightgbm_models_deterministic(
    deterministic_features, deterministic_forecast, thresholds, lead_times
):
    """Create sample lightgbm models for evaluating forecast probabilities."""
    training_data, fcst_column, obs_column, train_columns = prepare_dummy_training_data(
        deterministic_features, deterministic_forecast, lead_times
    )
    tree_models = train_models(
        training_data, obs_column, train_columns, thresholds, lead_times
    )
    return tree_models, lead_times, thresholds


@pytest.fixture
def dummy_treelite_models_deterministic(dummy_lightgbm_models_deterministic, tmp_path):
    """Create sample treelite models for evaluating forecast probabilities."""

    lightgbm_models, lead_times, thresholds = dummy_lightgbm_models_deterministic
    tree_models = compile_models(lightgbm_models, lead_times, thresholds, tmp_path)
    return tree_models, lead_times, thresholds


@pytest.fixture(params=["lightgbm", "treelite"])
def plugin_and_dummy_models_deterministic(request):
    if request.param == "lightgbm":
        _ = pytest.importorskip("lightgbm")
        return (
            ApplyRainForestsCalibrationLightGBM,
            request.getfixturevalue("dummy_lightgbm_models_deterministic"),
        )
    elif request.param == "treelite":
        _ = pytest.importorskip("treelite")
        return (
            ApplyRainForestsCalibrationTreelite,
            request.getfixturevalue("dummy_treelite_models_deterministic"),
        )
    else:
        pytest.fail("unknown plugin type")


@pytest.fixture
def threshold_cube(thresholds):
    """Create sample threshold cube"""
    prob = np.array([1.0, 0.8, 0.2, 0.0])
    data = np.broadcast_to(prob[:, np.newaxis, np.newaxis], (len(thresholds), 10, 10))
    probability_cube = set_up_probability_cube(
        data.astype(np.float32),
        thresholds=thresholds,
        variable_name="lwe_thickness_of_precipitation_amount",
        threshold_units="m",
        frt=datetime(2017, 11, 10, 0, 0),
        time=datetime(2017, 11, 11, 0, 0),
        attributes=ATTRIBUTES,
        spp__relative_to_threshold="greater_than_or_equal_to",
    )
    threshold_cube = add_coordinate(
        probability_cube,
        coord_points=np.arange(5),
        coord_name="realization",
        coord_units="1",
        order=[1, 0, 2, 3],
    )
    return threshold_cube
